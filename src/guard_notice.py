"""
Guard-pause notification gate.

Why this exists
---------------
The guard-pause notice was sent on *every* run. When the scanner is dispatched on
a short cadence (the external scheduler fires Stage-1 every 15 minutes), a market
that stays under the breadth floor for a few days produced an identical Telegram
message dozens of times — the user saw the same alert repeatedly and reasonably
read it as a bug.

A pause is one ongoing *state*, not a stream of events. This gate collapses it:
the notice is re-sent only when it carries new information —

  1. the pause reason changed (e.g. breadth -> NIFTY sell-off),
  2. the underlying metric moved meaningfully (>= `min_delta_pct`), or
  3. `repeat_hours` have elapsed (a low-frequency heartbeat that proves the
     scanner is still alive and still paused).

Everything else is suppressed, so the first notice of a pause is immediate and
subsequent runs are silent.

The state file MUST be committed by the workflow — GitHub Actions checks out a
fresh tree on every run, so an uncommitted gate would forget it had already
notified and defeat itself.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.market_hours import IST, now_ist

logger = logging.getLogger(__name__)

DEFAULT_FILENAME = "guard_notice.json"


def _metric_for(reason: str, info: Dict[str, Any]) -> Tuple[str, Optional[float]]:
    """Return (metric_name, value) used to decide whether a pause materially changed."""
    if reason == "nse_breadth_paused":
        v = info.get("breadth50_pct")
        return "breadth50_pct", float(v) if isinstance(v, (int, float)) else None
    if reason == "nifty_regime_paused":
        v = info.get("ret20_pct")
        return "ret20_pct", float(v) if isinstance(v, (int, float)) else None
    return "none", None


class GuardNoticeGate:
    """Persistent, idempotent gate around guard-pause notifications."""

    def __init__(
        self,
        state_dir: str | Path,
        repeat_hours: float = 20.0,
        min_delta_pct: float = 1.0,
    ) -> None:
        self.path = Path(state_dir) / DEFAULT_FILENAME
        self.repeat_hours = float(repeat_hours)
        self.min_delta_pct = float(min_delta_pct)
        self._data: Dict[str, Any] = self._load()

    # ------------------------------------------------------------------ io
    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except Exception as exc:
            logger.warning("Could not read guard-notice state (%s); treating as fresh.", exc)
            return {}

    def _parse_last(self) -> Optional[datetime]:
        raw = self._data.get("notified_at_ist")
        if not raw:
            return None
        txt = str(raw).strip()
        if txt.upper().endswith("IST"):
            txt = txt[:-3].strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(txt, fmt).replace(tzinfo=IST)
            except ValueError:
                continue
        return None

    # ---------------------------------------------------------- decision
    def should_notify(
        self,
        reason: str,
        info: Dict[str, Any],
        now: Optional[datetime] = None,
    ) -> Tuple[bool, str]:
        """
        Decide whether to send a pause notice now.

        Returns (notify, human_reason). `human_reason` explains the decision so it
        can be logged and shown in the job summary.
        """
        now = now or now_ist()
        last_reason = self._data.get("reason")
        metric_name, metric_value = _metric_for(reason, info)

        # 1. Never notified before (or state was lost) -> notify.
        if not self._data:
            return True, "first pause notice"

        # 2. Reason changed -> notify.
        if last_reason and last_reason != reason:
            return True, f"pause reason changed ({last_reason} -> {reason})"

        last_at = self._parse_last()

        # 3. Metric moved materially -> notify.
        last_metric = self._data.get("metric_value")
        if (
            metric_value is not None
            and isinstance(last_metric, (int, float))
            and abs(metric_value - float(last_metric)) >= self.min_delta_pct
        ):
            return True, (
                f"{metric_name} moved {last_metric:.1f} -> {metric_value:.1f} "
                f"(>= {self.min_delta_pct:.1f})"
            )

        # 4. Heartbeat interval elapsed -> notify.
        if last_at is not None:
            elapsed_h = (now - last_at).total_seconds() / 3600.0
            if elapsed_h >= self.repeat_hours:
                return True, f"{elapsed_h:.1f}h since last notice (>= {self.repeat_hours:.0f}h)"
            return False, f"already notified {elapsed_h:.1f}h ago; unchanged — suppressed"

        # Undated record -> treat as not-yet-notified so we can never end up muted.
        return True, "previous notice had no readable timestamp"

    def mark_notified(
        self,
        reason: str,
        info: Dict[str, Any],
        now: Optional[datetime] = None,
    ) -> None:
        now = now or now_ist()
        metric_name, metric_value = _metric_for(reason, info)
        self._data = {
            "reason": reason,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "notified_at_ist": now.strftime("%Y-%m-%d %H:%M:%S IST"),
            "message": self._summarise(reason, info),
        }

    @staticmethod
    def _summarise(reason: str, info: Dict[str, Any]) -> str:
        if reason == "nse_breadth_paused":
            v = info.get("breadth50_pct")
            thr = info.get("min_threshold_pct", 40.0)
            v_txt = f"{v:.1f}%" if isinstance(v, (int, float)) else "n/a"
            return f"{v_txt} below {thr:.1f}%"
        if reason == "nifty_regime_paused":
            v = info.get("ret20_pct")
            return f"20d {v:+.2f}%" if isinstance(v, (int, float)) else "regime breach"
        return reason

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    @property
    def last_summary(self) -> str:
        return str(self._data.get("message") or "")
