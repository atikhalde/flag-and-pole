"""
Symbol-level Alert Deduplication Ledger.

Prevents repeat Telegram alerts for the same setup. Two policies are supported,
selected by `deduplication.policy` in settings.yaml:

  * "cooldown"  (DEFAULT, recommended)
        `strict_never_repeat_symbol: true` + `repeat_cooldown_days: N`
        Warning signs show the ONCE-PER-LIFETIME behaviour: a symbol that alerted
        six months ago is silenced forever, so the monitorable universe only ever
        shrinks. Under "cooldown" a symbol is suppressed for N calendar days after
        its last alert and then becomes eligible again for a fresh, genuinely new
        setup — this keeps the anti-spam intent without permanently blinding the
        scanner.

  * "lifetime"
        `strict_never_repeat_symbol: true` + `repeat_cooldown_days: 0`
        Legacy behaviour: a symbol is alerted once and NEVER again, ever.

  * "per_trigger"
        `strict_never_repeat_symbol: false`
        Deduplicate on (symbol, trigger_type) instead of symbol alone.

Eligibility is determined by `is_eligible(symbol, trigger_type)`. The older
`is_sent(symbol)` is retained and deliberately keeps its strict lifetime
semantics so existing callers/tests are unaffected.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.market_hours import IST, now_ist

logger = logging.getLogger(__name__)


def normalize_trigger(trigger_type: Optional[str]) -> str:
    """Collapse trigger variants so they compare consistently in the ledger."""
    return str(trigger_type or "ALERT").strip().upper()


class AlertLedger:
    def __init__(
        self,
        path: str | Path,
        strict_never_repeat_symbol: bool = True,
        repeat_cooldown_days: int = 20,
    ) -> None:
        self.path = Path(path)
        self.alerted_symbols: Dict[str, Dict[str, Any]] = {}
        self.strict_never_repeat_symbol = bool(strict_never_repeat_symbol)
        self.repeat_cooldown_days = max(0, int(repeat_cooldown_days))
        self._load()

    # ------------------------------------------------------------------ load
    def _load(self) -> None:
        if not self.path.exists():
            self.alerted_symbols = {}
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to parse alert ledger %s (%s); starting fresh.", self.path, exc)
            self.alerted_symbols = {}
            return

        # New strict permanent schema
        stored = raw.get("alerted_symbols", {})
        if isinstance(stored, dict):
            for sym, meta in stored.items():
                norm = str(sym).strip().upper()
                if norm:
                    self.alerted_symbols[norm] = meta if isinstance(meta, dict) else {"meta": meta}

        # Auto-migrate any legacy "SYMBOL:TRIGGER_TYPE" keys so already-sent symbols never repeat
        legacy = raw.get("alerts", {})
        if isinstance(legacy, dict):
            for key, meta in legacy.items():
                sym = str(key).split(":")[0].strip().upper()
                if sym and sym not in self.alerted_symbols:
                    entry = dict(meta) if isinstance(meta, dict) else {}
                    entry.setdefault("stage", "LEGACY")
                    entry.setdefault("alert_type", str(key).split(":")[1] if ":" in str(key) else "ALERT")
                    self.alerted_symbols[sym] = entry

    # ----------------------------------------------------------- eligibility
    @property
    def policy_label(self) -> str:
        """Short human-readable description of the active dedup policy."""
        if not self.strict_never_repeat_symbol:
            return "per_trigger"
        if self.repeat_cooldown_days <= 0:
            return "lifetime"
        return f"cooldown_{self.repeat_cooldown_days}d"

    def is_sent(self, symbol: str) -> bool:
        """True if the symbol appears in the ledger AT ALL (strict lifetime semantics)."""
        return symbol.strip().upper() in self.alerted_symbols

    @staticmethod
    def _parse_timestamp(meta: Dict[str, Any]) -> Optional[datetime]:
        """Best-effort parse of a stored alert timestamp into an IST-aware datetime."""
        for key in ("timestamp_ist", "timestamp", "alerted_at"):
            raw = meta.get(key)
            if not raw:
                continue
            txt = str(raw).strip()
            # Tolerate trailing " IST" and ISO-8601 variants
            if txt.upper().endswith("IST"):
                txt = txt[:-3].strip()
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                try:
                    return datetime.strptime(txt, fmt).replace(tzinfo=IST)
                except ValueError:
                    continue
        return None

    def _last_alert_age_days(self, meta: Dict[str, Any], ref: datetime) -> Optional[float]:
        """Age in days of the most recent alert, or None when it cannot be dated."""
        ts = self._parse_timestamp(meta)
        if ts is None:
            return None
        return (ref - ts).total_seconds() / 86400.0

    def is_eligible(
        self,
        symbol: str,
        trigger_type: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> bool:
        """
        Return True when THIS symbol/trigger may be alerted right now.

        - "lifetime"     -> eligible only if the symbol has never alerted.
        - "cooldown"     -> eligible unless alerted within `repeat_cooldown_days`.
                            A record that cannot be dated is treated as NOT eligible
                            (fail-safe: stay quiet rather than spam).
        - "per_trigger"  -> eligible unless this exact (symbol, trigger_type) alerted.
        """
        sym = symbol.strip().upper()
        meta = self.alerted_symbols.get(sym)
        if meta is None:
            return True

        ref = now or now_ist()

        if not self.strict_never_repeat_symbol:
            # Deduplicate on (symbol, trigger_type) only.
            return normalize_trigger(meta.get("alert_type")) != normalize_trigger(trigger_type)

        if self.repeat_cooldown_days <= 0:
            return False  # legacy once-per-lifetime

        age = self._last_alert_age_days(meta, ref)
        if age is None:
            return False  # undated record -> cannot prove the cooldown elapsed
        return age >= self.repeat_cooldown_days

    # -------------------------------------------------------------- mutation
    def mark_sent(
        self,
        symbol: str,
        stage: str,
        alert_type: str,
        price: float,
        timestamp_ist: str,
        extra: Optional[Dict[str, Any]] = None,
        save_immediately: bool = True,
    ) -> None:
        norm = symbol.strip().upper()
        if not norm:
            return
        record: Dict[str, Any] = {
            "stage": stage,
            "alert_type": normalize_trigger(alert_type),
            "price": round(float(price), 2),
            "timestamp_ist": timestamp_ist,
        }
        prior = self.alerted_symbols.get(norm) or {}
        # Keep an audit trail of how many times this symbol has fired.
        try:
            seen = int(prior.get("alert_count", 1)) + 1 if prior else 1
        except (TypeError, ValueError):
            seen = 1
        record["alert_count"] = seen
        if extra:
            record.update(extra)
        self.alerted_symbols[norm] = record
        if save_immediately:
            self.save()

    def merge_from(self, other_path: str | Path) -> int:
        """
        Losslessly union symbols from another alert ledger file into this ledger.
        Preserves existing first-alert records intact and appends any newly alerted symbols.
        Returns the count of newly added symbols.
        """
        other = AlertLedger(other_path)
        added = 0
        for sym, meta in other.alerted_symbols.items():
            if sym not in self.alerted_symbols:
                self.alerted_symbols[sym] = meta
                added += 1
        return added

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "policy": self.policy_label,
            "total_alerted_symbols": len(self.alerted_symbols),
            "alerted_symbols": dict(sorted(self.alerted_symbols.items())),
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ledger_from_config(cfg: Dict[str, Any], state_dir: str | Path) -> AlertLedger:
    """Build an AlertLedger honouring the `deduplication` block in settings.yaml."""
    d_cfg = (cfg or {}).get("deduplication", {}) or {}
    return AlertLedger(
        Path(state_dir) / "sent_alerts.json",
        strict_never_repeat_symbol=bool(d_cfg.get("strict_never_repeat_symbol", True)),
        repeat_cooldown_days=int(d_cfg.get("repeat_cooldown_days", 20)),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Losslessly union-merge alert deduplication ledgers")
    parser.add_argument("--merge-into", required=True, help="Target ledger path (e.g. state/sent_alerts.json)")
    parser.add_argument("--from", dest="src_path", required=True, help="Incoming ledger path to merge from")
    args = parser.parse_args()

    target = AlertLedger(args.merge_into)
    added_count = target.merge_from(args.src_path)
    if added_count > 0 or not Path(args.merge_into).exists():
        target.save()
    print(f"AlertLedger union-merge complete: added={added_count}, total={len(target.alerted_symbols)}")
