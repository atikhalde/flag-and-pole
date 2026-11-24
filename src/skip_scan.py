"""
Once-per-session gate for Stage-1 EOD scans.

An external scheduler can dispatch the EOD workflow far more often than an EOD
scan needs — it was firing every 15 minutes, so ~2,400 symbols were re-downloaded
and the same guard-pause notice re-sent dozens of times a day.

This module decides whether an EOD scan is actually needed:
  * already completed for today's session  -> skip
  * inside the market session and we have not yet taken a *closed* scan -> skip
    (the post-close cron will take it once the day's candle is final)
  * otherwise -> proceed

It writes `needed=true|false` to `$GITHUB_OUTPUT` for the workflow to branch on,
and sends nothing to Telegram. Exit code is always 0 so a skip never fails CI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

from src.market_hours import is_nse_candle_incomplete_today, is_nse_trading_day, now_ist


def _load(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def decide(state_path: str | Path, now=None) -> Tuple[bool, str]:
    """
    Return (needed, reason).

    `needed=False` only when we can prove today's completed EOD scan already ran.
    """
    now = now or now_ist()
    today = now.date().isoformat()

    if not is_nse_trading_day(now.date()):
        return False, f"{today} is not an NSE trading day"

    state = _load(Path(state_path))
    scan_date = str(state.get("scan_date") or "")
    stripped = bool(state.get("dropped_incomplete_live_bar"))
    updated = str(state.get("updated_at_ist") or "")

    # A scan dated today that did NOT have to strip an intraday bar means it saw
    # the completed close. That is the one scan per session that matters.
    if scan_date == today and not stripped and not is_nse_candle_incomplete_today(now):
        return False, f"completed EOD scan already done for {today} (state updated {updated})"

    # Before the close, any scan would only be an interim read; the post-close
    # cron will take the authoritative one. Avoid burning the universe repeatedly.
    if is_nse_candle_incomplete_today(now):
        return False, (
            f"NSE session still open ({now.strftime('%H:%M')} IST < 15:30) — "
            "waiting for the post-close scan"
        )

    return True, "no completed EOD scan recorded for this session yet"


def _write_output(needed: bool, reason: str) -> None:
    out = os.environ.get("GITHUB_OUTPUT", "").strip()
    print(f"EOD scan needed: {needed} — {reason}")
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(f"needed={'true' if needed else 'false'}\n")
            fh.write(f"reason={reason}\n")
    summary = os.environ.get("GITHUB_STEP_SUMMARY", "").strip()
    if summary:
        icon = "▶" if needed else "⏭"
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(f"### {icon} EOD scan gate\n\n- Needed: **{needed}**\n- Reason: {reason}\n\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Decide whether a Stage-1 EOD scan is needed")
    ap.add_argument("--check", default="state/coiled_candidates.json", help="Path to coiled_candidates.json")
    args = ap.parse_args()
    needed, reason = decide(args.check)
    _write_output(needed, reason)
    sys.exit(0)


if __name__ == "__main__":
    main()
