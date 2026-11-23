"""
Run-visibility helpers.

Previously a paused market guard and a genuine "no setups found" day looked
identical: the GitHub Actions job went green with no indication that the
scanner had deliberately muted itself. That made an idle scanner impossible to
distinguish from a healthy one.

Two sinks are used:
  * the GitHub Actions job summary (when $GITHUB_STEP_SUMMARY is set), so the
    run page itself explains the pause;
  * an optional Telegram notice, so the user hears about it too.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def write_job_summary(title: str, body: str) -> bool:
    """Append markdown to the GitHub Actions job summary if available. Returns True on success."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY", "").strip()
    if not summary_path:
        return False
    try:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(f"## {title}\n\n{body}\n\n")
        return True
    except Exception as exc:  # pragma: no cover - filesystem edge case
        logger.warning("Could not write GitHub job summary: %s", exc)
        return False


def describe_pause(reason: str, info: Dict[str, Any]) -> str:
    """Build a human-readable one-liner explaining why a guard paused the scan."""
    if reason == "nifty_regime_paused":
        ret20 = info.get("ret20_pct")
        ret20_txt = f"{ret20:+.2f}%" if isinstance(ret20, (int, float)) else "N/A"
        return f"NIFTY 50 20-day return {ret20_txt} breached the regime guard threshold"
    if reason == "nse_breadth_paused":
        pct = info.get("breadth50_pct")
        thr = info.get("min_threshold_pct", 40.0)
        pct_txt = f"{pct:.1f}%" if isinstance(pct, (int, float)) else "N/A"
        return f"NSE Breadth50 {pct_txt} is below the {thr:.1f}% floor (weak broad market)"
    if reason == "data_fetch_failed":
        return (
            f"Only {info.get('fetched', 0)} of {info.get('requested', 0)} symbols could be fetched "
            f"({info.get('pct', 0.0):.1f}%) — treating this as a data outage and leaving state untouched"
        )
    return f"Scan paused ({reason})"


def publish_pause(
    reason: str,
    info: Dict[str, Any],
    notifier: Optional[Any] = None,
    enabled: bool = True,
) -> str:
    """
    Record a pause in the job summary and (optionally) announce it on Telegram.
    Returns the description used.
    """
    desc = describe_pause(reason, info)

    body = (
        f"⏸ **Scanner paused — no alerts were evaluated.**\n\n"
        f"- Reason: {desc}\n"
        f"- Detail: `{info}`\n"
    )
    write_job_summary("⏸ Scanner paused", body)

    logger.warning("Scanner paused: %s", desc)

    if enabled and notifier is not None:
        try:
            notifier.send_message(f"⏸ <b>Scanner paused</b>\n{desc}\n<i>No alerts were evaluated this run.</i>")
        except Exception as exc:  # never let notification failure mask the pause
            logger.warning("Could not send pause notice: %s", exc)

    return desc


def publish_abort(reason: str, info: Dict[str, Any]) -> None:
    """Record a run abort in the GitHub job summary."""
    write_job_summary(
        "⛔ Scanner aborted",
        f"**State was left untouched.**\n\n- Reason: {reason}\n- Detail: `{info}`\n",
    )
