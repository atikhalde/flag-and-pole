"""
Post-run verifier for CI.

Why this exists
---------------
A guard pause and a data-outage abort are both legitimate, non-crashing outcomes.
But they must never look like success in the Actions UI: an entire week of green
runs previously produced nothing at all.

This verifier reads the **run summary** emitted by the scanner (not just the
state file), because on an abort the state file is deliberately left UNTOUCHED —
so it still contains the last healthy scan. Inspecting only the state file would
therefore happily report "OK" for a run that fetched nothing.

Exit codes:
    0  healthy (0 candidates is fine; a guard pause is fine)
    1  run aborted on a data-integrity failure -> the job must go red
    2  summary/state missing or unparseable
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from src.run_visibility import write_job_summary


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def verify(summary_path: str | Path, state_path: str | Path) -> int:
    summary_p = Path(summary_path)
    state_p = Path(state_path)

    summary = _load_json(summary_p) if summary_p.exists() else None
    state = _load_json(state_p) if state_p.exists() else None

    # The authoritative signal is the run summary. Fall back to state only if
    # the scanner never got far enough to write one.
    signal = summary if summary is not None else state
    if signal is None:
        write_job_summary(
            "⛔ State check failed",
            f"Neither `{summary_p}` nor `{state_p}` exists or is valid JSON.\n",
        )
        print(f"::error::No readable run summary or state file ({summary_p}, {state_p}).")
        return 2

    if signal.get("aborted"):
        reason = signal.get("reason", "data_fetch_failed")
        fetched = signal.get("fetched")
        requested = signal.get("requested")
        detail = f" ({fetched}/{requested} symbols)" if fetched is not None else ""
        write_job_summary(
            "⛔ Stage-1 aborted — data integrity",
            f"- Reason: `{reason}`{detail}\n"
            "- **State was left untouched** and no alerts were evaluated.\n"
            "- This is a red run on purpose: a silent abort must not look green.\n",
        )
        print(f"::error::Stage-1 aborted ({reason}){detail}; state left untouched.")
        return 1

    candidates = (state or {}).get("candidates", []) if state else signal.get("candidates", [])
    paused = bool(signal.get("alerts_paused"))
    pause_reason = signal.get("pause_reason")
    scanned = signal.get("universe_scanned", (state or {}).get("universe_scanned", "n/a"))

    lines = [
        f"- Universe scanned: **{scanned}**",
        f"- Candidates saved: **{len(candidates or [])}**",
        f"- Alerts paused: **{paused}**" + (f" (`{pause_reason}`)" if pause_reason else ""),
    ]
    if paused:
        lines.append(
            "\n> The watchlist was still refreshed — Stage-2 monitoring stays current and "
            "resumes automatically when the guard clears."
        )
    write_job_summary("✅ State check passed", "\n".join(lines) + "\n")
    print(
        f"State OK: {len(candidates or [])} candidate(s), scanned={scanned}, "
        f"alerts_paused={paused}" + (f" ({pause_reason})" if pause_reason else "")
    )
    return 0


def verify_state(path: str | Path) -> int:
    """Back-compat helper: verify a state file alone."""
    return verify(summary_path="__missing_summary__", state_path=path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a Stage-1 scan after a CI run")
    parser.add_argument(
        "--summary",
        default="output/scan_summary.json",
        help="Path to the run summary JSON written by --summary-out",
    )
    parser.add_argument(
        "--path",
        default="state/coiled_candidates.json",
        help="Path to coiled_candidates.json",
    )
    args = parser.parse_args()
    sys.exit(verify(args.summary, args.path))


if __name__ == "__main__":
    main()
