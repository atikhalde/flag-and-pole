"""
Post-run state verifier for CI.

The scanner deliberately returns exit code 0 when it aborts on a data outage so
that the workflow's commit step still runs and state is preserved. That correct
behaviour is indistinguishable from success in the Actions UI, which is how an
entire week of "green" runs produced nothing.

This module converts that soft abort into a non-zero exit so the run goes red,
and annotates the job summary with what actually happened.

Exit codes:
    0  state is healthy (candidates may legitimately be 0)
    1  run aborted on a data-integrity failure
    2  state file missing or unparseable
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from src.run_visibility import write_job_summary


def verify_state(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        write_job_summary("⛔ State check failed", f"`{p}` does not exist.\n")
        print(f"::error::State file {p} is missing.")
        return 2

    try:
        data: Dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        write_job_summary("⛔ State check failed", f"`{p}` is not valid JSON: {exc}\n")
        print(f"::error::State file {p} is unparseable: {exc}")
        return 2

    candidates = data.get("candidates", []) or []
    paused = bool(data.get("alerts_paused"))
    reason = data.get("pause_reason")

    # An abort writes no state at all, but defensive handling costs nothing.
    if data.get("aborted"):
        write_job_summary(
            "⛔ Stage-1 aborted (data integrity)",
            "State was left untouched and no alerts were evaluated.\n",
        )
        print("::error::Stage-1 aborted due to a data fetch outage; state was left untouched.")
        return 1

    lines = [
        f"- Candidates saved: **{len(candidates)}**",
        f"- Alerts paused: **{paused}**" + (f" (`{reason}`)" if reason else ""),
        f"- Universe scanned: **{data.get('universe_scanned', 'n/a')}**",
    ]
    write_job_summary("✅ State check passed", "\n".join(lines) + "\n")
    print(
        f"State OK: {len(candidates)} candidate(s), alerts_paused={paused}"
        + (f" ({reason})" if reason else "")
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Stage-1 scan state after a CI run")
    parser.add_argument(
        "--path",
        default="state/coiled_candidates.json",
        help="Path to coiled_candidates.json",
    )
    args = parser.parse_args()
    sys.exit(verify_state(args.path))


if __name__ == "__main__":
    main()
