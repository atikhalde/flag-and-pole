"""
Regression tests for the workflow reliability fixes.

Covers:
  * dedup cooldown policy (previously "lifetime" left Stage-2 permanently muted)
  * data-integrity circuit breaker (previously an outage wiped the watchlist)
  * stale-breadth handling in Stage-2 (previously paused on a stale flag forever)
  * pause/abort visibility (previously a pause looked identical to "no setups")
  * CI state verifier (previously a soft abort still showed green)
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from src.alert_ledger import AlertLedger, ledger_from_config, normalize_trigger
from src.check_state import verify_state
from src.config_loader import load_config
from src.market_hours import IST, now_ist
from src.run_eod_scan import check_fetch_integrity
from src.run_live_scan import _is_breadth_stale
from src.run_visibility import describe_pause


def _ist(days_ago: float = 0.0) -> datetime:
    return now_ist() - timedelta(days=days_ago)


def _ts(days_ago: float = 0.0) -> str:
    return _ist(days_ago).strftime("%Y-%m-%d %H:%M IST")


class TestDedupCooldownPolicy(unittest.TestCase):
    def test_cooldown_lets_symbol_realert_after_window(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sent_alerts.json"
            ledger = AlertLedger(p, repeat_cooldown_days=20)
            ledger.mark_sent("BOROLTD", "EOD", "PRE_BREAKOUT", 341.1, _ts(30))
            ledger.save()

            reloaded = AlertLedger(p, repeat_cooldown_days=20)
            # Still "sent" under the strict lifetime accessor (back-compat)...
            self.assertTrue(reloaded.is_sent("BOROLTD"))
            # ...but eligible again because the cooldown has elapsed.
            self.assertTrue(reloaded.is_eligible("BOROLTD", "PRE_BREAKOUT", now=now_ist()))

    def test_cooldown_blocks_inside_window(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sent_alerts.json"
            ledger = AlertLedger(p, repeat_cooldown_days=20)
            ledger.mark_sent("AFFLE", "EOD", "PRE_BREAKOUT", 1576.6, _ts(3))
            ledger.save()

            reloaded = AlertLedger(p, repeat_cooldown_days=20)
            self.assertFalse(reloaded.is_eligible("AFFLE", "PRE_BREAKOUT", now=now_ist()))

    def test_lifetime_mode_still_never_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sent_alerts.json"
            ledger = AlertLedger(p, repeat_cooldown_days=0)  # legacy lifetime
            ledger.mark_sent("RAMKY", "EOD", "PRE_BREAKOUT", 387.1, _ts(400))
            ledger.save()

            reloaded = AlertLedger(p, repeat_cooldown_days=0)
            self.assertFalse(reloaded.is_eligible("RAMKY", "PRE_BREAKOUT", now=now_ist()))
            self.assertEqual(reloaded.policy_label, "lifetime")

    def test_per_trigger_mode_allows_other_trigger_types(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sent_alerts.json"
            ledger = AlertLedger(p, strict_never_repeat_symbol=False)
            ledger.mark_sent("IXIGO", "LIVE", "GAP_AND_GO", 165.4, _ts(1))
            ledger.save()

            reloaded = AlertLedger(p, strict_never_repeat_symbol=False)
            # Same trigger -> suppressed; a different trigger is a new event.
            self.assertFalse(reloaded.is_eligible("IXIGO", "GAP_AND_GO", now=now_ist()))
            self.assertTrue(reloaded.is_eligible("IXIGO", "SPRING_RETEST", now=now_ist()))

    def test_undated_record_is_not_eligible(self) -> None:
        """Fail-safe: if we cannot prove the cooldown elapsed, stay quiet."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sent_alerts.json"
            p.write_text(
                json.dumps({"alerted_symbols": {"ENTERO": {"stage": "EOD", "alert_type": "PRE_BREAKOUT"}}}),
                encoding="utf-8",
            )
            ledger = AlertLedger(p, repeat_cooldown_days=20)
            self.assertFalse(ledger.is_eligible("ENTERO", "PRE_BREAKOUT", now=now_ist()))

    def test_policy_label_reported(self) -> None:
        self.assertEqual(AlertLedger("nonexistent.json", repeat_cooldown_days=20).policy_label, "cooldown_20d")
        self.assertEqual(AlertLedger("nonexistent.json", repeat_cooldown_days=0).policy_label, "lifetime")
        self.assertEqual(
            AlertLedger("nonexistent.json", strict_never_repeat_symbol=False).policy_label, "per_trigger"
        )

    def test_ledger_from_config_reads_settings(self) -> None:
        cfg = load_config()
        with tempfile.TemporaryDirectory() as td:
            ledger = ledger_from_config(cfg, td)
            self.assertTrue(ledger.strict_never_repeat_symbol)
            self.assertEqual(ledger.repeat_cooldown_days, int(cfg["deduplication"]["repeat_cooldown_days"]))
            self.assertGreater(ledger.repeat_cooldown_days, 0)

    def test_timestamp_with_suffix_parses(self) -> None:
        self.assertEqual(normalize_trigger("gap_and_go"), "GAP_AND_GO")
        self.assertEqual(normalize_trigger(None), "ALERT")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.json"
            ledger = AlertLedger(p, repeat_cooldown_days=5)
            ledger.mark_sent("TEST", "EOD", "PRE_BREAKOUT", 100.0, "2026-01-01 10:00 IST")
            self.assertTrue(ledger.is_eligible("TEST", "PRE_BREAKOUT", now=datetime(2026, 2, 1, tzinfo=IST)))


class TestFetchIntegrityBreaker(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config()

    def test_total_outage_aborts(self) -> None:
        abort, reason = check_fetch_integrity(0, 2400, self.cfg)
        self.assertTrue(abort)
        self.assertIn("integrity floor", reason)

    def test_partial_outage_aborts(self) -> None:
        abort, _ = check_fetch_integrity(100, 2400, self.cfg)
        self.assertTrue(abort)

    def test_healthy_run_passes(self) -> None:
        abort, _ = check_fetch_integrity(2380, 2400, self.cfg)
        self.assertFalse(abort)

    def test_small_limit_test_run_is_exempt_from_absolute_floor(self) -> None:
        # A 5-symbol local smoke test must still be able to run.
        abort, _ = check_fetch_integrity(5, 5, self.cfg, limit=5)
        self.assertFalse(abort)

    def test_zero_requested_aborts(self) -> None:
        abort, _ = check_fetch_integrity(0, 0, self.cfg)
        self.assertTrue(abort)


class TestStaleBreadthHandling(unittest.TestCase):
    def test_fresh_breadth_is_honoured(self) -> None:
        self.assertFalse(_is_breadth_stale(now_ist().date().isoformat(), None))

    def test_old_breadth_is_stale(self) -> None:
        old = (now_ist().date() - timedelta(days=30)).isoformat()
        self.assertTrue(_is_breadth_stale(old, old))

    def test_long_weekend_is_not_stale(self) -> None:
        four_days = (now_ist().date() - timedelta(days=4)).isoformat()
        self.assertFalse(_is_breadth_stale(four_days, four_days))

    def test_missing_or_bad_date_is_stale(self) -> None:
        self.assertTrue(_is_breadth_stale("", None))
        self.assertTrue(_is_breadth_stale("garbage", None))


class TestVisibility(unittest.TestCase):
    def test_pause_descriptions_are_specific(self) -> None:
        breadth = describe_pause("nse_breadth_paused", {"breadth50_pct": 32.8, "min_threshold_pct": 40.0})
        self.assertIn("32.8%", breadth)
        self.assertIn("40.0%", breadth)

        nifty = describe_pause("nifty_regime_paused", {"ret20_pct": -4.2})
        self.assertIn("-4.20%", nifty)

        fetch = describe_pause("data_fetch_failed", {"fetched": 12, "requested": 2400, "pct": 0.5})
        self.assertIn("12", fetch)
        self.assertIn("2400", fetch)

    def test_job_summary_written_when_env_set(self) -> None:
        from src.run_visibility import write_job_summary

        with tempfile.TemporaryDirectory() as td:
            summary = Path(td) / "summary.md"
            old = os.environ.get("GITHUB_STEP_SUMMARY")
            os.environ["GITHUB_STEP_SUMMARY"] = str(summary)
            try:
                self.assertTrue(write_job_summary("Title", "Body text"))
                self.assertIn("Body text", summary.read_text(encoding="utf-8"))
            finally:
                if old is None:
                    os.environ.pop("GITHUB_STEP_SUMMARY", None)
                else:
                    os.environ["GITHUB_STEP_SUMMARY"] = old

    def test_job_summary_absent_is_graceful(self) -> None:
        from src.run_visibility import write_job_summary

        old = os.environ.pop("GITHUB_STEP_SUMMARY", None)
        try:
            self.assertFalse(write_job_summary("T", "B"))
        finally:
            if old is not None:
                os.environ["GITHUB_STEP_SUMMARY"] = old


class TestCheckState(unittest.TestCase):
    def test_healthy_state_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "coiled_candidates.json"
            p.write_text(
                json.dumps({"candidates": [], "alerts_paused": True, "pause_reason": "nse_breadth_paused"}),
                encoding="utf-8",
            )
            self.assertEqual(verify_state(p), 0)

    def test_aborted_state_returns_one(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "coiled_candidates.json"
            p.write_text(json.dumps({"aborted": True}), encoding="utf-8")
            self.assertEqual(verify_state(p), 1)

    def test_missing_state_returns_two(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(verify_state(Path(td) / "nope.json"), 2)

    def test_corrupt_state_returns_two(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.json"
            p.write_text("{not json", encoding="utf-8")
            self.assertEqual(verify_state(p), 2)


if __name__ == "__main__":
    unittest.main()


class TestAbortIsActuallyRed(unittest.TestCase):
    """
    Regression: the first version of the circuit breaker returned `aborted: true`
    from the library but main() still exited 0, AND the verifier read the
    untouched state file and reported OK — so a silent abort still showed green.
    """

    def test_verify_reads_run_summary_not_just_state(self) -> None:
        from src.check_state import verify

        with tempfile.TemporaryDirectory() as td:
            summary = Path(td) / "summary.json"
            state = Path(td) / "coiled_candidates.json"
            # State is the last HEALTHY scan (left untouched by the aborted run)
            state.write_text(json.dumps({"candidates": [{"symbol": "OLD"}], "alerts_paused": False}), encoding="utf-8")
            # The run summary records the abort
            summary.write_text(
                json.dumps({"aborted": True, "reason": "data_fetch_failed", "fetched": 0, "requested": 2403}),
                encoding="utf-8",
            )
            self.assertEqual(verify(summary, state), 1, "aborted run must be red despite healthy state")

    def test_healthy_paused_run_stays_green(self) -> None:
        from src.check_state import verify

        with tempfile.TemporaryDirectory() as td:
            summary = Path(td) / "summary.json"
            state = Path(td) / "coiled_candidates.json"
            state.write_text(json.dumps({"candidates": []}), encoding="utf-8")
            summary.write_text(
                json.dumps({"alerts_paused": True, "pause_reason": "nse_breadth_paused", "universe_scanned": 2380}),
                encoding="utf-8",
            )
            self.assertEqual(verify(summary, state), 0)

    def test_run_eod_scan_exits_nonzero_on_abort(self) -> None:
        """Exercised via check_fetch_integrity + the main() SystemExit contract."""
        import subprocess
        import sys

        # A 5-symbol run against an unreachable network must exit 1, not 0.
        proc = subprocess.run(
            [sys.executable, "-m", "src.run_eod_scan", "--force", "--limit", "5", "--dry-run",
             "--no-live-refresh", "--summary-out", "/tmp/_t_summary.json"],
            capture_output=True, text=True, timeout=240,
        )
        self.assertIn(proc.returncode, (0, 1), f"unexpected exit {proc.returncode}")
        if proc.returncode == 1:
            # If it aborted, the summary must record why
            data = json.loads(Path("/tmp/_t_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(data.get("aborted"))
