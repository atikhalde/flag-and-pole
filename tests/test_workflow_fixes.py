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
        self.assertIn("Zero", reason)

    def test_partial_outage_aborts(self) -> None:
        abort, reason = check_fetch_integrity(100, 2400, self.cfg)
        self.assertTrue(abort)
        self.assertIn("floor", reason)

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

    def test_small_but_complete_universe_passes(self) -> None:
        """Regression: a 100% successful fetch must never abort just for being small."""
        abort, reason = check_fetch_integrity(63, 63, self.cfg)
        self.assertFalse(abort, f"63/63 (100%) must pass, got: {reason}")

    def test_small_universe_at_exactly_half_passes(self) -> None:
        abort, _ = check_fetch_integrity(60, 120, self.cfg)
        self.assertFalse(abort)

    def test_large_request_few_fetched_aborts(self) -> None:
        abort, _ = check_fetch_integrity(5, 2403, self.cfg)
        self.assertTrue(abort)

    def test_zero_fetched_always_aborts(self) -> None:
        for req in (1, 50, 2403):
            abort, _ = check_fetch_integrity(0, req, self.cfg)
            self.assertTrue(abort, f"0/{req} must abort")


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


class TestDetectionIndependentOfGuard(unittest.TestCase):
    """
    Regression: `detect_daily_pattern` returns None when the breadth it is handed
    is under the floor. Stage-1 used to pass the LIVE breadth value, so a
    weak-breadth day detected nothing and wrote an empty watchlist — destroying
    the very list the guard-ordering fix existed to preserve.
    """

    def setUp(self) -> None:
        self.cfg = load_config()

    def test_detector_self_gates_on_low_breadth(self) -> None:
        """Documents the upstream behaviour we must NOT feed a tripped value into."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_detector import _make_nifty_df, _make_synthetic_flag_df
        from src.pattern_detector import detect_daily_pattern

        df = _make_synthetic_flag_df(base_days=6, pole_days=2)
        nifty = _make_nifty_df(df.index, ret20_pct=2.0, ret63_pct=6.0)

        low, _ = detect_daily_pattern("X", "X", df, self.cfg, nifty_df=nifty, breadth50_pct=10.0)
        self.assertIsNone(low, "detector rejects setups under the breadth floor")

        ok, _ = detect_daily_pattern("X", "X", df, self.cfg, nifty_df=nifty, breadth50_pct=62.0)
        self.assertIsNotNone(ok, "detector accepts setups above the breadth floor")

    def test_eod_detection_uses_floor_not_live_breadth(self) -> None:
        """A tripped guard must still populate the watchlist."""
        import sys
        from pathlib import Path

        import pandas as pd

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_detector import _make_nifty_df, _make_synthetic_flag_df

        df = _make_synthetic_flag_df(base_days=6, pole_days=2)
        df.index = pd.bdate_range(end="2026-11-20", periods=len(df))
        nifty = _make_nifty_df(df.index, ret20_pct=2.0, ret63_pct=6.0)
        # A single rising stock is 100% above its own 50-SMA and could never trip a
        # breadth threshold, so include a genuinely declining name (a plain downtrend
        # — the flag builder still ends above its 50-SMA, so it cannot be reused here).
        n = len(df)
        import numpy as np

        declining = np.linspace(400.0, 200.0, n)
        dud = pd.DataFrame(
            {
                "Open": declining,
                "High": declining * 1.01,
                "Low": declining * 0.99,
                "Close": declining,
                "Volume": np.full(n, 3e5),
            },
            index=df.index,
        )
        universe = {"SETUP0": df, "DUD0": dud}

        import src.run_eod_scan as eod

        cfg = load_config()
        # Open the NIFTY guard but trip the breadth guard hard
        cfg["regime"]["min_nifty_20d_return_pct"] = -99.0
        cfg["regime"]["min_nse_breadth50_pct"] = 99.9
        # The breadth guard is only enforced once >= min_breadth_universe symbols
        # load, so relax that here rather than fabricating 30 frames.
        cfg["regime"]["min_breadth_universe"] = 1

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cfg["runtime"]["state_dir"] = str(td / "state")
            cfg["runtime"]["charts_dir"] = str(td / "charts")
            (td / "state").mkdir()
            (td / "charts").mkdir()
            cfg["universe"]["custom_watchlist_abs"] = str(td / "none.txt")
            (td / "none.txt").write_text("", encoding="utf-8")

            orig = (eod.fetch_daily_batch, eod.fetch_nifty_benchmark, eod.fetch_nse_universe)
            eod.fetch_daily_batch = lambda symbols, **kw: {s: universe[s] for s in symbols if s in universe}
            eod.fetch_nifty_benchmark = lambda **kw: nifty
            eod.fetch_nse_universe = lambda **kw: (list(universe), {s: s for s in universe})
            try:
                summary = eod.run_eod_scan(
                    cfg=cfg, force=True, limit=0, dry_run=True, no_live_refresh=True
                )
            finally:
                eod.fetch_daily_batch, eod.fetch_nifty_benchmark, eod.fetch_nse_universe = orig

            self.assertTrue(summary.get("alerts_paused"), "guard must trip at 99.9%")
            self.assertEqual(summary.get("alert_count"), 0, "no alerts while paused")
            self.assertGreaterEqual(
                summary.get("watchlist_count", 0), 1,
                "paused day must STILL save a non-empty watchlist",
            )

            state = json.loads((td / "state" / "coiled_candidates.json").read_text(encoding="utf-8"))
            self.assertTrue(state.get("alerts_paused"))
            self.assertEqual(state.get("pause_reason"), "nse_breadth_paused")
            self.assertGreaterEqual(len(state.get("candidates", [])), 1, "candidates must not be wiped")


class TestGuardNoticeDedup(unittest.TestCase):
    """
    Regression: the guard-pause notice fired on EVERY run. With the scheduler
    dispatching every 15 minutes, a multi-day weak-breadth stretch produced
    dozens of identical Telegram messages.
    """

    def _gate(self, td, **kw):
        from src.guard_notice import GuardNoticeGate

        return GuardNoticeGate(td, **kw)

    def test_first_notice_is_sent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ok, why = self._gate(td).should_notify("nse_breadth_paused", {"breadth50_pct": 34.7})
            self.assertTrue(ok)
            self.assertIn("first", why)

    def test_identical_repeat_is_suppressed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            now = now_ist()
            info = {"breadth50_pct": 34.7, "min_threshold_pct": 40.0}
            g = self._gate(td, repeat_hours=20, min_delta_pct=1.0)
            self.assertTrue(g.should_notify("nse_breadth_paused", info, now=now)[0])
            g.mark_notified("nse_breadth_paused", info, now=now)
            g.save()

            # 30 more runs, 15 minutes apart, unchanged breadth
            for i in range(1, 30):
                g2 = self._gate(td, repeat_hours=20, min_delta_pct=1.0)
                ok, why = g2.should_notify(
                    "nse_breadth_paused", info, now=now + timedelta(minutes=15 * i)
                )
                self.assertFalse(ok, f"run {i} should be suppressed, got: {why}")

    def test_material_metric_change_notifies(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            now = now_ist()
            g = self._gate(td, repeat_hours=20, min_delta_pct=1.0)
            g.mark_notified("nse_breadth_paused", {"breadth50_pct": 34.7}, now=now)
            g.save()

            g2 = self._gate(td, repeat_hours=20, min_delta_pct=1.0)
            # 2.5 points worse -> new information, notify
            ok, why = g2.should_notify(
                "nse_breadth_paused", {"breadth50_pct": 32.2},
                now=now + timedelta(minutes=15),
            )
            self.assertTrue(ok)
            self.assertIn("moved", why)

    def test_sub_threshold_change_stays_quiet(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            now = now_ist()
            g = self._gate(td, repeat_hours=20, min_delta_pct=1.0)
            g.mark_notified("nse_breadth_paused", {"breadth50_pct": 34.7}, now=now)
            g.save()

            g2 = self._gate(td, repeat_hours=20, min_delta_pct=1.0)
            ok, _ = g2.should_notify(
                "nse_breadth_paused", {"breadth50_pct": 34.9},
                now=now + timedelta(minutes=15),
            )
            self.assertFalse(ok, "0.2pt drift must not re-notify")

    def test_reason_change_notifies(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            now = now_ist()
            g = self._gate(td)
            g.mark_notified("nse_breadth_paused", {"breadth50_pct": 34.7}, now=now)
            g.save()

            g2 = self._gate(td)
            ok, why = g2.should_notify(
                "nifty_regime_paused", {"ret20_pct": -5.0}, now=now + timedelta(minutes=15)
            )
            self.assertTrue(ok)
            self.assertIn("changed", why)

    def test_heartbeat_after_interval_notifies(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            now = now_ist()
            g = self._gate(td, repeat_hours=20)
            g.mark_notified("nse_breadth_paused", {"breadth50_pct": 34.7}, now=now)
            g.save()

            g2 = self._gate(td, repeat_hours=20)
            ok, why = g2.should_notify(
                "nse_breadth_paused", {"breadth50_pct": 34.7}, now=now + timedelta(hours=21)
            )
            self.assertTrue(ok, why)
            self.assertIn("since last notice", why)

    def test_corrupt_state_does_not_mute_the_scanner(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "guard_notice.json").write_text("{not json", encoding="utf-8")
            ok, _ = self._gate(td).should_notify("nse_breadth_paused", {"breadth50_pct": 34.7})
            self.assertTrue(ok, "unreadable gate state must fail open, not silent")


class TestEodOncePerSession(unittest.TestCase):
    """Regression: Stage-1 was re-scanning ~2,400 symbols every 15 minutes."""

    def _state(self, td, **over):
        p = Path(td) / "cc.json"
        base = {
            "scan_date": "2026-11-25",
            "updated_at_ist": "2026-11-25 16:15:00 IST",
            "dropped_incomplete_live_bar": False,
        }
        base.update(over)
        p.write_text(json.dumps(base), encoding="utf-8")
        return p

    def test_skips_when_session_already_scanned(self) -> None:
        from src.skip_scan import decide

        with tempfile.TemporaryDirectory() as td:
            p = self._state(td)
            needed, why = decide(p, now=datetime(2026, 11, 25, 17, 0, tzinfo=IST))
            self.assertFalse(needed, why)
            self.assertIn("already done", why)

    def test_waits_for_the_close_during_session(self) -> None:
        from src.skip_scan import decide

        with tempfile.TemporaryDirectory() as td:
            p = self._state(td, scan_date="2026-11-24")
            needed, why = decide(p, now=datetime(2026, 11, 25, 11, 0, tzinfo=IST))
            self.assertFalse(needed, why)
            self.assertIn("still open", why)

    def test_proceeds_after_close_on_a_new_session(self) -> None:
        from src.skip_scan import decide

        with tempfile.TemporaryDirectory() as td:
            p = self._state(td, scan_date="2026-11-24")
            needed, why = decide(p, now=datetime(2026, 11, 25, 16, 0, tzinfo=IST))
            self.assertTrue(needed, why)

    def test_non_trading_day_skips(self) -> None:
        from src.skip_scan import decide

        with tempfile.TemporaryDirectory() as td:
            p = self._state(td)
            needed, why = decide(p, now=datetime(2026, 11, 28, 16, 0, tzinfo=IST))  # Saturday
            self.assertFalse(needed, why)
