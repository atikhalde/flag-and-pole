"""Unit tests for the pure Daily-TF <=20-day V2.1 Core + Coil High-Tight Flag & Backtest Scanner."""

from __future__ import annotations

import unittest
from datetime import datetime

import numpy as np
import pandas as pd

from src.config_loader import load_config
from src.live_triggers import evaluate_live_daily_triggers
from src.market_hours import IST
from src.pattern_detector import (
    compute_breadth50_series,
    compute_nse_breadth50,
    detect_daily_pattern,
)


def _make_synthetic_flag_df(
    base_days: int = 8,
    pole_days: int = 2,
    uptrend: bool = True,
    pole_top: float = 328.0,
    close_at_bottom: bool = False,
    tight_micro_box: bool = False,
    deep_retrace_base: bool = False,
    pole_rvol_mult: float = 1.0,
) -> pd.DataFrame:
    """Create a realistic synthetic Daily OHLCV High-Tight Flag series with 210+ bars for 150/200 SMA."""
    rng = np.random.default_rng(42)
    n_pre = 210
    dates = pd.bdate_range("2025-11-01", periods=n_pre + pole_days + base_days)

    rows = []
    # Smooth Stage-2 uptrend (175 -> 248) or downtrend (470 -> 248 below 150/200 SMA)
    start_px = 175.0 if uptrend else 470.0
    end_px = 248.0
    trend = np.linspace(start_px, end_px, n_pre)
    for i in range(n_pre):
        base_p = float(trend[i])
        o = base_p + rng.uniform(-2.0, 2.0)
        c = base_p + rng.uniform(-2.5, 2.5)
        h = max(o, c) + rng.uniform(1.5, 4.0)
        l = min(o, c) - rng.uniform(1.5, 4.0)
        v = rng.uniform(120_000, 220_000)
        rows.append((o, h, l, c, v))

    # Breakout Pole (~30% gain over pre-pole close ~248 -> 322-328, ~21x institutional RVOL * mult)
    rows.append((255.0, 290.0, 252.0, 288.0, 3_600_000.0 * pole_rvol_mult))
    if pole_days >= 2:
        rows.append((292.0, pole_top, 286.0, min(312.0, pole_top - 4.0), 2_800_000.0 * pole_rvol_mult))
    if pole_days >= 3:
        rows.append((314.0, pole_top + 6.0, 310.0, pole_top + 2.0, 2_200_000.0 * pole_rvol_mult))

    # High-Tight Flag Shelf
    for j in range(base_days):
        tight = 1.0 - (j / max(base_days, 1)) * 0.45
        if tight_micro_box:
            # Ultra-narrow <3.5% micro-box stop-hunt trap -> must be rejected by V2.1 box_w >= 5.5%!
            o, h, l, c = 313.0, 316.0, 309.0, 314.5
        elif deep_retrace_base:
            # Drops to ~288 (only ~50% upper hold of 244..328 pole -> rejected by V2.1 upper_hold >= 0.65!)
            o, h, l, c = 292.0, 305.0, 286.0, 290.0 if j == 0 else 298.0
        elif j == base_days - 1 and close_at_bottom:
            o, h, l, c = 310.0, 312.0, 302.0, 303.0  # bottom 5% of box -> falling knife!
        elif j == base_days - 1:
            o, h, l, c = 313.0, 319.5, 311.5, 317.8  # coiled in upper 75% of box!
        else:
            o = 311.0 + rng.uniform(-2.0, 2.0) * tight
            c = 313.0 + rng.uniform(-2.0, 2.0) * tight
            h = max(o, c) + (9.5 if j == 0 else 4.5) * tight
            l = min(o, c) - (9.5 if j == 0 else 3.5) * tight
        v = 550_000.0 * (0.72 ** j) + 120_000.0
        rows.append((o, h, l, c, v))

    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=dates)


def _make_nifty_df(dates: pd.DatetimeIndex, ret20_pct: float = 2.0, ret63_pct: float = 5.0) -> pd.DataFrame:
    """Create a synthetic NIFTY 50 series with desired 20-day and 63-day returns at the final bar."""
    n = len(dates)
    closes = np.full(n, 24000.0, dtype=float)
    if n >= 70:
        c_end = 24000.0 * (1.0 + ret63_pct / 100.0)
        c_20ago = c_end / (1.0 + ret20_pct / 100.0)
        closes[-64:-20] = np.linspace(24000.0, c_20ago, 44)
        closes[-21:] = np.linspace(c_20ago, c_end, 21)
    elif n >= 25:
        closes[-21:] = np.linspace(24000.0, 24000.0 * (1.0 + ret20_pct / 100.0), 21)
    return pd.DataFrame(
        {"Open": closes, "High": closes * 1.005, "Low": closes * 0.995, "Close": closes, "Volume": np.full(n, 1e8)},
        index=dates,
    )


class TestDailyFlagScannerV21(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config()

    def test_catches_clean_v21_within_20_day_flag(self) -> None:
        df = _make_synthetic_flag_df(base_days=8, pole_days=2)  # 10 total days <= 20
        nifty = _make_nifty_df(df.index, ret20_pct=1.8, ret63_pct=4.0)
        setup, vp = detect_daily_pattern("SYNTH", "Synthetic Ltd", df, self.cfg, nifty_df=nifty, breadth50_pct=62.0)
        self.assertIsNotNone(setup)
        self.assertIsNotNone(vp)
        assert setup is not None
        self.assertLessEqual(setup.total_pattern_days, 20)
        self.assertGreaterEqual(setup.total_score, 78.0)
        self.assertEqual(setup.grade, "A+")
        self.assertTrue(setup.above_150sma)
        self.assertTrue(setup.above_200sma)
        self.assertTrue(setup.sma50_rising)
        self.assertGreaterEqual(setup.rs63_vs_nifty, 15.0)
        self.assertGreaterEqual(setup.pct_from_52w_high, -15.0)
        self.assertGreaterEqual(setup.shelf_wick_box_pct, 5.5)
        self.assertGreaterEqual(setup.upper_pole_hold_pct, 65.0)
        self.assertGreaterEqual(setup.sig_pos_in_box, 0.20)
        self.assertEqual(setup.nse_breadth50_pct, 62.0)

        # Verify V2.1 Option A Trade Plan invariants
        self.assertEqual(setup.sl_mode, "daily_close")
        self.assertLess(setup.entry_price, setup.breakout_level)
        self.assertLessEqual(setup.entry_zone_high, setup.breakout_level)
        self.assertLess(setup.invalidation_stop, setup.entry_price)
        self.assertLessEqual(setup.hard_stop_gtt, setup.invalidation_stop)
        self.assertGreater(setup.risk_pct, 3.5)
        self.assertLessEqual(setup.risk_pct, 6.7)
        self.assertLess(setup.entry_price, setup.tp1)
        self.assertLess(setup.tp1, setup.tp2)
        self.assertLess(setup.tp2, setup.tp3)

    def test_rejects_pattern_older_than_20_days(self) -> None:
        df = _make_synthetic_flag_df(base_days=22, pole_days=2)  # 24 total days > 20!
        setup, _ = detect_daily_pattern("STALE", "Stale Base Ltd", df, self.cfg)
        self.assertIsNone(setup)

    def test_v21_core_and_coil_gates_reject_all_traps(self) -> None:
        df_ok = _make_synthetic_flag_df()
        nifty_ok = _make_nifty_df(df_ok.index, ret20_pct=1.5, ret63_pct=4.0)

        # 1. Rejects Stage-4 downtrend below 150/200 SMA
        s_bear, _ = detect_daily_pattern("BEAR", "Bear Ltd", _make_synthetic_flag_df(uptrend=False), self.cfg, nifty_df=nifty_ok)
        self.assertIsNone(s_bear)

        # 2. Rejects falling knife closing in bottom 20% of flag box
        s_bot, _ = detect_daily_pattern("KNIFE", "Knife Ltd", _make_synthetic_flag_df(close_at_bottom=True), self.cfg, nifty_df=nifty_ok)
        self.assertIsNone(s_bot)

        # 3. Rejects parabolic blow-off pole (>42% high gain)
        s_blow, _ = detect_daily_pattern("BLOW", "Blowoff Ltd", _make_synthetic_flag_df(pole_top=375.0), self.cfg, nifty_df=nifty_ok)
        self.assertIsNone(s_blow)

        # 4. Rejects entries during NIFTY 50 sharp sell-off (20d < -3.0%)
        nifty_crash = _make_nifty_df(df_ok.index, ret20_pct=-4.5, ret63_pct=-2.0)
        s_crash, _ = detect_daily_pattern("SYNTH", "Synthetic Ltd", df_ok, self.cfg, nifty_df=nifty_crash, breadth50_pct=55.0)
        self.assertIsNone(s_crash)

        # 5. Rejects weak NSE Market Breadth (< 40% stocks above 50-SMA)
        s_weak_br, _ = detect_daily_pattern("SYNTH", "Synthetic Ltd", df_ok, self.cfg, nifty_df=nifty_ok, breadth50_pct=34.0)
        self.assertIsNone(s_weak_br)

        # 6. Rejects 3-month RS laggards (stock lags or fails +15% RS vs NIFTY)
        nifty_hot = _make_nifty_df(df_ok.index, ret20_pct=2.0, ret63_pct=45.0)
        s_lag, _ = detect_daily_pattern("SYNTH", "Synthetic Ltd", df_ok, self.cfg, nifty_df=nifty_hot, breadth50_pct=60.0)
        self.assertIsNone(s_lag)

        # 7. Rejects ultra-tight micro-box stop-hunt trap (< 5.5% wick box)
        s_micro, _ = detect_daily_pattern("MICRO", "MicroBox Ltd", _make_synthetic_flag_df(tight_micro_box=True), self.cfg, nifty_df=nifty_ok)
        self.assertIsNone(s_micro)

        # 8. Rejects deep flag retracement (upper_hold < 0.65)
        s_deep, _ = detect_daily_pattern("DEEP", "DeepFlag Ltd", _make_synthetic_flag_df(deep_retrace_base=True), self.cfg, nifty_df=nifty_ok)
        self.assertIsNone(s_deep)

        # 9. Soft climax RVOL penalty (>60x trims Pillar 1 by 8.5 pts without hard-killing elite setups)
        s_norm, _ = detect_daily_pattern("NORM", "Norm Ltd", _make_synthetic_flag_df(pole_rvol_mult=1.0), self.cfg, nifty_df=nifty_ok, breadth50_pct=55.0)
        s_climax, _ = detect_daily_pattern("CLIMAX", "Climax Ltd", _make_synthetic_flag_df(pole_rvol_mult=4.0), self.cfg, nifty_df=nifty_ok, breadth50_pct=55.0)
        self.assertIsNotNone(s_norm)
        self.assertIsNotNone(s_climax)
        assert s_norm is not None and s_climax is not None
        self.assertGreater(s_climax.pole_rvol, 60.0)
        self.assertAlmostEqual(s_norm.total_score - s_climax.total_score, 8.5, places=1)

    def test_live_and_historical_nse_breadth50_engine(self) -> None:
        dates = pd.bdate_range("2026-01-01", periods=70)
        # Build 40 synthetic stocks: 15 uptrending (> 50-SMA) + 25 downtrending (< 50-SMA) => 37.5% breadth!
        daily_map = {}
        for i in range(40):
            trend = np.linspace(100.0, 140.0, 70) if i < 15 else np.linspace(140.0, 100.0, 70)
            daily_map[f"STK{i:02d}"] = pd.DataFrame(
                {"Open": trend, "High": trend * 1.01, "Low": trend * 0.99, "Close": trend, "Volume": np.full(70, 1e5)},
                index=dates,
            )
        info = compute_nse_breadth50(daily_map, cfg=self.cfg)
        self.assertTrue(info["available"])
        self.assertTrue(info["enforced"])
        self.assertAlmostEqual(info["breadth50_pct"], 37.5, places=1)
        self.assertFalse(info["breadth_ok"])

        series = compute_breadth50_series(daily_map, min_stocks=30)
        self.assertIsNotNone(series)
        assert series is not None
        self.assertAlmostEqual(float(series.iloc[-1]), 37.5, places=1)

    def test_live_gap_and_go_trigger_fires(self) -> None:
        df_pre = _make_synthetic_flag_df(base_days=8, pole_days=2)
        setup, _ = detect_daily_pattern("SYNTH", "Synthetic Ltd", df_pre, self.cfg)
        self.assertIsNotNone(setup)
        assert setup is not None

        next_day = df_pre.index[-1] + pd.Timedelta(days=1)
        bo_row = pd.DataFrame(
            {"Open": [324.0], "High": [349.5], "Low": [321.5], "Close": [344.0], "Volume": [4_800_000.0]},
            index=[next_day],
        )
        df_live = pd.concat([df_pre, bo_row])
        sim_dt = datetime(next_day.year, next_day.month, next_day.day, 15, 0, tzinfo=IST)
        events = evaluate_live_daily_triggers(setup.to_dict(), df_live, self.cfg, as_of_dt=sim_dt)
        self.assertTrue(len(events) >= 1)
        self.assertEqual(events[0].trigger_type, "GAP_AND_GO")
        self.assertEqual(events[0].sl_mode, "daily_close")
        self.assertEqual(events[0].invalidation_stop, setup.invalidation_stop)
        self.assertEqual(events[0].hard_stop_gtt, setup.hard_stop_gtt)
        self.assertLessEqual(events[0].day_of_pattern, 21)

    def test_multi_bot_telegram_notifier_and_v21_captions(self) -> None:
        import os
        from src.telegram_notifier import TelegramNotifier

        tn = TelegramNotifier(
            bot_token="tok1",
            chat_id="chat1",
            enabled=True,
            extra_targets=[("tok2", "chat2"), ("tok1", "chat1")],
        )
        self.assertTrue(tn.enabled)
        self.assertEqual(tn.targets, [("tok1", "chat1"), ("tok2", "chat2")])

        df = _make_synthetic_flag_df()
        setup, _ = detect_daily_pattern("SYNTH", "Synthetic Industries Ltd", df, self.cfg, breadth50_pct=58.0)
        assert setup is not None
        cap = tn.format_eod_setup_caption(setup.to_dict())
        self.assertLess(len(cap), 1020)
        self.assertIn("V2.1 PRE-BREAKOUT", cap)
        self.assertIn("Close SL (3:20 PM)", cap)
        self.assertIn("Emergency GTT", cap)

        digest = tn.format_eod_digest(
            "2026-11-16",
            2400,
            [setup.to_dict()],
            nifty_regime={"available": True, "ret20_pct": 1.5, "above_50sma": True, "nse_breadth50_pct": 58.0},
        )
        self.assertIn("NSE Breadth50", digest)
        self.assertIn("58.0%", digest)

        prev = {k: os.environ.get(k) for k in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "TELEGRAM_BOT_TOKEN_2", "TELEGRAM_CHAT_ID_2")}
        try:
            os.environ["TELEGRAM_BOT_TOKEN"] = "t1"
            os.environ["TELEGRAM_CHAT_ID"] = "c1"
            os.environ["TELEGRAM_BOT_TOKEN_2"] = "t2"
            os.environ["TELEGRAM_CHAT_ID_2"] = "c2"
            cfg2 = load_config()
            self.assertEqual(cfg2["runtime"]["telegram_extra_targets"], [("t2", "c2")])
        finally:
            for k, v in prev.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_strict_never_repeat_symbol_ledger(self) -> None:
        import json
        import tempfile
        from pathlib import Path
        from src.alert_ledger import AlertLedger

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sent_alerts.json"
            p.write_text(
                json.dumps({"session_date": "2026-11-16", "alerts": {"BOROLTD:GAP_AND_GO": {"price": 341.1}}}),
                encoding="utf-8",
            )
            ledger = AlertLedger(p)
            self.assertTrue(ledger.is_sent("BOROLTD"))
            self.assertTrue(ledger.is_sent("boroltd"))
            self.assertFalse(ledger.is_sent("AFFLE"))

            ledger.mark_sent("AFFLE", stage="EOD", alert_type="PRE_BREAKOUT", price=1576.60, timestamp_ist="2026-11-17 02:00 IST")

            ledger2 = AlertLedger(p)
            self.assertTrue(ledger2.is_sent("AFFLE"))
            self.assertTrue(ledger2.is_sent("BOROLTD"))
            self.assertFalse(ledger2.is_sent("RAMKY"))

    def test_v21_daily_close_sl_dodges_stop_hunt_and_generates_pdf(self) -> None:
        import tempfile
        from pathlib import Path
        from src.backtester import backtest_single_symbol, compute_backtest_summary
        from src.pdf_report import generate_backtest_pdf

        df_base = _make_synthetic_flag_df(base_days=8)
        last_dt = df_base.index[-1]
        fut_dates = pd.bdate_range(last_dt + pd.Timedelta(days=1), periods=11)

        fut_rows = [
            (315.5, 319.0, 312.0, 316.0, 900_000.0),   # Entry fill day
            (314.0, 316.0, 294.0, 312.5, 1_100_000.0), # Intraday stop-hunt wick below Close SL, closes above!
        ]
        px = 315.0
        for _ in range(9):
            o = px
            h = px + 11.0
            l = px - 1.5
            c = px + 9.0
            fut_rows.append((o, h, l, c, 2_500_000.0))
            px = c

        df_fut = pd.DataFrame(fut_rows, index=fut_dates, columns=["Open", "High", "Low", "Close", "Volume"])
        df_all = pd.concat([df_base, df_fut])
        nifty_all = _make_nifty_df(df_all.index, ret20_pct=1.5, ret63_pct=4.0)
        breadth_ok = pd.Series(58.0, index=df_all.index)
        breadth_bad = pd.Series(32.0, index=df_all.index)

        # 1. Breadth < 40% must cleanly suppress backtest trades
        trades_blocked = backtest_single_symbol(
            "SYNTH", "Synthetic Flag Ltd", df_all, self.cfg, min_score=66.0, nifty_df=nifty_all, breadth50_series=breadth_bad
        )
        self.assertEqual(len(trades_blocked), 0)

        # 2. Healthy Breadth >= 40% executes trade, dodges stop-hunt wick, and hits TP3
        trades = backtest_single_symbol(
            "SYNTH", "Synthetic Flag Ltd", df_all, self.cfg, min_score=66.0, nifty_df=nifty_all, breadth50_series=breadth_ok
        )
        self.assertGreaterEqual(len(trades), 1)
        tr = trades[0]
        self.assertEqual(tr.sl_mode, "daily_close")
        self.assertEqual(tr.nse_breadth50_pct, 58.0)
        self.assertGreaterEqual(tr.rs63_vs_nifty, 15.0)
        self.assertNotEqual(tr.pole_start_date, "-")
        self.assertNotEqual(tr.entry_date, "-")
        self.assertNotEqual(tr.breakout_date, "-")
        self.assertNotEqual(tr.tp1_date, "-")
        self.assertNotEqual(tr.tp2_date, "-")
        self.assertNotEqual(tr.tp3_date, "-")
        self.assertEqual(tr.outcome, "TP3_HIT")
        self.assertGreater(tr.return_pct, 8.0)

        summary = compute_backtest_summary(trades, round_trip_cost_pct=0.25)
        self.assertEqual(summary["total_trades"], len(trades))
        self.assertEqual(summary["tp3_hits"], len(trades))
        self.assertGreater(summary["net_final_equity_1l"], 100000.0)

        with tempfile.TemporaryDirectory() as td:
            pdf_out = Path(td) / "test_v21_backtest.pdf"
            generate_backtest_pdf(
                trades=trades,
                summary=summary,
                meta={
                    "period": "1y",
                    "universe_scanned": 1,
                    "min_score": 66,
                    "sl_mode": "daily_close",
                    "version": "V2.1 Core + Coil",
                    "date_from": "2026-01-01",
                    "date_to": "2026-11-16",
                },
                output_pdf_path=pdf_out,
                daily_map={"SYNTH": df_all},
            )
            self.assertTrue(pdf_out.exists())
            self.assertGreater(pdf_out.stat().st_size, 15_000)

    def test_alert_ledger_lossless_union_merge(self) -> None:
        import tempfile
        from pathlib import Path
        from src.alert_ledger import AlertLedger

        with tempfile.TemporaryDirectory() as td:
            remote_path = Path(td) / "remote_sent_alerts.json"
            local_path = Path(td) / "local_sent_alerts.json"

            remote = AlertLedger(remote_path)
            remote.mark_sent("BOROLTD", stage="EOD", alert_type="PRE_BREAKOUT", price=311.5, timestamp_ist="2026-11-13 16:00 IST")
            remote.mark_sent("PARAGMILK", stage="LIVE", alert_type="GAP_AND_GO", price=344.1, timestamp_ist="2026-11-16 09:30 IST")

            local = AlertLedger(local_path)
            local.mark_sent("BOROLTD", stage="LIVE", alert_type="RETEST", price=341.0, timestamp_ist="2026-11-16 10:00 IST")
            local.mark_sent("GYFTR", stage="EOD", alert_type="PRE_BREAKOUT", price=219.6, timestamp_ist="2026-11-16 16:00 IST")

            merged = AlertLedger(remote_path)
            added = merged.merge_from(local_path)
            merged.save()

            self.assertEqual(added, 1)
            reloaded = AlertLedger(remote_path)
            self.assertTrue(reloaded.is_sent("BOROLTD"))
            self.assertTrue(reloaded.is_sent("PARAGMILK"))
            self.assertTrue(reloaded.is_sent("GYFTR"))
            # Original first-alert record for BOROLTD must remain untouched
            self.assertEqual(reloaded.alerted_symbols["BOROLTD"]["timestamp_ist"], "2026-11-13 16:00 IST")

    def test_eod_incomplete_intraday_bar_guard(self) -> None:
        from datetime import date
        from src.market_hours import is_nse_candle_incomplete_today
        from src.run_eod_scan import strip_incomplete_today_bar

        # Tue 2026-11-17: incomplete before 15:30 IST, complete at/after 15:30 IST
        self.assertTrue(is_nse_candle_incomplete_today(datetime(2026, 11, 17, 9, 20, tzinfo=IST)))
        self.assertTrue(is_nse_candle_incomplete_today(datetime(2026, 11, 17, 15, 29, tzinfo=IST)))
        self.assertFalse(is_nse_candle_incomplete_today(datetime(2026, 11, 17, 15, 30, tzinfo=IST)))
        self.assertFalse(is_nse_candle_incomplete_today(datetime(2026, 11, 17, 15, 45, tzinfo=IST)))
        # Weekend (Sun 2026-11-15) & NSE holiday (Tue 2026-11-10) -> never incomplete
        self.assertFalse(is_nse_candle_incomplete_today(datetime(2026, 11, 15, 11, 0, tzinfo=IST)))
        self.assertFalse(is_nse_candle_incomplete_today(datetime(2026, 11, 10, 11, 0, tzinfo=IST)))

        dates = pd.to_datetime(["2026-11-13", "2026-11-16", "2026-11-17"])
        df = pd.DataFrame(
            {"Open": [100, 102, 103], "High": [103, 105, 104], "Low": [99, 101, 102], "Close": [102, 104, 103], "Volume": [1e6, 1e6, 2e4]},
            index=dates,
        )
        trimmed, dropped = strip_incomplete_today_bar(df, date(2026, 11, 17))
        self.assertTrue(dropped)
        self.assertEqual(len(trimmed), 2)
        self.assertEqual(trimmed.index[-1].strftime("%Y-%m-%d"), "2026-11-16")

        # Calling again with next day cutoff does not drop completed bars
        same_df, dropped2 = strip_incomplete_today_bar(trimmed, date(2026, 11, 17))
        self.assertFalse(dropped2)
        self.assertEqual(len(same_df), 2)


if __name__ == "__main__":
    unittest.main()
