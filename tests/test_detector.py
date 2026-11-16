"""Unit tests for the pure Daily-TF <=20-day V2 Option A High-Tight Flag & Backtest Scanner."""

from __future__ import annotations

import unittest
from datetime import datetime

import numpy as np
import pandas as pd

from src.config_loader import load_config
from src.live_triggers import evaluate_live_daily_triggers
from src.market_hours import IST
from src.pattern_detector import detect_daily_pattern


def _make_synthetic_flag_df(
    base_days: int = 8,
    pole_days: int = 2,
    uptrend: bool = True,
    pole_top: float = 328.0,
    close_at_bottom: bool = False,
) -> pd.DataFrame:
    """Create a realistic synthetic Daily OHLCV High-Tight Flag series with 210+ bars for 200 SMA."""
    rng = np.random.default_rng(42)
    n_pre = 210
    dates = pd.bdate_range("2025-11-01", periods=n_pre + pole_days + base_days)

    rows = []
    # Smoothly drift from 170 -> 248 (uptrend) or 360 -> 248 (downtrend below 200 SMA)
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

    # Breakout Pole (~30% gain over pre-pole close ~248 -> 322-328, ~18x institutional RVOL)
    rows.append((255.0, 290.0, 252.0, 288.0, 3_600_000.0))
    if pole_days >= 2:
        rows.append((292.0, pole_top, 286.0, min(312.0, pole_top - 4.0), 2_800_000.0))
    if pole_days >= 3:
        rows.append((314.0, pole_top + 6.0, 310.0, pole_top + 2.0, 2_200_000.0))

    # Tight High-Flag Shelf (box ~302..322 = ~6.6% wick box, healthy ~78% dry-up)
    for j in range(base_days):
        tight = 1.0 - (j / max(base_days, 1)) * 0.45
        if j == base_days - 1 and close_at_bottom:
            o, h, l, c = 310.0, 312.0, 302.0, 303.5  # bottom 7% of box -> falling knife!
        elif j == base_days - 1:
            o, h, l, c = 313.0, 319.5, 311.5, 317.8  # coiled in upper 75% of box!
        else:
            o = 311.0 + rng.uniform(-2.0, 2.0) * tight
            c = 313.0 + rng.uniform(-2.0, 2.0) * tight
            h = max(o, c) + (7.5 if j == 0 else 4.5) * tight
            l = min(o, c) - (8.5 if j == 0 else 3.5) * tight
        v = 550_000.0 * (0.72 ** j) + 120_000.0
        rows.append((o, h, l, c, v))

    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=dates)


def _make_nifty_df(dates: pd.DatetimeIndex, ret20_pct: float = 2.0) -> pd.DataFrame:
    """Create a synthetic NIFTY 50 series with desired 20-day return at the final bar."""
    n = len(dates)
    closes = np.full(n, 24000.0, dtype=float)
    if n >= 25:
        closes[-21:] = np.linspace(24000.0, 24000.0 * (1.0 + ret20_pct / 100.0), 21)
    return pd.DataFrame(
        {"Open": closes, "High": closes * 1.005, "Low": closes * 0.995, "Close": closes, "Volume": np.full(n, 1e8)},
        index=dates,
    )


class TestDailyFlagScannerV2(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config()

    def test_catches_clean_v2_within_20_day_flag(self) -> None:
        df = _make_synthetic_flag_df(base_days=8, pole_days=2)  # 10 total days <= 20
        nifty = _make_nifty_df(df.index, ret20_pct=1.8)
        setup, vp = detect_daily_pattern("SYNTH", "Synthetic Ltd", df, self.cfg, nifty_df=nifty)
        self.assertIsNotNone(setup)
        self.assertIsNotNone(vp)
        assert setup is not None
        self.assertLessEqual(setup.total_pattern_days, 20)
        self.assertGreaterEqual(setup.total_score, 66.0)
        self.assertIn(setup.grade, ("A+", "A"))
        self.assertTrue(setup.above_200sma)
        self.assertGreaterEqual(setup.sig_pos_in_box, 0.25)

        # Verify V2 Option A Trade Plan invariants
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

    def test_v2_gates_reject_traps(self) -> None:
        # 1. Rejects Stage-4 downtrend below 200 SMA
        df_bear = _make_synthetic_flag_df(uptrend=False)
        s_bear, _ = detect_daily_pattern("BEAR", "Bear Ltd", df_bear, self.cfg)
        self.assertIsNone(s_bear)

        # 2. Rejects falling knife closing in bottom 25% of flag box
        df_bot = _make_synthetic_flag_df(close_at_bottom=True)
        s_bot, _ = detect_daily_pattern("KNIFE", "Knife Ltd", df_bot, self.cfg)
        self.assertIsNone(s_bot)

        # 3. Rejects parabolic blow-off pole (>42% high gain)
        df_blow = _make_synthetic_flag_df(pole_top=375.0)
        s_blow, _ = detect_daily_pattern("BLOW", "Blowoff Ltd", df_blow, self.cfg)
        self.assertIsNone(s_blow)

        # 4. Rejects entries during NIFTY 50 sharp sell-off (20d < -3.0%)
        df_ok = _make_synthetic_flag_df()
        nifty_crash = _make_nifty_df(df_ok.index, ret20_pct=-4.5)
        s_crash, _ = detect_daily_pattern("SYNTH", "Synthetic Ltd", df_ok, self.cfg, nifty_df=nifty_crash)
        self.assertIsNone(s_crash)

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

    def test_multi_bot_telegram_notifier_and_v2_captions(self) -> None:
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

        # Verify V2 EOD caption fits cleanly within Telegram's 1024-char photo caption cap
        df = _make_synthetic_flag_df()
        setup, _ = detect_daily_pattern("SYNTH", "Synthetic Industries Ltd", df, self.cfg)
        assert setup is not None
        cap = tn.format_eod_setup_caption(setup.to_dict())
        self.assertLess(len(cap), 1020)
        self.assertIn("Close SL (3:20 PM)", cap)
        self.assertIn("Emergency GTT", cap)

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

    def test_v2_daily_close_sl_dodges_stop_hunt_and_generates_pdf(self) -> None:
        import tempfile
        from pathlib import Path
        from src.backtester import backtest_single_symbol, compute_backtest_summary
        from src.pdf_report import generate_backtest_pdf

        df_base = _make_synthetic_flag_df(base_days=8)
        last_dt = df_base.index[-1]
        fut_dates = pd.bdate_range(last_dt + pd.Timedelta(days=1), periods=11)

        # Day 1: fills planned entry (~316); Day 2: intraday stop-hunt wicks down to 294 (below Close SL ~299, above -9% GTT ~287)
        #        but CLOSES green at 312 -> Option A Daily-Close SL MUST SURVIVE!
        # Days 3..11: rallies smoothly through BO, TP1 (+4.5%), TP2 (+10%), TP3 (+18%)!
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
        nifty_all = _make_nifty_df(df_all.index, ret20_pct=1.5)

        trades = backtest_single_symbol("SYNTH", "Synthetic Flag Ltd", df_all, self.cfg, min_score=66.0, nifty_df=nifty_all)
        self.assertGreaterEqual(len(trades), 1)
        tr = trades[0]
        self.assertEqual(tr.sl_mode, "daily_close")
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
            pdf_out = Path(td) / "test_v2_backtest.pdf"
            generate_backtest_pdf(
                trades=trades,
                summary=summary,
                meta={"period": "1y", "universe_scanned": 1, "min_score": 66, "sl_mode": "daily_close", "date_from": "2026-01-01", "date_to": "2026-11-16"},
                output_pdf_path=pdf_out,
                daily_map={"SYNTH": df_all},
            )
            self.assertTrue(pdf_out.exists())
            self.assertGreater(pdf_out.stat().st_size, 15_000)


if __name__ == "__main__":
    unittest.main()
