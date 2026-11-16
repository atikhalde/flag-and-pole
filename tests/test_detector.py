"""Unit tests for the pure Daily-TF <=20-day High-Tight Flag & Live Trigger Scanner."""

from __future__ import annotations

import unittest
from datetime import datetime

import numpy as np
import pandas as pd

from src.config_loader import load_config
from src.live_triggers import evaluate_live_daily_triggers
from src.market_hours import IST
from src.pattern_detector import detect_daily_pattern


def _make_synthetic_flag_df(base_days: int = 8, pole_days: int = 2) -> pd.DataFrame:
    """Create a realistic synthetic Daily OHLCV High-Tight Flag series."""
    rng = np.random.default_rng(42)
    n_pre = 45
    dates = pd.bdate_range("2026-08-01", periods=n_pre + pole_days + base_days)

    rows = []
    price = 240.0
    for i in range(n_pre):
        o = price + rng.uniform(-2, 2)
        c = o + rng.uniform(-3, 3)
        h = max(o, c) + rng.uniform(1, 4)
        l = min(o, c) - rng.uniform(1, 4)
        v = rng.uniform(80_000, 160_000)
        rows.append((o, h, l, c, v))
        price = np.clip(c, 225, 255)

    # Breakout Pole (strong surge over prior 40d high on massive volume)
    rows.append((255.0, 286.0, 250.0, 286.0, 14_000_000.0))
    if pole_days >= 2:
        rows.append((292.0, 334.0, 283.0, 312.0, 8_500_000.0))
    if pole_days >= 3:
        rows.append((314.0, 342.0, 310.0, 336.0, 6_000_000.0))

    # Tight High-Flag Shelf (drying volume + tightening ranges)
    for j in range(base_days):
        tight = 1.0 - (j / max(base_days, 1)) * 0.65
        o = 310.0 + rng.uniform(-1.5, 1.5) * tight
        c = 311.0 + rng.uniform(-1.5, 1.5) * tight
        h = max(o, c) + 4.5 * tight
        l = min(o, c) - 3.5 * tight
        v = 900_000.0 * (0.65 ** j) + 80_000.0
        rows.append((o, h, l, c, v))

    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=dates)


class TestDailyFlagScanner(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config()

    def test_catches_clean_within_20_day_flag(self) -> None:
        df = _make_synthetic_flag_df(base_days=8, pole_days=2)  # 10 total days <= 20
        setup, vp = detect_daily_pattern("SYNTH", "Synthetic Ltd", df, self.cfg)
        self.assertIsNotNone(setup)
        self.assertIsNotNone(vp)
        assert setup is not None
        self.assertLessEqual(setup.total_pattern_days, 20)
        self.assertGreaterEqual(setup.total_score, 75.0)
        self.assertGreaterEqual(setup.dryup_pct, 85.0)

    def test_rejects_pattern_older_than_20_days(self) -> None:
        df = _make_synthetic_flag_df(base_days=22, pole_days=2)  # 24 total days > 20!
        setup, _ = detect_daily_pattern("STALE", "Stale Base Ltd", df, self.cfg)
        self.assertIsNone(setup)

    def test_live_gap_and_go_trigger_fires(self) -> None:
        df_pre = _make_synthetic_flag_df(base_days=8, pole_days=2)
        setup, _ = detect_daily_pattern("SYNTH", "Synthetic Ltd", df_pre, self.cfg)
        self.assertIsNotNone(setup)
        assert setup is not None

        # Append a BOROLTD-style Gap-and-Go breakout day
        next_day = df_pre.index[-1] + pd.Timedelta(days=1)
        bo_row = pd.DataFrame(
            {"Open": [321.0], "High": [346.5], "Low": [318.0], "Close": [341.0], "Volume": [4_800_000.0]},
            index=[next_day],
        )
        df_live = pd.concat([df_pre, bo_row])
        sim_dt = datetime(next_day.year, next_day.month, next_day.day, 15, 0, tzinfo=IST)
        events = evaluate_live_daily_triggers(setup.to_dict(), df_live, self.cfg, as_of_dt=sim_dt)
        self.assertTrue(len(events) >= 1)
        self.assertEqual(events[0].trigger_type, "GAP_AND_GO")
        self.assertLessEqual(events[0].day_of_pattern, 21)


if __name__ == "__main__":
    unittest.main()
