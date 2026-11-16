"""
Point-in-Time Historical Validator.
Proves the pure Daily-TF scanner catches BOROLTD:
  1. BEFORE the breakout — as of Fri 13 Nov 2026 (Day 10/20 from 30 Oct pole)
  2. ON the breakout — simulating Mon 16 Nov 2026 Live Daily Trigger (GAP_AND_GO)
Saves both proof chart PNGs to output/charts/.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.chart_generator import generate_pattern_chart
from src.config_loader import load_config
from src.data_fetcher import fetch_daily_single
from src.live_triggers import evaluate_live_daily_triggers
from src.market_hours import IST
from src.pattern_detector import detect_daily_pattern


def validate_symbol(
    symbol: str = "BOROLTD",
    pre_breakout_date: str = "2026-11-13",
    breakout_date: str = "2026-11-16",
) -> None:
    cfg = load_config()
    df_full = fetch_daily_single(symbol, period="5mo", repair_nan_today=True)
    if df_full.empty:
        raise RuntimeError(f"Could not fetch daily OHLCV for {symbol}")

    # 1. Strict Point-in-Time EOD Scan BEFORE Breakout
    df_pre = df_full.loc[:pre_breakout_date]
    setup, vp = detect_daily_pattern(symbol, f"{symbol} (NSE)", df_pre, cfg)
    if setup is None:
        print(f"❌ {symbol} did NOT match as of {pre_breakout_date}")
        return

    setup_dict = setup.to_dict()
    charts_dir = Path(cfg["runtime"]["charts_dir"])
    pre_chart = charts_dir / f"PROOF_PRE_{symbol}_{pre_breakout_date}.png"
    generate_pattern_chart(symbol, df_pre, setup_dict, pre_chart, vp=vp)

    print("=" * 72)
    print(f"✅ STAGE-1 PRE-BREAKOUT MATCH: {symbol} as of {pre_breakout_date}")
    print("=" * 72)
    print(f"  Pattern       : {setup.pattern_label} | Grade {setup.grade} ({setup.total_score:.1f}/100)")
    print(f"  Window        : Day {setup.total_pattern_days}/20 ({setup.pole_days}d Pole [{setup.pole_start_date}->{setup.pole_end_date}] + {setup.base_days}d Shelf)")
    print(f"  Pillar Scores : {json.dumps(setup.pillar_scores)}")
    print(f"  Pole Surge    : +{setup.pole_high_gain_pct:.1f}% high / +{setup.pole_close_gain_pct:.1f}% close on {setup.pole_rvol:.1f}x RVOL")
    print(f"  Shelf Box     : ₹{setup.shelf_low:.2f}–₹{setup.shelf_high:.2f} (Close spread {setup.shelf_close_spread_pct:.1f}%, upper {setup.upper_pole_hold_pct:.0f}% hold)")
    print(f"  Volume Dry-Up : {setup.dryup_pct:.1f}% collapse (Base/Pole vol {setup.base_to_pole_vol_pct:.1f}%)")
    print(f"  Daily Profile : VAH ₹{setup.vah} | POC ₹{setup.poc} | VAL ₹{setup.val} | LVN {setup.lvn_zone}")
    print(f"  Trade Plan    : Close ₹{setup.last_close:.2f} | BO > ₹{setup.entry_breakout:.2f} | SL ₹{setup.invalidation_stop:.2f} | T1 ₹{setup.target_1_bos:.2f} | T2 ₹{setup.target_2_ext:.2f}")
    print(f"  Proof Chart   : {pre_chart}")

    # 2. Simulate Stage-2 Live Daily Trigger ON Breakout Date
    df_bo = df_full.loc[:breakout_date]
    if len(df_bo) > len(df_pre):
        sim_dt = datetime.strptime(f"{breakout_date} 15:20", "%Y-%m-%d %H:%M").replace(tzinfo=IST)
        events = evaluate_live_daily_triggers(setup_dict, df_bo, cfg, as_of_dt=sim_dt)
        print("\n" + "=" * 72)
        print(f"🚀 STAGE-2 LIVE DAILY TRIGGER SIMULATION: {symbol} on {breakout_date}")
        print("=" * 72)
        if events:
            ev = events[0]
            bo_chart = charts_dir / f"PROOF_LIVE_{symbol}_{breakout_date}.png"
            generate_pattern_chart(symbol, df_bo, setup_dict, bo_chart, live_trigger=ev.to_dict())
            print(f"  Trigger       : {ev.trigger_type} (Day {ev.day_of_pattern}/20)")
            print(f"  Live Daily    : O ₹{ev.day_open} | H ₹{ev.day_high} | L ₹{ev.day_low} | C ₹{ev.live_price} ({ev.day_chg_pct:+.2f}%)")
            print(f"  RVOL vs Base  : {ev.projected_rvol_vs_base:.1f}x | Daily CLV {ev.daily_clv:.0%}")
            print(f"  Reason        : {ev.reason}")
            print(f"  Proof Chart   : {bo_chart}")
        else:
            print("  No trigger fired on breakout date.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Daily Pattern Scanner on historical breakout")
    parser.add_argument("--symbol", default="BOROLTD")
    parser.add_argument("--pre-date", default="2026-11-13")
    parser.add_argument("--bo-date", default="2026-11-16")
    args = parser.parse_args()
    validate_symbol(args.symbol, args.pre_date, args.bo_date)
