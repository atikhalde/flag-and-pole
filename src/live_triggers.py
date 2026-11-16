"""
Stage-2 Live Market Scanner (Pure Daily Forming Candle Trigger Engine).
Monitors every coiled <=20-day flag candidate in state/coiled_candidates.json
using TODAY's live forming Daily candle from yfinance and fires alerts:
  1. GAP_AND_GO        - Opens above shelf ceiling without filling gap + strong RVOL
  2. BREAKOUT_TRIGGER  - Live price breaks above shelf ceiling on expanding Daily RVOL
  3. SPRING_RETEST     - Dips to test shelf floor, then snaps back green into upper half
  4. TIGHT_COIL_WATCH  - Ultra-tight inside-day / NR bar hovering right under ceiling
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.market_hours import expected_intraday_volume_fraction, now_ist, session_elapsed_minutes


@dataclass
class LiveTriggerEvent:
    symbol: str
    company_name: str
    trigger_type: str        # GAP_AND_GO | BREAKOUT_TRIGGER | SPRING_RETEST | TIGHT_COIL_WATCH
    priority: int            # 1 (highest) to 4
    timestamp_ist: str
    session_date: str
    # Live Forming Daily Candle
    live_price: float
    day_open: float
    day_high: float
    day_low: float
    prev_close: float
    day_chg_pct: float
    gap_pct: float
    daily_clv: float         # 0.0 (closed at low) .. 1.0 (closed at high)
    # Volume Expansion vs Quiet Base
    day_volume: float
    projected_rvol_vs_base: float
    # Pattern Context
    setup_score: float
    setup_grade: str
    pattern_label: str
    day_of_pattern: int      # Strictly tracked against the 20-day pole window
    shelf_high: float
    shelf_low: float
    vah: Optional[float]
    poc: Optional[float]
    invalidation_stop: float
    target_1_bos: float
    target_2_ext: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_live_daily_triggers(
    candidate: Dict[str, Any],
    df_daily: pd.DataFrame,
    cfg: Dict[str, Any],
    as_of_dt: Optional[datetime] = None,
) -> List[LiveTriggerEvent]:
    """
    Evaluate Daily-TF live forming-candle triggers for a single coiled candidate.
    """
    if df_daily is None or len(df_daily) < 2:
        return []

    lt_cfg = cfg["live_triggers"]
    max_total_days = int(cfg["pattern"]["max_total_pattern_days"])
    df = df_daily.sort_index()

    today_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    session_date = pd.Timestamp(df.index[-1]).strftime("%Y-%m-%d")

    # Track how many trading days elapsed since first breakout (pole_start_date)
    pole_start_str = candidate.get("pole_start_date")
    if pole_start_str:
        pole_start_ts = pd.Timestamp(pole_start_str)
        day_of_pattern = int((df.index >= pole_start_ts).sum())
    else:
        day_of_pattern = int(candidate.get("total_pattern_days", 10)) + 1

    # Enforce the strict 20-day pattern window on live triggers too (allow +1 day for breakout bar)
    if day_of_pattern > max_total_days + 1:
        return []

    day_open = float(today_row["Open"])
    day_high = float(today_row["High"])
    day_low = float(today_row["Low"])
    live_price = float(today_row["Close"])
    day_vol = float(today_row["Volume"])
    prev_close = float(prev_row["Close"])
    if prev_close <= 0 or live_price <= 0:
        return []

    day_chg_pct = (live_price / prev_close - 1.0) * 100.0
    gap_pct = (day_open / prev_close - 1.0) * 100.0
    day_rng = day_high - day_low
    daily_clv = ((live_price - day_low) / day_rng) if day_rng > 0 else 0.5

    ref_dt = as_of_dt or now_ist()
    elapsed_min = session_elapsed_minutes(ref_dt)
    # If running after close or in historical replay with full-day bar, use 1.0
    if elapsed_min <= 0 or ref_dt.strftime("%Y-%m-%d") != session_date:
        expected_frac = 1.0
        elapsed_min = 375.0
    else:
        expected_frac = expected_intraday_volume_fraction(elapsed_min)

    base_ref_vol = max(
        float(candidate.get("recent_avg_vol", 0) or 0),
        float(candidate.get("base_avg_vol", 0) or 0) * 0.5,
        1000.0,
    )
    projected_vol = day_vol / max(expected_frac, 0.08)
    rvol_vs_base = projected_vol / base_ref_vol

    shelf_high = float(candidate["shelf_high"])
    shelf_low = float(candidate["shelf_low"])
    vah = candidate.get("vah")
    poc = candidate.get("poc")
    stop = float(candidate["invalidation_stop"])
    t1 = float(candidate["target_1_bos"])
    t2 = float(candidate["target_2_ext"])
    ts_str = ref_dt.strftime("%Y-%m-%d %H:%M IST")

    events: List[LiveTriggerEvent] = []

    def _make(trig_type: str, prio: int, reason: str) -> LiveTriggerEvent:
        return LiveTriggerEvent(
            symbol=candidate["symbol"],
            company_name=candidate.get("company_name", candidate["symbol"]),
            trigger_type=trig_type,
            priority=prio,
            timestamp_ist=ts_str,
            session_date=session_date,
            live_price=round(live_price, 2),
            day_open=round(day_open, 2),
            day_high=round(day_high, 2),
            day_low=round(day_low, 2),
            prev_close=round(prev_close, 2),
            day_chg_pct=round(day_chg_pct, 2),
            gap_pct=round(gap_pct, 2),
            daily_clv=round(daily_clv, 2),
            day_volume=round(day_vol, 0),
            projected_rvol_vs_base=round(rvol_vs_base, 2),
            setup_score=float(candidate["total_score"]),
            setup_grade=str(candidate["grade"]),
            pattern_label=str(candidate.get("pattern_label", "High-Tight Flag")),
            day_of_pattern=day_of_pattern,
            shelf_high=shelf_high,
            shelf_low=shelf_low,
            vah=vah,
            poc=poc,
            invalidation_stop=stop,
            target_1_bos=t1,
            target_2_ext=t2,
            reason=reason,
        )

    # 1. GAP_AND_GO (BOROLTD 16 Nov: Gap up near/over shelf ceiling + strong expansion)
    min_gap = float(lt_cfg["gap_and_go_min_gap_pct"])
    if (
        day_open >= shelf_high * 0.992
        and gap_pct >= min_gap
        and day_low >= prev_close * 1.003
        and live_price >= day_open * 1.003
        and live_price > shelf_high
        and rvol_vs_base >= float(lt_cfg["breakout_min_rvol"])
        and daily_clv >= float(lt_cfg["breakout_min_clv"])
    ):
        events.append(
            _make(
                "GAP_AND_GO",
                1,
                f"Day {day_of_pattern}/20 Gap-and-Go: opened +{gap_pct:.1f}% near/above flag ceiling "
                f"({shelf_high:.2f}), never filled gap (low {day_low:.2f}), surging +{day_chg_pct:.1f}% "
                f"on {rvol_vs_base:.1f}x projected base RVOL.",
            )
        )

    # 2. BREAKOUT_TRIGGER (Intraday break cleanly above Daily Flag Ceiling)
    bo_level = shelf_high * (1.0 + float(lt_cfg["breakout_buffer_pct"]) / 100.0)
    if (
        not events
        and elapsed_min >= float(lt_cfg["min_minutes_after_open_for_rvol"])
        and live_price >= bo_level
        and live_price >= day_open
        and rvol_vs_base >= float(lt_cfg["breakout_min_rvol"])
        and daily_clv >= float(lt_cfg["breakout_min_clv"])
    ):
        events.append(
            _make(
                "BREAKOUT_TRIGGER",
                1,
                f"Day {day_of_pattern}/20 Flag Breakout: cleared shelf ceiling {shelf_high:.2f} -> "
                f"{live_price:.2f} (+{day_chg_pct:.1f}%) with strong candle close (CLV {daily_clv:.0%}) "
                f"on {rvol_vs_base:.1f}x base RVOL.",
            )
        )

    # 3. SPRING_RETEST (BOROLTD 13 Nov: Dips to shelf floor / VAH, then snaps back green)
    sweep_buf = float(lt_cfg["spring_sweep_buffer_pct"]) / 100.0
    floor_ref = shelf_low
    if vah is not None and float(vah) >= shelf_low * 0.99:
        floor_ref = max(shelf_low, float(vah))
    if (
        not events
        and elapsed_min >= 30
        and day_low <= floor_ref * (1.0 + sweep_buf)
        and day_low >= stop * 0.995
        and live_price >= day_open
        and live_price > floor_ref
        and live_price <= shelf_high * 1.005
        and daily_clv >= float(lt_cfg["spring_min_clv"])
    ):
        events.append(
            _make(
                "SPRING_RETEST",
                2,
                f"Day {day_of_pattern}/20 Spring Retest: swept floor ({day_low:.2f} vs support "
                f"{floor_ref:.2f}) and snapped back green to {live_price:.2f} (CLV {daily_clv:.0%}) "
                f"— classic pre-breakout shakeout.",
            )
        )

    # 4. TIGHT_COIL_WATCH (Inside/NR bar coiled right under ceiling near close)
    dist_to_ceil = (shelf_high - live_price) / live_price * 100.0
    day_range_pct = (day_rng / live_price * 100.0) if live_price > 0 else 99.0
    if (
        not events
        and elapsed_min >= 180
        and 0.0 <= dist_to_ceil <= float(lt_cfg["coil_max_dist_to_ceiling_pct"])
        and day_range_pct <= 2.5
        and live_price >= day_open * 0.995
        and rvol_vs_base <= 1.25
    ):
        events.append(
            _make(
                "TIGHT_COIL_WATCH",
                3,
                f"Day {day_of_pattern}/20 Ultra-Tight Coil: range compressed to {day_range_pct:.1f}%, "
                f"hugging {dist_to_ceil:.1f}% below ceiling ({shelf_high:.2f}) on dry volume — ready to pop.",
            )
        )

    events.sort(key=lambda e: e.priority)
    return events
