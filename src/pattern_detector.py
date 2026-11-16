"""
Pure Daily-TF Chart Pattern Detection Engine.
Catches High-Tight Flag / Power Pole + Tight Shelf Consolidation + VCP Range Contraction
+ Volume Dry-Up setups BEFORE breakout, strictly within <= 20 trading days of the first
breakout (the Pole).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.volume_profile import DailyVolumeProfile, compute_daily_volume_profile


@dataclass
class DailyPatternSetup:
    symbol: str
    company_name: str
    scan_date: str
    pattern_label: str              # e.g., "High-Tight Flag + NR Coil"
    grade: str                      # A+ | A | B
    total_score: float              # 0 - 100
    pillar_scores: Dict[str, float]
    # Timing Window (Strictly <= 20 days from first breakout pole)
    pole_start_date: str
    pole_end_date: str
    pole_dates: List[str]
    base_dates: List[str]
    pole_days: int
    base_days: int
    total_pattern_days: int         # pole_days + base_days <= 20
    # Pillar 1: Pole Breakout Metrics
    pre_pole_close: float
    pre_pole_low: float
    pole_high: float
    pole_close_gain_pct: float
    pole_high_gain_pct: float
    pole_peak_vol: float
    pole_total_vol: float
    pole_rvol: float
    pole_value_cr: float
    broke_prior_20d_high: bool
    broke_prior_40d_high: bool
    has_pole_gap_up: bool
    # Pillar 2 & 3: Tight Shelf & Volatility Contraction (VCP / NR)
    last_close: float
    shelf_high: float
    shelf_low: float
    shelf_close_spread_pct: float
    shelf_wick_box_pct: float
    upper_pole_hold_pct: float      # % up the pole where lowest base close held
    dist_to_breakout_pct: float     # % below shelf ceiling (closer = ready to pop)
    recent_3d_range_pct: float
    range_contraction_ratio: float  # recent_3d_range / pole_or_early_range (lower = tighter coil)
    tight_doji_count: int           # narrow-range / inside / spinning-top bars in base
    has_spring_bar: bool            # recent lower-wick sweep below prior lows + green reclaim
    # Pillar 4: Volume Dry-Up
    base_avg_vol: float
    recent_avg_vol: float
    dryup_pct: float                # % collapse from peak pole vol to recent base avg vol
    base_to_pole_vol_pct: float     # Total base vol / Total pole vol (%)
    # Pillar 5: Daily Volume Profile
    vah: Optional[float]
    poc: Optional[float]
    val: Optional[float]
    lvn_zone: Optional[List[float]]
    has_unfilled_lvn: bool
    above_vah: bool
    # Trade Plan Levels (Pre-Breakout Entry strictly BELOW breakout level, SL = 0.5% below flag low, TP1/TP2/TP3)
    breakout_level: float
    entry_price: float
    entry_zone_low: float
    entry_zone_high: float
    entry_below_bo_pct: float
    entry_breakout: float
    entry_anticipatory: float
    invalidation_stop: float
    sl_buffer_pct: float
    tp1: float
    tp2: float
    tp3: float
    tp1_pct: float
    tp2_pct: float
    tp3_pct: float
    risk_pct: float
    risk_pct_anticipatory: float
    risk_pct_breakout: float
    rr_to_tp1: float
    rr_to_tp2: float
    rr_to_tp3: float
    highlights: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _grade(score: float) -> str:
    if score >= 82:
        return "A+"
    if score >= 70:
        return "A"
    if score >= 60:
        return "B"
    return "C"


def _round_price(val: float, use_nse_tick: bool = True) -> float:
    if use_nse_tick:
        return round(round(val * 20.0) / 20.0, 2)
    return round(val, 2)


def detect_daily_pattern(
    symbol: str,
    company_name: str,
    df_daily: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Tuple[Optional[DailyPatternSetup], Optional[DailyVolumeProfile]]:
    """
    Pure Daily-TF chart pattern detector.
    Searches all valid (pole_days in 1..3, base_days >= 3) windows satisfying:
        total_pattern_days = pole_days + base_days <= max_total_pattern_days (20)
    Returns the highest-scoring pre-breakout setup + its Daily Volume Profile.
    """
    u_cfg = cfg["universe"]
    p_cfg = cfg["pattern"]

    if df_daily is None or len(df_daily) < int(u_cfg["min_history_days"]):
        return None, None

    df = df_daily.sort_index().copy()
    last_close = float(df["Close"].iloc[-1])
    if last_close < float(u_cfg["min_price"]) or last_close > float(u_cfg["max_price"]):
        return None, None

    n = len(df)
    max_total_days = int(p_cfg["max_total_pattern_days"])  # 20 days!
    min_pole = int(p_cfg["min_pole_days"])
    max_pole = int(p_cfg["max_pole_days"])
    min_base = int(p_cfg["min_base_days"])

    opens = df["Open"].to_numpy(dtype=float)
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)
    vols = df["Volume"].to_numpy(dtype=float)
    dates = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in df.index]

    best_candidate: Optional[Tuple[DailyPatternSetup, Optional[DailyVolumeProfile]]] = None

    # Iterate strictly within the <= 20-day full pattern window from first breakout (pole_start)
    for pole_days in range(min_pole, max_pole + 1):
        max_base_for_pole = max_total_days - pole_days
        for base_days in range(min_base, max_base_for_pole + 1):
            total_pattern_days = pole_days + base_days
            if total_pattern_days > max_total_days:
                continue

            pole_end = n - 1 - base_days
            pole_start = pole_end - pole_days + 1
            if pole_start < 25:
                continue

            pre_idx = pole_start - 1
            pre_close = closes[pre_idx]
            if pre_close <= 0:
                continue

            # Pre-pole 20d baseline
            pre_vols = vols[max(0, pole_start - 20) : pole_start]
            pos_pre_vols = pre_vols[pre_vols > 0]
            if len(pos_pre_vols) < 5:
                continue
            pre_med_vol = float(np.median(pos_pre_vols))
            if pre_med_vol <= 0:
                continue

            prior_20d_high = float(np.max(highs[max(0, pole_start - 20) : pole_start]))
            prior_40d_high = float(np.max(highs[max(0, pole_start - 40) : pole_start]))
            pre_pole_low = float(np.min(lows[max(0, pole_start - 5) : pole_start + 1]))

            # Pole slice
            p_opens = opens[pole_start : pole_end + 1]
            p_highs = highs[pole_start : pole_end + 1]
            p_lows = lows[pole_start : pole_end + 1]
            p_closes = closes[pole_start : pole_end + 1]
            p_vols = vols[pole_start : pole_end + 1]

            # Day 1 of the pole MUST itself be a strong green expansion bar (the "first breakout")
            d1_chg_pct = (p_closes[0] / pre_close - 1.0) * 100.0
            d1_rvol = p_vols[0] / pre_med_vol
            if d1_chg_pct < 5.0 or p_closes[0] <= p_opens[0] or d1_rvol < 2.0:
                continue

            # For multi-day poles, ensure each subsequent pole day pushes higher (not already consolidating)
            if pole_days >= 2:
                valid_multi = True
                for k in range(1, pole_days):
                    step_gain = (p_closes[k] / p_closes[k - 1] - 1.0) * 100.0
                    step_new_high = p_highs[k] > p_highs[k - 1] * 1.02
                    step_vol_ok = p_vols[k] >= pre_med_vol * 2.0
                    if not ((step_gain >= 3.0 or step_new_high) and step_vol_ok and p_closes[k] >= p_opens[k] * 0.99):
                        valid_multi = False
                        break
                if not valid_multi:
                    continue

            pole_high = float(np.max(p_highs))
            pole_close = float(p_closes[-1])
            pole_peak_vol = float(np.max(p_vols))
            pole_total_vol = float(np.sum(p_vols))
            pole_rvol = pole_peak_vol / pre_med_vol
            pole_value_cr = (pole_peak_vol * pole_close) / 1e7

            if pole_value_cr < float(u_cfg["min_pole_value_cr"]):
                continue

            pole_close_gain_pct = (pole_close / pre_close - 1.0) * 100.0
            pole_high_gain_pct = (pole_high / pre_close - 1.0) * 100.0
            broke_20d = pole_high > prior_20d_high and pole_close > prior_20d_high * 0.97
            broke_40d = pole_high > prior_40d_high

            if pole_close_gain_pct < float(p_cfg["min_pole_close_gain_pct"]):
                continue
            if pole_high_gain_pct < float(p_cfg["min_pole_high_gain_pct"]):
                continue
            if pole_rvol < float(p_cfg["min_pole_rvol"]):
                continue
            if bool(p_cfg["require_breakout_over_20d_high"]) and not broke_20d:
                continue

            has_pole_gap_up = bool(p_opens[0] > highs[pre_idx] * 1.005) or (
                pole_days >= 2 and bool(p_opens[1] > p_closes[0] * 1.01)
            )

            # ------------------------------------------------------------------
            # Tight High-Flag / Shelf Slice (strictly AFTER pole_end up to today)
            # ------------------------------------------------------------------
            b_opens = opens[pole_end + 1 :]
            b_highs = highs[pole_end + 1 :]
            b_lows = lows[pole_end + 1 :]
            b_closes = closes[pole_end + 1 :]
            b_vols = vols[pole_end + 1 :]

            shelf_high = float(np.max(b_highs))
            shelf_low = float(np.min(b_lows))
            b_max_close = float(np.max(b_closes))
            b_min_close = float(np.min(b_closes))
            if shelf_low <= 0 or b_min_close <= 0:
                continue

            # Pre-breakout gate: must NOT have already exploded above the flag ceiling
            prior_shelf_high = float(np.max(b_highs[:-1])) if len(b_highs) >= 2 else shelf_high
            if last_close > prior_shelf_high * (1.0 + float(p_cfg["max_already_broken_out_pct"]) / 100.0):
                continue
            if last_close > pole_high * 1.015:
                continue
            # Exclude stocks that are ALREADY printing a massive breakout surge today
            today_chg = (closes[-1] / closes[-2] - 1.0) * 100.0 if n >= 2 else 0.0
            prior_base_med_vol = float(np.median(b_vols[:-1])) if len(b_vols) >= 3 else float(np.mean(b_vols))
            if today_chg >= 5.5 and prior_base_med_vol > 0 and vols[-1] >= prior_base_med_vol * 2.2:
                continue

            # Retracement discipline: every base close holds in upper half of the pole
            impulse_span = pole_high - pre_pole_low
            if impulse_span <= 0:
                continue
            upper_hold = float((b_min_close - pre_pole_low) / impulse_span)
            if upper_hold < float(p_cfg["min_upper_pole_hold"]):
                continue

            # Shelf tightness
            close_spread_pct = (b_max_close - b_min_close) / b_min_close * 100.0
            wick_box_pct = (shelf_high - shelf_low) / shelf_low * 100.0
            if close_spread_pct > float(p_cfg["max_base_close_spread_pct"]):
                continue
            if wick_box_pct > float(p_cfg["max_base_wick_box_pct"]):
                continue

            # Volume Dry-Up (Supply Exhaustion)
            recent_k = min(4, len(b_vols))
            recent_avg_vol = float(np.mean(b_vols[-recent_k:]))
            base_avg_vol = float(np.mean(b_vols))
            dryup_ratio = recent_avg_vol / pole_peak_vol if pole_peak_vol > 0 else 1.0
            base_to_pole_ratio = float(np.sum(b_vols)) / pole_total_vol if pole_total_vol > 0 else 1.0

            if dryup_ratio > float(p_cfg["max_dryup_ratio"]):
                continue
            if base_to_pole_ratio > float(p_cfg["max_base_to_pole_vol_ratio"]):
                continue

            # Volatility Contraction (VCP / NR Coiling)
            b_ranges_pct = (b_highs - b_lows) / np.maximum(b_closes, 1e-6) * 100.0
            b_bodies_pct = np.abs(b_closes - b_opens) / np.maximum(b_closes, 1e-6) * 100.0
            p_ranges_pct = (p_highs - p_lows) / np.maximum(p_closes, 1e-6) * 100.0

            recent_3d_range_pct = float(np.mean(b_ranges_pct[-min(3, len(b_ranges_pct)) :]))
            if recent_3d_range_pct > float(p_cfg["max_recent_avg_range_pct"]):
                continue

            early_ref_range = max(
                float(np.mean(p_ranges_pct)),
                float(np.mean(b_ranges_pct[: max(1, len(b_ranges_pct) // 2)])),
                1.0,
            )
            contraction_ratio = recent_3d_range_pct / early_ref_range

            # Count tight dojis / spinning tops / inside bars in base (body <= 1.2% or inside bar)
            tight_dojis = 0
            for j in range(len(b_closes)):
                is_small_body = b_bodies_pct[j] <= 1.25
                is_inside = j > 0 and (b_highs[j] <= b_highs[j - 1] * 1.003 and b_lows[j] >= b_lows[j - 1] * 0.997)
                if is_small_body or is_inside:
                    tight_dojis += 1

            # Check for recent Spring / Dry Stop-Sweep bar (like BOROLTD 13 Nov)
            has_spring_bar = False
            if len(b_lows) >= 3:
                for offset in (1, 2):
                    idx = len(b_lows) - offset
                    if idx < 2:
                        continue
                    prior_3_low = float(np.min(b_lows[max(0, idx - 3) : idx]))
                    swept = b_lows[idx] < prior_3_low or b_lows[idx] <= shelf_low * 1.01
                    rng_i = b_highs[idx] - b_lows[idx]
                    clv_i = (b_closes[idx] - b_lows[idx]) / rng_i if rng_i > 0 else 0.5
                    green_reclaim = b_closes[idx] >= b_opens[idx] and clv_i >= 0.40
                    quiet_vol = b_vols[idx] <= pole_peak_vol * 0.15
                    if swept and green_reclaim and quiet_vol:
                        has_spring_bar = True
                        break

            # ------------------------------------------------------------------
            # Compute Pure Daily Volume Profile (last 35 trading days through today)
            # ------------------------------------------------------------------
            vp_window = df.iloc[max(0, pole_start - 15) :]
            vp = compute_daily_volume_profile(
                vp_window,
                num_bins=int(p_cfg["vp_bins"]),
                value_area_pct=float(p_cfg["value_area_pct"]),
                shelf_low=shelf_low,
            )

            # ==================================================================
            # 5-PILLAR DAILY CHART-PATTERN SCORING (100 POINTS)
            # ==================================================================
            highlights: List[str] = []
            dryup_pct = (1.0 - dryup_ratio) * 100.0
            base_to_pole_pct = base_to_pole_ratio * 100.0
            dist_to_bo_pct = max(0.0, (shelf_high - last_close) / last_close * 100.0)

            # --- Pillar 1: Breakout Pole Strength (25 pts) ---
            s_pole = 0.0
            s_pole += min(9.0, max(0.0, (pole_high_gain_pct - 12.0) * 0.35))
            s_pole += min(8.0, max(0.0, np.log1p(pole_rvol) * 2.2))
            if broke_20d:
                s_pole += 3.0
            if broke_40d:
                s_pole += 3.0
            if has_pole_gap_up:
                s_pole += 2.0
            s_pole = round(min(25.0, s_pole), 1)
            highlights.append(
                f"Pole ({pole_days}d): +{pole_high_gain_pct:.1f}% high / +{pole_close_gain_pct:.1f}% close "
                f"on {pole_rvol:.1f}x RVOL (₹{pole_value_cr:.1f} Cr)"
            )

            # --- Pillar 2: High-Tight Shelf Geometry (25 pts) ---
            s_shelf = 0.0
            # Upper-pole hold (0.50 -> 0 pts, 0.78+ -> 10 pts)
            s_shelf += min(10.0, max(0.0, (upper_hold - 0.50) / 0.28 * 10.0))
            # Tight closing range (9% -> 0 pts, <=3.5% -> 9 pts)
            s_shelf += min(9.0, max(0.0, (9.0 - close_spread_pct) / 5.5 * 9.0))
            # Tight wick box (15% -> 0 pts, <=6.5% -> 4 pts)
            s_shelf += min(4.0, max(0.0, (15.0 - wick_box_pct) / 8.5 * 4.0))
            # Proximity to breakout ceiling (<=3.5% below shelf high -> 2 pts)
            if dist_to_bo_pct <= 3.5:
                s_shelf += 2.0
            s_shelf = round(min(25.0, s_shelf), 1)
            highlights.append(
                f"High-Tight Shelf ({base_days}d base, Day {total_pattern_days}/20): "
                f"close box {close_spread_pct:.1f}%, holding upper {upper_hold*100:.0f}% of pole"
            )

            # --- Pillar 3: Volatility Contraction / VCP & NR Coiling (20 pts) ---
            s_vcp = 0.0
            # Contraction ratio (<=0.30 -> 10 pts, 0.90 -> 0 pts)
            s_vcp += min(10.0, max(0.0, (0.90 - contraction_ratio) / 0.60 * 10.0))
            # Absolute recent 3d range tightness (<=2.2% -> 5 pts, 5.5% -> 0 pts)
            s_vcp += min(5.0, max(0.0, (5.5 - recent_3d_range_pct) / 3.3 * 5.0))
            # Tight doji / inside bar cluster
            s_vcp += min(3.0, tight_dojis * 0.9)
            if has_spring_bar:
                s_vcp += 2.0
                highlights.append("Dry stop-sweep / spring reclaim candle near shelf floor")
            s_vcp = round(min(20.0, s_vcp), 1)
            if contraction_ratio <= 0.45:
                highlights.append(
                    f"Strong VCP range coil: 3d avg range down to {recent_3d_range_pct:.1f}% "
                    f"({(1.0 - contraction_ratio)*100:.0f}% tighter, {tight_dojis} tight/doji bars)"
                )

            # --- Pillar 4: Volume Dry-Up / Supply Exhaustion (20 pts) ---
            s_vol = 0.0
            # Dry-up % (70% -> 2 pts, 95%+ -> 13 pts)
            s_vol += min(13.0, max(0.0, (dryup_pct - 65.0) / 30.0 * 13.0))
            # Base-to-pole cumulative volume retention (<=15% -> 7 pts, 65% -> 0 pts)
            s_vol += min(7.0, max(0.0, (65.0 - base_to_pole_pct) / 50.0 * 7.0))
            s_vol = round(min(20.0, s_vol), 1)
            highlights.append(
                f"Volume Dry-Up: {dryup_pct:.1f}% collapse vs pole peak "
                f"(entire base traded only {base_to_pole_pct:.1f}% of pole volume)"
            )

            # --- Pillar 5: Daily Volume Profile Confluence (10 pts) ---
            s_vp = 0.0
            above_vah = False
            if vp is not None:
                above_vah = last_close >= vp.vah * 0.992
                if above_vah:
                    s_vp += 5.0
                    highlights.append(f"Shelf accepted ABOVE Daily VAH ({vp.vah:.2f}; POC {vp.poc:.2f})")
                elif last_close > vp.poc * 1.03:
                    s_vp += 2.5
                if vp.has_unfilled_lvn and vp.lvn_zone is not None:
                    s_vp += 5.0
                    highlights.append(f"Unfilled LVN air-pocket below shelf ({vp.lvn_zone[0]:.1f}–{vp.lvn_zone[1]:.1f})")
                elif vp.lvn_zone is not None:
                    s_vp += 2.5
            s_vp = round(min(10.0, s_vp), 1)

            total_score = round(s_pole + s_shelf + s_vcp + s_vol + s_vp, 1)
            grade = _grade(total_score)

            # Pattern label
            if has_spring_bar and contraction_ratio <= 0.45:
                pat_label = "High-Tight Flag + Spring Coil"
            elif contraction_ratio <= 0.40 and dryup_pct >= 85.0:
                pat_label = "High-Tight Flag + VCP Dry-Up"
            elif tight_dojis >= 3:
                pat_label = "Tight Shelf + NR Doji Coil"
            else:
                pat_label = "Post-Pole Tight Flag"

            # Actionable Full Structural Trade Plan:
            #   Breakout Level = shelf_high (flag ceiling)
            #   Entry          = Pre-breakout anticipatory entry STRICTLY BELOW breakout level
            #   SL             = shelf_low - sl_buffer_pct (default 0.5% below flag low!)
            #   TP1 / TP2 / TP3 = Structural Breakout/Pole targets + R-multiples above breakout level
            tp_cfg = cfg.get("trade_plan", {})
            below_bo_min_pct = float(tp_cfg.get("entry_below_bo_min_pct", 0.8))
            low_frac = float(tp_cfg.get("entry_flag_low_frac", 0.15))
            high_frac = float(tp_cfg.get("entry_flag_high_frac", 0.72))
            sl_buf_pct = float(tp_cfg.get("sl_buffer_pct", 0.5))
            tp1_r = float(tp_cfg.get("tp1_r_multiple", 1.5))
            tp2_r = float(tp_cfg.get("tp2_r_multiple", 3.0))
            tp3_r = float(tp_cfg.get("tp3_r_multiple", 4.5))
            use_tick = bool(tp_cfg.get("round_to_nse_tick", True))

            box_w = max(shelf_high - shelf_low, shelf_high * 0.02)
            bo_lvl = _round_price(shelf_high, use_tick)
            stop = _round_price(shelf_low * (1.0 - sl_buf_pct / 100.0), use_tick)

            # Entry zone strictly inside the flag shelf and below breakout level
            raw_zone_low = max(shelf_low * 1.003, shelf_low + low_frac * box_w)
            raw_zone_high = min(shelf_high * (1.0 - below_bo_min_pct / 100.0), shelf_low + high_frac * box_w)
            if raw_zone_high <= raw_zone_low:
                raw_zone_low = shelf_low * 1.003
                raw_zone_high = shelf_high * (1.0 - below_bo_min_pct / 100.0)
            # Allow lower edge to capture CMP if CMP sits calmly near flag support
            if shelf_low * 1.002 <= last_close < raw_zone_low:
                raw_zone_low = last_close

            entry_zone_low = _round_price(raw_zone_low, use_tick)
            entry_zone_high = _round_price(raw_zone_high, use_tick)
            if entry_zone_high >= bo_lvl:
                entry_zone_high = _round_price(bo_lvl * (1.0 - below_bo_min_pct / 100.0), use_tick)

            # Primary Pre-Breakout Entry: CMP if inside/below zone cap, else pullback cap strictly below BO
            entry_price = _round_price(min(max(last_close, entry_zone_low), entry_zone_high), use_tick)
            if entry_price >= bo_lvl:
                entry_price = _round_price(bo_lvl * (1.0 - below_bo_min_pct / 100.0), use_tick)

            entry_below_bo_pct = round((bo_lvl - entry_price) / bo_lvl * 100.0, 2)
            entry_bo = bo_lvl
            entry_ant = entry_price

            risk_amt = max(entry_price - stop, entry_price * 0.01)
            risk_pct = round((entry_price - stop) / entry_price * 100.0, 2)
            risk_pct_ant = risk_pct
            risk_pct_bo = round((bo_lvl - stop) / bo_lvl * 100.0, 2)

            pole_height = max(pole_high - float(pre_close), box_w * 1.8)
            step_unit = max(risk_amt, box_w * 0.85, entry_price * 0.04)

            # TP1: cleanly above Breakout / Pole High and >= tp1_r
            raw_tp1 = max(
                pole_high if pole_high >= bo_lvl * 1.02 else bo_lvl * 1.03,
                bo_lvl + 0.5 * box_w,
                entry_price + tp1_r * risk_amt,
            )
            # TP2: Swing extension / 0.618x Pole Measured Move
            raw_tp2 = max(
                raw_tp1 + step_unit,
                shelf_low + 0.618 * pole_height,
                entry_price + tp2_r * risk_amt,
            )
            # TP3: Full 1.0x Flag-and-Pole Measured Move
            raw_tp3 = max(
                raw_tp2 + step_unit,
                shelf_low + 1.00 * pole_height,
                entry_price + tp3_r * risk_amt,
            )

            tp1 = _round_price(raw_tp1, use_tick)
            tp2 = _round_price(raw_tp2, use_tick)
            tp3 = _round_price(raw_tp3, use_tick)

            tp1_pct = round((tp1 - entry_price) / entry_price * 100.0, 1)
            tp2_pct = round((tp2 - entry_price) / entry_price * 100.0, 1)
            tp3_pct = round((tp3 - entry_price) / entry_price * 100.0, 1)

            rr_tp1 = round((tp1 - entry_price) / risk_amt, 2)
            rr_tp2 = round((tp2 - entry_price) / risk_amt, 2)
            rr_tp3 = round((tp3 - entry_price) / risk_amt, 2)

            setup = DailyPatternSetup(
                symbol=symbol,
                company_name=company_name,
                scan_date=dates[-1],
                pattern_label=pat_label,
                grade=grade,
                total_score=total_score,
                pillar_scores={
                    "pole_strength": s_pole,
                    "shelf_geometry": s_shelf,
                    "vcp_contraction": s_vcp,
                    "volume_dryup": s_vol,
                    "daily_profile": s_vp,
                },
                pole_start_date=dates[pole_start],
                pole_end_date=dates[pole_end],
                pole_dates=dates[pole_start : pole_end + 1],
                base_dates=dates[pole_end + 1 :],
                pole_days=int(pole_days),
                base_days=int(base_days),
                total_pattern_days=int(total_pattern_days),
                pre_pole_close=round(float(pre_close), 2),
                pre_pole_low=round(float(pre_pole_low), 2),
                pole_high=round(float(pole_high), 2),
                pole_close_gain_pct=round(float(pole_close_gain_pct), 2),
                pole_high_gain_pct=round(float(pole_high_gain_pct), 2),
                pole_peak_vol=float(pole_peak_vol),
                pole_total_vol=float(pole_total_vol),
                pole_rvol=round(float(pole_rvol), 2),
                pole_value_cr=round(float(pole_value_cr), 2),
                broke_prior_20d_high=bool(broke_20d),
                broke_prior_40d_high=bool(broke_40d),
                has_pole_gap_up=bool(has_pole_gap_up),
                last_close=round(float(last_close), 2),
                shelf_high=round(float(shelf_high), 2),
                shelf_low=round(float(shelf_low), 2),
                shelf_close_spread_pct=round(float(close_spread_pct), 2),
                shelf_wick_box_pct=round(float(wick_box_pct), 2),
                upper_pole_hold_pct=round(float(upper_hold * 100.0), 1),
                dist_to_breakout_pct=round(float(dist_to_bo_pct), 2),
                recent_3d_range_pct=round(float(recent_3d_range_pct), 2),
                range_contraction_ratio=round(float(contraction_ratio), 3),
                tight_doji_count=int(tight_dojis),
                has_spring_bar=bool(has_spring_bar),
                base_avg_vol=round(float(base_avg_vol), 1),
                recent_avg_vol=round(float(recent_avg_vol), 1),
                dryup_pct=round(float(dryup_pct), 2),
                base_to_pole_vol_pct=round(float(base_to_pole_pct), 2),
                vah=round(float(vp.vah), 2) if vp else None,
                poc=round(float(vp.poc), 2) if vp else None,
                val=round(float(vp.val), 2) if vp else None,
                lvn_zone=[round(float(vp.lvn_zone[0]), 2), round(float(vp.lvn_zone[1]), 2)] if (vp and vp.lvn_zone) else None,
                has_unfilled_lvn=bool(vp.has_unfilled_lvn) if vp else False,
                above_vah=bool(above_vah),
                breakout_level=bo_lvl,
                entry_price=entry_price,
                entry_zone_low=entry_zone_low,
                entry_zone_high=entry_zone_high,
                entry_below_bo_pct=entry_below_bo_pct,
                entry_breakout=entry_bo,
                entry_anticipatory=entry_ant,
                invalidation_stop=stop,
                sl_buffer_pct=sl_buf_pct,
                tp1=tp1,
                tp2=tp2,
                tp3=tp3,
                tp1_pct=tp1_pct,
                tp2_pct=tp2_pct,
                tp3_pct=tp3_pct,
                risk_pct=risk_pct,
                risk_pct_anticipatory=risk_pct_ant,
                risk_pct_breakout=risk_pct_bo,
                rr_to_tp1=rr_tp1,
                rr_to_tp2=rr_tp2,
                rr_to_tp3=rr_tp3,
                highlights=highlights,
            )

            if best_candidate is None or setup.total_score > best_candidate[0].total_score:
                best_candidate = (setup, vp)

    if best_candidate is None:
        return None, None
    return best_candidate
