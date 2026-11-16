"""
Pure Daily-TF Chart Pattern Detection Engine (V2 — 5-Year Validated Option A).

Catches High-Tight Flag / Power Pole + Tight Shelf Consolidation + VCP Range Contraction
+ Volume Dry-Up setups BEFORE breakout, strictly within <= 20 trading days of the first
breakout (the Pole), enhanced with 5-year out-of-sample validated institutional filters:
  1. Primary Uptrend Gate       : Close > 200-day SMA
  2. 52-Week Leadership Gate    : Close within -25% of 52-Week High
  3. NIFTY 50 Regime Guard      : NIFTY 50 20-day return >= -3.0%
  4. Anti-Blowoff Pole Cap      : Pole High Gain <= 42.0%
  5. Upper-Flag Coil Gate       : Signal Close in upper 75% of flag box (pos >= 0.25)

Trade Plan (Option A — Daily-Close SL + Hard Emergency Broker GTT):
  - Entry strictly BELOW Breakout Level (inside coiled flag shelf)
  - Daily-Close SL (-1.5% below flag low, min 4.0% / max 6.5% risk; exit on 3:20 PM IST close)
  - Hard Emergency Intraday Broker GTT (-9.0% safety net)
  - Reference Hands-Off Intraday GTT SL (-2.5% below flag low, 4.5%-6.5% risk)
  - High-Win-Rate 3-Tranche Targets: TP1 +4.5% (45% -> SL to BE), TP2 +10.0% (30% -> SL to TP1), TP3 +18.0% (25%)
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
    total_score: float              # 0 - 100 (V2 Recalibrated 5-Pillar Score)
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
    # Pillar 2 & 3: Tight Shelf, Coil Position & Volatility Contraction
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
    # Daily Volume Profile (visual chart & spring support context)
    vah: Optional[float]
    poc: Optional[float]
    val: Optional[float]
    lvn_zone: Optional[List[float]]
    has_unfilled_lvn: bool
    above_vah: bool
    # Trade Plan Levels (Pre-Breakout Entry strictly BELOW breakout level)
    breakout_level: float
    entry_price: float
    entry_zone_low: float
    entry_zone_high: float
    entry_below_bo_pct: float
    entry_breakout: float
    entry_anticipatory: float
    invalidation_stop: float        # Active SL (Option A: Daily-Close SL)
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
    # V2 Institutional Edge & Option A Protection Fields
    sl_mode: str = "daily_close"
    hard_stop_gtt: float = 0.0      # Catastrophic intraday broker GTT (-9.0%)
    hard_stop_gtt_pct: float = 9.0
    intraday_gtt_sl: float = 0.0    # Reference 100% hands-off Intraday GTT SL
    intraday_gtt_risk_pct: float = 0.0
    tp1_weight: float = 0.45
    tp2_weight: float = 0.30
    tp3_weight: float = 0.25
    above_200sma: bool = True
    stage2_uptrend: bool = False
    pct_from_52w_high: float = 0.0
    sig_pos_in_box: float = 0.5
    sig_clv: float = 0.5
    atr14_pct: float = 4.5
    med_turnover_cr: float = 5.0
    rs63_vs_nifty: float = 15.0
    nifty_ret20_pct: Optional[float] = None
    nifty_above_50sma: Optional[bool] = None
    highlights: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _grade(score: float) -> str:
    """V2 Recalibrated Grades (strict A+ > A > B monotonicity proven IS & OOS)."""
    if score >= 78.0:
        return "A+"
    if score >= 66.0:
        return "A"
    if score >= 58.0:
        return "B"
    return "C"


def _round_price(val: float, use_nse_tick: bool = True) -> float:
    if use_nse_tick:
        return round(round(val * 20.0) / 20.0, 2)
    return round(val, 2)


def evaluate_nifty_regime(
    nifty_df: Optional[pd.DataFrame],
    as_of_ts: Optional[pd.Timestamp] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Strictly point-in-time NIFTY 50 regime snapshot as of `as_of_ts`.
    Returns {"available": bool, "regime_ok": bool, "ret20_pct": float|None, "ret63_pct": float|None, "above_50sma": bool|None}.
    """
    r_cfg = (cfg or {}).get("regime", {})
    enable_guard = bool(r_cfg.get("enable_nifty_guard", True))
    min_ret20 = float(r_cfg.get("min_nifty_20d_return_pct", -3.0))

    if nifty_df is None or nifty_df.empty or "Close" not in nifty_df.columns:
        return {
            "available": False,
            "regime_ok": True,
            "ret20_pct": None,
            "ret63_pct": None,
            "above_50sma": None,
            "nifty_close": None,
            "as_of": None,
        }

    if as_of_ts is not None:
        loc = int(nifty_df.index.searchsorted(as_of_ts, side="right")) - 1
    else:
        loc = len(nifty_df) - 1

    if loc < 20:
        return {
            "available": False,
            "regime_ok": True,
            "ret20_pct": None,
            "ret63_pct": None,
            "above_50sma": None,
            "nifty_close": None,
            "as_of": None,
        }

    nc = nifty_df["Close"].to_numpy(dtype=float)
    cur = float(nc[loc])
    ret20 = (cur / float(nc[loc - 20]) - 1.0) * 100.0 if nc[loc - 20] > 0 else 0.0
    ret63 = (cur / float(nc[loc - 63]) - 1.0) * 100.0 if (loc >= 63 and nc[loc - 63] > 0) else ret20 * 2.5
    sma50 = float(np.mean(nc[max(0, loc - 49) : loc + 1]))
    above_50 = cur > sma50
    regime_ok = (not enable_guard) or (ret20 >= min_ret20)

    return {
        "available": True,
        "regime_ok": bool(regime_ok),
        "ret20_pct": round(ret20, 2),
        "ret63_pct": round(ret63, 2),
        "above_50sma": bool(above_50),
        "nifty_close": round(cur, 2),
        "as_of": pd.Timestamp(nifty_df.index[loc]).strftime("%Y-%m-%d"),
    }


def compute_trade_plan(
    entry_price: float,
    shelf_low: float,
    shelf_high: float,
    pole_high: float,
    pre_close: float,
    cfg: Dict[str, Any],
) -> Dict[str, float | str]:
    """
    Shared V2 Trade Plan builder used consistently by Detector, Live Triggers, and Backtester.
    Supports Option A ("daily_close" - Recommended), Option B ("intraday_gtt"), and Option C ("strict_05").
    """
    tp_cfg = cfg.get("trade_plan", {})
    sl_mode = str(tp_cfg.get("sl_mode", "daily_close")).strip().lower()
    use_tick = bool(tp_cfg.get("round_to_nse_tick", True))

    sl_buf_pct = float(tp_cfg.get("sl_buffer_pct", 1.5))
    sl_min_risk = float(tp_cfg.get("sl_min_risk_pct", 4.0))
    sl_max_risk = float(tp_cfg.get("sl_max_risk_pct", 6.5))
    hard_gtt_pct = float(tp_cfg.get("hard_gtt_stop_pct", 9.0))

    # Reference 100% hands-off Intraday GTT SL (-2.5% below flag low, 4.5%..6.5% risk)
    ig_buf = float(tp_cfg.get("intraday_gtt_ref_buffer_pct", 2.5))
    ig_min = float(tp_cfg.get("intraday_gtt_ref_min_risk_pct", 4.5))
    ig_max = float(tp_cfg.get("intraday_gtt_ref_max_risk_pct", 6.5))
    raw_intra_gtt = max(
        min(shelf_low * (1.0 - ig_buf / 100.0), entry_price * (1.0 - ig_min / 100.0)),
        entry_price * (1.0 - ig_max / 100.0),
    )
    intraday_gtt_sl = _round_price(raw_intra_gtt, use_tick)

    if sl_mode == "strict_05":
        raw_sl = shelf_low * (1.0 - 0.5 / 100.0)
        sl_buf_used = 0.5
        stop = _round_price(raw_sl, use_tick)
        hard_gtt = stop
    elif sl_mode == "intraday_gtt":
        stop = intraday_gtt_sl
        sl_buf_used = ig_buf
        hard_gtt = stop
    else:
        # Option A (Default): Daily-Close SL (-1.5% below flag low, 4.0%..6.5% risk) + Hard -9.0% GTT
        sl_mode = "daily_close"
        raw_sl = max(
            min(shelf_low * (1.0 - sl_buf_pct / 100.0), entry_price * (1.0 - sl_min_risk / 100.0)),
            entry_price * (1.0 - sl_max_risk / 100.0),
        )
        sl_buf_used = sl_buf_pct
        stop = _round_price(raw_sl, use_tick)
        raw_hard_gtt = min(stop * 0.96, entry_price * (1.0 - hard_gtt_pct / 100.0))
        hard_gtt = _round_price(raw_hard_gtt, use_tick)

    # Guard against degenerate SL >= entry
    if stop >= entry_price:
        stop = _round_price(entry_price * (1.0 - sl_min_risk / 100.0), use_tick)
    if hard_gtt > stop:
        hard_gtt = stop

    # V2 High-Win-Rate 3-Tranche Targets (+4.5% / +10.0% / +18.0% from Entry)
    tp1_t = float(tp_cfg.get("tp1_pct_target", 4.5))
    tp2_t = float(tp_cfg.get("tp2_pct_target", 10.0))
    tp3_t = float(tp_cfg.get("tp3_pct_target", 18.0))
    w1 = float(tp_cfg.get("tp1_weight", 0.45))
    w2 = float(tp_cfg.get("tp2_weight", 0.30))
    w3 = float(tp_cfg.get("tp3_weight", 0.25))

    tp1 = _round_price(entry_price * (1.0 + tp1_t / 100.0), use_tick)
    tp2 = _round_price(entry_price * (1.0 + tp2_t / 100.0), use_tick)
    tp3 = _round_price(entry_price * (1.0 + tp3_t / 100.0), use_tick)
    if tp2 <= tp1:
        tp2 = _round_price(tp1 * 1.05, use_tick)
    if tp3 <= tp2:
        tp3 = _round_price(tp2 * 1.07, use_tick)

    risk_amt = max(entry_price - stop, entry_price * 0.01)
    risk_pct = round((entry_price - stop) / entry_price * 100.0, 2)
    hard_gtt_risk = round((entry_price - hard_gtt) / entry_price * 100.0, 2)
    intra_gtt_risk = round((entry_price - intraday_gtt_sl) / entry_price * 100.0, 2)

    return {
        "sl_mode": sl_mode,
        "invalidation_stop": stop,
        "sl_buffer_pct": sl_buf_used,
        "hard_stop_gtt": hard_gtt,
        "hard_stop_gtt_pct": hard_gtt_risk,
        "intraday_gtt_sl": intraday_gtt_sl,
        "intraday_gtt_risk_pct": intra_gtt_risk,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "tp1_pct": round((tp1 - entry_price) / entry_price * 100.0, 1),
        "tp2_pct": round((tp2 - entry_price) / entry_price * 100.0, 1),
        "tp3_pct": round((tp3 - entry_price) / entry_price * 100.0, 1),
        "tp1_weight": w1,
        "tp2_weight": w2,
        "tp3_weight": w3,
        "risk_pct": risk_pct,
        "rr_to_tp1": round((tp1 - entry_price) / risk_amt, 2),
        "rr_to_tp2": round((tp2 - entry_price) / risk_amt, 2),
        "rr_to_tp3": round((tp3 - entry_price) / risk_amt, 2),
    }


def detect_daily_pattern(
    symbol: str,
    company_name: str,
    df_daily: pd.DataFrame,
    cfg: Dict[str, Any],
    nifty_df: Optional[pd.DataFrame] = None,
    compute_vp: bool = True,
) -> Tuple[Optional[DailyPatternSetup], Optional[DailyVolumeProfile]]:
    """
    Pure Daily-TF V2 chart pattern detector (strictly point-in-time on df_daily).
    Searches all valid (pole_days in 1..3, base_days >= 3) windows satisfying:
        total_pattern_days = pole_days + base_days <= max_total_pattern_days (20)
    Enforces the 5-year validated V2 institutional edge filters and returns the
    highest-scoring pre-breakout setup + its Daily Volume Profile.
    """
    u_cfg = cfg["universe"]
    p_cfg = cfg["pattern"]

    if df_daily is None or len(df_daily) < int(u_cfg["min_history_days"]):
        return None, None

    df = df_daily if df_daily.index.is_monotonic_increasing else df_daily.sort_index()
    opens = df["Open"].to_numpy(dtype=float)
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)
    vols = df["Volume"].to_numpy(dtype=float)
    n = len(closes)

    last_close = float(closes[-1])
    if last_close < float(u_cfg["min_price"]) or last_close > float(u_cfg["max_price"]):
        return None, None

    # ------------------------------------------------------------------
    # V2 Pre-Signal Institutional Trend, Leadership & Regime Checks (O(1))
    # ------------------------------------------------------------------
    as_of_ts = pd.Timestamp(df.index[-1])
    regime = evaluate_nifty_regime(nifty_df, as_of_ts=as_of_ts, cfg=cfg)
    if not regime["regime_ok"]:
        return None, None

    # Primary Uptrend (200 SMA; falls back cleanly to available history if young listing <200 bars)
    sma200_len = min(200, n)
    sma200 = float(np.mean(closes[-sma200_len:]))
    above_200sma = last_close > sma200
    if bool(p_cfg.get("require_above_200sma", True)) and not above_200sma:
        return None, None

    # 52-Week High Proximity (252 trading days or available history)
    hi_52w = float(np.max(highs[-min(252, n) :]))
    pct_from_52w = ((last_close / hi_52w) - 1.0) * 100.0 if hi_52w > 0 else 0.0
    max_below_52w = float(p_cfg.get("max_pct_below_52w_high", 25.0))
    if pct_from_52w < -abs(max_below_52w):
        return None, None

    # Minervini Stage-2 Uptrend badge (informational + highlight)
    sma50 = float(np.mean(closes[-min(50, n) :]))
    sma150 = float(np.mean(closes[-min(150, n) :]))
    sma200_20ago = float(np.mean(closes[max(0, n - 220) : max(1, n - 20)])) if n >= 220 else sma200 * 0.99
    stage2_uptrend = bool(
        above_200sma and last_close > sma50 and sma50 > sma150 and sma150 > sma200 and sma200 >= sma200_20ago
    )

    # ATR14% (Wilder/SMA 14-day true range as % of close)
    tr_slice_lo = max(1, n - 14)
    tr_vals = np.maximum.reduce(
        [
            highs[tr_slice_lo:] - lows[tr_slice_lo:],
            np.abs(highs[tr_slice_lo:] - closes[tr_slice_lo - 1 : n - 1]),
            np.abs(lows[tr_slice_lo:] - closes[tr_slice_lo - 1 : n - 1]),
        ]
    )
    atr14_pct = (float(np.mean(tr_vals)) / last_close * 100.0) if (len(tr_vals) > 0 and last_close > 0) else 4.0

    # 20-day Median Daily Turnover (INR Crores)
    turn_20 = (closes[-min(20, n) :] * vols[-min(20, n) :]) / 1e7
    med_turnover_cr = float(np.median(turn_20)) if len(turn_20) > 0 else 5.0

    # 63-day Relative Strength vs NIFTY 50 (if available; else vs 0% baseline)
    stock_ret63 = ((last_close / float(closes[n - 64])) - 1.0) * 100.0 if (n >= 64 and closes[n - 64] > 0) else (
        ((last_close / float(closes[0])) - 1.0) * 100.0 if closes[0] > 0 else 0.0
    )
    nifty_ret63 = float(regime["ret63_pct"]) if regime["ret63_pct"] is not None else 0.0
    rs63_vs_nifty = stock_ret63 - nifty_ret63

    # Signal Bar CLV (Close Location Value within today's range)
    sig_rng = highs[-1] - lows[-1]
    sig_clv = float((closes[-1] - lows[-1]) / sig_rng) if sig_rng > 0 else 0.5

    max_total_days = int(p_cfg["max_total_pattern_days"])  # 20 days!
    min_pole = int(p_cfg["min_pole_days"])
    max_pole = int(p_cfg["max_pole_days"])
    min_base = int(p_cfg["min_base_days"])
    max_pole_gain = float(p_cfg.get("max_pole_high_gain_pct", 42.0))
    min_sig_pos = float(p_cfg.get("min_signal_pos_in_box", 0.25))

    dates: Optional[List[str]] = None
    best_candidate: Optional[Tuple[DailyPatternSetup, int]] = None

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

            # Pole slice — fast O(1) price-action reject BEFORE computing 20d median/max!
            p_opens = opens[pole_start : pole_end + 1]
            p_closes = closes[pole_start : pole_end + 1]
            d1_chg_pct = (p_closes[0] / pre_close - 1.0) * 100.0
            if d1_chg_pct < 5.0 or p_closes[0] <= p_opens[0]:
                continue

            p_highs = highs[pole_start : pole_end + 1]
            p_lows = lows[pole_start : pole_end + 1]
            p_vols = vols[pole_start : pole_end + 1]

            pole_close = float(p_closes[-1])
            pole_close_gain_pct = (pole_close / pre_close - 1.0) * 100.0
            if pole_close_gain_pct < float(p_cfg["min_pole_close_gain_pct"]):
                continue

            pole_high = float(np.max(p_highs))
            pole_high_gain_pct = (pole_high / pre_close - 1.0) * 100.0
            if pole_high_gain_pct < float(p_cfg["min_pole_high_gain_pct"]) or pole_high_gain_pct > max_pole_gain:
                continue

            # Pre-pole 20d volume baseline (only computed for genuine +15% pole candidates!)
            pre_vols = vols[max(0, pole_start - 20) : pole_start]
            pos_pre_vols = pre_vols[pre_vols > 0]
            if len(pos_pre_vols) < 5:
                continue
            pre_med_vol = float(np.median(pos_pre_vols))
            if pre_med_vol <= 0:
                continue

            d1_rvol = p_vols[0] / pre_med_vol
            if d1_rvol < 2.0:
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

            pole_peak_vol = float(np.max(p_vols))
            pole_total_vol = float(np.sum(p_vols))
            pole_rvol = pole_peak_vol / pre_med_vol
            if pole_rvol < float(p_cfg["min_pole_rvol"]):
                continue

            pole_value_cr = (pole_peak_vol * pole_close) / 1e7
            if pole_value_cr < float(u_cfg["min_pole_value_cr"]):
                continue

            prior_20d_high = float(np.max(highs[max(0, pole_start - 20) : pole_start]))
            prior_40d_high = float(np.max(highs[max(0, pole_start - 40) : pole_start]))
            pre_pole_low = float(np.min(lows[max(0, pole_start - 5) : pole_start + 1]))
            broke_20d = pole_high > prior_20d_high and pole_close > prior_20d_high * 0.97
            broke_40d = pole_high > prior_40d_high
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

            # V2 Gate 4: Upper-Flag Coil Position (eliminates bottom-25% falling knives)
            box_span = max(shelf_high - shelf_low, 1e-6)
            sig_pos_in_box = float((last_close - shelf_low) / box_span)
            if sig_pos_in_box < min_sig_pos:
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

            # Count tight dojis / spinning tops / inside bars in base
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

            # Lazy date string formatting (only runs if a pattern actually passes all hard gates!)
            if dates is None:
                dates = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in df.index]

            # ==================================================================
            # V2 EVIDENCE-CALIBRATED 5-PILLAR SCORE (100 POINTS)
            # Proven monotonic A+ > A > B on BOTH IS (2022-24) & OOS (2025-26)
            # ==================================================================
            highlights: List[str] = []
            dryup_pct = (1.0 - dryup_ratio) * 100.0
            base_to_pole_pct = base_to_pole_ratio * 100.0
            dist_to_bo_pct = max(0.0, (shelf_high - last_close) / last_close * 100.0)

            # Pillar 1: Institutional Pole Quality (25 pts)
            # Sweet spot: 24-36% pole gain & 5-28x institutional RVOL (penalizes >50x news blow-offs)
            if 24.0 <= pole_high_gain_pct <= 36.0:
                s_pg = 14.0
            elif (19.0 <= pole_high_gain_pct < 24.0) or (36.0 < pole_high_gain_pct <= 42.0):
                s_pg = 10.5
            else:
                s_pg = 7.0

            if 5.0 <= pole_rvol <= 28.0:
                s_pr = 11.0
            elif 28.0 < pole_rvol <= 50.0:
                s_pr = 8.0
            else:
                s_pr = 4.5
            s_pole = round(s_pg + s_pr, 1)

            # Pillar 2: Upper-Flag Coil & Box Geometry (25 pts)
            # Sweet spot: Close coiled in upper 35% of box (pos >= 0.65) + 5.5-9.5% healthy flag box
            if sig_pos_in_box >= 0.65:
                s_pos = 14.0
            elif sig_pos_in_box >= 0.45:
                s_pos = 10.0
            else:
                s_pos = 7.0

            if 5.5 <= wick_box_pct <= 9.5:
                s_bw = 11.0
            elif 9.5 < wick_box_pct <= 12.0:
                s_bw = 8.5
            else:
                s_bw = 5.5
            s_flag = round(s_pos + s_bw, 1)

            # Pillar 3: Momentum & Volatility Leadership (20 pts)
            if atr14_pct >= 5.2:
                s_atr = 11.0
            elif atr14_pct >= 4.0:
                s_atr = 8.0
            else:
                s_atr = 4.0

            if rs63_vs_nifty >= 25.0:
                s_rs = 9.0
            elif rs63_vs_nifty >= 10.0:
                s_rs = 6.5
            else:
                s_rs = 3.5
            s_mom = round(s_atr + s_rs, 1)

            # Pillar 4: Healthy Supply Dry-Up & Active Bid (20 pts)
            if dryup_pct <= 85.0:
                s_du = 12.0
            elif dryup_pct <= 94.0:
                s_du = 8.5
            else:
                s_du = 5.5
            s_clv = 8.0 if sig_clv >= 0.35 else 5.0
            s_vol = round(s_du + s_clv, 1)

            # Pillar 5: Liquidity Sweet Spot & 52w Leadership (10 pts)
            s_turn = 6.0 if med_turnover_cr <= 25.0 else 2.0
            s_52 = 4.0 if pct_from_52w >= -12.0 else 2.5
            s_lead = round(s_turn + s_52, 1)

            total_score = round(s_pole + s_flag + s_mom + s_vol + s_lead, 1)
            grade = _grade(total_score)

            highlights.append(
                f"Pole ({pole_days}d): +{pole_high_gain_pct:.1f}% high on {pole_rvol:.1f}x RVOL (₹{pole_value_cr:.1f} Cr)"
            )
            highlights.append(
                f"Upper-Flag Coil (Day {total_pattern_days}/20): pos {sig_pos_in_box*100:.0f}% of box, "
                f"wick box {wick_box_pct:.1f}%, close box {close_spread_pct:.1f}%"
            )
            highlights.append(
                f"Trend/Leader: Above 200 SMA, {pct_from_52w:+.1f}% from 52wH, 3m RS {rs63_vs_nifty:+.0f}% vs NIFTY"
                + (" (Stage-2)" if stage2_uptrend else "")
            )
            if has_spring_bar:
                highlights.append("Dry stop-sweep / spring reclaim candle near shelf floor")
            highlights.append(f"Volume Dry-Up: {dryup_pct:.1f}% collapse | ATR14 {atr14_pct:.1f}%")

            # Pattern label
            if has_spring_bar and contraction_ratio <= 0.45:
                pat_label = "High-Tight Flag + Spring Coil"
            elif sig_pos_in_box >= 0.65 and dryup_pct >= 75.0:
                pat_label = "Upper-Shelf Coiled Flag"
            elif contraction_ratio <= 0.40 and dryup_pct >= 85.0:
                pat_label = "High-Tight Flag + VCP Dry-Up"
            elif tight_dojis >= 3:
                pat_label = "Tight Shelf + NR Doji Coil"
            else:
                pat_label = "Post-Pole Tight Flag"

            # ------------------------------------------------------------------
            # Actionable V2 Trade Plan (Pre-Breakout Entry < BO + Option A SL/TP)
            # ------------------------------------------------------------------
            tp_cfg = cfg.get("trade_plan", {})
            below_bo_min_pct = float(tp_cfg.get("entry_below_bo_min_pct", 0.8))
            low_frac = float(tp_cfg.get("entry_flag_low_frac", 0.15))
            high_frac = float(tp_cfg.get("entry_flag_high_frac", 0.72))
            use_tick = bool(tp_cfg.get("round_to_nse_tick", True))

            box_w = max(shelf_high - shelf_low, shelf_high * 0.02)
            bo_lvl = _round_price(shelf_high, use_tick)

            # Entry zone strictly inside the flag shelf and below breakout level
            raw_zone_low = max(shelf_low * 1.003, shelf_low + low_frac * box_w)
            raw_zone_high = min(shelf_high * (1.0 - below_bo_min_pct / 100.0), shelf_low + high_frac * box_w)
            if raw_zone_high <= raw_zone_low:
                raw_zone_low = shelf_low * 1.003
                raw_zone_high = shelf_high * (1.0 - below_bo_min_pct / 100.0)
            if shelf_low * 1.002 <= last_close < raw_zone_low:
                raw_zone_low = last_close

            entry_zone_low = _round_price(raw_zone_low, use_tick)
            entry_zone_high = _round_price(raw_zone_high, use_tick)
            if entry_zone_high >= bo_lvl:
                entry_zone_high = _round_price(bo_lvl * (1.0 - below_bo_min_pct / 100.0), use_tick)

            entry_price = _round_price(min(max(last_close, entry_zone_low), entry_zone_high), use_tick)
            if entry_price >= bo_lvl:
                entry_price = _round_price(bo_lvl * (1.0 - below_bo_min_pct / 100.0), use_tick)

            entry_below_bo_pct = round((bo_lvl - entry_price) / bo_lvl * 100.0, 2)

            plan = compute_trade_plan(
                entry_price=entry_price,
                shelf_low=shelf_low,
                shelf_high=shelf_high,
                pole_high=pole_high,
                pre_close=float(pre_close),
                cfg=cfg,
            )
            stop = float(plan["invalidation_stop"])
            risk_pct_bo = round((bo_lvl - stop) / bo_lvl * 100.0, 2)

            setup = DailyPatternSetup(
                symbol=symbol,
                company_name=company_name,
                scan_date=dates[-1],
                pattern_label=pat_label,
                grade=grade,
                total_score=total_score,
                pillar_scores={
                    "pole_quality": s_pole,
                    "flag_coil": s_flag,
                    "momentum_atr": s_mom,
                    "supply_dryup": s_vol,
                    "leadership_liq": s_lead,
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
                vah=None,
                poc=None,
                val=None,
                lvn_zone=None,
                has_unfilled_lvn=False,
                above_vah=False,
                breakout_level=bo_lvl,
                entry_price=entry_price,
                entry_zone_low=entry_zone_low,
                entry_zone_high=entry_zone_high,
                entry_below_bo_pct=entry_below_bo_pct,
                entry_breakout=bo_lvl,
                entry_anticipatory=entry_price,
                invalidation_stop=stop,
                sl_buffer_pct=float(plan["sl_buffer_pct"]),
                tp1=float(plan["tp1"]),
                tp2=float(plan["tp2"]),
                tp3=float(plan["tp3"]),
                tp1_pct=float(plan["tp1_pct"]),
                tp2_pct=float(plan["tp2_pct"]),
                tp3_pct=float(plan["tp3_pct"]),
                risk_pct=float(plan["risk_pct"]),
                risk_pct_anticipatory=float(plan["risk_pct"]),
                risk_pct_breakout=risk_pct_bo,
                rr_to_tp1=float(plan["rr_to_tp1"]),
                rr_to_tp2=float(plan["rr_to_tp2"]),
                rr_to_tp3=float(plan["rr_to_tp3"]),
                sl_mode=str(plan["sl_mode"]),
                hard_stop_gtt=float(plan["hard_stop_gtt"]),
                hard_stop_gtt_pct=float(plan["hard_stop_gtt_pct"]),
                intraday_gtt_sl=float(plan["intraday_gtt_sl"]),
                intraday_gtt_risk_pct=float(plan["intraday_gtt_risk_pct"]),
                tp1_weight=float(plan["tp1_weight"]),
                tp2_weight=float(plan["tp2_weight"]),
                tp3_weight=float(plan["tp3_weight"]),
                above_200sma=bool(above_200sma),
                stage2_uptrend=bool(stage2_uptrend),
                pct_from_52w_high=round(float(pct_from_52w), 2),
                sig_pos_in_box=round(float(sig_pos_in_box), 3),
                sig_clv=round(float(sig_clv), 3),
                atr14_pct=round(float(atr14_pct), 2),
                med_turnover_cr=round(float(med_turnover_cr), 2),
                rs63_vs_nifty=round(float(rs63_vs_nifty), 1),
                nifty_ret20_pct=regime["ret20_pct"],
                nifty_above_50sma=regime["above_50sma"],
                highlights=highlights,
            )

            if best_candidate is None or setup.total_score > best_candidate[0].total_score:
                best_candidate = (setup, pole_start)

    if best_candidate is None:
        return None, None

    best_setup, best_pole_start = best_candidate
    best_vp: Optional[DailyVolumeProfile] = None
    if compute_vp:
        vp_window = df.iloc[max(0, best_pole_start - 15) :]
        best_vp = compute_daily_volume_profile(
            vp_window,
            num_bins=int(p_cfg["vp_bins"]),
            value_area_pct=float(p_cfg["value_area_pct"]),
            shelf_low=best_setup.shelf_low,
        )
        if best_vp is not None:
            best_setup.vah = round(float(best_vp.vah), 2)
            best_setup.poc = round(float(best_vp.poc), 2)
            best_setup.val = round(float(best_vp.val), 2)
            best_setup.lvn_zone = (
                [round(float(best_vp.lvn_zone[0]), 2), round(float(best_vp.lvn_zone[1]), 2)]
                if best_vp.lvn_zone
                else None
            )
            best_setup.has_unfilled_lvn = bool(best_vp.has_unfilled_lvn)
            best_setup.above_vah = bool(last_close >= best_vp.vah * 0.992)

    return best_setup, best_vp
