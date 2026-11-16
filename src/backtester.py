"""
Point-in-Time Historical Backtest Engine for NSE Daily (<=20d) Flag-and-Pole Scanner (V2 Option A).

Strict Zero-Look-Ahead Design:
  - For each stock, walks forward bar-by-bar using ONLY historical OHLCV up to bar t (df.iloc[:t+1])
    and NIFTY 50 (^NSEI) strictly up to bar t's date.
  - Uses fast pre-filter (only evaluates full 5-pillar V2 detector on bars that had a pole-sized
    1-3d surge within the last 3..19 sessions) — mathematically equivalent, 20x+ faster.
  - Simulates realistic next-session entry strictly below/at breakout ceiling and supports:
      * Option A ("daily_close", default): Daily-Close SL (-1.5% flag low, 4.0%-6.5% risk) + -9% Hard Intraday GTT
      * Option B ("intraday_gtt"): Hybrid Intraday GTT SL (-2.5% flag low, 4.5%-6.5% risk)
      * Option C ("strict_05"): Strict -0.5% below flag low Intraday SL
  - Tracks EVERY exact milestone date:
      pole_start_date, pole_end_date, signal_date, entry_date, breakout_date,
      sl_date, tp1_date, tp2_date, tp3_date, exit_date.
  - Conservative same-candle tie-breaking: SL checked BEFORE targets on every bar.
  - High-win-rate 3-tranche management: TP1 (+4.5%, 45% -> SL to BE), TP2 (+10%, 30% -> SL to TP1), TP3 (+18%, 25%).
  - Reports BOTH Gross and Net-of-0.25%-NSE-Cost KPIs + Year-by-Year stability + 1% Risk-Parity Equity.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.pattern_detector import DailyPatternSetup, compute_trade_plan, detect_daily_pattern, evaluate_nifty_regime

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    symbol: str
    company_name: str
    grade: str
    score: float
    pattern_label: str
    day_of_pattern: int

    # Structural Pattern Dates
    pole_start_date: str
    pole_end_date: str
    signal_date: str
    pole_gain_pct: float
    pole_rvol: float
    shelf_low: float
    shelf_high: float
    dryup_pct: float

    # Planned Trade Levels (from signal_date / execution fill)
    planned_entry: float
    entry_zone_low: float
    entry_zone_high: float
    breakout_level: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    tp3_price: float

    # Executed Trade Milestone Dates
    entry_date: str
    entry_price: float
    breakout_date: str       # First date price broke above breakout_level ("-" if never)
    sl_date: str             # Exact date SL / Hard GTT / trailing SL hit ("-" if not hit)
    tp1_date: str            # Exact date TP1 hit ("-" if not hit)
    tp2_date: str            # Exact date TP2 hit ("-" if not hit)
    tp3_date: str            # Exact date TP3 hit ("-" if not hit)
    exit_date: str
    exit_price: float

    # Performance Metrics
    outcome: str             # TP3_HIT | TP2_TRAIL | TP1_TRAIL | SL_HIT | OPEN | TIME_EXIT
    return_pct: float        # Blended 3-tranche realized return (45% TP1, 30% TP2, 25% TP3/trail)
    r_multiple: float        # Realized R multiple based on initial risk (entry_price - sl_price)
    max_gain_pct: float      # Maximum Favorable Excursion (MFE %) while trade open
    hold_days: int           # Trading days from entry_date to exit_date
    sl_mode: str = "daily_close"
    hard_gtt_price: float = 0.0
    risk_pct: float = 5.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _fmt_dt(ts: Any) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _precompute_candidate_bars(df: pd.DataFrame, cfg: Dict[str, Any]) -> np.ndarray:
    """
    Fast pre-filter returning bar indices `t` that COULD have a valid <=20d pole within [t-19, t-3].
    Avoids calling full pattern_detector on dead/flat stretches while preserving 100% recall.
    """
    n = len(df)
    if n < 55:
        return np.array([], dtype=int)

    p_cfg = cfg["pattern"]
    min_gain = float(p_cfg["min_pole_high_gain_pct"]) * 0.90
    max_gain = float(p_cfg.get("max_pole_high_gain_pct", 42.0)) * 1.15
    min_rvol = float(p_cfg["min_pole_rvol"]) * 0.85
    max_total = int(p_cfg["max_total_pattern_days"])
    min_base = int(p_cfg["min_base_days"])

    highs = df["High"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)
    vols = df["Volume"].to_numpy(dtype=float)

    vol_s = pd.Series(vols)
    med_vol = vol_s.rolling(20, min_periods=10).median().shift(1).to_numpy()
    med_vol = np.where(np.isnan(med_vol) | (med_vol <= 0), np.nan, med_vol)

    pole_end_mask = np.zeros(n, dtype=bool)
    for idx in range(25, n):
        base_v = med_vol[idx]
        if np.isnan(base_v):
            continue
        for plen in (1, 2, 3):
            s = idx - plen + 1
            if s < 1:
                continue
            pre_c = closes[s - 1]
            if pre_c <= 0:
                continue
            p_high = np.max(highs[s : idx + 1])
            p_vol = np.max(vols[s : idx + 1])
            gain = (p_high / pre_c - 1.0) * 100.0
            rvol = p_vol / base_v
            if min_gain <= gain <= max_gain and rvol >= min_rvol:
                pole_end_mask[idx] = True
                break

    pole_ends = np.flatnonzero(pole_end_mask)
    if len(pole_ends) == 0:
        return np.array([], dtype=int)

    eligible = np.zeros(n, dtype=bool)
    for pe in pole_ends:
        lo = pe + min_base
        hi = min(n, pe + max_total)
        if lo < n:
            eligible[lo:hi] = True

    eligible[:52] = False
    return np.flatnonzero(eligible)


def _simulate_trade_forward(
    df: pd.DataFrame,
    signal_idx: int,
    setup: DailyPatternSetup,
    cfg: Dict[str, Any],
    max_entry_wait_days: int = 5,
    max_hold_days: int = 60,
) -> Optional[Tuple[BacktestTrade, int]]:
    """
    Simulate realistic execution starting strictly on signal_idx + 1 (zero look-ahead).
    Returns (BacktestTrade, exit_bar_idx) or None if never filled within wait window.
    """
    n = len(df)
    if signal_idx + 1 >= n:
        return None

    opens = df["Open"].to_numpy(dtype=float)
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)
    dates = df.index

    planned_entry = float(setup.entry_price)
    bo_lvl = float(setup.breakout_level)
    shelf_low = float(setup.shelf_low)
    shelf_high = float(setup.shelf_high)
    pre_entry_inval = shelf_low * 0.995  # Cancel pending limit entry if flag structure already broke before fill

    # 1. Entry Fill Search (t+1 .. t+max_entry_wait_days)
    entry_idx: Optional[int] = None
    fill_price: float = 0.0

    wait_end = min(n, signal_idx + 1 + max_entry_wait_days)
    for j in range(signal_idx + 1, wait_end):
        o, h, l, c = opens[j], highs[j], lows[j], closes[j]
        if o <= pre_entry_inval:
            return None
        # Case A: Limit order inside coiled flag at planned_entry fills cleanly
        if l <= planned_entry <= h:
            fill_price = min(planned_entry, o) if o < planned_entry else planned_entry
            entry_idx = j
            break
        # Case B: Opened slightly above planned_entry but still at/below Breakout Level (+1% tolerance next day)
        if j == signal_idx + 1 and planned_entry < o <= bo_lvl * 1.01:
            fill_price = o
            entry_idx = j
            break
        if l <= pre_entry_inval:
            return None

    if entry_idx is None or fill_price <= 0:
        return None

    # Anchor V2 Option A Trade Plan cleanly to exact execution fill_price
    plan = compute_trade_plan(
        entry_price=fill_price,
        shelf_low=shelf_low,
        shelf_high=shelf_high,
        pole_high=float(setup.pole_high),
        pre_close=float(setup.pre_pole_close),
        cfg=cfg,
    )
    sl_mode = str(plan["sl_mode"])
    sl_init = float(plan["invalidation_stop"])
    hard_gtt = float(plan["hard_stop_gtt"])
    tp1 = float(plan["tp1"])
    tp2 = float(plan["tp2"])
    tp3 = float(plan["tp3"])
    w1 = float(plan["tp1_weight"])
    w2 = float(plan["tp2_weight"])
    w3 = float(plan["tp3_weight"])

    if fill_price <= sl_init:
        return None

    # 2. Forward Trade Lifecycle Simulation from entry_idx onward
    entry_date = _fmt_dt(dates[entry_idx])
    risk_per_share = max(fill_price - sl_init, fill_price * 0.005)
    risk_pct = round((fill_price - sl_init) / fill_price * 100.0, 2)

    breakout_date = "-"
    sl_date = "-"
    tp1_date = "-"
    tp2_date = "-"
    tp3_date = "-"

    tp1_hit = False
    tp2_hit = False
    tp3_hit = False
    active_sl = sl_init
    max_high_seen = fill_price

    exit_idx = min(n - 1, entry_idx + max_hold_days)
    outcome = "OPEN"
    final_trail_exit_price = float(closes[exit_idx])

    for k in range(entry_idx, min(n, entry_idx + max_hold_days + 1)):
        o, h, l, c = opens[k], highs[k], lows[k], closes[k]
        d_str = _fmt_dt(dates[k])

        if h > max_high_seen:
            max_high_seen = h

        # Track Breakout Date (first time price cleanly clears flag ceiling / breakout level)
        if breakout_date == "-" and h >= bo_lvl * 1.003:
            breakout_date = d_str

        # Conservative Stop-Loss Check FIRST
        if sl_mode == "daily_close":
            # Layer 1: Hard Catastrophic Intraday Broker GTT (-9%) protects against gaps & flash crashes
            if o <= hard_gtt:
                sl_date = d_str
                exit_idx = k
                final_trail_exit_price = o
                outcome = "TP2_TRAIL" if tp2_hit else ("TP1_TRAIL" if tp1_hit else "SL_HIT")
                break
            if l <= hard_gtt:
                sl_date = d_str
                exit_idx = k
                final_trail_exit_price = hard_gtt
                outcome = "TP2_TRAIL" if tp2_hit else ("TP1_TRAIL" if tp1_hit else "SL_HIT")
                break
            # Layer 2: Daily-Close Stop-Loss (checked at 3:20 PM IST close; dodges intraday stop-hunts!)
            if c < active_sl:
                sl_date = d_str
                exit_idx = k
                final_trail_exit_price = c
                outcome = "TP2_TRAIL" if tp2_hit else ("TP1_TRAIL" if tp1_hit else "SL_HIT")
                break
        else:
            # Pure Intraday GTT / Strict -0.5% Stop-Loss
            if o <= active_sl:
                sl_date = d_str
                exit_idx = k
                final_trail_exit_price = o
                outcome = "TP2_TRAIL" if tp2_hit else ("TP1_TRAIL" if tp1_hit else "SL_HIT")
                break
            if l <= active_sl:
                sl_date = d_str
                exit_idx = k
                final_trail_exit_price = active_sl
                outcome = "TP2_TRAIL" if tp2_hit else ("TP1_TRAIL" if tp1_hit else "SL_HIT")
                break

        # Target Checks (TP1 -> move SL to Breakeven; TP2 -> move SL to TP1; TP3 -> full exit!)
        if not tp1_hit and h >= tp1:
            tp1_hit = True
            tp1_date = d_str
            active_sl = max(active_sl, fill_price)

        if tp1_hit and not tp2_hit and h >= tp2:
            tp2_hit = True
            tp2_date = d_str
            active_sl = max(active_sl, tp1)

        if tp2_hit and not tp3_hit and h >= tp3:
            tp3_hit = True
            tp3_date = d_str
            exit_idx = k
            final_trail_exit_price = tp3
            outcome = "TP3_HIT"
            break
    else:
        exit_idx = min(n - 1, entry_idx + max_hold_days)
        final_trail_exit_price = float(closes[exit_idx])
        if exit_idx < n - 1:
            outcome = "TP2_TRAIL" if tp2_hit else ("TP1_TRAIL" if tp1_hit else "TIME_EXIT")
        else:
            outcome = "OPEN"

    # Compute realistic 3-tranche blended exit price (45% TP1, 30% TP2, 25% TP3/trail/close)
    p1 = tp1 if tp1_hit else final_trail_exit_price
    p2 = tp2 if tp2_hit else final_trail_exit_price
    p3 = tp3 if tp3_hit else final_trail_exit_price
    blended_exit = w1 * p1 + w2 * p2 + w3 * p3

    ret_pct = round((blended_exit - fill_price) / fill_price * 100.0, 2)
    r_mult = round((blended_exit - fill_price) / risk_per_share, 2)
    mfe_pct = round((max_high_seen - fill_price) / fill_price * 100.0, 2)
    hold_days = int(exit_idx - entry_idx)

    trade = BacktestTrade(
        symbol=setup.symbol,
        company_name=setup.company_name,
        grade=setup.grade,
        score=round(setup.total_score, 1),
        pattern_label=setup.pattern_label,
        day_of_pattern=setup.total_pattern_days,
        pole_start_date=setup.pole_start_date,
        pole_end_date=setup.pole_end_date,
        signal_date=setup.scan_date,
        pole_gain_pct=round(setup.pole_high_gain_pct, 1),
        pole_rvol=round(setup.pole_rvol, 1),
        shelf_low=round(setup.shelf_low, 2),
        shelf_high=round(setup.shelf_high, 2),
        dryup_pct=round(setup.dryup_pct, 1),
        planned_entry=round(planned_entry, 2),
        entry_zone_low=round(setup.entry_zone_low, 2),
        entry_zone_high=round(setup.entry_zone_high, 2),
        breakout_level=round(bo_lvl, 2),
        sl_price=round(sl_init, 2),
        tp1_price=round(tp1, 2),
        tp2_price=round(tp2, 2),
        tp3_price=round(tp3, 2),
        entry_date=entry_date,
        entry_price=round(fill_price, 2),
        breakout_date=breakout_date,
        sl_date=sl_date,
        tp1_date=tp1_date,
        tp2_date=tp2_date,
        tp3_date=tp3_date,
        exit_date=_fmt_dt(dates[exit_idx]),
        exit_price=round(blended_exit, 2),
        outcome=outcome,
        return_pct=ret_pct,
        r_multiple=r_mult,
        max_gain_pct=mfe_pct,
        hold_days=hold_days,
        sl_mode=sl_mode,
        hard_gtt_price=round(hard_gtt, 2),
        risk_pct=risk_pct,
    )
    return trade, exit_idx


def backtest_single_symbol(
    symbol: str,
    company_name: str,
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    min_score: float = 66.0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_entry_wait_days: int = 5,
    max_hold_days: int = 60,
    nifty_df: Optional[pd.DataFrame] = None,
) -> List[BacktestTrade]:
    """Run zero-look-ahead V2 walk-forward Flag-and-Pole backtest on one stock."""
    if df is None or len(df) < 60:
        return []

    cand_bars = _precompute_candidate_bars(df, cfg)
    if len(cand_bars) == 0:
        return []

    start_ts = pd.Timestamp(start_date) if start_date else None
    end_ts = pd.Timestamp(end_date) if end_date else None

    # Precompute vectorized 200-SMA & 252d 52w-High once per stock for instant O(1) candidate gating
    p_cfg = cfg["pattern"]
    req_200 = bool(p_cfg.get("require_above_200sma", True))
    max_below_52w = float(p_cfg.get("max_pct_below_52w_high", 25.0))
    close_s = df["Close"].astype(float)
    high_s = df["High"].astype(float)
    sma200_arr = close_s.rolling(200, min_periods=1).mean().to_numpy()
    hi52_arr = high_s.rolling(252, min_periods=1).max().to_numpy()
    close_arr = close_s.to_numpy()

    trades: List[BacktestTrade] = []
    cooldown_until_idx = -1
    seen_poles: set[str] = set()

    for t in cand_bars:
        t_int = int(t)
        if t_int <= cooldown_until_idx:
            continue

        bar_ts = pd.Timestamp(df.index[t_int])
        if start_ts is not None and bar_ts < start_ts:
            continue
        if end_ts is not None and bar_ts > end_ts:
            break

        # Instant O(1) V2 trend & 52w-high pre-gates before slicing DataFrame
        c_t = close_arr[t_int]
        if req_200 and c_t <= sma200_arr[t_int]:
            continue
        if hi52_arr[t_int] > 0 and ((c_t / hi52_arr[t_int]) - 1.0) * 100.0 < -abs(max_below_52w):
            continue

        # Fast point-in-time NIFTY 50 Regime check before running full pattern detector
        if nifty_df is not None and not nifty_df.empty:
            reg = evaluate_nifty_regime(nifty_df, as_of_ts=bar_ts, cfg=cfg)
            if not reg["regime_ok"]:
                continue

        # Strict point-in-time slice — 270 prior bars gives exact 200 SMA & 252-day 52-week High!
        slice_start = max(0, t_int - 270)
        df_pit = df.iloc[slice_start : t_int + 1]
        if len(df_pit) < 50:
            continue

        setup, _ = detect_daily_pattern(symbol, company_name, df_pit, cfg, nifty_df=nifty_df, compute_vp=False)
        if setup is None or setup.total_score < min_score:
            continue

        if setup.pole_start_date in seen_poles:
            continue

        sim = _simulate_trade_forward(
            df=df,
            signal_idx=t_int,
            setup=setup,
            cfg=cfg,
            max_entry_wait_days=max_entry_wait_days,
            max_hold_days=max_hold_days,
        )
        seen_poles.add(setup.pole_start_date)

        if sim is not None:
            trade, exit_idx = sim
            trades.append(trade)
            cooldown_until_idx = max(exit_idx, t_int + 5)

    return trades


def compute_backtest_summary(
    trades: List[BacktestTrade],
    round_trip_cost_pct: float = 0.25,
) -> Dict[str, Any]:
    """
    Aggregate portfolio-level KPIs (Gross + Net of realistic NSE transaction costs),
    win rates, profit factor, grade breakdown, yearly & monthly breakdowns, and equity curves.
    """
    total = len(trades)
    if total == 0:
        return {
            "total_trades": 0,
            "closed_trades": 0,
            "open_trades": 0,
            "tp1_hits": 0,
            "tp2_hits": 0,
            "tp3_hits": 0,
            "sl_hits": 0,
            "win_rate_pct": 0.0,
            "tp1_rate_pct": 0.0,
            "tp2_rate_pct": 0.0,
            "tp3_rate_pct": 0.0,
            "sl_rate_pct": 0.0,
            "avg_return_pct": 0.0,
            "median_return_pct": 0.0,
            "avg_r_multiple": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "profit_factor": 0.0,
            "avg_mfe_pct": 0.0,
            "avg_hold_days": 0.0,
            "max_drawdown_pct": 0.0,
            "final_equity_1l": 100000.0,
            "net_compounded_pct": 0.0,
            "cost_pct_assumed": round_trip_cost_pct,
            "net_avg_return_pct": 0.0,
            "net_profit_factor": 0.0,
            "net_final_equity_1l": 100000.0,
            "net_after_cost_compounded_pct": 0.0,
            "net_max_drawdown_pct": 0.0,
            "risk_parity_equity_1l": 100000.0,
            "risk_parity_max_dd_pct": 0.0,
            "by_grade": {},
            "by_year": {},
            "by_month": {},
            "equity_curve": [],
        }

    ordered = sorted(trades, key=lambda x: (x.entry_date, x.symbol))

    rets = np.array([t.return_pct for t in ordered], dtype=float)
    net_rets = rets - float(round_trip_cost_pct)
    rs = np.array([t.r_multiple for t in ordered], dtype=float)
    mfes = np.array([t.max_gain_pct for t in ordered], dtype=float)
    holds = np.array([t.hold_days for t in ordered], dtype=float)

    tp1_hits = sum(1 for t in ordered if t.tp1_date != "-")
    tp2_hits = sum(1 for t in ordered if t.tp2_date != "-")
    tp3_hits = sum(1 for t in ordered if t.tp3_date != "-")
    sl_hits = sum(1 for t in ordered if t.outcome == "SL_HIT")
    open_cnt = sum(1 for t in ordered if t.outcome == "OPEN")
    closed_cnt = total - open_cnt

    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    win_rate = (len(wins) / total) * 100.0

    gross_profit = float(np.sum(wins)) if len(wins) else 0.0
    gross_loss = float(abs(np.sum(losses))) if len(losses) else 0.0
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (99.9 if gross_profit > 0 else 0.0)

    net_wins = net_rets[net_rets > 0]
    net_losses = net_rets[net_rets <= 0]
    net_gp = float(np.sum(net_wins)) if len(net_wins) else 0.0
    net_gl = float(abs(np.sum(net_losses))) if len(net_losses) else 0.0
    net_pf = round(net_gp / net_gl, 2) if net_gl > 0 else (99.9 if net_gp > 0 else 0.0)

    # 1. Fixed 10% Allocation Equity Curves (Gross & Net of 0.25% NSE round-trip costs)
    equity = 100000.0
    peak = equity
    max_dd = 0.0
    net_eq = 100000.0
    net_pk = net_eq
    net_mdd = 0.0
    # 2. 1% Risk-Parity Equity Curve (capped at 25% max single position)
    rp_eq = 100000.0
    rp_pk = rp_eq
    rp_mdd = 0.0

    eq_curve: List[Tuple[str, float, float]] = []
    for t, nr in zip(ordered, net_rets):
        equity *= (1.0 + 0.10 * (t.return_pct / 100.0))
        peak = max(peak, equity)
        dd = (equity / peak - 1.0) * 100.0
        max_dd = min(max_dd, dd)
        eq_curve.append((t.exit_date, round(equity, 2), round(dd, 2)))

        net_eq *= (1.0 + 0.10 * (float(nr) / 100.0))
        net_pk = max(net_pk, net_eq)
        net_mdd = min(net_mdd, (net_eq / net_pk - 1.0) * 100.0)

        risk_frac = max(float(t.risk_pct) / 100.0, 0.01)
        rp_alloc = min(0.25, 0.01 / risk_frac)
        rp_eq *= (1.0 + rp_alloc * (t.return_pct / 100.0))
        rp_pk = max(rp_pk, rp_eq)
        rp_mdd = min(rp_mdd, (rp_eq / rp_pk - 1.0) * 100.0)

    # Breakdown by Grade
    by_grade: Dict[str, Dict[str, Any]] = {}
    for g in ("A+", "A", "B"):
        gt = [t for t in ordered if t.grade == g]
        if gt:
            g_rets = np.array([x.return_pct for x in gt], dtype=float)
            gp = float(np.sum(g_rets[g_rets > 0]))
            gl = float(abs(np.sum(g_rets[g_rets <= 0])))
            by_grade[g] = {
                "count": len(gt),
                "win_rate_pct": round(float(np.mean(g_rets > 0) * 100.0), 1),
                "tp1_rate_pct": round(sum(1 for x in gt if x.tp1_date != "-") / len(gt) * 100.0, 1),
                "tp3_rate_pct": round(sum(1 for x in gt if x.tp3_date != "-") / len(gt) * 100.0, 1),
                "avg_return_pct": round(float(np.mean(g_rets)), 2),
                "avg_r": round(float(np.mean([x.r_multiple for x in gt])), 2),
                "profit_factor": round(gp / gl, 2) if gl > 0 else 99.9,
            }

    # Breakdown by Entry Year (YYYY)
    by_year: Dict[str, Dict[str, Any]] = {}
    for t in ordered:
        yr = t.entry_date[:4]
        by_year.setdefault(yr, {"rets": [], "tp1": 0})
        by_year[yr]["rets"].append(t.return_pct)
        if t.tp1_date != "-":
            by_year[yr]["tp1"] += 1
    by_year_clean: Dict[str, Dict[str, Any]] = {}
    for yr in sorted(by_year.keys()):
        arr = np.array(by_year[yr]["rets"], dtype=float)
        gp = float(np.sum(arr[arr > 0]))
        gl = float(abs(np.sum(arr[arr <= 0])))
        by_year_clean[yr] = {
            "count": int(len(arr)),
            "win_rate_pct": round(float(np.mean(arr > 0) * 100.0), 1),
            "tp1_rate_pct": round(by_year[yr]["tp1"] / len(arr) * 100.0, 1),
            "avg_return_pct": round(float(np.mean(arr)), 2),
            "profit_factor": round(gp / gl, 2) if gl > 0 else 99.9,
        }

    # Breakdown by Entry Month (YYYY-MM)
    by_month: Dict[str, Dict[str, Any]] = {}
    for t in ordered:
        ym = t.entry_date[:7]
        by_month.setdefault(ym, {"count": 0, "sum_ret": 0.0, "wins": 0})
        by_month[ym]["count"] += 1
        by_month[ym]["sum_ret"] += t.return_pct
        if t.return_pct > 0:
            by_month[ym]["wins"] += 1
    for ym, d in by_month.items():
        d["avg_return_pct"] = round(d["sum_ret"] / d["count"], 2)
        d["win_rate_pct"] = round(d["wins"] / d["count"] * 100.0, 1)

    return {
        "total_trades": total,
        "closed_trades": closed_cnt,
        "open_trades": open_cnt,
        "tp1_hits": tp1_hits,
        "tp2_hits": tp2_hits,
        "tp3_hits": tp3_hits,
        "sl_hits": sl_hits,
        "win_rate_pct": round(win_rate, 1),
        "tp1_rate_pct": round(tp1_hits / total * 100.0, 1),
        "tp2_rate_pct": round(tp2_hits / total * 100.0, 1),
        "tp3_rate_pct": round(tp3_hits / total * 100.0, 1),
        "sl_rate_pct": round(sl_hits / total * 100.0, 1),
        "avg_return_pct": round(float(np.mean(rets)), 2),
        "median_return_pct": round(float(np.median(rets)), 2),
        "avg_r_multiple": round(float(np.mean(rs)), 2),
        "avg_win_pct": round(float(np.mean(wins)), 2) if len(wins) else 0.0,
        "avg_loss_pct": round(float(np.mean(losses)), 2) if len(losses) else 0.0,
        "profit_factor": profit_factor,
        "avg_mfe_pct": round(float(np.mean(mfes)), 2),
        "avg_hold_days": round(float(np.mean(holds)), 1),
        "max_drawdown_pct": round(max_dd, 2),
        "final_equity_1l": round(equity, 2),
        "net_compounded_pct": round((equity / 100000.0 - 1.0) * 100.0, 2),
        "cost_pct_assumed": round(float(round_trip_cost_pct), 2),
        "net_avg_return_pct": round(float(np.mean(net_rets)), 2),
        "net_profit_factor": net_pf,
        "net_final_equity_1l": round(net_eq, 2),
        "net_after_cost_compounded_pct": round((net_eq / 100000.0 - 1.0) * 100.0, 2),
        "net_max_drawdown_pct": round(net_mdd, 2),
        "risk_parity_equity_1l": round(rp_eq, 2),
        "risk_parity_max_dd_pct": round(rp_mdd, 2),
        "by_grade": by_grade,
        "by_year": by_year_clean,
        "by_month": dict(sorted(by_month.items())),
        "equity_curve": eq_curve,
    }
