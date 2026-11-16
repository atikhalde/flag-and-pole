"""
Point-in-Time Historical Backtest Engine for NSE Daily (<=20d) Flag-and-Pole Scanner.

Strict Zero-Look-Ahead Design:
  - For each stock, walks forward bar-by-bar using ONLY historical OHLCV up to bar t (df.iloc[:t+1]).
  - Uses fast pre-filter (only evaluates full 5-pillar detector on bars that had a pole-sized
    1-3d surge within the last 3..19 sessions) — mathematically equivalent, 20x+ faster.
  - Simulates realistic next-session entry strictly below/at breakout ceiling, 0.5%-below-flag-low SL,
    and tracks EVERY exact milestone date:
      pole_start_date, pole_end_date, signal_date, entry_date, breakout_date,
      sl_date, tp1_date, tp2_date, tp3_date, exit_date.
  - Conservative same-candle tie-breaking: if both SL and TP occur on the same bar, assumes SL first.
  - Trailing management after TP1 (SL -> Breakeven) and after TP2 (SL -> TP1) + blended 3-tranche P&L.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.pattern_detector import DailyPatternSetup, detect_daily_pattern

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

    # Planned Trade Levels (from signal_date)
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
    sl_date: str             # Exact date SL / trailing SL hit ("-" if not hit)
    tp1_date: str            # Exact date TP1 hit ("-" if not hit)
    tp2_date: str            # Exact date TP2 hit ("-" if not hit)
    tp3_date: str            # Exact date TP3 hit ("-" if not hit)
    exit_date: str
    exit_price: float

    # Performance Metrics
    outcome: str             # TP3_HIT | TP2_TRAIL | TP1_TRAIL | SL_HIT | OPEN | TIME_EXIT
    return_pct: float        # Blended 3-tranche realized return (33% TP1, 33% TP2, 34% TP3/trail)
    r_multiple: float        # Realized R multiple based on initial risk (entry_price - sl_price)
    max_gain_pct: float      # Maximum Favorable Excursion (MFE %) while trade open
    hold_days: int           # Trading days from entry_date to exit_date

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
    min_gain = float(p_cfg["min_pole_high_gain_pct"]) * 0.90  # slight safety margin
    min_rvol = float(p_cfg["min_pole_rvol"]) * 0.85
    max_total = int(p_cfg["max_total_pattern_days"])
    min_base = int(p_cfg["min_base_days"])

    highs = df["High"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)
    vols = df["Volume"].to_numpy(dtype=float)

    vol_s = pd.Series(vols)
    med_vol = vol_s.rolling(20, min_periods=10).median().shift(1).to_numpy()
    med_vol = np.where(np.isnan(med_vol) | (med_vol <= 0), np.nan, med_vol)

    # Identify bars that could be the end of a 1..3 day pole
    pole_end_mask = np.zeros(n, dtype=bool)
    for idx in range(25, n):
        base_v = med_vol[idx]
        if np.isnan(base_v):
            continue
        # Check 1d, 2d, or 3d cumulative pole ending at idx
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
            if gain >= min_gain and rvol >= min_rvol:
                pole_end_mask[idx] = True
                break

    # Any bar `t` where a pole ended between `t - (max_total - 1)` and `t - min_base` is eligible
    cand_indices: List[int] = []
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
    sl_init = float(setup.invalidation_stop)
    tp1 = float(setup.tp1)
    tp2 = float(setup.tp2)
    tp3 = float(setup.tp3)

    # 1. Entry Fill Search (t+1 .. t+max_entry_wait_days)
    entry_idx: Optional[int] = None
    fill_price: float = 0.0

    wait_end = min(n, signal_idx + 1 + max_entry_wait_days)
    for j in range(signal_idx + 1, wait_end):
        o, h, l, c = opens[j], highs[j], lows[j], closes[j]
        # Pattern invalidated before entry if price breaks SL first on open
        if o <= sl_init:
            return None
        # Case A: Limit order inside flag at planned_entry fills cleanly
        if l <= planned_entry <= h:
            # Conservative check: if gap opened below planned_entry (still > SL), fill at Open
            fill_price = min(planned_entry, o) if o < planned_entry else planned_entry
            entry_idx = j
            break
        # Case B: Opened slightly above planned_entry but STILL at/below Breakout Level (+1% tolerance on next day)
        # (e.g., BOROLTD 16 Nov opened at ₹318.90 vs planned ₹311.50 and BO ₹320.20!)
        if j == signal_idx + 1 and planned_entry < o <= bo_lvl * 1.01:
            fill_price = o
            entry_idx = j
            break
        # If broke way out without filling, stop waiting
        if h >= tp1:
            return None
        if l <= sl_init:
            return None

    if entry_idx is None or fill_price <= sl_init:
        return None

    # 2. Forward Trade Lifecycle Simulation from entry_idx onward
    entry_date = _fmt_dt(dates[entry_idx])
    risk_per_share = max(fill_price - sl_init, fill_price * 0.005)

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

        # On the entry candle itself, only consider prices achievable after fill
        bar_low = l
        bar_high = h
        if k == entry_idx:
            # If filled at planned_entry on a green breakout candle whose open was >= planned_entry,
            # intraday low could have touched SL; conservatively keep bar_low = l!
            pass

        if bar_high > max_high_seen:
            max_high_seen = bar_high

        # Track Breakout Date (first time price cleanly clears flag ceiling / breakout level)
        if breakout_date == "-" and bar_high >= bo_lvl * 1.003:
            breakout_date = d_str

        # Conservative Stop-Loss Check FIRST (on gap-down open or intraday low)
        if o <= active_sl:
            sl_date = d_str
            exit_idx = k
            final_trail_exit_price = o
            if tp2_hit:
                outcome = "TP2_TRAIL"
            elif tp1_hit:
                outcome = "TP1_TRAIL"
            else:
                outcome = "SL_HIT"
            break

        if bar_low <= active_sl:
            sl_date = d_str
            exit_idx = k
            final_trail_exit_price = active_sl
            if tp2_hit:
                outcome = "TP2_TRAIL"
            elif tp1_hit:
                outcome = "TP1_TRAIL"
            else:
                outcome = "SL_HIT"
            break

        # Target Checks (TP1 -> move SL to Breakeven; TP2 -> move SL to TP1; TP3 -> full exit!)
        if not tp1_hit and bar_high >= tp1:
            tp1_hit = True
            tp1_date = d_str
            active_sl = max(active_sl, fill_price)  # Move stop to breakeven on remaining tranches

        if tp1_hit and not tp2_hit and bar_high >= tp2:
            tp2_hit = True
            tp2_date = d_str
            active_sl = max(active_sl, tp1)         # Trail stop up to TP1 on final runner tranche

        if tp2_hit and not tp3_hit and bar_high >= tp3:
            tp3_hit = True
            tp3_date = d_str
            exit_idx = k
            final_trail_exit_price = tp3
            outcome = "TP3_HIT"
            break
    else:
        # Reached end of available data or max_hold_days
        exit_idx = min(n - 1, entry_idx + max_hold_days)
        final_trail_exit_price = float(closes[exit_idx])
        if exit_idx < n - 1:
            outcome = "TP2_TRAIL" if tp2_hit else ("TP1_TRAIL" if tp1_hit else "TIME_EXIT")
        else:
            outcome = "OPEN"

    # Compute realistic 3-tranche blended exit price (33% at TP1, 33% at TP2, 34% at TP3/trail/close)
    p1 = tp1 if tp1_hit else final_trail_exit_price
    p2 = tp2 if tp2_hit else final_trail_exit_price
    p3 = tp3 if tp3_hit else final_trail_exit_price
    blended_exit = 0.33 * p1 + 0.33 * p2 + 0.34 * p3

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
    )
    return trade, exit_idx


def backtest_single_symbol(
    symbol: str,
    company_name: str,
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    min_score: float = 68.0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_entry_wait_days: int = 5,
    max_hold_days: int = 60,
) -> List[BacktestTrade]:
    """Run zero-look-ahead walk-forward Flag-and-Pole backtest on one stock."""
    if df is None or len(df) < 60:
        return []

    cand_bars = _precompute_candidate_bars(df, cfg)
    if len(cand_bars) == 0:
        return []

    start_ts = pd.Timestamp(start_date) if start_date else None
    end_ts = pd.Timestamp(end_date) if end_date else None

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

        # Strict point-in-time slice — ONLY data up to bar t (no future bars!)
        slice_start = max(0, t_int - 135)
        df_pit = df.iloc[slice_start : t_int + 1]
        if len(df_pit) < 50:
            continue

        setup, _ = detect_daily_pattern(symbol, company_name, df_pit, cfg)
        if setup is None or setup.total_score < min_score:
            continue

        # Never trigger twice on the same pole
        if setup.pole_start_date in seen_poles:
            continue

        sim = _simulate_trade_forward(
            df=df,
            signal_idx=t_int,
            setup=setup,
            max_entry_wait_days=max_entry_wait_days,
            max_hold_days=max_hold_days,
        )
        seen_poles.add(setup.pole_start_date)

        if sim is not None:
            trade, exit_idx = sim
            trades.append(trade)
            # Cooldown until trade exits so positions on the same stock never overlap
            cooldown_until_idx = max(exit_idx, t_int + 5)

    return trades


def compute_backtest_summary(trades: List[BacktestTrade]) -> Dict[str, Any]:
    """Aggregate portfolio-level KPIs, win rates, profit factor, grade breakdown, and equity curve."""
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
            "by_grade": {},
            "by_month": {},
            "equity_curve": [],
        }

    # Sort chronologically by entry_date for equity curve
    ordered = sorted(trades, key=lambda x: (x.entry_date, x.symbol))

    rets = np.array([t.return_pct for t in ordered], dtype=float)
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

    # Equal-weight 10% capital per trade equity curve (starting ₹1,00,000)
    equity = 100000.0
    peak = equity
    max_dd = 0.0
    eq_curve: List[Tuple[str, float, float]] = []
    for t in ordered:
        # Allocate 10% of current equity per swing trade (realistic max 10 concurrent positions)
        pnl_frac = 0.10 * (t.return_pct / 100.0)
        equity *= (1.0 + pnl_frac)
        if equity > peak:
            peak = equity
        dd = (equity / peak - 1.0) * 100.0
        if dd < max_dd:
            max_dd = dd
        eq_curve.append((t.exit_date, round(equity, 2), round(dd, 2)))

    # Breakdown by Grade
    by_grade: Dict[str, Dict[str, Any]] = {}
    for g in ("A+", "A", "B"):
        gt = [t for t in ordered if t.grade == g]
        if gt:
            g_rets = np.array([x.return_pct for x in gt], dtype=float)
            by_grade[g] = {
                "count": len(gt),
                "win_rate_pct": round(float(np.mean(g_rets > 0) * 100.0), 1),
                "tp1_rate_pct": round(sum(1 for x in gt if x.tp1_date != "-") / len(gt) * 100.0, 1),
                "tp3_rate_pct": round(sum(1 for x in gt if x.tp3_date != "-") / len(gt) * 100.0, 1),
                "avg_return_pct": round(float(np.mean(g_rets)), 2),
                "avg_r": round(float(np.mean([x.r_multiple for x in gt])), 2),
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
        "by_grade": by_grade,
        "by_month": dict(sorted(by_month.items())),
        "equity_curve": eq_curve,
    }
