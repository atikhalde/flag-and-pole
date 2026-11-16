"""
Institutional Multi-Page PDF Backtest Report Generator (Pure Matplotlib PdfPages).

Produces a crisp Landscape A4 PDF containing:
  - Page 1: Executive KPI Dashboard + Equity Curve + Outcome Breakdown + Grade & Monthly Charts
  - Pages 2..N: Complete Trade-by-Trade Audit Table with EVERY exact milestone date:
      Pole Date, Entry Date, Breakout Date, SL Date, TP1 Date, TP2 Date, TP3 Date, Result, Ret %, R
  - Final Page: Visual Candlestick Proof Charts of Top Winning Flag-and-Pole Trades
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

from src.backtester import BacktestTrade

# Clean Print/Screen Palette (Navy Header + Crisp White Body + Institutional Accents)
NAVY = "#0f172a"
SLATE_DARK = "#1e293b"
SLATE_TEXT = "#334155"
MUTED = "#64748b"
BORDER = "#cbd5e1"
ROW_ALT = "#f8fafc"
BG_CARD = "#f1f5f9"
GREEN = "#059669"
GREEN_BG = "#d1fae5"
TEAL = "#0d9488"
TEAL_BG = "#ccfbf1"
RED = "#dc2626"
RED_BG = "#fee2e2"
AMBER = "#d97706"
AMBER_BG = "#fef3c7"
BLUE = "#2563eb"
PURPLE = "#7c3aed"


def _short_date(d: str) -> str:
    """Format YYYY-MM-DD cleanly as YY-MM-DD (or '-') to fit 18 columns with generous whitespace."""
    if not d or d == "-":
        return "-"
    if len(d) == 10 and d[4] == "-":
        return d[2:]
    return d


def _outcome_style(outcome: str) -> tuple[str, str, str]:
    mapping = {
        "TP3_HIT": ("TP3 HIT", GREEN, GREEN_BG),
        "TP2_TRAIL": ("TP2 TRAIL", GREEN, GREEN_BG),
        "TP1_TRAIL": ("TP1 TRAIL", TEAL, TEAL_BG),
        "SL_HIT": ("SL HIT", RED, RED_BG),
        "OPEN": ("OPEN", AMBER, AMBER_BG),
        "TIME_EXIT": ("TIME EXIT", SLATE_TEXT, BG_CARD),
    }
    return mapping.get(outcome, (outcome, SLATE_TEXT, BG_CARD))


def _draw_header_banner(fig: plt.Figure, title: str, subtitle: str, page_label: str) -> None:
    ax_hdr = fig.add_axes([0.0, 0.905, 1.0, 0.095])
    ax_hdr.set_xlim(0, 1)
    ax_hdr.set_ylim(0, 1)
    ax_hdr.axis("off")
    ax_hdr.add_patch(Rectangle((0, 0), 1, 1, facecolor=NAVY, edgecolor="none"))
    ax_hdr.add_patch(Rectangle((0, 0), 1, 0.06, facecolor="#38bdf8", edgecolor="none"))
    ax_hdr.text(0.025, 0.62, title, color="#ffffff", fontsize=14.5, fontweight="bold", va="center")
    ax_hdr.text(0.025, 0.24, subtitle, color="#cbd5e1", fontsize=8.5, va="center")
    ax_hdr.text(0.975, 0.50, page_label, color="#94a3b8", fontsize=8.5, ha="right", va="center", fontweight="bold")


def _draw_kpi_card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    value: str,
    sub: str,
    accent: str = BLUE,
) -> None:
    card = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        facecolor=BG_CARD,
        edgecolor=BORDER,
        linewidth=0.9,
    )
    ax.add_patch(card)
    # Left accent bar
    ax.add_patch(Rectangle((x, y + 0.01), 0.006, h - 0.02, facecolor=accent, edgecolor="none"))
    ax.text(x + 0.016, y + h - 0.022, label.upper(), color=MUTED, fontsize=7.0, fontweight="bold", va="top")
    ax.text(x + 0.016, y + h * 0.48, value, color=NAVY, fontsize=14.0, fontweight="bold", va="center")
    ax.text(x + 0.016, y + 0.016, sub, color=SLATE_TEXT, fontsize=7.2, va="bottom")


def _render_summary_page(
    pdf: PdfPages,
    trades: List[BacktestTrade],
    summary: Dict[str, Any],
    meta: Dict[str, Any],
    total_pages: int,
) -> None:
    fig = plt.figure(figsize=(11.69, 8.27), facecolor="#ffffff")
    period_str = meta.get("period", "2y")
    univ_str = f"{meta.get('universe_scanned', 0)} NSE EQ stocks"
    date_range = f"{meta.get('date_from', 'N/A')} to {meta.get('date_to', 'N/A')}"
    min_score = meta.get("min_score", 68)

    _draw_header_banner(
        fig,
        "NSE DAILY FLAG-AND-POLE (<=20d) — HISTORICAL BACKTEST REPORT",
        f"Window: {date_range} ({period_str})  |  Universe: {univ_str}  |  Min Score: {min_score}  |  "
        f"Rules: Entry < BO, SL -0.5% Flag Low, 3-Tranche TP1/TP2/TP3 Trailing",
        f"Page 1 / {total_pages}",
    )

    # KPI Cards Canvas
    ax_kpi = fig.add_axes([0.025, 0.65, 0.95, 0.235])
    ax_kpi.set_xlim(0, 1)
    ax_kpi.set_ylim(0, 1)
    ax_kpi.axis("off")

    cw, ch = 0.238, 0.44
    xs = [0.0, 0.254, 0.508, 0.762]
    y_top, y_bot = 0.53, 0.03

    # Row 1 KPIs
    _draw_kpi_card(
        ax_kpi, xs[0], y_top, cw, ch,
        "Total Trades Executed",
        f"{summary['total_trades']}",
        f"Closed: {summary['closed_trades']}  |  Open: {summary['open_trades']}",
        BLUE,
    )
    _draw_kpi_card(
        ax_kpi, xs[1], y_top, cw, ch,
        "Overall Win Rate",
        f"{summary['win_rate_pct']:.1f}%",
        f"TP1 Reached: {summary['tp1_hits']} ({summary['tp1_rate_pct']:.1f}%)",
        GREEN if summary["win_rate_pct"] >= 50 else AMBER,
    )
    _draw_kpi_card(
        ax_kpi, xs[2], y_top, cw, ch,
        "TP2 / TP3 Hit Rates",
        f"{summary['tp2_rate_pct']:.1f}% / {summary['tp3_rate_pct']:.1f}%",
        f"TP2: {summary['tp2_hits']} trades  |  TP3 Full: {summary['tp3_hits']} trades",
        PURPLE,
    )
    _draw_kpi_card(
        ax_kpi, xs[3], y_top, cw, ch,
        "Profit Factor",
        f"{summary['profit_factor']:.2f}",
        f"Avg Win: +{summary['avg_win_pct']:.1f}%  |  Avg Loss: {summary['avg_loss_pct']:.1f}%",
        GREEN if summary["profit_factor"] >= 1.5 else AMBER,
    )

    # Row 2 KPIs
    _draw_kpi_card(
        ax_kpi, xs[0], y_bot, cw, ch,
        "Avg Return / Trade",
        f"{summary['avg_return_pct']:+.2f}%",
        f"Median: {summary['median_return_pct']:+.2f}%  |  Avg MFE: +{summary['avg_mfe_pct']:.1f}%",
        GREEN if summary["avg_return_pct"] > 0 else RED,
    )
    _draw_kpi_card(
        ax_kpi, xs[1], y_bot, cw, ch,
        "Expectancy (Avg R)",
        f"{summary['avg_r_multiple']:+.2f} R",
        f"Initial SL Hit Rate: {summary['sl_rate_pct']:.1f}% ({summary['sl_hits']} trades)",
        TEAL if summary["avg_r_multiple"] > 0 else RED,
    )
    _draw_kpi_card(
        ax_kpi, xs[2], y_bot, cw, ch,
        "₹1.0L Portfolio Sim",
        f"₹{summary['final_equity_1l']:,.0f}",
        f"Net: {summary['net_compounded_pct']:+.1f}% (10% alloc/trade) | Max DD: {summary['max_drawdown_pct']:.1f}%",
        GREEN if summary["net_compounded_pct"] >= 0 else RED,
    )
    _draw_kpi_card(
        ax_kpi, xs[3], y_bot, cw, ch,
        "Avg Holding Period",
        f"{summary['avg_hold_days']:.1f} days",
        f"Generated: {meta.get('generated_at_ist', '')}",
        SLATE_DARK,
    )

    # 2x2 Charts Grid
    # Chart 1 (Top-Left): Equity Curve
    ax1 = fig.add_axes([0.055, 0.365, 0.43, 0.245])
    eq = summary.get("equity_curve", [])
    if eq:
        eq_vals = [100000.0] + [pt[1] for pt in eq]
        xs_eq = np.arange(len(eq_vals))
        ax1.plot(xs_eq, np.array(eq_vals) / 1000.0, color=BLUE, lw=2.0)
        ax1.fill_between(xs_eq, 100.0, np.array(eq_vals) / 1000.0, color=BLUE, alpha=0.10)
        ax1.axhline(100.0, color=MUTED, ls="--", lw=0.8)
        ax1.set_title("Simulated Equity Curve (₹100K Start, 10% Allocation/Trade)", fontsize=9.2, fontweight="bold", color=NAVY, loc="left")
        ax1.set_ylabel("Equity (₹ Thousands)", fontsize=8, color=SLATE_TEXT)
        ax1.set_xlabel("Trade # (Chronological)", fontsize=7.5, color=MUTED)
        ax1.grid(color=BORDER, alpha=0.5, lw=0.6)
    else:
        ax1.text(0.5, 0.5, "No trades executed in period", ha="center", va="center", color=MUTED)
    ax1.tick_params(labelsize=7.5, colors=SLATE_TEXT)
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False)

    # Chart 2 (Top-Right): Outcome Distribution
    ax2 = fig.add_axes([0.555, 0.365, 0.415, 0.245])
    outcomes_order = ["TP3_HIT", "TP2_TRAIL", "TP1_TRAIL", "SL_HIT", "OPEN", "TIME_EXIT"]
    out_labels = ["TP3 Full", "TP2 Trail", "TP1 Trail", "SL Hit", "Open", "Time Exit"]
    out_colors = [GREEN, "#10b981", TEAL, RED, AMBER, MUTED]
    counts = [sum(1 for t in trades if t.outcome == k) for k in outcomes_order]
    bars = ax2.bar(out_labels, counts, color=out_colors, width=0.58, edgecolor="none")
    for b, cnt in zip(bars, counts):
        if cnt > 0:
            ax2.text(b.get_x() + b.get_width() / 2, cnt + max(counts) * 0.02, str(cnt), ha="center", va="bottom", fontsize=8, fontweight="bold", color=NAVY)
    ax2.set_title("Trade Outcome Distribution", fontsize=9.2, fontweight="bold", color=NAVY, loc="left")
    ax2.set_ylabel("Trades", fontsize=8, color=SLATE_TEXT)
    ax2.set_ylim(0, max(counts + [1]) * 1.18)
    ax2.grid(axis="y", color=BORDER, alpha=0.5, lw=0.6)
    ax2.tick_params(labelsize=7.5, colors=SLATE_TEXT)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    # Chart 3 (Bottom-Left): Grade Performance (A+ vs A vs B)
    ax3 = fig.add_axes([0.055, 0.065, 0.43, 0.225])
    by_grade = summary.get("by_grade", {})
    if by_grade:
        g_keys = list(by_grade.keys())
        x_g = np.arange(len(g_keys))
        w_g = 0.34
        wr_vals = [by_grade[g]["win_rate_pct"] for g in g_keys]
        tp1_vals = [by_grade[g]["tp1_rate_pct"] for g in g_keys]
        b1 = ax3.bar(x_g - w_g / 2, wr_vals, width=w_g, color=BLUE, label="Win Rate %")
        b2 = ax3.bar(x_g + w_g / 2, tp1_vals, width=w_g, color=GREEN, label="TP1 Hit %")
        for bar, g in zip(b1, g_keys):
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.0f}%", ha="center", fontsize=7.5, fontweight="bold", color=NAVY)
        for bar, g in zip(b2, g_keys):
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.0f}%", ha="center", fontsize=7.5, fontweight="bold", color=GREEN)
        x_lbls = [f"Grade {g}\n(n={by_grade[g]['count']}, avg {by_grade[g]['avg_return_pct']:+.1f}%)" for g in g_keys]
        ax3.set_xticks(x_g)
        ax3.set_xticklabels(x_lbls, fontsize=7.6)
        ax3.set_ylim(0, 115)
        ax3.legend(fontsize=7.3, frameon=False, loc="upper right")
        ax3.grid(axis="y", color=BORDER, alpha=0.5, lw=0.6)
    else:
        ax3.text(0.5, 0.5, "No grade data", ha="center", va="center", color=MUTED)
    ax3.set_title("Performance by Scanner Grade (A+ vs A vs B)", fontsize=9.2, fontweight="bold", color=NAVY, loc="left")
    ax3.tick_params(labelsize=7.5, colors=SLATE_TEXT)
    for sp in ("top", "right"):
        ax3.spines[sp].set_visible(False)

    # Chart 4 (Bottom-Right): Monthly Avg Return %
    ax4 = fig.add_axes([0.555, 0.065, 0.415, 0.225])
    by_month = summary.get("by_month", {})
    if by_month:
        months = list(by_month.keys())[-14:]  # Last 14 active months cleanly readable
        m_vals = [by_month[m]["avg_return_pct"] for m in months]
        m_cols = [GREEN if v >= 0 else RED for v in m_vals]
        ax4.bar(np.arange(len(months)), m_vals, color=m_cols, width=0.62)
        ax4.axhline(0, color=SLATE_TEXT, lw=0.8)
        ax4.set_xticks(np.arange(len(months)))
        ax4.set_xticklabels([m[2:] for m in months], rotation=35, ha="right", fontsize=7.0)
        ax4.set_ylabel("Avg Return %", fontsize=8, color=SLATE_TEXT)
        ax4.grid(axis="y", color=BORDER, alpha=0.5, lw=0.6)
    else:
        ax4.text(0.5, 0.5, "No monthly data", ha="center", va="center", color=MUTED)
    ax4.set_title("Average Trade Return % by Entry Month", fontsize=9.2, fontweight="bold", color=NAVY, loc="left")
    ax4.tick_params(labelsize=7.5, colors=SLATE_TEXT)
    for sp in ("top", "right"):
        ax4.spines[sp].set_visible(False)

    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def _render_trade_table_pages(
    pdf: PdfPages,
    trades: List[BacktestTrade],
    meta: Dict[str, Any],
    start_page_num: int,
    total_pages: int,
    rows_per_page: int = 22,
) -> int:
    """Render cleanly paginated Trade-by-Trade Audit Tables showing ALL milestone dates."""
    if not trades:
        return start_page_num

    # Order newest-first or chronological? Chronological newest-first lets user inspect recent trades immediately!
    ordered = sorted(trades, key=lambda t: (t.entry_date, t.symbol), reverse=True)

    # 18 crisp columns covering EVERY requested date & price level
    col_specs = [
        ("#", 0.025, "center"),
        ("Symbol", 0.088, "left"),
        ("Grd", 0.032, "center"),
        ("Pole Date", 0.062, "center"),
        ("Entry Date", 0.062, "center"),
        ("Entry ₹", 0.056, "right"),
        ("BO Date", 0.062, "center"),
        ("SL ₹", 0.054, "right"),
        ("SL Date", 0.062, "center"),
        ("TP1 ₹", 0.054, "right"),
        ("TP1 Date", 0.062, "center"),
        ("TP2 ₹", 0.054, "right"),
        ("TP2 Date", 0.062, "center"),
        ("TP3 ₹", 0.054, "right"),
        ("TP3 Date", 0.062, "center"),
        ("Result", 0.066, "center"),
        ("Ret %", 0.046, "right"),
        ("R", 0.037, "right"),
    ]
    # Normalize widths to sum to 1.0
    total_w = sum(c[1] for c in col_specs)
    col_xs = []
    cur_x = 0.0
    norm_specs = []
    for lbl, w, al in col_specs:
        nw = w / total_w
        col_xs.append(cur_x)
        norm_specs.append((lbl, nw, al))
        cur_x += nw

    page_num = start_page_num
    for chunk_start in range(0, len(ordered), rows_per_page):
        chunk = ordered[chunk_start : chunk_start + rows_per_page]
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="#ffffff")
        _draw_header_banner(
            fig,
            "TRADE-BY-TRADE AUDIT LOG — ALL DATES & TARGET MILESTONES",
            f"Showing trades {chunk_start + 1}–{chunk_start + len(chunk)} of {len(ordered)} (Newest First)  |  "
            f"Dates formatted YY-MM-DD  |  SL = -0.5% Flag Low  |  3-Tranche Return (33% TP1 / 33% TP2 / 34% TP3 or Trail)",
            f"Page {page_num} / {total_pages}",
        )

        ax = fig.add_axes([0.018, 0.04, 0.964, 0.845])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, rows_per_page + 1.35)
        ax.axis("off")

        # Table Header Row
        hdr_y = rows_per_page + 0.35
        ax.add_patch(Rectangle((0, hdr_y), 1.0, 0.92, facecolor=SLATE_DARK, edgecolor="none"))
        for x0, (lbl, nw, al) in zip(col_xs, norm_specs):
            tx = x0 + (0.004 if al == "left" else (nw - 0.004 if al == "right" else nw / 2.0))
            ax.text(tx, hdr_y + 0.46, lbl, color="#ffffff", fontsize=6.9, fontweight="bold", ha=al, va="center")

        # Data Rows
        for r_idx, t in enumerate(chunk):
            global_idx = chunk_start + r_idx + 1
            y0 = rows_per_page - 1 - r_idx + 0.28
            bg = ROW_ALT if r_idx % 2 == 1 else "#ffffff"
            ax.add_patch(Rectangle((0, y0), 1.0, 0.96, facecolor=bg, edgecolor="none"))
            ax.plot([0, 1], [y0, y0], color="#e2e8f0", lw=0.45)

            res_lbl, res_fg, res_bg = _outcome_style(t.outcome)
            vals = [
                str(global_idx),
                t.symbol[:12],
                t.grade,
                _short_date(t.pole_start_date),
                _short_date(t.entry_date),
                f"{t.entry_price:.1f}",
                _short_date(t.breakout_date),
                f"{t.sl_price:.1f}",
                _short_date(t.sl_date),
                f"{t.tp1_price:.1f}",
                _short_date(t.tp1_date),
                f"{t.tp2_price:.1f}",
                _short_date(t.tp2_date),
                f"{t.tp3_price:.1f}",
                _short_date(t.tp3_date),
                res_lbl,
                f"{t.return_pct:+.1f}%",
                f"{t.r_multiple:+.1f}R",
            ]

            for c_i, (val, x0, (_, nw, al)) in enumerate(zip(vals, col_xs, norm_specs)):
                tx = x0 + (0.004 if al == "left" else (nw - 0.004 if al == "right" else nw / 2.0))
                cy = y0 + 0.48
                if c_i == 15:  # Result badge pill
                    ax.add_patch(
                        FancyBboxPatch(
                            (x0 + 0.003, y0 + 0.16),
                            nw - 0.006,
                            0.64,
                            boxstyle="round,pad=0.0,rounding_size=0.004",
                            facecolor=res_bg,
                            edgecolor="none",
                        )
                    )
                    ax.text(tx, cy, val, color=res_fg, fontsize=6.3, fontweight="bold", ha="center", va="center")
                else:
                    col_txt = SLATE_DARK
                    fw = "normal"
                    if c_i == 1:
                        fw = "bold"
                        col_txt = NAVY
                    elif c_i == 2:
                        fw = "bold"
                        col_txt = GREEN if val == "A+" else (BLUE if val == "A" else MUTED)
                    elif c_i == 8 and val != "-":  # SL Date
                        col_txt = RED
                        fw = "bold"
                    elif c_i in (10, 12, 14) and val != "-":  # TP1/TP2/TP3 Dates
                        col_txt = GREEN
                        fw = "bold"
                    elif c_i in (16, 17):
                        col_txt = GREEN if t.return_pct > 0 else RED
                        fw = "bold"
                    elif val == "-":
                        col_txt = "#94a3b8"
                    ax.text(tx, cy, val, color=col_txt, fontsize=6.6, fontweight=fw, ha=al, va="center")

        # Footer note
        fig.text(
            0.02,
            0.015,
            "Note: Zero-look-ahead walk-forward simulation. SL date shows initial SL hit (red SL HIT) or trailing BE/TP1 stop exit after partial target hit.",
            fontsize=6.8,
            color=MUTED,
        )
        pdf.savefig(fig, dpi=150)
        plt.close(fig)
        page_num += 1

    return page_num


def _render_top_trades_chart_page(
    pdf: PdfPages,
    trades: List[BacktestTrade],
    daily_map: Dict[str, pd.DataFrame],
    page_num: int,
    total_pages: int,
) -> None:
    """Render a 2x2 visual proof gallery of top winning trades on actual Daily candles."""
    winners = [t for t in trades if t.symbol in daily_map and t.return_pct > 0]
    winners.sort(key=lambda t: (t.r_multiple, t.return_pct), reverse=True)
    showcase = winners[:4]
    if not showcase:
        return

    fig = plt.figure(figsize=(11.69, 8.27), facecolor="#ffffff")
    _draw_header_banner(
        fig,
        "TOP WINNING FLAG-AND-POLE TRADES — VISUAL PROOF GALLERY",
        "Actual historical Daily OHLCV showing Pole, Flag Box, Pre-Breakout Entry, Stop Loss (-0.5% flag low), and Target Hits",
        f"Page {page_num} / {total_pages}",
    )

    positions = [
        [0.05, 0.49, 0.435, 0.36],
        [0.535, 0.49, 0.435, 0.36],
        [0.05, 0.065, 0.435, 0.36],
        [0.535, 0.065, 0.435, 0.36],
    ]

    for idx, t in enumerate(showcase):
        ax = fig.add_axes(positions[idx])
        ax.set_facecolor("#0f172a")
        df_full = daily_map[t.symbol]

        # Window: 12 bars before pole_start_date through 8 bars after exit_date (max ~65 bars)
        p_ts = pd.Timestamp(t.pole_start_date)
        e_ts = pd.Timestamp(t.exit_date)
        idx_arr = df_full.index
        p_loc = int(idx_arr.searchsorted(p_ts))
        e_loc = int(idx_arr.searchsorted(e_ts))
        lo = max(0, p_loc - 12)
        hi = min(len(df_full), max(e_loc + 8, p_loc + 35))
        if hi - lo > 75:
            hi = lo + 75
        sub = df_full.iloc[lo:hi]
        n = len(sub)
        if n < 5:
            continue

        o = sub["Open"].to_numpy(dtype=float)
        h = sub["High"].to_numpy(dtype=float)
        l = sub["Low"].to_numpy(dtype=float)
        c = sub["Close"].to_numpy(dtype=float)
        dates = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in sub.index]

        # Plot candlesticks
        for i in range(n):
            col = "#22c55e" if c[i] >= o[i] else "#ef4444"
            ax.plot([i, i], [l[i], h[i]], color=col, lw=0.9, zorder=3)
            bh = max(abs(c[i] - o[i]), (h[i] - l[i]) * 0.06, 1e-4)
            ax.add_patch(Rectangle((i - 0.32, min(o[i], c[i])), 0.64, bh, facecolor=col, edgecolor=col, lw=0.4, zorder=4))

        # Shade pole & flag box
        pole_xs = [i for i, ds in enumerate(dates) if t.pole_start_date <= ds <= t.pole_end_date]
        flag_xs = [i for i, ds in enumerate(dates) if t.pole_end_date < ds <= t.signal_date]
        if pole_xs:
            ax.axvspan(min(pole_xs) - 0.45, max(pole_xs) + 0.45, color="#10b981", alpha=0.14, zorder=1)
        if flag_xs:
            ax.add_patch(
                Rectangle(
                    (min(flag_xs) - 0.45, t.shelf_low),
                    (max(flag_xs) - min(flag_xs)) + 0.90,
                    max(t.shelf_high - t.shelf_low, 0.01),
                    facecolor="#38bdf8",
                    alpha=0.16,
                    edgecolor="#38bdf8",
                    lw=0.8,
                    zorder=2,
                )
            )

        # Horizontal Trade Plan Lines
        x_start = min(flag_xs) if flag_xs else max(0, n // 3)
        ax.hlines(t.sl_price, x_start, n - 0.3, colors="#f87171", linestyles="--", lw=0.9, zorder=5)
        ax.hlines(t.entry_price, x_start, n - 0.3, colors="#38bdf8", linestyles="-.", lw=0.9, zorder=5)
        ax.hlines(t.tp1_price, x_start, n - 0.3, colors="#4ade80", linestyles="--", lw=0.85, zorder=5)
        if t.tp2_date != "-":
            ax.hlines(t.tp2_price, x_start, n - 0.3, colors="#a78bfa", linestyles="--", lw=0.8, zorder=5)
        if t.tp3_date != "-":
            ax.hlines(t.tp3_price, x_start, n - 0.3, colors="#fbbf24", linestyles=":", lw=0.85, zorder=5)

        # Entry marker
        ent_xs = [i for i, ds in enumerate(dates) if ds == t.entry_date]
        if ent_xs:
            ax.scatter([ent_xs[0]], [t.entry_price], marker="^", s=45, color="#38bdf8", edgecolor="#ffffff", lw=0.6, zorder=7)

        ax.set_title(
            f"{t.symbol} ({t.grade})  |  Pole {t.pole_start_date[2:]} → Entry {t.entry_date[2:]}  |  "
            f"{t.outcome.replace('_', ' ')} ({t.return_pct:+.1f}% / {t.r_multiple:+.1f}R)",
            fontsize=8.0,
            fontweight="bold",
            color=NAVY,
            loc="left",
            pad=5,
        )
        step = max(1, n // 6)
        ticks = list(range(0, n, step))
        ax.set_xticks(ticks)
        ax.set_xticklabels([pd.Timestamp(sub.index[i]).strftime("%d %b") for i in ticks], fontsize=6.5, color=SLATE_TEXT)
        ax.tick_params(axis="y", labelsize=6.8, colors=SLATE_TEXT)
        ax.grid(color="#1e293b", lw=0.5, alpha=0.8)

    fig.text(
        0.5,
        0.014,
        "Chart Legend:  Green Shade = Breakout Pole   |   Blue Box = Coiled Flag Shelf   |   ▲ Cyan = Entry (<BO)   |   -- Red = SL (-0.5% Flag Low)   |   -- Green = TP1   |   -- Purple = TP2   |   ·· Gold = TP3",
        ha="center",
        fontsize=7.2,
        color=SLATE_TEXT,
        fontweight="bold",
    )
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def generate_backtest_pdf(
    trades: List[BacktestTrade],
    summary: Dict[str, Any],
    meta: Dict[str, Any],
    output_pdf_path: str | Path,
    daily_map: Optional[Dict[str, pd.DataFrame]] = None,
    rows_per_page: int = 22,
) -> Path:
    """Build complete multi-page PDF Backtest Report and return saved Path."""
    out = Path(output_pdf_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_table_pages = (len(trades) + rows_per_page - 1) // rows_per_page if trades else 0
    has_gallery = bool(daily_map and any(t.symbol in daily_map and t.return_pct > 0 for t in trades))
    total_pages = 1 + n_table_pages + (1 if has_gallery else 0)

    with PdfPages(out) as pdf:
        # Page 1: Executive Summary & Charts
        _render_summary_page(pdf, trades, summary, meta, total_pages=total_pages)

        # Pages 2..N: Full Trade-by-Trade Audit Log (Pole Date, Entry Date, BO Date, SL Date, TP1/TP2/TP3 Dates)
        next_page = _render_trade_table_pages(
            pdf, trades, meta, start_page_num=2, total_pages=total_pages, rows_per_page=rows_per_page
        )

        # Final Page: Visual Proof Gallery of Top Winners
        if has_gallery and daily_map is not None:
            _render_top_trades_chart_page(pdf, trades, daily_map, page_num=next_page, total_pages=total_pages)

        # PDF Metadata
        d = pdf.infodict()
        d["Title"] = "NSE Daily Flag-and-Pole (<=20d) Backtest Report"
        d["Author"] = "NSE Flag-and-Pole Scanner"
        d["Subject"] = "Point-in-time historical backtest with Pole, Entry, Breakout, SL, and TP dates"

    return out
