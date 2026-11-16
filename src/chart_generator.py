"""
Institutional Daily Chart PNG Generator (TradingCharts-inspired dark theme).
Renders:
  - Main Panel: Daily Candlesticks + Pole Highlight + Tight Flag/Shelf Box + VAH/POC/VAL
                + LVN Vacuum Shading + Breakout/Stop/Target Lines + Right-Side Volume Profile
  - Bottom Panel: Daily Volume Histogram + 20d Median Baseline + Volume Dry-Up Callout
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")  # Headless backend for GitHub Actions / servers
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.volume_profile import DailyVolumeProfile, compute_daily_volume_profile

# Theme palette (clean institutional dark charcoal)
BG_FIG = "#0d1117"
BG_AX = "#11161f"
GRID_COL = "#1f2937"
TEXT_MAIN = "#e5e7eb"
TEXT_MUTED = "#9ca3af"
UP_COL = "#10b981"      # Emerald green
DOWN_COL = "#ef4444"    # Crimson red
POLE_BG = "#10b981"
SHELF_BG = "#38bdf8"    # Sky blue
VAH_COL = "#94a3b8"
POC_COL = "#f97316"     # Orange
LVN_COL = "#f59e0b"     # Amber vacuum shading
STOP_COL = "#f43f5e"
TGT_COL = "#22d3ee"


def _fmt_vol(v: float) -> str:
    if v >= 1e6:
        val = v / 1e6
        return f"{val:.0f}M" if abs(val - round(val)) < 0.05 else f"{val:.1f}M"
    if v >= 1e3:
        return f"{v / 1e3:.0f}K"
    return f"{v:.0f}"


def generate_pattern_chart(
    symbol: str,
    df_daily: pd.DataFrame,
    setup: Dict[str, Any],
    output_path: str | Path,
    vp: Optional[DailyVolumeProfile] = None,
    live_trigger: Optional[Dict[str, Any]] = None,
    lookback_bars: int = 45,
) -> Path:
    """Render an annotated Daily chart PNG with right-side Volume Profile & Volume Dry-Up panel."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = df_daily.sort_index().tail(lookback_bars).copy()
    n = len(df)
    dates_str = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in df.index]
    date_to_x = {d: i for i, d in enumerate(dates_str)}

    if vp is None:
        vp = compute_daily_volume_profile(df, num_bins=30, shelf_low=setup.get("shelf_low"))

    fig = plt.figure(figsize=(13.2, 7.6), dpi=140, facecolor=BG_FIG)
    gs = gridspec.GridSpec(
        2,
        2,
        height_ratios=[3.3, 1.15],
        width_ratios=[4.8, 1.0],
        hspace=0.06,
        wspace=0.02,
        left=0.06,
        right=0.94,
        top=0.88,
        bottom=0.08,
    )
    ax_price = fig.add_subplot(gs[0, 0], facecolor=BG_AX)
    ax_vp = fig.add_subplot(gs[0, 1], sharey=ax_price, facecolor=BG_AX)
    ax_vol = fig.add_subplot(gs[1, 0], sharex=ax_price, facecolor=BG_AX)
    ax_info = fig.add_subplot(gs[1, 1], facecolor=BG_AX)

    for ax in (ax_price, ax_vp, ax_vol, ax_info):
        for spine in ax.spines.values():
            spine.set_color("#263040")
            spine.set_linewidth(0.8)

    # ------------------------------------------------------------------
    # 1. Background Zones: Pole Highlight, Shelf Box, LVN Air-Pocket
    # ------------------------------------------------------------------
    pole_dates = setup.get("pole_dates", [])
    base_dates = setup.get("base_dates", [])
    pole_xs = [date_to_x[d] for d in pole_dates if d in date_to_x]
    base_xs = [date_to_x[d] for d in base_dates if d in date_to_x]

    if pole_xs:
        px0, px1 = min(pole_xs) - 0.45, max(pole_xs) + 0.45
        ax_price.axvspan(px0, px1, color=POLE_BG, alpha=0.10, zorder=0)
        ax_vol.axvspan(px0, px1, color=POLE_BG, alpha=0.10, zorder=0)

    shelf_high = float(setup["shelf_high"])
    shelf_low = float(setup["shelf_low"])
    if base_xs:
        bx0 = min(base_xs) - 0.45
        bx1 = max(base_xs) + 0.45
        rect = patches.Rectangle(
            (bx0, shelf_low),
            bx1 - bx0,
            shelf_high - shelf_low,
            linewidth=1.2,
            edgecolor=SHELF_BG,
            facecolor=SHELF_BG,
            alpha=0.14,
            zorder=1,
        )
        ax_price.add_patch(rect)
        ax_price.hlines([shelf_high, shelf_low], bx0, n + 0.3, colors=SHELF_BG, linestyles="-", lw=1.0, alpha=0.75, zorder=2)

    lvn_zone = setup.get("lvn_zone")
    if lvn_zone and len(lvn_zone) == 2:
        ax_price.axhspan(lvn_zone[0], lvn_zone[1], color=LVN_COL, alpha=0.07, zorder=0)

    # ------------------------------------------------------------------
    # 2. Daily Candlesticks
    # ------------------------------------------------------------------
    opens = df["Open"].to_numpy(dtype=float)
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)
    vols = df["Volume"].to_numpy(dtype=float)

    for i in range(n):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        col = UP_COL if c >= o else DOWN_COL
        # Wick
        ax_price.plot([i, i], [l, h], color=col, lw=1.25, zorder=3)
        # Body
        body_lo = min(o, c)
        body_h = max(abs(c - o), (h - l) * 0.03, c * 0.0012)
        rect_c = patches.Rectangle(
            (i - 0.32, body_lo),
            0.64,
            body_h,
            facecolor=col,
            edgecolor=col,
            lw=0.6,
            zorder=4,
        )
        ax_price.add_patch(rect_c)

    # Highlight latest breakout candle on live trigger
    if live_trigger is not None:
        ax_price.axvline(n - 1, color="#facc15", ls="--", lw=1.0, alpha=0.7, zorder=2)

    # ------------------------------------------------------------------
    # 3. Structural Price Lines: VAH, POC, VAL, SL, Entry (Below BO), TP1, TP2, TP3
    # ------------------------------------------------------------------
    vah = setup.get("vah")
    poc = setup.get("poc")
    val = setup.get("val")
    bo_lvl = float(setup.get("breakout_level", shelf_high))
    entry_price = float(setup.get("entry_price", setup.get("last_close", shelf_high * 0.985)))
    stop = float(setup["invalidation_stop"])
    tp1 = float(setup.get("tp1", setup.get("target_1_bos", bo_lvl * 1.05)))
    tp2 = float(setup.get("tp2", setup.get("target_2_ext", bo_lvl * 1.12)))
    tp3 = float(setup.get("tp3", bo_lvl * 1.20))

    if poc:
        ax_price.axhline(poc, color=POC_COL, ls="-", lw=1.1, alpha=0.80, zorder=2)
        ax_price.text(0.4, poc, f" POC {poc:.1f}", color=POC_COL, fontsize=7.8, va="bottom", fontweight="bold")
    if vah:
        ax_price.axhline(vah, color=VAH_COL, ls="--", lw=0.95, alpha=0.75, zorder=2)
        ax_price.text(0.4, vah, f" VAH {vah:.1f}", color=VAH_COL, fontsize=7.6, va="bottom")
    if val:
        ax_price.axhline(val, color=VAH_COL, ls=":", lw=0.85, alpha=0.60, zorder=2)

    # Smart Y-axis limits to fit all candles + Entry/SL/TP1/TP2/TP3 cleanly without crushing candles
    y_min_val = min(float(np.min(lows)), stop) * 0.965
    y_max_val = max(float(np.max(highs)), tp1 * 1.02, min(tp3, float(np.max(highs)) * 1.24)) * 1.035

    # Stop, Pre-Breakout Entry (below BO) & Target lines extending across right portion of chart
    x_lvl_start = max(0, (min(base_xs) - 1) if base_xs else (n - 12))
    ax_price.hlines(stop, x_lvl_start, n + 0.3, colors=STOP_COL, linestyles="--", lw=1.1, alpha=0.92, zorder=3)
    ax_price.hlines(entry_price, x_lvl_start, n + 0.3, colors="#38bdf8", linestyles="-.", lw=1.1, alpha=0.92, zorder=3)
    ax_price.hlines(tp1, x_lvl_start, n + 0.3, colors=TGT_COL, linestyles="--", lw=1.0, alpha=0.88, zorder=3)

    # Collect right-margin labels and nudge slightly if two levels are very close vertically
    right_labels = [
        (stop, f"SL {stop:.2f}", STOP_COL),
        (entry_price, f"ENT {entry_price:.1f}", "#38bdf8"),
        (tp1, f"TP1 {tp1:.1f}", TGT_COL),
    ]
    if tp2 <= y_max_val * 0.985:
        ax_price.hlines(tp2, x_lvl_start, n + 0.3, colors="#a78bfa", linestyles="--", lw=0.95, alpha=0.85, zorder=3)
        right_labels.append((tp2, f"TP2 {tp2:.1f}", "#a78bfa"))
    if tp3 <= y_max_val * 0.985:
        ax_price.hlines(tp3, x_lvl_start, n + 0.3, colors="#fbbf24", linestyles=":", lw=0.95, alpha=0.85, zorder=3)
        right_labels.append((tp3, f"TP3 {tp3:.1f}", "#fbbf24"))

    right_labels.sort(key=lambda item: item[0])
    min_gap = (y_max_val - y_min_val) * 0.036
    placed_ys: list[float] = []
    for raw_y, lbl_txt, lbl_col in right_labels:
        draw_y = raw_y
        if placed_ys and (draw_y - placed_ys[-1]) < min_gap:
            draw_y = placed_ys[-1] + min_gap
        placed_ys.append(draw_y)
        if draw_y <= y_max_val * 0.99:
            ax_price.text(n + 0.50, draw_y, lbl_txt, color=lbl_col, fontsize=7.4, va="center", fontweight="bold", clip_on=True)

    # Pole & Shelf inline annotations (auto-offset POLE badge left when close to shelf_high so badges never collide)
    if pole_xs:
        pole_top_y = float(np.max(highs[pole_xs]))
        near_shelf_height = abs(pole_top_y - shelf_high) / max(shelf_high, 1e-6) < 0.06
        pole_label_x = (min(pole_xs) - 0.6) if near_shelf_height else float(np.mean(pole_xs))
        pole_ha = "right" if near_shelf_height else "center"
        ax_price.text(
            pole_label_x,
            pole_top_y * 1.012,
            f"POLE +{setup['pole_high_gain_pct']:.0f}%\n({setup['pole_rvol']:.0f}x Vol)",
            color=UP_COL,
            fontsize=7.8,
            ha=pole_ha,
            va="bottom",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#064e3b", edgecolor=UP_COL, alpha=0.85, lw=0.8),
            zorder=6,
        )
    if base_xs:
        ax_price.text(
            float(np.mean(base_xs)),
            shelf_high * 1.012,
            f"SHELF ({setup['base_days']}d | Day {setup['total_pattern_days']}/20)\nBox {setup['shelf_close_spread_pct']:.1f}%",
            color="#e0f2fe",
            fontsize=7.6,
            ha="center",
            va="bottom",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#0c4a6e", edgecolor=SHELF_BG, alpha=0.85, lw=0.8),
            zorder=6,
        )

    ax_price.set_ylim(y_min_val, y_max_val)
    ax_price.set_xlim(-1.0, n + 4.5)
    ax_price.grid(color=GRID_COL, ls="-", lw=0.55, alpha=0.7)
    ax_price.tick_params(axis="x", labelbottom=False, length=0)
    ax_price.tick_params(axis="y", colors=TEXT_MUTED, labelsize=8.5)
    ax_price.set_ylabel("Price (INR)", color=TEXT_MUTED, fontsize=8.8)

    # ------------------------------------------------------------------
    # 4. Right-Side Daily Volume Profile
    # ------------------------------------------------------------------
    if vp is not None:
        bin_h = float(vp.bin_edges[1] - vp.bin_edges[0]) * 0.84
        max_v = max(float(vp.total_vol.max()), 1.0)
        up_norm = vp.up_vol / max_v
        dn_norm = vp.down_vol / max_v

        ax_vp.barh(vp.bin_centers, up_norm, height=bin_h, color="#14b8a6", alpha=0.82, edgecolor="none")
        ax_vp.barh(vp.bin_centers, dn_norm, left=up_norm, height=bin_h, color="#ec4899", alpha=0.82, edgecolor="none")
        ax_vp.axhline(vp.poc, color=POC_COL, lw=1.1, alpha=0.9)
        ax_vp.axhline(vp.vah, color=VAH_COL, ls="--", lw=0.9, alpha=0.8)
        ax_vp.axhline(vp.val, color=VAH_COL, ls=":", lw=0.8, alpha=0.6)
        ax_vp.set_xlim(0, 1.08)
        ax_vp.invert_xaxis()  # Bars grow leftward toward candles like TradingCharts!
        ax_vp.text(0.54, 0.97, "Daily VP", transform=ax_vp.transAxes, color=TEXT_MUTED, fontsize=8, ha="center", va="top")
    ax_vp.grid(color=GRID_COL, ls="-", lw=0.4, alpha=0.5)
    ax_vp.tick_params(axis="both", labelbottom=False, labelleft=False, length=0)

    # ------------------------------------------------------------------
    # 5. Bottom Panel: Daily Volume Histogram + Dry-Up Annotation
    # ------------------------------------------------------------------
    bar_colors = [UP_COL if closes[i] >= opens[i] else DOWN_COL for i in range(n)]
    ax_vol.bar(np.arange(n), vols, width=0.66, color=bar_colors, alpha=0.85, zorder=3)

    # 20d rolling median baseline
    vol_s = pd.Series(vols)
    vol_med = vol_s.rolling(20, min_periods=5).median().to_numpy()
    ax_vol.plot(np.arange(n), vol_med, color="#facc15", lw=1.1, ls="--", alpha=0.85, label="20d Med Vol", zorder=4)

    # Volume dry-up visual callout
    ax_vol.text(
        0.015,
        0.86,
        f"Pole Peak: {_fmt_vol(setup['pole_peak_vol'])}  →  Recent Base Avg: {_fmt_vol(setup['recent_avg_vol'])}  "
        f"(Dry-Up: {setup['dryup_pct']:.1f}% | Base/Pole Vol: {setup['base_to_pole_vol_pct']:.1f}%)",
        transform=ax_vol.transAxes,
        color="#fef08a",
        fontsize=8.0,
        fontweight="bold",
        va="top",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#1e293b", edgecolor="#334155", alpha=0.9),
    )

    ax_vol.grid(color=GRID_COL, ls="-", lw=0.5, alpha=0.65)
    ax_vol.set_ylabel("Volume", color=TEXT_MUTED, fontsize=8.5)
    ax_vol.tick_params(axis="y", colors=TEXT_MUTED, labelsize=8)
    ax_vol.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: _fmt_vol(x)))

    # Clean x-axis date ticks
    step = max(1, n // 9)
    ticks = list(range(0, n, step))
    if (n - 1) not in ticks:
        ticks.append(n - 1)
    ax_vol.set_xticks(ticks)
    ax_vol.set_xticklabels([pd.Timestamp(df.index[i]).strftime("%d %b") for i in ticks], color=TEXT_MUTED, fontsize=8)

    # ------------------------------------------------------------------
    # 6. Bottom-Right Scorecard Box (5-Pillar Breakdown)
    # ------------------------------------------------------------------
    ax_info.axis("off")
    ps = setup.get("pillar_scores", {})
    score_lines = [
        f"GRADE {setup['grade']}  ({setup['total_score']:.1f}/100)",
        f"Pole SOS : {ps.get('pole_strength', 0):.1f}/25",
        f"Shelf Box: {ps.get('shelf_geometry', 0):.1f}/25",
        f"VCP Coil : {ps.get('vcp_contraction', 0):.1f}/20",
        f"Dry-Up   : {ps.get('volume_dryup', 0):.1f}/20",
        f"Profile  : {ps.get('daily_profile', 0):.1f}/10",
    ]
    ax_info.text(
        0.08,
        0.93,
        "\n".join(score_lines),
        transform=ax_info.transAxes,
        color=TEXT_MAIN,
        fontsize=7.9,
        family="monospace",
        va="top",
        linespacing=1.42,
    )

    # ------------------------------------------------------------------
    # 7. Title Banner
    # ------------------------------------------------------------------
    last_c = float(closes[-1])
    prev_c = float(closes[-2]) if n >= 2 else last_c
    chg_pct = (last_c / prev_c - 1.0) * 100.0 if prev_c > 0 else 0.0
    chg_sign = "+" if chg_pct >= 0 else ""

    if live_trigger is not None:
        trig_name = live_trigger["trigger_type"].replace("_", " ")
        title_str = (
            f"NSE:{symbol}  |  LIVE TRIGGER: {trig_name}  |  "
            f"₹{last_c:.2f} ({chg_sign}{chg_pct:.2f}%)  |  Day {live_trigger['day_of_pattern']}/20"
        )
        title_col = "#facc15"
        subtitle = (
            f"LTP ₹{live_trigger['live_price']:.2f} (RVOL {live_trigger['projected_rvol_vs_base']:.1f}x)  •  "
            f"SL ₹{stop:.2f} (-{setup.get('sl_buffer_pct', 0.5):.1f}% flag)  |  "
            f"TP1 ₹{tp1:.1f}  |  TP2 ₹{tp2:.1f}  |  TP3 ₹{tp3:.1f}  •  {live_trigger['timestamp_ist']}"
        )
    else:
        title_str = (
            f"NSE:{symbol} (Daily)  |  {setup['pattern_label']}  |  "
            f"₹{last_c:.2f} ({chg_sign}{chg_pct:.2f}%)  |  Grade {setup['grade']} ({setup['total_score']:.0f}/100)"
        )
        title_col = TEXT_MAIN
        subtitle = (
            f"{setup.get('company_name', symbol)}  •  As of {setup['scan_date']}  •  "
            f"Entry ₹{entry_price:.2f} (<BO ₹{bo_lvl:.1f})  |  SL ₹{stop:.2f} (-{setup.get('sl_buffer_pct', 0.5):.1f}% flag)  |  "
            f"TP1 ₹{tp1:.1f}  |  TP2 ₹{tp2:.1f}  |  TP3 ₹{tp3:.1f}"
        )

    fig.text(0.06, 0.955, title_str, color=title_col, fontsize=12.2, fontweight="bold", va="top")
    fig.text(0.06, 0.916, subtitle, color=TEXT_MUTED, fontsize=8.8, va="top")

    fig.savefig(out, dpi=140, facecolor=BG_FIG)
    plt.close(fig)
    return out
