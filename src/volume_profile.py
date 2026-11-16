"""Pure Daily-OHLCV Visible-Range Volume Profile (POC, VAH, VAL, and LVN Gap Detector)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DailyVolumeProfile:
    bin_centers: np.ndarray
    bin_edges: np.ndarray
    total_vol: np.ndarray
    up_vol: np.ndarray
    down_vol: np.ndarray
    poc: float
    vah: float
    val: float
    lvn_zone: Optional[Tuple[float, float]]
    has_unfilled_lvn: bool


def compute_daily_volume_profile(
    df_daily: pd.DataFrame,
    num_bins: int = 32,
    value_area_pct: float = 0.70,
    shelf_low: Optional[float] = None,
) -> Optional[DailyVolumeProfile]:
    """
    Build a visible-range Volume Profile directly from Daily OHLCV bars.
    Distributes each daily bar's volume smoothly across its [Low, High] price span
    and separates up-day / down-day volume (teal/pink bars).
    """
    if df_daily is None or df_daily.empty or len(df_daily) < 10:
        return None

    lows = df_daily["Low"].to_numpy(dtype=float)
    highs = df_daily["High"].to_numpy(dtype=float)
    opens = df_daily["Open"].to_numpy(dtype=float)
    closes = df_daily["Close"].to_numpy(dtype=float)
    vols = df_daily["Volume"].to_numpy(dtype=float)

    p_min = float(np.nanmin(lows))
    p_max = float(np.nanmax(highs))
    if not np.isfinite(p_min) or not np.isfinite(p_max) or p_max <= p_min:
        return None

    bin_edges = np.linspace(p_min, p_max, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    up_vol = np.zeros(num_bins, dtype=float)
    down_vol = np.zeros(num_bins, dtype=float)

    prev_close = np.roll(closes, 1)
    prev_close[0] = opens[0]

    for i in range(len(vols)):
        v = vols[i]
        if v <= 0 or not np.isfinite(v):
            continue
        lo, hi = lows[i], highs[i]
        is_up = (closes[i] > opens[i]) or (closes[i] == opens[i] and closes[i] >= prev_close[i])

        b_start = max(0, min(num_bins - 1, int(np.searchsorted(bin_edges, lo, side="right") - 1)))
        b_end = max(0, min(num_bins - 1, int(np.searchsorted(bin_edges, hi, side="right") - 1)))
        span = b_end - b_start + 1

        # Weight slightly more volume around the real body [min(O,C), max(O,C)] vs outer wicks
        body_lo, body_hi = min(opens[i], closes[i]), max(opens[i], closes[i])
        weights = np.ones(span, dtype=float)
        for j, b_idx in enumerate(range(b_start, b_end + 1)):
            c_price = bin_centers[b_idx]
            if body_lo <= c_price <= body_hi:
                weights[j] = 1.6
        weights /= weights.sum()

        if is_up:
            up_vol[b_start : b_end + 1] += v * weights
        else:
            down_vol[b_start : b_end + 1] += v * weights

    total_vol = up_vol + down_vol
    grand_total = float(total_vol.sum())
    if grand_total <= 0:
        return None

    poc_idx = int(np.argmax(total_vol))
    poc_price = float(bin_centers[poc_idx])

    # Expand 70% Value Area outward from POC
    target_vol = grand_total * value_area_pct
    va_vol = float(total_vol[poc_idx])
    low_idx = poc_idx
    high_idx = poc_idx

    while va_vol < target_vol and (low_idx > 0 or high_idx < num_bins - 1):
        up_v = float(total_vol[high_idx + 1 : min(num_bins, high_idx + 3)].sum()) if high_idx < num_bins - 1 else -1.0
        dn_v = float(total_vol[max(0, low_idx - 2) : low_idx].sum()) if low_idx > 0 else -1.0
        if up_v >= dn_v and high_idx < num_bins - 1:
            next_hi = min(num_bins - 1, high_idx + 2)
            va_vol += float(total_vol[high_idx + 1 : next_hi + 1].sum())
            high_idx = next_hi
        elif low_idx > 0:
            next_lo = max(0, low_idx - 2)
            va_vol += float(total_vol[next_lo:low_idx].sum())
            low_idx = next_lo
        else:
            break

    vah = float(bin_edges[high_idx + 1])
    val = float(bin_edges[low_idx])

    # Detect Low Volume Node (LVN) vacuum left behind by a fast breakout pole below shelf_low
    lvn_zone: Optional[Tuple[float, float]] = None
    has_unfilled_lvn = False
    if shelf_low is not None and shelf_low > poc_price:
        shelf_bin = int(np.searchsorted(bin_centers, shelf_low))
        if shelf_bin > poc_idx + 2:
            mid_slice = total_vol[poc_idx + 1 : shelf_bin]
            if len(mid_slice) >= 2:
                poc_v = float(total_vol[poc_idx])
                valley_mask = mid_slice <= poc_v * 0.45
                if np.any(valley_mask):
                    v_idxs = np.where(valley_mask)[0] + (poc_idx + 1)
                    lvn_low = float(bin_edges[v_idxs[0]])
                    lvn_high = float(bin_edges[v_idxs[-1] + 1])
                    lvn_zone = (lvn_low, lvn_high)
                    has_unfilled_lvn = shelf_low >= (lvn_low + 0.45 * (lvn_high - lvn_low))

    return DailyVolumeProfile(
        bin_centers=bin_centers,
        bin_edges=bin_edges,
        total_vol=total_vol,
        up_vol=up_vol,
        down_vol=down_vol,
        poc=poc_price,
        vah=vah,
        val=val,
        lvn_zone=lvn_zone,
        has_unfilled_lvn=has_unfilled_lvn,
    )
