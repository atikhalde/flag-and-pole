"""
Pure Daily-TF yfinance Data Fetcher.
- Batched daily OHLCV downloads for all ~2,400 NSE equities
- Handles Yahoo Finance's NSE live-day quirk where today's `1d` row can briefly
  return NaN OHLC while Volume is populated (repairs that single day cleanly
  from a tiny 2d/15m fallback only when needed).
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Iterable, List

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def _normalize_daily_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    if len(cols) < 5:
        return pd.DataFrame()
    out = df[cols].copy()
    out = out.dropna(subset=["Open", "High", "Low", "Close"], how="any")
    out["Volume"] = out["Volume"].fillna(0).astype(float)
    out = out[out["Close"] > 0]
    if out.empty:
        return out
    if out.index.tz is not None:
        out.index = pd.to_datetime(out.index.tz_convert("Asia/Kolkata").date)
    else:
        out.index = pd.to_datetime(out.index.date)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def _repair_latest_day_if_nan(symbol: str, raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> pd.DataFrame:
    """
    If Yahoo's latest 1d row had NaN OHLC (common on NSE during/right after market hours),
    aggregate today's daily OHLCV cleanly from a minimal 2d/15m pull.
    """
    if raw_df is None or raw_df.empty:
        return clean_df
    last_raw = raw_df.iloc[-1]
    if not pd.isna(last_raw.get("Close")):
        return clean_df

    try:
        intra = yf.Ticker(f"{symbol}.NS").history(period="2d", interval="15m", auto_adjust=False)
        if intra is None or intra.empty:
            return clean_df
        intra = intra.dropna(subset=["Open", "High", "Low", "Close"], how="any")
        if intra.empty:
            return clean_df
        if intra.index.tz is not None:
            intra.index = intra.index.tz_convert("Asia/Kolkata")
        latest_d = intra.index[-1].date()
        day_bars = intra[intra.index.date == latest_d]
        if day_bars.empty:
            return clean_df

        raw_vol = float(last_raw.get("Volume", 0) or 0)
        intra_vol = float(day_bars["Volume"].sum())
        row = pd.DataFrame(
            {
                "Open": [float(day_bars["Open"].iloc[0])],
                "High": [float(day_bars["High"].max())],
                "Low": [float(day_bars["Low"].min())],
                "Close": [float(day_bars["Close"].iloc[-1])],
                "Volume": [max(raw_vol, intra_vol)],
            },
            index=pd.to_datetime([latest_d]),
        )
        if clean_df.empty:
            return row
        merged = pd.concat([clean_df[clean_df.index != row.index[0]], row]).sort_index()
        return merged
    except Exception as exc:
        logger.debug("Live-day NaN repair failed for %s: %s", symbol, exc)
        return clean_df


def fetch_daily_batch(
    symbols: Iterable[str],
    period: str = "5mo",
    batch_size: int = 100,
    batch_sleep_sec: float = 1.0,
    max_retries: int = 2,
    repair_nan_today: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Download daily (1d) OHLCV for a list of NSE symbols (without .NS suffix).
    Returns {symbol: clean_daily_df} with timezone-naive IST date index.
    """
    sym_list = list(symbols)
    results: Dict[str, pd.DataFrame] = {}
    if not sym_list:
        return results

    total_batches = (len(sym_list) + batch_size - 1) // batch_size

    for b_idx in range(total_batches):
        chunk = sym_list[b_idx * batch_size : (b_idx + 1) * batch_size]
        tickers = [f"{s}.NS" for s in chunk]
        raw = None

        for attempt in range(max_retries + 1):
            try:
                raw = yf.download(
                    tickers=tickers,
                    period=period,
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=False,
                    threads=True,
                    progress=False,
                )
                if raw is not None and not raw.empty:
                    break
            except Exception as exc:
                logger.warning("Daily batch %d/%d attempt %d failed: %s", b_idx + 1, total_batches, attempt + 1, exc)
                time.sleep(2.0 * (attempt + 1))

        if raw is None or raw.empty:
            continue

        if len(tickers) == 1:
            sym = chunk[0]
            sub = raw.copy()
            if isinstance(sub.columns, pd.MultiIndex):
                sub.columns = sub.columns.get_level_values(-1)
            cleaned = _normalize_daily_df(sub)
            if repair_nan_today:
                cleaned = _repair_latest_day_if_nan(sym, sub, cleaned)
            if not cleaned.empty:
                results[sym] = cleaned
        else:
            lvl0 = set(raw.columns.get_level_values(0))
            for sym, tk in zip(chunk, tickers):
                if tk not in lvl0:
                    continue
                sub = raw[tk]
                cleaned = _normalize_daily_df(sub)
                if repair_nan_today:
                    cleaned = _repair_latest_day_if_nan(sym, sub, cleaned)
                if not cleaned.empty:
                    results[sym] = cleaned

        if b_idx < total_batches - 1 and batch_sleep_sec > 0:
            time.sleep(batch_sleep_sec)

    return results


def fetch_daily_single(
    symbol: str,
    period: str = "5mo",
    repair_nan_today: bool = True,
) -> pd.DataFrame:
    """Fetch clean Daily OHLCV for a single NSE symbol (with live-day NaN repair)."""
    data = fetch_daily_batch(
        [symbol],
        period=period,
        batch_size=1,
        batch_sleep_sec=0.0,
        repair_nan_today=repair_nan_today,
    )
    return data.get(symbol, pd.DataFrame())
