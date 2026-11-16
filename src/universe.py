"""NSE equity universe loader with live refresh & bundled CSV fallback."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)


def load_custom_watchlist(path: str | Path) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    symbols: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip().upper()
        if not s or s.startswith("#"):
            continue
        if s.endswith(".NS"):
            s = s[:-3]
        symbols.append(s)
    return list(dict.fromkeys(symbols))


def fetch_nse_universe(
    nse_url: str,
    fallback_csv: str | Path,
    custom_watchlist_path: str | Path | None = None,
    refresh_live: bool = True,
) -> Tuple[List[str], Dict[str, str]]:
    """
    Returns (symbols_without_suffix, symbol_to_company_name_map).
    Attempts live fetch from NSE official EQUITY_L.csv; gracefully falls back to bundled CSV
    if NSE rate-limits or blocks cloud runner IPs.
    """
    df: pd.DataFrame | None = None

    if refresh_live:
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
                ),
                "Accept": "text/csv,text/plain,*/*",
            }
            resp = requests.get(nse_url, headers=headers, timeout=15)
            resp.raise_for_status()
            if "SYMBOL" in resp.text[:250]:
                df = pd.read_csv(io.StringIO(resp.text))
                logger.info("Loaded fresh NSE universe from %s (%d raw rows)", nse_url, len(df))
        except Exception as exc:
            logger.warning("Live NSE CSV fetch failed (%s); falling back to bundled CSV.", exc)

    if df is None:
        fb = Path(fallback_csv)
        if not fb.exists():
            raise FileNotFoundError(f"Fallback universe CSV not found: {fb}")
        df = pd.read_csv(fb)
        logger.info("Loaded bundled NSE universe from %s (%d rows)", fb, len(df))

    df.columns = [str(c).strip() for c in df.columns]
    if "SERIES" in df.columns:
        df = df[df["SERIES"].astype(str).str.strip() == "EQ"].copy()

    symbols: List[str] = []
    names: Dict[str, str] = {}
    for _, row in df.iterrows():
        sym = str(row["SYMBOL"]).strip().upper()
        if not sym or sym == "NAN":
            continue
        # Skip odd/test/bond-like symbols with spaces
        if " " in sym:
            continue
        symbols.append(sym)
        if "NAME OF COMPANY" in df.columns:
            names[sym] = str(row["NAME OF COMPANY"]).strip()

    # Merge custom watchlist symbols at front so they're always scanned
    if custom_watchlist_path:
        custom = load_custom_watchlist(custom_watchlist_path)
        for c in reversed(custom):
            if c not in names:
                names[c] = c
            if c in symbols:
                symbols.remove(c)
            symbols.insert(0, c)

    symbols = list(dict.fromkeys(symbols))
    return symbols, names
