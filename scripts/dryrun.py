#!/usr/bin/env python3
"""
Offline end-to-end dry run.

Drives the real Stage-1, Stage-2 and Stage-3 code paths with realistic synthetic
OHLCV. Useful in sandboxes/CI where Yahoo & NSE are unreachable over TLS: only
the network boundary is patched, so detection gates, scoring, trade plans,
breadth, chart rendering, dedup ledger, state files and the PDF all run for real.

Usage:
    python scripts/dryrun.py            # all checks
    python scripts/dryrun.py --quick    # skip the Stage-3 PDF build
Exit code 0 = every check passed.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from test_detector import _make_nifty_df, _make_synthetic_flag_df  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((label, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    return bool(ok)


def build_universe(n_setups: int = 6, n_duds: int = 180) -> dict[str, pd.DataFrame]:
    """n_setups genuine V2.1 flags + n_duds quiet names that must be rejected."""
    rng = np.random.default_rng(7)
    out: dict[str, pd.DataFrame] = {}
    for i in range(n_setups):
        df = _make_synthetic_flag_df(base_days=6 + (i % 3), pole_days=2)
        df.index = pd.bdate_range(end="2026-11-20", periods=len(df))
        out[f"SETUP{i:02d}"] = df
    for i in range(n_duds):
        n = 240
        idx = pd.bdate_range(end="2026-11-20", periods=n)
        close = np.linspace(100, 108, n) + np.cumsum(rng.normal(0, 0.4, n))
        rows = [(c - 0.4, c + 1.2, c - 1.2, c, float(rng.uniform(2e5, 4e5))) for c in close]
        out[f"DUD{i:03d}"] = pd.DataFrame(
            rows, columns=["Open", "High", "Low", "Close", "Volume"], index=idx
        )
    return out


def flatten_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Guard against a MultiIndex (yfinance group_by='ticker') leaking in."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(-1)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the Stage-3 PDF build")
    args = ap.parse_args()

    universe = build_universe()

    import src.config_loader as cl
    import src.run_backtest as rb
    import src.run_eod_scan as eod
    import src.run_live_scan as live
    import src.telegram_notifier as tn

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        state, charts = td / "state", td / "charts"
        state.mkdir()
        charts.mkdir()
        (state / "sent_alerts.json").write_text(
            json.dumps({"alerted_symbols": {}}), encoding="utf-8"
        )
        wl = td / "empty_watchlist.txt"
        wl.write_text("# none\n", encoding="utf-8")

        cfg = cl.load_config()
        cfg["runtime"]["state_dir"] = str(state)
        cfg["runtime"]["charts_dir"] = str(charts)
        cfg["universe"]["custom_watchlist_abs"] = str(wl)
        cfg["regime"]["min_nifty_20d_return_pct"] = -99.0
        cfg["regime"]["min_nse_breadth50_pct"] = 0.0   # guards open for the happy path

        nifty = _make_nifty_df(universe["SETUP00"].index, ret20_pct=2.0, ret63_pct=6.0)

        # ---- patch ONLY the network boundary -----------------------------
        eod.fetch_daily_batch = lambda symbols, **kw: {
            s: universe[s] for s in symbols if s in universe
        }
        eod.fetch_daily_single = lambda sym, **kw: universe.get(sym, pd.DataFrame())
        eod.fetch_nifty_benchmark = lambda **kw: nifty
        eod.fetch_nse_universe = lambda **kw: (list(universe), {s: s for s in universe})
        live.fetch_daily_batch = eod.fetch_daily_batch
        live.fetch_nifty_benchmark = eod.fetch_nifty_benchmark
        # Simulate successful delivery so the ledger path can be exercised offline
        tn.TelegramNotifier.send_photo = lambda self, p, c: True
        tn.TelegramNotifier.send_message = lambda self, t, disable_preview=True: True

        print("\n── STAGE 1: EOD scan ──────────────────────────────────────────")
        s1 = eod.run_eod_scan(cfg=cfg, force=True, limit=0, dry_run=True, no_live_refresh=True)
        st = json.loads((state / "coiled_candidates.json").read_text(encoding="utf-8"))
        cands = st.get("candidates", [])

        check("scans the whole universe", s1.get("universe_scanned") == len(universe),
              f"{s1.get('universe_scanned')}/{len(universe)}")
        check("detects the genuine flags", len(cands) >= 1, f"{len(cands)} candidates")
        check("rejects every dud",
              all(c["symbol"].startswith("SETUP") for c in cands),
              f"{sum(1 for c in cands if not c['symbol'].startswith('SETUP'))} false positives")
        check("does not pause with guards open", s1.get("alerts_paused") is False)
        check("dispatches alerts", s1.get("alert_count", 0) >= 1, f"{s1.get('alert_count')}")
        check("renders charts", len(s1.get("charts_generated", [])) >= 1,
              f"{len(s1.get('charts_generated', []))} png")
        pngs = list(charts.glob("*.png"))
        check("chart files are non-trivial",
              bool(pngs) and pngs[0].stat().st_size > 5000,
              f"{pngs[0].stat().st_size} bytes" if pngs else "none")
        if cands:
            c0 = cands[0]
            check("candidate has the V2.1 schema",
                  all(k in c0 for k in ("symbol", "total_score", "sl_mode",
                                        "entry_price", "invalidation_stop",
                                        "tp1", "tp2", "tp3")))
            check("entry sits below the breakout (Option A)",
                  c0["entry_price"] < c0["breakout_level"],
                  f"{c0['entry_price']} < {c0['breakout_level']}")
            check("TP ladder is monotonic",
                  c0["entry_price"] < c0["tp1"] < c0["tp2"] < c0["tp3"])
        led = json.loads((state / "sent_alerts.json").read_text(encoding="utf-8"))
        check("dry run leaves the ledger untouched",
              led.get("total_alerted_symbols", 0) == 0)

        print("\n── STAGE 2: live scan ─────────────────────────────────────────")
        target = cands[0]["symbol"] if cands else "SETUP00"
        lu = dict(universe)
        df_t = flatten_frame(lu[target]).copy()
        bo = float(cands[0]["breakout_level"]) if cands else float(df_t["High"].iloc[-1])
        prev = float(df_t["Close"].iloc[-2])
        o, c = prev * 1.025, bo * 1.012
        df_t.iloc[-1] = [o, c * 1.006, o * 0.995, c, float(df_t["Volume"].iloc[-1]) * 6.0]
        lu[target] = df_t
        live.fetch_daily_batch = lambda symbols, **kw: {
            s: lu[s] for s in symbols if s in lu
        }

        s2 = live.run_live_scan(cfg=cfg, force=True, dry_run=True)
        check("monitors the Stage-1 watchlist", s2.get("monitored", 0) >= len(cands),
              f"{s2.get('monitored')}")
        check("does not abort", s2.get("aborted") is not True, str(s2.get("reason")))
        check("fires the live trigger", s2.get("triggers_fired", 0) >= 1,
              f"{s2.get('triggers_fired')} fired")
        if s2.get("events"):
            check("names a real trigger",
                  s2["events"][0].get("trigger") in
                  ("GAP_AND_GO", "BREAKOUT", "BREAKOUT_TRIGGER", "SPRING_RETEST",
                   "COIL_WATCH", "TIGHT_COIL"), str(s2["events"][0].get("trigger")))
        check("renders a live chart", len(list(charts.glob("LIVE_*.png"))) >= 1)

        if not args.quick:
            print("\n── STAGE 3: backtest PDF ──────────────────────────────────────")
            rb.REPO_ROOT = td
            rb.fetch_daily_batch = lambda symbols, **kw: {
                s: universe[s] for s in symbols if s in universe
            }
            rb.fetch_nifty_benchmark = lambda **kw: nifty
            res = rb.run_backtest(cfg=cfg, period="6mo", limit=0,
                                  symbols_override=list(universe), min_score=66.0)
            pdf = Path(res.get("pdf_path", ""))
            check("does not abort", res.get("aborted") is not True)
            check("writes a real PDF",
                  pdf.exists() and pdf.open("rb").read(5) == b"%PDF-"
                  and pdf.stat().st_size > 10000,
                  f"{pdf.stat().st_size if pdf.exists() else 0} bytes")
            check("finds the synthetic flags",
                  (res.get("summary", {}).get("total_trades") or 0) >= 1,
                  f"{res.get('summary', {}).get('total_trades')} trades")

    passed = sum(1 for _, ok, _ in CHECKS if ok)
    total = len(CHECKS)
    print("\n" + "=" * 62)
    print(f"DRY RUN: {passed}/{total} checks passed")
    if passed != total:
        for label, ok, detail in CHECKS:
            if not ok:
                print(f"  FAILED: {label} ({detail})")
        return 1
    print("✅ Stage-1 → Stage-2 → Stage-3 all produce output end-to-end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
