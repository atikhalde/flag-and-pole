"""
CLI Runner for Stage-3 Historical Flag-and-Pole Backtest Scanner & PDF Report Generator (V2 Option A).

Scans past Daily OHLCV history across NSE EQ equities with strict zero-look-ahead logic,
automatically fetches +1 year warmup data + NIFTY 50 (^NSEI) so 200 SMA, 52-week High, and
NIFTY Regime Guard are 100% exact from Day 1 of the requested backtest window, logs every
trade's exact Pole Date, Entry Date, Breakout Date, SL Date, TP1/TP2/TP3 Dates, exports
both CSV and a multi-page Executive PDF Report (Gross + Net of 0.25% NSE costs), and
optionally sends the PDF directly to your Telegram bots.

Examples:
  python -m src.run_backtest --period 2y --limit 300 --min-score 66
  python -m src.run_backtest --period 5y --limit 0 --min-score 66 --send-telegram
  python -m src.run_backtest --period 1y --symbols BOROLTD,APLLTD,POONAWALLA
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.backtester import BacktestTrade, backtest_single_symbol, compute_backtest_summary
from src.config_loader import REPO_ROOT, load_config
from src.data_fetcher import fetch_daily_batch, fetch_nifty_benchmark
from src.market_hours import now_ist
from src.pdf_report import generate_backtest_pdf
from src.telegram_notifier import TelegramNotifier
from src.universe import fetch_nse_universe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest_scanner")


def _compute_warmup_bounds(
    period: str,
    start_date: Optional[str],
    end_date: Optional[str],
    now_date: pd.Timestamp,
) -> Tuple[str, str, str]:
    """
    Given a user period (e.g. '6mo', '1y', '2y', '3y', '5y') or explicit start/end,
    return (download_start, effective_signal_start, effective_end) adding a 400-day warmup
    buffer so 200-SMA & 252-day 52-week High are exact on Day 1 of the backtest window.
    """
    eff_end_ts = pd.Timestamp(end_date) if end_date else now_date
    if start_date:
        eff_start_ts = pd.Timestamp(start_date)
    else:
        p = period.strip().lower()
        days_map = {
            "3mo": 92,
            "6mo": 183,
            "1y": 365,
            "2y": 730,
            "3y": 1095,
            "5y": 1826,
        }
        days_back = days_map.get(p, 730)
        eff_start_ts = eff_end_ts - timedelta(days=days_back)

    dl_start_ts = eff_start_ts - timedelta(days=400)
    return (
        dl_start_ts.strftime("%Y-%m-%d"),
        eff_start_ts.strftime("%Y-%m-%d"),
        eff_end_ts.strftime("%Y-%m-%d"),
    )


def run_backtest(
    cfg: Dict[str, Any],
    period: str = "2y",
    limit: int = 500,
    symbols_override: Optional[List[str]] = None,
    min_score: float = 78.0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_hold_days: int = 60,
    send_telegram: bool = False,
    no_live_refresh: bool = False,
) -> Dict[str, Any]:
    t0 = time.time()
    now = now_ist()
    u_cfg = cfg["universe"]
    f_cfg = cfg["fetcher"]
    r_cfg = cfg.get("regime", {})
    tp_cfg = cfg.get("trade_plan", {})

    if symbols_override:
        symbols = [s.strip().upper() for s in symbols_override if s.strip()]
        name_map = {s: s for s in symbols}
    else:
        symbols, name_map = fetch_nse_universe(
            nse_url=u_cfg["nse_equity_csv_url"],
            fallback_csv=u_cfg["fallback_csv_abs"],
            custom_watchlist_path=u_cfg["custom_watchlist_abs"],
            refresh_live=not no_live_refresh,
        )
        if limit > 0:
            symbols = symbols[:limit]

    dl_start, eff_start, eff_end = _compute_warmup_bounds(
        period=period,
        start_date=start_date,
        end_date=end_date,
        now_date=pd.Timestamp(now.date()),
    )
    # yfinance end date is exclusive, so add +2 days when explicitly specified
    dl_end = (pd.Timestamp(eff_end) + timedelta(days=2)).strftime("%Y-%m-%d")

    logger.info(
        "V2 Historical Flag-and-Pole Backtest starting | Period=%s (%s -> %s, warmup from %s) | Symbols=%d | MinScore=%.1f | SL=%s",
        period,
        eff_start,
        eff_end,
        dl_start,
        len(symbols),
        min_score,
        tp_cfg.get("sl_mode", "daily_close"),
    )

    nifty_df = fetch_nifty_benchmark(
        ticker=str(r_cfg.get("benchmark_ticker", "^NSEI")),
        start=dl_start,
        end=dl_end,
    )
    logger.info("Downloaded NIFTY 50 (^NSEI) benchmark: %d daily bars", len(nifty_df))

    daily_map = fetch_daily_batch(
        symbols=symbols,
        start=dl_start,
        end=dl_end,
        batch_size=min(int(f_cfg["batch_size"]), 80),
        batch_sleep_sec=float(f_cfg["batch_sleep_sec"]),
        max_retries=int(f_cfg["max_retries"]),
        repair_nan_today=False,
    )
    logger.info("Downloaded historical Daily OHLCV for %d / %d symbols in %.1fs", len(daily_map), len(symbols), time.time() - t0)

    all_trades: List[BacktestTrade] = []
    t_scan = time.time()
    for idx, (sym, df_d) in enumerate(daily_map.items(), start=1):
        if len(df_d) >= 60:
            sym_trades = backtest_single_symbol(
                symbol=sym,
                company_name=name_map.get(sym, sym),
                df=df_d,
                cfg=cfg,
                min_score=min_score,
                start_date=eff_start,
                end_date=eff_end,
                max_hold_days=max_hold_days,
                nifty_df=nifty_df,
            )
            all_trades.extend(sym_trades)
        if idx % 250 == 0:
            logger.info("Backtested %d / %d symbols | Trades found so far: %d (%.1fs)", idx, len(daily_map), len(all_trades), time.time() - t_scan)

    # Sort newest entry first for clean audit readability
    all_trades.sort(key=lambda tr: (tr.entry_date, tr.symbol), reverse=True)
    cost_pct = float(tp_cfg.get("round_trip_cost_pct", 0.25))
    summary = compute_backtest_summary(all_trades, round_trip_cost_pct=cost_pct)

    out_dir = REPO_ROOT / "output" / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y-%m-%d")
    csv_path = out_dir / f"backtest_trades_{period}_{stamp}.csv"
    pdf_path = out_dir / f"backtest_report_{period}_{stamp}.pdf"

    if all_trades:
        fieldnames = list(all_trades[0].to_dict().keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for tr in all_trades:
                writer.writerow(tr.to_dict())
    else:
        csv_path.write_text("symbol,pole_start_date,entry_date,breakout_date,sl_date,tp1_date,tp2_date,tp3_date,outcome,return_pct\n", encoding="utf-8")

    meta = {
        "period": period,
        "universe_scanned": len(daily_map),
        "min_score": min_score,
        "sl_mode": tp_cfg.get("sl_mode", "daily_close"),
        "date_from": eff_start,
        "date_to": eff_end,
        "generated_at_ist": now.strftime("%Y-%m-%d %H:%M IST"),
    }

    generate_backtest_pdf(
        trades=all_trades,
        summary=summary,
        meta=meta,
        output_pdf_path=pdf_path,
        daily_map=daily_map,
    )

    if send_telegram:
        notifier = TelegramNotifier(
            bot_token=cfg["runtime"]["telegram_bot_token"],
            chat_id=cfg["runtime"]["telegram_chat_id"],
            enabled=True,
            tv_url_template=cfg["telegram"]["tradingview_url_template"],
            extra_targets=cfg["runtime"].get("telegram_extra_targets", []),
        )
        caption = (
            f"📊 <b>NSE V2 FLAG-AND-POLE (≤20d) BACKTEST REPORT ({period.upper()})</b>\n"
            f"📅 {eff_start} → {eff_end} | 🔎 Scanned: <b>{len(daily_map)}</b> NSE stocks (Score ≥ {min_score:.0f})\n"
            f"🛡 Rules: <b>Daily-Close SL + -9% GTT</b> | 45/30/25 TP (+4.5/+10/+18%)\n\n"
            f"📈 <b>Key Results (Gross / Net 0.25% NSE Cost):</b>\n"
            f"• Trades: <b>{summary['total_trades']}</b> (Win Rate <b>{summary['win_rate_pct']:.1f}%</b>)\n"
            f"• TP1 Hit: <b>{summary['tp1_rate_pct']:.1f}%</b> | TP2: <b>{summary['tp2_rate_pct']:.1f}%</b> | TP3: <b>{summary['tp3_rate_pct']:.1f}%</b>\n"
            f"• Profit Factor: <b>{summary['profit_factor']:.2f}</b> (Net <b>{summary['net_profit_factor']:.2f}</b>) | Avg: <b>{summary['avg_return_pct']:+.2f}%</b> ({summary['avg_r_multiple']:+.2f}R)\n"
            f"• ₹1L Sim: <b>₹{summary['final_equity_1l']:,.0f}</b> (Net <b>₹{summary['net_final_equity_1l']:,.0f}</b> | Max DD {summary['max_drawdown_pct']:.1f}%)\n\n"
            f"<i>Full PDF attached with Pole, Entry, Breakout, SL, TP1, TP2 & TP3 dates for every trade.</i>"
        )
        notifier.send_document(pdf_path, caption)

    elapsed = time.time() - t0
    logger.info(
        "V2 Backtest complete in %.1fs | Scanned=%d | Trades=%d | WinRate=%.1f%% | TP1=%.1f%% | PF=%.2f (Net %.2f) | PDF=%s",
        elapsed,
        len(daily_map),
        summary["total_trades"],
        summary["win_rate_pct"],
        summary["tp1_rate_pct"],
        summary["profit_factor"],
        summary["net_profit_factor"],
        pdf_path,
    )
    return {
        "elapsed_sec": round(elapsed, 1),
        "universe_scanned": len(daily_map),
        "period": period,
        "date_from": eff_start,
        "date_to": eff_end,
        "pdf_path": str(pdf_path),
        "csv_path": str(csv_path),
        "summary": {k: v for k, v in summary.items() if k not in ("equity_curve", "by_month")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Historical NSE Daily (<=20d) V2 Flag-and-Pole Backtest + PDF Report")
    parser.add_argument("--config", default=None, help="Path to settings.yaml")
    parser.add_argument("--period", default="2y", help="Backtest window (e.g., 6mo, 1y, 2y, 3y, 5y)")
    parser.add_argument("--limit", type=int, default=500, help="Max NSE stocks to backtest (0 = all ~2,403 NSE EQ)")
    parser.add_argument("--symbols", default="", help="Optional comma-separated symbols (e.g., BOROLTD,APLLTD)")
    parser.add_argument("--min-score", type=float, default=78.0, help="Min V2 score threshold (default 78 = Grade A+, 66 = Grade A/A+)")
    parser.add_argument("--start-date", default=None, help="Optional YYYY-MM-DD start filter")
    parser.add_argument("--end-date", default=None, help="Optional YYYY-MM-DD end filter")
    parser.add_argument("--max-hold-days", type=int, default=60, help="Max trading days to hold a trade")
    parser.add_argument("--send-telegram", action="store_true", help="Send generated PDF report to Telegram bots")
    parser.add_argument("--no-live-refresh", action="store_true", help="Use bundled data/nse_universe.csv only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sym_list = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else None

    res = run_backtest(
        cfg=cfg,
        period=args.period,
        limit=args.limit,
        symbols_override=sym_list,
        min_score=args.min_score,
        start_date=args.start_date,
        end_date=args.end_date,
        max_hold_days=args.max_hold_days,
        send_telegram=args.send_telegram,
        no_live_refresh=args.no_live_refresh,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
