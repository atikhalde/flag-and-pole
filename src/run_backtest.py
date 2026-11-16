"""
CLI Runner for Stage-3 Historical Flag-and-Pole Backtest Scanner & PDF Report Generator.

Scans past Daily OHLCV history across NSE EQ equities with strict zero-look-ahead logic,
logs every trade's exact Pole Date, Entry Date, Breakout Date, SL Date, TP1/TP2/TP3 Dates,
exports both a clean CSV and a multi-page Executive PDF Report, and optionally sends the
PDF directly to your Telegram bots.

Examples:
  python -m src.run_backtest --period 2y --limit 300 --min-score 68
  python -m src.run_backtest --period 1y --symbols BOROLTD,AFFLE,RAMKY,IXIGO
  python -m src.run_backtest --period 2y --limit 0 --send-telegram
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.backtester import BacktestTrade, backtest_single_symbol, compute_backtest_summary
from src.config_loader import REPO_ROOT, load_config
from src.data_fetcher import fetch_daily_batch
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


def run_backtest(
    cfg: Dict[str, Any],
    period: str = "2y",
    limit: int = 500,
    symbols_override: Optional[List[str]] = None,
    min_score: float = 68.0,
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

    logger.info(
        "Historical Flag-and-Pole Backtest starting | Period=%s | Symbols=%d | MinScore=%.1f",
        period,
        len(symbols),
        min_score,
    )

    daily_map = fetch_daily_batch(
        symbols=symbols,
        period=period,
        batch_size=min(int(f_cfg["batch_size"]), 80),
        batch_sleep_sec=float(f_cfg["batch_sleep_sec"]),
        max_retries=int(f_cfg["max_retries"]),
        repair_nan_today=False,
    )
    logger.info("Downloaded historical Daily OHLCV for %d / %d symbols in %.1fs", len(daily_map), len(symbols), time.time() - t0)

    all_trades: List[BacktestTrade] = []
    date_mins: List[pd.Timestamp] = []
    date_maxs: List[pd.Timestamp] = []

    t_scan = time.time()
    for idx, (sym, df_d) in enumerate(daily_map.items(), start=1):
        if len(df_d) >= 60:
            date_mins.append(pd.Timestamp(df_d.index[0]))
            date_maxs.append(pd.Timestamp(df_d.index[-1]))
            sym_trades = backtest_single_symbol(
                symbol=sym,
                company_name=name_map.get(sym, sym),
                df=df_d,
                cfg=cfg,
                min_score=min_score,
                start_date=start_date,
                end_date=end_date,
                max_hold_days=max_hold_days,
            )
            all_trades.extend(sym_trades)
        if idx % 250 == 0:
            logger.info("Backtested %d / %d symbols | Trades found so far: %d (%.1fs)", idx, len(daily_map), len(all_trades), time.time() - t_scan)

    # Sort newest entry first for clean audit readability
    all_trades.sort(key=lambda tr: (tr.entry_date, tr.symbol), reverse=True)
    summary = compute_backtest_summary(all_trades)

    date_from = start_date or (min(date_mins).strftime("%Y-%m-%d") if date_mins else "N/A")
    date_to = end_date or (max(date_maxs).strftime("%Y-%m-%d") if date_maxs else now.strftime("%Y-%m-%d"))

    out_dir = REPO_ROOT / "output" / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y-%m-%d")
    csv_path = out_dir / f"backtest_trades_{period}_{stamp}.csv"
    pdf_path = out_dir / f"backtest_report_{period}_{stamp}.pdf"

    # Export full CSV with every date and trade metric
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
        "date_from": date_from,
        "date_to": date_to,
        "generated_at_ist": now.strftime("%Y-%m-%d %H:%M IST"),
    }

    generate_backtest_pdf(
        trades=all_trades,
        summary=summary,
        meta=meta,
        output_pdf_path=pdf_path,
        daily_map=daily_map,
    )

    # Optional: Send PDF Backtest Report directly to Telegram Bot #1 & Bot #2
    if send_telegram:
        notifier = TelegramNotifier(
            bot_token=cfg["runtime"]["telegram_bot_token"],
            chat_id=cfg["runtime"]["telegram_chat_id"],
            enabled=True,
            tv_url_template=cfg["telegram"]["tradingview_url_template"],
            extra_targets=cfg["runtime"].get("telegram_extra_targets", []),
        )
        caption = (
            f"📊 <b>NSE FLAG-AND-POLE (≤20d) BACKTEST REPORT ({period.upper()})</b>\n"
            f"📅 {date_from} → {date_to} | 🔎 Scanned: <b>{len(daily_map)}</b> NSE stocks (Score ≥ {min_score:.0f})\n\n"
            f"📈 <b>Key Results:</b>\n"
            f"• Trades: <b>{summary['total_trades']}</b> (Win Rate <b>{summary['win_rate_pct']:.1f}%</b>)\n"
            f"• TP1 Hit: <b>{summary['tp1_rate_pct']:.1f}%</b> | TP2: <b>{summary['tp2_rate_pct']:.1f}%</b> | TP3: <b>{summary['tp3_rate_pct']:.1f}%</b>\n"
            f"• Profit Factor: <b>{summary['profit_factor']:.2f}</b> | Avg Return: <b>{summary['avg_return_pct']:+.2f}%</b> ({summary['avg_r_multiple']:+.2f}R)\n"
            f"• ₹1L Sim: <b>₹{summary['final_equity_1l']:,.0f}</b> ({summary['net_compounded_pct']:+.1f}% | Max DD {summary['max_drawdown_pct']:.1f}%)\n\n"
            f"<i>Full PDF attached with Pole, Entry, Breakout, SL, TP1, TP2 & TP3 dates for every trade.</i>"
        )
        notifier.send_document(pdf_path, caption)

    elapsed = time.time() - t0
    logger.info(
        "Backtest complete in %.1fs | Scanned=%d | Trades=%d | WinRate=%.1f%% | PF=%.2f | PDF=%s",
        elapsed,
        len(daily_map),
        summary["total_trades"],
        summary["win_rate_pct"],
        summary["profit_factor"],
        pdf_path,
    )
    return {
        "elapsed_sec": round(elapsed, 1),
        "universe_scanned": len(daily_map),
        "period": period,
        "date_from": date_from,
        "date_to": date_to,
        "pdf_path": str(pdf_path),
        "csv_path": str(csv_path),
        "summary": {k: v for k, v in summary.items() if k not in ("equity_curve", "by_month")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Historical NSE Daily (<=20d) Flag-and-Pole Backtest + PDF Report")
    parser.add_argument("--config", default=None, help="Path to settings.yaml")
    parser.add_argument("--period", default="2y", help="yfinance history period (e.g., 6mo, 1y, 2y, 3y, 5y)")
    parser.add_argument("--limit", type=int, default=500, help="Max NSE stocks to backtest (0 = all 2,403 NSE EQ)")
    parser.add_argument("--symbols", default="", help="Optional comma-separated symbols (e.g., BOROLTD,AFFLE,RAMKY)")
    parser.add_argument("--min-score", type=float, default=68.0, help="Min scanner score threshold (default 68)")
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
