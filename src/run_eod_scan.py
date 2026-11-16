"""
Stage-1 EOD Full NSE Daily Pattern Scanner.
Scans the entire NSE EQ liquid universe on Daily timeframe (1d) to catch high-tight
post-pole coiled flags strictly within <= 20 trading days of the first breakout (pole).

Outputs:
  1. Updates state/coiled_candidates.json for Stage-2 Live Market Scanner
  2. Generates dark-theme annotated Daily Chart PNGs for top setups
  3. Sends Telegram EOD Digest + Individual Annotated Chart Photo Alerts
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.chart_generator import generate_pattern_chart
from src.config_loader import load_config
from src.data_fetcher import fetch_daily_batch, fetch_daily_single
from src.market_hours import is_nse_trading_day, now_ist
from src.pattern_detector import DailyPatternSetup, detect_daily_pattern
from src.telegram_notifier import TelegramNotifier
from src.universe import fetch_nse_universe
from src.volume_profile import DailyVolumeProfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eod_scanner")


def run_eod_scan(
    cfg: Dict[str, Any],
    force: bool = False,
    limit: int = 0,
    dry_run: bool = False,
    no_live_refresh: bool = False,
) -> Dict[str, Any]:
    now = now_ist()
    if not force and not is_nse_trading_day(now.date()):
        logger.info("Today (%s) is not an NSE trading day. Use --force to override.", now.date())
        return {"skipped": True, "reason": "non_trading_day"}

    t0 = time.time()
    u_cfg = cfg["universe"]
    f_cfg = cfg["fetcher"]
    s_cfg = cfg["scoring"]
    t_cfg = cfg["telegram"]

    symbols, name_map = fetch_nse_universe(
        nse_url=u_cfg["nse_equity_csv_url"],
        fallback_csv=u_cfg["fallback_csv_abs"],
        custom_watchlist_path=u_cfg["custom_watchlist_abs"],
        refresh_live=not no_live_refresh,
    )
    if limit > 0:
        symbols = symbols[:limit]
    logger.info("Stage-1 EOD Daily Scan starting for %d NSE symbols...", len(symbols))

    daily_map = fetch_daily_batch(
        symbols=symbols,
        period=f_cfg["daily_period"],
        batch_size=int(f_cfg["batch_size"]),
        batch_sleep_sec=float(f_cfg["batch_sleep_sec"]),
        max_retries=int(f_cfg["max_retries"]),
        repair_nan_today=False,  # Fast bulk pass; we repair surviving hits individually
    )
    logger.info("Fetched clean Daily OHLCV for %d / %d symbols in %.1fs", len(daily_map), len(symbols), time.time() - t0)

    # Detect pure Daily chart patterns (strictly <= 20 trading days from first breakout pole)
    min_wl_score = float(s_cfg["min_watchlist_score"])
    prelim_hits: List[Tuple[str, pd.DataFrame]] = []

    for sym, df_d in daily_map.items():
        setup, _ = detect_daily_pattern(sym, name_map.get(sym, sym), df_d, cfg)
        if setup is not None and setup.total_score >= min_wl_score:
            prelim_hits.append((sym, df_d))

    # Re-verify surviving hits with live-day NaN repair so stocks that ALREADY broke out today
    # are cleanly filtered out and coiled stocks include today's exact closing candle!
    repair_enabled = bool(f_cfg.get("repair_nan_today_with_15m", True))
    matched: List[Tuple[DailyPatternSetup, DailyVolumeProfile | None, pd.DataFrame]] = []
    for sym, df_d in prelim_hits:
        df_verified = df_d
        if repair_enabled:
            repaired = fetch_daily_single(sym, period=f_cfg["daily_period"], repair_nan_today=True)
            if not repaired.empty and len(repaired) >= len(df_d):
                df_verified = repaired
        setup, vp = detect_daily_pattern(sym, name_map.get(sym, sym), df_verified, cfg)
        if setup is not None and setup.total_score >= min_wl_score:
            matched.append((setup, vp, df_verified))

    # Sort best setups first
    matched.sort(key=lambda item: (item[0].total_score, item[0].dryup_pct, item[0].pole_rvol), reverse=True)
    max_wl = int(s_cfg["max_watchlist_size"])
    matched = matched[:max_wl]

    scan_date = matched[0][0].scan_date if matched else now.strftime("%Y-%m-%d")
    candidates_payload = [m[0].to_dict() for m in matched]

    # Persist watchlist state for Stage-2 Live Scanner
    state_dir = Path(cfg["runtime"]["state_dir"])
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / "coiled_candidates.json"

    state_obj = {
        "updated_at_ist": now.strftime("%Y-%m-%d %H:%M:%S IST"),
        "scan_date": scan_date,
        "universe_scanned": len(daily_map),
        "candidates": candidates_payload,
    }
    state_file.write_text(json.dumps(state_obj, indent=2), encoding="utf-8")
    logger.info("Saved %d qualified coiled candidates to %s", len(candidates_payload), state_file)

    # Filter setups that meet EOD Telegram alert threshold
    min_alert_score = float(s_cfg["min_eod_alert_score"])
    alert_items = [m for m in matched if m[0].total_score >= min_alert_score]
    max_charts = int(s_cfg["max_eod_chart_alerts"])

    notifier = TelegramNotifier(
        bot_token=cfg["runtime"]["telegram_bot_token"],
        chat_id=cfg["runtime"]["telegram_chat_id"],
        enabled=bool(t_cfg["enabled"]) and not dry_run,
        tv_url_template=t_cfg["tradingview_url_template"],
        extra_targets=cfg["runtime"].get("telegram_extra_targets", []),
    )

    if alert_items or bool(t_cfg.get("send_eod_summary_even_if_empty", True)):
        digest_html = notifier.format_eod_digest(scan_date, len(daily_map), [m[0].to_dict() for m in alert_items])
        notifier.send_message(digest_html)

    charts_dir = Path(cfg["runtime"]["charts_dir"])
    charts_dir.mkdir(parents=True, exist_ok=True)
    charts_generated: List[str] = []

    for setup, vp, df_d in alert_items[:max_charts]:
        setup_dict = setup.to_dict()
        chart_path = charts_dir / f"EOD_{setup.symbol}_{scan_date}.png"
        if bool(t_cfg.get("send_charts", True)):
            generate_pattern_chart(
                symbol=setup.symbol,
                df_daily=df_d,
                setup=setup_dict,
                output_path=chart_path,
                vp=vp,
            )
            charts_generated.append(str(chart_path))
            caption = notifier.format_eod_setup_caption(setup_dict)
            notifier.send_photo(chart_path, caption)
        else:
            notifier.send_message(notifier.format_eod_setup_caption(setup_dict))

    elapsed = time.time() - t0
    logger.info(
        "Stage-1 EOD Scan complete in %.1fs | Scanned=%d | Watchlist=%d | Alerted=%d | Charts=%d",
        elapsed,
        len(daily_map),
        len(candidates_payload),
        len(alert_items),
        len(charts_generated),
    )
    return {
        "skipped": False,
        "scan_date": scan_date,
        "universe_scanned": len(daily_map),
        "watchlist_count": len(candidates_payload),
        "alert_count": len(alert_items),
        "top_symbols": [c["symbol"] for c in candidates_payload[:10]],
        "charts_generated": charts_generated,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-1 Full NSE Daily Coiled Pre-Breakout Scanner")
    parser.add_argument("--config", default=None, help="Path to settings.yaml")
    parser.add_argument("--force", action="store_true", help="Run even on weekends/holidays")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of symbols scanned (for testing)")
    parser.add_argument("--dry-run", action="store_true", help="Do not send Telegram messages")
    parser.add_argument("--no-live-refresh", action="store_true", help="Use bundled data/nse_universe.csv only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    summary = run_eod_scan(
        cfg=cfg,
        force=args.force,
        limit=args.limit,
        dry_run=args.dry_run,
        no_live_refresh=args.no_live_refresh,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
