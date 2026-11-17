"""
Stage-1 EOD Full NSE Daily Pattern Scanner (V2.1 Core + Coil — 5-Year Full-NSE Edge).
Scans the entire NSE EQ liquid universe on Daily timeframe (1d) to catch high-tight
post-pole coiled flags strictly within <= 20 trading days of the first breakout (pole),
protected by V2.1 Dual Market Regime Guard (NIFTY 50 20d + Live NSE Breadth50 >= 40%),
Stage-2 150/200-SMA + Rising 50-SMA, 3m RS >= +15%, 52wH >= -15%, Box >= 5.5%,
Upper-Pole Hold >= 65%, Pole 18-42%, and Upper-Flag Coil >= 20%.

Outputs:
  1. Updates state/coiled_candidates.json for Stage-2 Live Market Scanner
  2. Generates dark-theme annotated Daily Chart PNGs for top Grade A / A+ setups
  3. Sends Telegram EOD Digest + Individual Annotated Chart Photo Alerts (strict never-repeat)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.alert_ledger import AlertLedger
from src.chart_generator import generate_pattern_chart
from src.config_loader import load_config
from src.data_fetcher import fetch_daily_batch, fetch_daily_single, fetch_nifty_benchmark
from src.market_hours import is_nse_candle_incomplete_today, is_nse_trading_day, now_ist
from src.pattern_detector import (
    DailyPatternSetup,
    compute_nse_breadth50,
    detect_daily_pattern,
    evaluate_nifty_regime,
)
from src.telegram_notifier import TelegramNotifier
from src.universe import fetch_nse_universe
from src.volume_profile import DailyVolumeProfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eod_scanner")


def strip_incomplete_today_bar(df: pd.DataFrame, cutoff_date: date) -> Tuple[pd.DataFrame, bool]:
    """
    Drop any daily bar with calendar date >= cutoff_date so Stage-1 EOD never
    evaluates a partial/incomplete intraday candle when run before 15:30 IST.
    """
    if df is None or df.empty:
        return df, False
    idx_dates = pd.DatetimeIndex(df.index).date
    if idx_dates[-1] >= cutoff_date:
        trimmed = df.loc[idx_dates < cutoff_date]
        return trimmed, True
    return df, False


def run_eod_scan(
    cfg: Dict[str, Any],
    force: bool = False,
    limit: int = 0,
    dry_run: bool = False,
    no_live_refresh: bool = False,
    include_live_bar: bool = False,
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
    r_cfg = cfg.get("regime", {})

    drop_incomplete_today = (not include_live_bar) and is_nse_candle_incomplete_today(now)
    today_ist = now.date()
    if drop_incomplete_today:
        logger.info(
            "NSE session for %s not yet closed (%s IST < 15:30 IST) — Stage-1 EOD will strip any incomplete %s intraday bar and scan the latest completed daily close.",
            today_ist.isoformat(),
            now.strftime("%H:%M"),
            today_ist.isoformat(),
        )

    # 1. Fetch NIFTY 50 (^NSEI) Benchmark for V2 Market Regime Guard & Relative Strength
    nifty_ticker = str(r_cfg.get("benchmark_ticker", "^NSEI"))
    nifty_df = fetch_nifty_benchmark(ticker=nifty_ticker, period=f_cfg["daily_period"])
    if drop_incomplete_today and nifty_df is not None and not nifty_df.empty:
        nifty_df, nifty_dropped = strip_incomplete_today_bar(nifty_df, today_ist)
        if nifty_dropped:
            logger.info("Stripped incomplete %s live bar from %s benchmark.", today_ist.isoformat(), nifty_ticker)
    regime = evaluate_nifty_regime(nifty_df, cfg=cfg)
    if regime["available"]:
        logger.info(
            "NIFTY 50 Regime (%s): Close=%.1f | 20d=%+.2f%% | Above 50 SMA=%s | Guard=%s",
            regime["as_of"],
            regime["nifty_close"],
            regime["ret20_pct"],
            regime["above_50sma"],
            "OPEN (OK)" if regime["regime_ok"] else "PAUSED (Sharp Sell-off)",
        )

    state_dir = Path(cfg["runtime"]["state_dir"])
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / "coiled_candidates.json"

    if not regime["regime_ok"]:
        logger.warning(
            "V2 NIFTY Regime Guard PAUSED scan: NIFTY 20d return %.2f%% < %.2f%% threshold. Skipping alerts.",
            regime["ret20_pct"],
            float(r_cfg.get("min_nifty_20d_return_pct", -3.0)),
        )
        state_obj = {
            "version": "V2.1 Core + Coil",
            "updated_at_ist": now.strftime("%Y-%m-%d %H:%M:%S IST"),
            "scan_date": regime.get("as_of") or now.strftime("%Y-%m-%d"),
            "dropped_incomplete_live_bar": drop_incomplete_today,
            "universe_scanned": 0,
            "nifty_regime": regime,
            "candidates": [],
        }
        state_file.write_text(json.dumps(state_obj, indent=2), encoding="utf-8")
        return {
            "skipped": True,
            "reason": "nifty_regime_paused",
            "nifty_regime": regime,
            "watchlist_count": 0,
            "alert_count": 0,
        }

    symbols, name_map = fetch_nse_universe(
        nse_url=u_cfg["nse_equity_csv_url"],
        fallback_csv=u_cfg["fallback_csv_abs"],
        custom_watchlist_path=u_cfg["custom_watchlist_abs"],
        refresh_live=not no_live_refresh,
    )
    if limit > 0:
        symbols = symbols[:limit]
    logger.info("Stage-1 V2.1 EOD Daily Scan starting for %d NSE symbols (period=%s)...", len(symbols), f_cfg["daily_period"])

    daily_map = fetch_daily_batch(
        symbols=symbols,
        period=f_cfg["daily_period"],
        batch_size=int(f_cfg["batch_size"]),
        batch_sleep_sec=float(f_cfg["batch_sleep_sec"]),
        max_retries=int(f_cfg["max_retries"]),
        repair_nan_today=False,  # Fast bulk pass; we repair surviving hits individually
    )
    logger.info("Fetched clean Daily OHLCV for %d / %d symbols in %.1fs", len(daily_map), len(symbols), time.time() - t0)

    if drop_incomplete_today:
        stripped_count = 0
        cleaned_map: Dict[str, pd.DataFrame] = {}
        for sym, df_d in daily_map.items():
            trimmed, was_dropped = strip_incomplete_today_bar(df_d, today_ist)
            if was_dropped:
                stripped_count += 1
            if not trimmed.empty:
                cleaned_map[sym] = trimmed
        daily_map = cleaned_map
        if stripped_count > 0:
            logger.info(
                "Stripped incomplete %s live bar from %d / %d symbols (scanning completed EOD closes).",
                today_ist.isoformat(),
                stripped_count,
                len(daily_map),
            )

    # 2. Compute Live NSE Market Breadth50 (% of NSE stocks > 50-SMA) with ZERO extra API calls!
    breadth_info = compute_nse_breadth50(daily_map, cfg=cfg)
    breadth50_pct = breadth_info["breadth50_pct"] if breadth_info["enforced"] else None
    if breadth_info["available"]:
        logger.info(
            "NSE Live Breadth50 (%s): %.1f%% (%d / %d stocks > 50 SMA) | Threshold>=%.0f%% | Guard=%s",
            breadth_info["as_of"],
            breadth_info["breadth50_pct"],
            breadth_info["above_count"],
            breadth_info["total_count"],
            breadth_info["min_threshold_pct"],
            "OPEN (OK)" if breadth_info["breadth_ok"] else "PAUSED (Weak Broad Market)",
        )
    regime["nse_breadth50_pct"] = breadth_info["breadth50_pct"]
    regime["nse_breadth_above"] = breadth_info["above_count"]
    regime["nse_breadth_total"] = breadth_info["total_count"]
    regime["breadth_ok"] = breadth_info["breadth_ok"]

    if not breadth_info["breadth_ok"]:
        logger.warning(
            "V2.1 NSE Breadth50 Guard PAUSED scan: %.1f%% (< %.1f%% threshold). Skipping breakout alerts in bear breadth.",
            breadth_info["breadth50_pct"],
            breadth_info["min_threshold_pct"],
        )
        state_obj = {
            "version": "V2.1 Core + Coil",
            "updated_at_ist": now.strftime("%Y-%m-%d %H:%M:%S IST"),
            "scan_date": breadth_info.get("as_of") or regime.get("as_of") or now.strftime("%Y-%m-%d"),
            "dropped_incomplete_live_bar": drop_incomplete_today,
            "universe_scanned": len(daily_map),
            "nifty_regime": regime,
            "nse_breadth50": breadth_info,
            "candidates": [],
        }
        state_file.write_text(json.dumps(state_obj, indent=2), encoding="utf-8")
        return {
            "skipped": True,
            "reason": "nse_breadth_paused",
            "nifty_regime": regime,
            "nse_breadth50": breadth_info,
            "watchlist_count": 0,
            "alert_count": 0,
        }

    # Detect V2.1 Core + Coil pure Daily chart patterns (strictly <= 20 trading days + 11 institutional gates)
    min_wl_score = float(s_cfg["min_watchlist_score"])
    prelim_hits: List[Tuple[str, pd.DataFrame]] = []

    for sym, df_d in daily_map.items():
        setup, _ = detect_daily_pattern(
            sym,
            name_map.get(sym, sym),
            df_d,
            cfg,
            nifty_df=nifty_df,
            breadth50_pct=breadth50_pct,
            compute_vp=False,
        )
        if setup is not None and setup.total_score >= min_wl_score:
            prelim_hits.append((sym, df_d))

    # Re-verify surviving hits with live-day NaN repair and full Daily Volume Profile
    repair_enabled = bool(f_cfg.get("repair_nan_today_with_15m", True)) and not drop_incomplete_today
    matched: List[Tuple[DailyPatternSetup, DailyVolumeProfile | None, pd.DataFrame]] = []
    for sym, df_d in prelim_hits:
        df_verified = df_d
        if repair_enabled:
            repaired = fetch_daily_single(sym, period=f_cfg["daily_period"], repair_nan_today=True)
            if drop_incomplete_today and not repaired.empty:
                repaired, _ = strip_incomplete_today_bar(repaired, today_ist)
            if not repaired.empty and len(repaired) >= len(df_d):
                df_verified = repaired
        setup, vp = detect_daily_pattern(
            sym,
            name_map.get(sym, sym),
            df_verified,
            cfg,
            nifty_df=nifty_df,
            breadth50_pct=breadth50_pct,
            compute_vp=True,
        )
        if setup is not None and setup.total_score >= min_wl_score:
            matched.append((setup, vp, df_verified))

    # Sort best V2.1 setups first (Score -> RS vs NIFTY -> Upper-Flag Coil Position)
    matched.sort(key=lambda item: (item[0].total_score, item[0].rs63_vs_nifty, item[0].sig_pos_in_box), reverse=True)
    max_wl = int(s_cfg["max_watchlist_size"])
    matched = matched[:max_wl]

    scan_date = matched[0][0].scan_date if matched else (regime.get("as_of") or now.strftime("%Y-%m-%d"))
    candidates_payload = [m[0].to_dict() for m in matched]

    state_obj = {
        "version": "V2.1 Core + Coil",
        "updated_at_ist": now.strftime("%Y-%m-%d %H:%M:%S IST"),
        "scan_date": scan_date,
        "dropped_incomplete_live_bar": drop_incomplete_today,
        "universe_scanned": len(daily_map),
        "nifty_regime": regime,
        "nse_breadth50": breadth_info,
        "candidates": candidates_payload,
    }
    state_file.write_text(json.dumps(state_obj, indent=2), encoding="utf-8")
    logger.info("Saved %d qualified V2.1 coiled candidates to %s", len(candidates_payload), state_file)

    # Filter setups that meet EOD Telegram alert threshold (Grade A/A+ >= 66) AND have NEVER been alerted before
    min_alert_score = float(s_cfg["min_eod_alert_score"])
    eligible_items = [m for m in matched if m[0].total_score >= min_alert_score]
    max_charts = int(s_cfg["max_eod_chart_alerts"])

    strict_dedup = bool(cfg.get("deduplication", {}).get("strict_never_repeat_symbol", True))
    ledger = AlertLedger(state_dir / "sent_alerts.json")

    alert_items = []
    skipped_already_sent: List[str] = []
    seen_this_run: set[str] = set()
    for item in eligible_items:
        sym_key = item[0].symbol.strip().upper()
        if sym_key in seen_this_run or (strict_dedup and ledger.is_sent(sym_key)):
            skipped_already_sent.append(sym_key)
            continue
        seen_this_run.add(sym_key)
        alert_items.append(item)

    if skipped_already_sent:
        logger.info(
            "Strict dedup skipped %d already-alerted symbol(s): %s",
            len(skipped_already_sent),
            ", ".join(skipped_already_sent),
        )

    notifier = TelegramNotifier(
        bot_token=cfg["runtime"]["telegram_bot_token"],
        chat_id=cfg["runtime"]["telegram_chat_id"],
        enabled=bool(t_cfg["enabled"]) and not dry_run,
        tv_url_template=t_cfg["tradingview_url_template"],
        extra_targets=cfg["runtime"].get("telegram_extra_targets", []),
    )

    if alert_items or bool(t_cfg.get("send_eod_summary_even_if_empty", False)):
        digest_html = notifier.format_eod_digest(
            scan_date,
            len(daily_map),
            [m[0].to_dict() for m in alert_items],
            nifty_regime=regime,
        )
        notifier.send_message(digest_html)

    charts_dir = Path(cfg["runtime"]["charts_dir"])
    charts_dir.mkdir(parents=True, exist_ok=True)
    charts_generated: List[str] = []
    newly_alerted: List[str] = []
    ts_ist = now.strftime("%Y-%m-%d %H:%M IST")

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
            delivered = notifier.send_photo(chart_path, caption)
        else:
            delivered = notifier.send_message(notifier.format_eod_setup_caption(setup_dict))

        if not dry_run and delivered:
            ledger.mark_sent(
                symbol=setup.symbol,
                stage="EOD",
                alert_type="PRE_BREAKOUT",
                price=setup.entry_price,
                timestamp_ist=ts_ist,
                extra={"scan_date": scan_date, "score": setup.total_score, "pole_start_date": setup.pole_start_date},
            )
            newly_alerted.append(setup.symbol)

    if not dry_run:
        ledger.save()

    elapsed = time.time() - t0
    logger.info(
        "Stage-1 V2.1 EOD Scan complete in %.1fs | Scanned=%d | Breadth50=%s | Watchlist=%d | NewAlerts=%d | SkippedDup=%d | Charts=%d",
        elapsed,
        len(daily_map),
        f"{breadth_info['breadth50_pct']:.1f}%" if breadth_info["available"] else "N/A",
        len(candidates_payload),
        len(newly_alerted),
        len(skipped_already_sent),
        len(charts_generated),
    )
    return {
        "skipped": False,
        "scan_date": scan_date,
        "dropped_incomplete_live_bar": drop_incomplete_today,
        "universe_scanned": len(daily_map),
        "nifty_regime": regime,
        "nse_breadth50": breadth_info,
        "watchlist_count": len(candidates_payload),
        "alert_count": len(alert_items),
        "newly_alerted": newly_alerted,
        "skipped_already_sent": skipped_already_sent,
        "top_symbols": [c["symbol"] for c in candidates_payload[:10]],
        "charts_generated": charts_generated,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-1 Full NSE Daily Coiled Pre-Breakout Scanner (V2.1 Core + Coil)")
    parser.add_argument("--config", default=None, help="Path to settings.yaml")
    parser.add_argument("--force", action="store_true", help="Run even on weekends/holidays")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of symbols scanned (for testing)")
    parser.add_argument("--dry-run", action="store_true", help="Do not send Telegram messages")
    parser.add_argument("--no-live-refresh", action="store_true", help="Use bundled data/nse_universe.csv only")
    parser.add_argument(
        "--include-live-bar",
        action="store_true",
        help="Include today's incomplete intraday bar if run before 15:30 IST (default: strip incomplete bar)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    summary = run_eod_scan(
        cfg=cfg,
        force=args.force,
        limit=args.limit,
        dry_run=args.dry_run,
        no_live_refresh=args.no_live_refresh,
        include_live_bar=args.include_live_bar,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
