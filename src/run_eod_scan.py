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

from src.alert_ledger import ledger_from_config
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
from src.run_visibility import publish_abort, publish_pause, write_job_summary
from src.telegram_notifier import TelegramNotifier
from src.universe import fetch_nse_universe
from src.volume_profile import DailyVolumeProfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eod_scanner")


def check_fetch_integrity(
    fetched: int,
    requested: int,
    cfg: Dict[str, Any],
    limit: int = 0,
) -> Tuple[bool, str]:
    """
    Decide whether a run should ABORT rather than commit an empty/degraded watchlist.

    Returns (should_abort, human_readable_reason). Skipped entirely for small
    `--limit` test runs so local smoke tests on 5 symbols still work.
    """
    f_cfg = (cfg or {}).get("fetcher", {}) or {}
    min_pct = float(f_cfg.get("min_fetch_success_pct", 50.0))
    min_abs = int(f_cfg.get("min_fetched_symbols", 100))

    if requested <= 0:
        return True, "No symbols requested from the NSE universe"

    if fetched <= 0:
        return True, f"Zero of {requested} requested symbols returned data"

    pct = fetched / requested * 100.0

    # Coverage floor: a partial download is a data outage, whatever its size.
    if pct < min_pct:
        return (
            True,
            f"Only {fetched}/{requested} symbols fetched ({pct:.1f}%) — below the "
            f"{min_pct:.0f}% coverage floor",
        )

    # Absolute floor: guards against a run that asked for a LARGE universe but
    # got back only a handful. It is deliberately scoped to large universes, so a
    # small-but-complete run (a --limit test, or a 63-symbol watchlist fetching
    # all 63) is never aborted for the crime of being small — that would make a
    # 100% successful fetch fail, which is both wrong and self-contradictory.
    if requested >= max(min_abs * 2, 200) and fetched < min_abs:
        return (
            True,
            f"Only {fetched}/{requested} symbols fetched ({pct:.1f}%) — below the "
            f"{min_abs}-symbol absolute floor",
        )
    return False, ""


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

    # NOTE ON ORDERING (important):
    # Detection is deliberately run BEFORE the breadth gate decides whether to
    # dispatch alerts. Pausing on breadth used to return early and write
    # `candidates: []`, which (a) threw away that day's watchlist and (b) handed
    # Stage-2 a stale/false "breadth_ok: false" that kept live scanning muted
    # long after the market recovered. Now the watchlist is always refreshed;
    # only the *alert dispatch* is suppressed while the guards are tripped.

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

    # --- Data-integrity circuit breaker -------------------------------------
    # A Yahoo/NSE outage or a throttled run must never overwrite a good
    # watchlist with an empty one. Abort loudly, leaving state untouched.
    abort, abort_reason = check_fetch_integrity(
        fetched=len(daily_map),
        requested=len(symbols),
        cfg=cfg,
        limit=limit,
    )
    if abort:
        publish_abort(abort_reason, {"fetched": len(daily_map), "requested": len(symbols)})
        logger.error("ABORTING scan: %s — state left untouched.", abort_reason)
        return {
            "skipped": True,
            "aborted": True,
            "reason": "data_fetch_failed",
            "fetched": len(daily_map),
            "requested": len(symbols),
        }

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
    # IMPORTANT: Stage-1 detection must be INDEPENDENT of the breadth guard.
    # `detect_daily_pattern` rejects every setup whose passed breadth is under the
    # floor (pattern_detector.py:~462). Feeding it the live value therefore made a
    # weak-breadth day write an EMPTY watchlist — silently wiping the very list
    # this reordering exists to preserve, and leaving Stage-2 with nothing to
    # monitor. Detection gets the configured floor (always satisfied here); the
    # guard is applied further down to decide DISPATCH only.
    detection_breadth_pct = float(r_cfg.get("min_nse_breadth50_pct", 40.0))
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

    # Guards suppress ALERT DISPATCH only — detection below still runs so the
    # watchlist handed to Stage-2 is always current.
    pause_reason: str | None = None
    pause_info: Dict[str, Any] = {}
    if not regime["regime_ok"]:
        pause_reason = "nifty_regime_paused"
        pause_info = {"ret20_pct": regime.get("ret20_pct"), "min_ret20_pct": float(r_cfg.get("min_nifty_20d_return_pct", -3.0))}
        logger.warning(
            "V2 NIFTY Regime Guard TRIPPED: NIFTY 20d return %.2f%% < %.2f%% threshold. Watchlist will still refresh; alerts suppressed.",
            regime["ret20_pct"] if regime["ret20_pct"] is not None else float("nan"),
            float(r_cfg.get("min_nifty_20d_return_pct", -3.0)),
        )
    elif not breadth_info["breadth_ok"]:
        pause_reason = "nse_breadth_paused"
        pause_info = {
            "breadth50_pct": breadth_info.get("breadth50_pct"),
            "min_threshold_pct": breadth_info.get("min_threshold_pct", 40.0),
        }
        logger.warning(
            "V2.1 NSE Breadth50 Guard TRIPPED: %.1f%% (< %.1f%% threshold). Watchlist will still refresh; alerts suppressed.",
            breadth_info["breadth50_pct"] if breadth_info["breadth50_pct"] is not None else float("nan"),
            breadth_info.get("min_threshold_pct", 40.0),
        )

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
            breadth50_pct=detection_breadth_pct,
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
            breadth50_pct=detection_breadth_pct,
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
        "alerts_paused": pause_reason is not None,
        "pause_reason": pause_reason,
        "candidates": candidates_payload,
    }
    state_file.write_text(json.dumps(state_obj, indent=2), encoding="utf-8")
    logger.info("Saved %d qualified V2.1 coiled candidates to %s", len(candidates_payload), state_file)

    # Filter setups that meet EOD Telegram alert threshold (Grade A/A+ >= 66) AND are eligible to alert
    min_alert_score = float(s_cfg["min_eod_alert_score"])
    eligible_items = [m for m in matched if m[0].total_score >= min_alert_score]
    max_charts = int(s_cfg["max_eod_chart_alerts"])

    ledger = ledger_from_config(cfg, state_dir)

    alert_items = []
    suppressed: List[str] = []
    seen_this_run: set[str] = set()
    for item in eligible_items:
        sym_key = item[0].symbol.strip().upper()
        if sym_key in seen_this_run or not ledger.is_eligible(sym_key, "PRE_BREAKOUT", now=now):
            suppressed.append(sym_key)
            continue
        seen_this_run.add(sym_key)
        alert_items.append(item)

    if suppressed:
        logger.info(
            "Dedup suppressed %d symbol(s) (policy=%s): %s",
            len(suppressed),
            ledger.policy_label,
            ", ".join(suppressed),
        )

    # A tripped guard suppresses dispatch but NOT detection (see ordering note above).
    if pause_reason is not None:
        alert_items = []

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

    # Make a paused guard visible instead of indistinguishable from "no setups".
    if pause_reason is not None:
        pause_desc = publish_pause(
            pause_reason,
            pause_info,
            notifier=notifier,
            enabled=bool(t_cfg.get("notify_on_guard_pause", True)),
        )
        write_job_summary(
            "Stage-1 EOD scan — watchlist refreshed, alerts paused",
            f"- {pause_desc}\n"
            f"- **{len(candidates_payload)}** candidate(s) still saved to `state/coiled_candidates.json`\n"
            f"- Stage-2 live monitoring stays current and will resume the moment the guard clears\n",
        )
    else:
        write_job_summary(
            "Stage-1 EOD scan — complete",
            f"- Candidates: **{len(candidates_payload)}**\n"
            f"- New alerts: **{len(newly_alerted)}**\n"
            f"- Breadth50: **{breadth_info['breadth50_pct']:.1f}%**\n"
            if breadth_info["available"]
            else f"- Candidates: **{len(candidates_payload)}**\n",
        )

    elapsed = time.time() - t0
    logger.info(
        "Stage-1 V2.1 EOD Scan complete in %.1fs | Scanned=%d | Breadth50=%s | Watchlist=%d | NewAlerts=%d | SuppressedDup=%d | Charts=%d | Paused=%s",
        elapsed,
        len(daily_map),
        f"{breadth_info['breadth50_pct']:.1f}%" if breadth_info["available"] else "N/A",
        len(candidates_payload),
        len(newly_alerted),
        len(suppressed),
        len(charts_generated),
        pause_reason or "no",
    )
    return {
        "skipped": pause_reason is not None,
        "scan_date": scan_date,
        "dropped_incomplete_live_bar": drop_incomplete_today,
        "universe_scanned": len(daily_map),
        "nifty_regime": regime,
        "nse_breadth50": breadth_info,
        "alerts_paused": pause_reason is not None,
        "pause_reason": pause_reason,
        "watchlist_count": len(candidates_payload),
        "alert_count": len(alert_items),
        "newly_alerted": newly_alerted,
        "suppressed_by_dedup": suppressed,
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
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Write the run summary JSON here (used by CI to detect a data-integrity abort)",
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

    # Persist the run summary so the post-run verifier can distinguish
    # "healthy scan with 0 candidates" from "aborted on a data outage".
    if args.summary_out:
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # A data-integrity abort must turn the job RED. Returning 0 here is what
    # previously let an entire week of runs look successful while producing
    # nothing. A guard pause is legitimate, so it stays green (but summarised).
    if summary.get("aborted"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
