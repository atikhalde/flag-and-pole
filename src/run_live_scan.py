"""
Stage-2 Live Market Scanner (Pure Daily Forming Candle Triggers — V2 Option A).
Runs every 15 minutes during NSE trading hours (09:15 - 15:30 IST) via GitHub Actions.
Reads state/coiled_candidates.json (+ config/custom_watchlist.txt), verifies NIFTY 50 Regime
Guard, fetches today's live forming Daily candle via yfinance, evaluates Breakout / Gap-and-Go
/ Spring / Coil triggers, strictly deduplicates alerts via state/sent_alerts.json, and sends
annotated Telegram chart alerts.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from src.alert_ledger import AlertLedger
from src.chart_generator import generate_pattern_chart
from src.config_loader import load_config
from src.data_fetcher import fetch_daily_batch, fetch_nifty_benchmark
from src.live_triggers import evaluate_live_daily_triggers
from src.market_hours import is_nse_market_open, now_ist
from src.pattern_detector import detect_daily_pattern, evaluate_nifty_regime
from src.telegram_notifier import TelegramNotifier
from src.universe import load_custom_watchlist

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("live_scanner")


def _load_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def run_live_scan(
    cfg: Dict[str, Any],
    force: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    now = now_ist()
    if not force and not is_nse_market_open(now):
        logger.info("NSE market is currently closed (%s). Use --force to override.", now.strftime("%Y-%m-%d %H:%M IST"))
        return {"skipped": True, "reason": "market_closed"}

    t0 = time.time()
    r_cfg = cfg.get("regime", {})
    nifty_df = fetch_nifty_benchmark(
        ticker=str(r_cfg.get("benchmark_ticker", "^NSEI")),
        period=cfg["fetcher"].get("daily_period", "15mo"),
    )
    regime = evaluate_nifty_regime(nifty_df, cfg=cfg)
    if not regime["regime_ok"]:
        logger.warning(
            "V2 NIFTY Regime Guard PAUSED live triggers: NIFTY 20d return %.2f%% < %.2f%%.",
            regime["ret20_pct"],
            float(r_cfg.get("min_nifty_20d_return_pct", -3.0)),
        )
        return {"skipped": True, "reason": "nifty_regime_paused", "nifty_regime": regime}

    state_dir = Path(cfg["runtime"]["state_dir"])
    state_dir.mkdir(parents=True, exist_ok=True)

    candidates_file = state_dir / "coiled_candidates.json"
    alerts_file = state_dir / "sent_alerts.json"

    wl_state = _load_json(candidates_file, {"candidates": []})
    saved_breadth = wl_state.get("nse_breadth50") or {}
    if (
        bool(r_cfg.get("enable_breadth_guard", True))
        and saved_breadth.get("enforced")
        and saved_breadth.get("breadth_ok") is False
    ):
        logger.warning(
            "V2.1 NSE Breadth50 Guard PAUSED live triggers: EOD Breadth50=%.1f%% (< %.1f%%).",
            saved_breadth.get("breadth50_pct", 0.0),
            float(r_cfg.get("min_nse_breadth50_pct", 40.0)),
        )
        return {"skipped": True, "reason": "nse_breadth_paused", "nifty_regime": regime, "nse_breadth50": saved_breadth}

    saved_br_pct = saved_breadth.get("breadth50_pct") if saved_breadth.get("enforced") else None
    raw_candidates: List[Dict[str, Any]] = list(wl_state.get("candidates", []))
    # Skip any stale pre-V2.1 candidates missing Option A / V2.1 schema fields
    candidates: List[Dict[str, Any]] = [
        c for c in raw_candidates if isinstance(c, dict) and c.get("sl_mode") and "above_150sma" in c
    ]
    if len(candidates) < len(raw_candidates):
        logger.info("Filtered out %d legacy pre-V2.1 candidate(s) from coiled_candidates.json.", len(raw_candidates) - len(candidates))
    cand_map: Dict[str, Dict[str, Any]] = {c["symbol"]: c for c in candidates}

    # Merge custom watchlist symbols so they are always monitored live
    custom_syms = load_custom_watchlist(cfg["universe"]["custom_watchlist_abs"])
    all_syms = list(dict.fromkeys(list(cand_map.keys()) + custom_syms))

    if not all_syms:
        logger.info("No candidates in coiled_candidates.json or custom_watchlist.txt. Run Stage-1 EOD first.")
        return {"skipped": False, "monitored": 0, "triggers_fired": 0}

    logger.info("Stage-2 V2.1 Live Daily Scan checking %d symbols (%d coiled + %d custom)...", len(all_syms), len(cand_map), len(custom_syms))

    # Fetch 15mo Daily OHLCV with live-day NaN repair (fast because <= 40 symbols!)
    daily_map = fetch_daily_batch(
        symbols=all_syms,
        period=cfg["fetcher"].get("daily_period", "15mo"),
        batch_size=40,
        batch_sleep_sec=0.5,
        max_retries=2,
        repair_nan_today=bool(cfg["fetcher"].get("repair_nan_today_with_15m", True)),
    )

    # For custom watchlist symbols not already in coiled_candidates.json, evaluate prior-day V2.1 daily pattern
    min_wl_score = float(cfg["scoring"]["min_watchlist_score"])
    for sym in custom_syms:
        if sym not in cand_map and sym in daily_map and len(daily_map[sym]) >= 50:
            df_prior = daily_map[sym].iloc[:-1]
            setup, _ = detect_daily_pattern(sym, sym, df_prior, cfg, nifty_df=nifty_df, breadth50_pct=saved_br_pct)
            if setup is not None and setup.total_score >= min_wl_score:
                cand_map[sym] = setup.to_dict()

    # Load strict permanent symbol-level deduplication ledger (NEVER resets across scans or days)
    strict_dedup = bool(cfg.get("deduplication", {}).get("strict_never_repeat_symbol", True))
    ledger = AlertLedger(alerts_file)

    notifier = TelegramNotifier(
        bot_token=cfg["runtime"]["telegram_bot_token"],
        chat_id=cfg["runtime"]["telegram_chat_id"],
        enabled=bool(cfg["telegram"]["enabled"]) and not dry_run,
        tv_url_template=cfg["telegram"]["tradingview_url_template"],
        extra_targets=cfg["runtime"].get("telegram_extra_targets", []),
    )

    charts_dir = Path(cfg["runtime"]["charts_dir"])
    charts_dir.mkdir(parents=True, exist_ok=True)
    fired: List[Dict[str, Any]] = []
    skipped_already_sent: List[str] = []
    seen_this_run: set[str] = set()

    for sym, cand in cand_map.items():
        sym_key = sym.strip().upper()
        if sym_key in seen_this_run or (strict_dedup and ledger.is_sent(sym_key)):
            skipped_already_sent.append(sym_key)
            continue

        df_d = daily_map.get(sym)
        if df_d is None or len(df_d) < 5:
            continue

        events = evaluate_live_daily_triggers(cand, df_d, cfg, as_of_dt=now)
        if not events:
            continue

        ev = events[0]
        seen_this_run.add(sym_key)

        ev_dict = ev.to_dict()
        chart_path = charts_dir / f"LIVE_{sym}_{ev.trigger_type}_{ev.session_date}.png"
        caption = notifier.format_live_trigger_caption(ev_dict)

        if bool(cfg["telegram"].get("send_charts", True)):
            generate_pattern_chart(
                symbol=sym,
                df_daily=df_d,
                setup=cand,
                output_path=chart_path,
                live_trigger=ev_dict,
            )
            delivered = notifier.send_photo(chart_path, caption)
        else:
            delivered = notifier.send_message(caption)

        if not dry_run and delivered:
            ledger.mark_sent(
                symbol=sym_key,
                stage="LIVE",
                alert_type=ev.trigger_type,
                price=ev.live_price,
                timestamp_ist=ev.timestamp_ist,
                extra={"rvol": ev.projected_rvol_vs_base, "session_date": ev.session_date},
            )
        fired.append(ev_dict)

    if not dry_run:
        ledger.save()

    logger.info(
        "Stage-2 V2.1 Live Scan done in %.1fs | Monitored=%d | New Triggers=%d | SkippedDup=%d",
        time.time() - t0,
        len(cand_map),
        len(fired),
        len(skipped_already_sent),
    )
    return {
        "skipped": False,
        "timestamp_ist": now.strftime("%Y-%m-%d %H:%M IST"),
        "nifty_regime": regime,
        "monitored": len(cand_map),
        "triggers_fired": len(fired),
        "skipped_already_sent": skipped_already_sent,
        "events": [{"symbol": e["symbol"], "trigger": e["trigger_type"], "price": e["live_price"]} for e in fired],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-2 Live NSE Daily-Candle Trigger Scanner (V2)")
    parser.add_argument("--config", default=None, help="Path to settings.yaml")
    parser.add_argument("--force", action="store_true", help="Run even outside NSE market hours")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate triggers without sending Telegram messages")
    args = parser.parse_args()

    cfg = load_config(args.config)
    summary = run_live_scan(cfg=cfg, force=args.force, dry_run=args.dry_run)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
