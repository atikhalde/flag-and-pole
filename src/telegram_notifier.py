"""Rich HTML Telegram Bot Notifier with Annotated Chart Images & Retry Handling (V2 Option A)."""

from __future__ import annotations

import html
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        enabled: bool = True,
        tv_url_template: str = "https://www.tradingview.com/chart/?symbol=NSE%3A{symbol}",
        extra_targets: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        self.bot_token = bot_token.strip()
        self.chat_id = chat_id.strip()
        self.tv_url_template = tv_url_template

        # Build deduplicated list of (bot_token, chat_id) destinations (Bot #1, Bot #2, ...)
        targets: List[Tuple[str, str]] = []
        if self.bot_token and self.chat_id:
            targets.append((self.bot_token, self.chat_id))
        for tok, cid in (extra_targets or []):
            tok_c = str(tok).strip()
            cid_c = str(cid).strip()
            if tok_c and cid_c and (tok_c, cid_c) not in targets:
                targets.append((tok_c, cid_c))

        self.targets: List[Tuple[str, str]] = targets
        self.enabled = enabled and bool(self.targets)
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}" if self.bot_token else ""

        if enabled and not self.enabled:
            logger.warning(
                "Telegram credentials (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID) not set — running in DRY-RUN mode."
            )
        elif self.enabled:
            logger.info("TelegramNotifier active with %d destination bot/chat target(s).", len(self.targets))

    def _tv_link(self, symbol: str) -> str:
        return self.tv_url_template.format(symbol=symbol)

    # Status/operational phrases that must NEVER reach the Telegram channel.
    # Telegram is reserved strictly for actionable STOCK alerts; operational
    # notices (guard pauses, aborts, "no alerts today") belong in the Actions
    # job summary and the logs only.
    _BLOCKED_MARKERS = (
        "scanner paused",
        "scanner aborted",
        "no alerts were evaluated",
        "scan paused",
    )

    def _is_stock_alert(self, text_html: str) -> bool:
        """False when the text is an operational notice rather than a stock alert."""
        low = str(text_html or "").lower()
        return not any(marker in low for marker in self._BLOCKED_MARKERS)

    def send_message(self, text_html: str, disable_preview: bool = True) -> bool:
        if not self._is_stock_alert(text_html):
            logger.warning(
                "Blocked a non-stock (operational) Telegram message; "
                "status notices belong in the job summary/logs only."
            )
            return False
        if not self.enabled:
            logger.info("[TELEGRAM DRY-RUN MESSAGE]\n%s", text_html)
            return False

        any_ok = False
        for idx, (tok, cid) in enumerate(self.targets, start=1):
            url = f"https://api.telegram.org/bot{tok}/sendMessage"
            payload = {
                "chat_id": cid,
                "text": text_html[:4096],
                "parse_mode": "HTML",
                "disable_web_page_preview": disable_preview,
            }
            ok = self._post_with_retry(url, data=payload, label=f"Bot #{idx}")
            any_ok = any_ok or ok
        return any_ok

    def send_photo(self, image_path: str | Path, caption_html: str) -> bool:
        if not self._is_stock_alert(caption_html):
            logger.warning("Blocked a non-stock (operational) Telegram photo caption.")
            return False
        if not self.enabled:
            logger.info("[TELEGRAM DRY-RUN PHOTO: %s]\n%s", image_path, caption_html)
            return False

        p = Path(image_path)
        if not p.exists():
            logger.warning("Chart image missing (%s); falling back to text message.", p)
            return self.send_message(caption_html)

        # Telegram sendPhoto caption limit is 1024 characters
        safe_caption = caption_html[:1020]
        any_ok = False
        for idx, (tok, cid) in enumerate(self.targets, start=1):
            url = f"https://api.telegram.org/bot{tok}/sendPhoto"
            sent = False
            for attempt in range(3):
                try:
                    with open(p, "rb") as img_fh:
                        resp = requests.post(
                            url,
                            data={
                                "chat_id": cid,
                                "caption": safe_caption,
                                "parse_mode": "HTML",
                            },
                            files={"photo": img_fh},
                            timeout=30,
                        )
                    if resp.status_code == 200:
                        time.sleep(0.4)
                        sent = True
                        break
                    if resp.status_code == 429:
                        retry_after = int(resp.json().get("parameters", {}).get("retry_after", 3))
                        time.sleep(retry_after + 1)
                        continue
                    logger.warning("Telegram [Bot #%d] sendPhoto HTTP %d: %s", idx, resp.status_code, resp.text[:300])
                except Exception as exc:
                    logger.warning("Telegram [Bot #%d] sendPhoto attempt %d failed: %s", idx, attempt + 1, exc)
                time.sleep(1.5 * (attempt + 1))
            any_ok = any_ok or sent
        return any_ok

    def send_document(self, file_path: str | Path, caption_html: str = "") -> bool:
        """Send a PDF/CSV report document to all configured Telegram bots."""
        if not self.enabled:
            logger.info("[TELEGRAM DRY-RUN DOCUMENT: %s]\n%s", file_path, caption_html)
            return False

        p = Path(file_path)
        if not p.exists():
            logger.warning("Document missing (%s); cannot send to Telegram.", p)
            return False

        safe_caption = caption_html[:1020]
        any_ok = False
        for idx, (tok, cid) in enumerate(self.targets, start=1):
            url = f"https://api.telegram.org/bot{tok}/sendDocument"
            sent = False
            for attempt in range(3):
                try:
                    with open(p, "rb") as fh:
                        resp = requests.post(
                            url,
                            data={
                                "chat_id": cid,
                                "caption": safe_caption,
                                "parse_mode": "HTML",
                            },
                            files={"document": (p.name, fh)},
                            timeout=60,
                        )
                    if resp.status_code == 200:
                        time.sleep(0.5)
                        sent = True
                        break
                    if resp.status_code == 429:
                        retry_after = int(resp.json().get("parameters", {}).get("retry_after", 3))
                        time.sleep(retry_after + 1)
                        continue
                    logger.warning("Telegram [Bot #%d] sendDocument HTTP %d: %s", idx, resp.status_code, resp.text[:300])
                except Exception as exc:
                    logger.warning("Telegram [Bot #%d] sendDocument attempt %d failed: %s", idx, attempt + 1, exc)
                time.sleep(1.5 * (attempt + 1))
            any_ok = any_ok or sent
        return any_ok

    def _post_with_retry(self, url: str, data: Dict[str, Any], label: str = "Bot #1") -> bool:
        for attempt in range(3):
            try:
                resp = requests.post(url, data=data, timeout=20)
                if resp.status_code == 200:
                    time.sleep(0.3)
                    return True
                if resp.status_code == 429:
                    retry_after = int(resp.json().get("parameters", {}).get("retry_after", 3))
                    time.sleep(retry_after + 1)
                    continue
                logger.warning("Telegram [%s] API HTTP %d: %s", label, resp.status_code, resp.text[:300])
            except Exception as exc:
                logger.warning("Telegram [%s] API attempt %d failed: %s", label, attempt + 1, exc)
            time.sleep(1.5 * (attempt + 1))
        return False

    # ------------------------------------------------------------------
    # V2 Message Formatters (Option A: Daily-Close SL + Hard -9% GTT)
    # ------------------------------------------------------------------
    def format_eod_digest(
        self,
        scan_date: str,
        universe_count: int,
        candidates: List[Dict[str, Any]],
        nifty_regime: Optional[Dict[str, Any]] = None,
    ) -> str:
        regime_line = ""
        if nifty_regime and nifty_regime.get("available"):
            n_ret = nifty_regime.get("ret20_pct", 0.0)
            n_50 = "Above 50 SMA" if nifty_regime.get("above_50sma") else "Below 50 SMA"
            br_pct = nifty_regime.get("nse_breadth50_pct")
            br_seg = f" | NSE Breadth50: <b>{br_pct:.1f}%</b>" if br_pct is not None else ""
            regime_line = f"🟢 <b>Market Regime:</b> NIFTY 20d {n_ret:+.1f}% ({n_50}){br_seg} — OPEN\n"

        header = (
            f"🎯 <b>NSE V2.1 CORE+COIL FLAG-AND-POLE SCAN (≤20d | Grade A/A+)</b>\n"
            f"📅 <b>Date:</b> {html.escape(scan_date)} | 🔎 <b>Scanned:</b> {universe_count} NSE EQ stocks\n"
            f"{regime_line}"
            f"📦 <b>New High-Conviction Setups:</b> {len(candidates)} (Daily-Close SL + Hard GTT)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
        )
        if not candidates:
            return header + "<i>No new Grade A/A+ ≤20d coiled flags met all V2.1 Core+Coil filters today.</i>"

        lines = [header]
        for idx, c in enumerate(candidates[:15], start=1):
            sym = html.escape(c["symbol"])
            tv = self._tv_link(c["symbol"])
            badge = "🔥" if c["grade"] == "A+" else "⚡"
            entry_p = c.get("entry_price", c["last_close"])
            bo_lvl = c.get("breakout_level", c["shelf_high"])
            gtt = c.get("hard_stop_gtt", c["invalidation_stop"])
            lines.append(
                f"{idx}. {badge} <b><a href='{tv}'>{sym}</a></b> "
                f"[<b>{c['grade']} • {c['total_score']:.0f}</b>] — CMP ₹{c['last_close']:.2f} (Day {c['total_pattern_days']}/20)\n"
                f"   • Pole <b>+{c['pole_high_gain_pct']:.0f}%</b> ({c['pole_rvol']:.0f}x) | Box <b>{c['shelf_wick_box_pct']:.1f}%</b> | Dry-Up <b>{c['dryup_pct']:.0f}%</b>\n"
                f"   • 🎯 Entry <b>₹{entry_p:.2f}</b> <i>(&lt;BO ₹{bo_lvl:.2f})</i> | Close SL <b>₹{c['invalidation_stop']:.2f}</b> (GTT ₹{gtt:.2f})\n"
                f"   • 💰 TP1 <b>₹{c['tp1']:.2f}</b> (+4.5%) | TP2 <b>₹{c['tp2']:.2f}</b> (+10%) | TP3 <b>₹{c['tp3']:.2f}</b> (+18%)"
            )
        lines.append("\n<i>Annotated Daily Charts + Trade Plans follow below 👇</i>")
        return "\n".join(lines)

    def format_eod_setup_caption(self, c: Dict[str, Any]) -> str:
        sym = html.escape(c["symbol"])
        name = html.escape(str(c.get("company_name", sym))[:30])
        tv = self._tv_link(c["symbol"])
        badge = "🔥" if c["grade"] == "A+" else "⚡"
        spring_tag = " | 🟢 Spring" if c.get("has_spring_bar") else ""
        s2_tag = " | Stage-2" if c.get("stage2_uptrend") else ""
        entry_p = c.get("entry_price", c["last_close"])
        ez_lo = c.get("entry_zone_low", c["shelf_low"])
        ez_hi = c.get("entry_zone_high", entry_p)
        bo_lvl = c.get("breakout_level", c["shelf_high"])
        risk_pct = c.get("risk_pct", 0.0)
        sl_mode = c.get("sl_mode", "daily_close")
        hard_gtt = c.get("hard_stop_gtt", c["invalidation_stop"])
        hard_gtt_pct = c.get("hard_stop_gtt_pct", 9.0)
        intra_ref = c.get("intraday_gtt_sl", c["invalidation_stop"])

        if sl_mode == "daily_close":
            sl_lines = (
                f"• <b>Close SL (3:20 PM):</b> ₹{c['invalidation_stop']:.2f} <i>(-{risk_pct:.1f}% | exit only if Daily Close breaks)</i>\n"
                f"• <b>Emergency GTT:</b> ₹{hard_gtt:.2f} <i>(-{hard_gtt_pct:.1f}% broker safety net | Hands-off GTT: ₹{intra_ref:.2f})</i>\n"
            )
        else:
            sl_lines = f"• <b>Stop Loss (GTT):</b> ₹{c['invalidation_stop']:.2f} <i>(-{risk_pct:.1f}%)</i>\n"

        return (
            f"{badge} <b>V2.1 PRE-BREAKOUT: <a href='{tv}'>{sym}</a></b> ({name})\n"
            f"🏆 <b>Grade {c['grade']} ({c['total_score']:.0f}/100)</b> — {html.escape(c['pattern_label'])}{spring_tag}{s2_tag}\n"
            f"⏱ <b>Day {c['total_pattern_days']}/20</b> ({c['pole_days']}d Pole +{c['pole_high_gain_pct']:.0f}% on {c['pole_rvol']:.0f}x Vol + {c['base_days']}d Flag)\n"
            f"📊 Flag ₹{c['shelf_low']:.2f}–₹{c['shelf_high']:.2f} (Dry-Up <b>{c['dryup_pct']:.0f}%</b> | 52wH {c.get('pct_from_52w_high', 0):+.0f}% | 3m RS {c.get('rs63_vs_nifty', 0):+.0f}%)\n\n"
            f"🎯 <b>V2.1 TRADE PLAN (Entry Below Breakout):</b>\n"
            f"• <b>Entry:</b> ₹{entry_p:.2f} <i>(Zone ₹{ez_lo:.2f}–₹{ez_hi:.2f} | BO ₹{bo_lvl:.2f})</i>\n"
            f"{sl_lines}"
            f"• <b>TP1 (+{c['tp1_pct']:.1f}%):</b> ₹{c['tp1']:.2f} <i>(Book 45% → Move SL to Breakeven)</i>\n"
            f"• <b>TP2 (+{c['tp2_pct']:.1f}%):</b> ₹{c['tp2']:.2f} <i>(Book 30% → Trail SL to TP1)</i>\n"
            f"• <b>TP3 (+{c['tp3_pct']:.1f}%):</b> ₹{c['tp3']:.2f} <i>(Ride final 25% runner)</i>"
        )

    def format_live_trigger_caption(self, ev: Dict[str, Any]) -> str:
        sym = html.escape(ev["symbol"])
        name = html.escape(str(ev.get("company_name", sym))[:28])
        tv = self._tv_link(ev["symbol"])

        icons = {
            "GAP_AND_GO": "🚀",
            "BREAKOUT_TRIGGER": "💥",
            "SPRING_RETEST": "🔥",
            "TIGHT_COIL_WATCH": "⚡",
        }
        icon = icons.get(ev["trigger_type"], "🔔")
        label = ev["trigger_type"].replace("_", " ")
        chg_sign = "+" if ev["day_chg_pct"] >= 0 else ""
        bo_lvl = ev.get("breakout_level", ev["shelf_high"])
        ez_lo = ev.get("entry_zone_low", ev["shelf_low"])
        ez_hi = ev.get("entry_zone_high", ev.get("entry_price", ev["live_price"]))
        sl_mode = ev.get("sl_mode", "daily_close")
        hard_gtt = ev.get("hard_stop_gtt", ev["invalidation_stop"])
        hard_gtt_pct = ev.get("hard_stop_gtt_pct", 9.0)
        intra_ref = ev.get("intraday_gtt_sl", ev["invalidation_stop"])

        def _fmt_tp(label_tp: str, price: float, pct: float, action: str) -> str:
            if ev["live_price"] >= price:
                return f"• <b>{label_tp}:</b> ₹{price:.2f} <i>(✅ Already Hit)</i>"
            return f"• <b>{label_tp} ({pct:+.1f}%):</b> ₹{price:.2f} <i>({action})</i>"

        entry_line = (
            f"• <b>Entry (Below BO):</b> ₹{ev['live_price']:.2f} <i>(Zone ₹{ez_lo:.2f}–₹{ez_hi:.2f} | BO ₹{bo_lvl:.2f})</i>"
            if ev["live_price"] < bo_lvl
            else f"• <b>Entry / LTP:</b> ₹{ev['live_price']:.2f} <i>(Pre-BO Zone ₹{ez_lo:.2f}–₹{ez_hi:.2f} | BO ₹{bo_lvl:.2f})</i>"
        )
        if sl_mode == "daily_close":
            sl_lines = (
                f"• <b>Close SL (3:20 PM):</b> ₹{ev['invalidation_stop']:.2f} <i>(-{ev['live_risk_pct']:.1f}% on Daily Close)</i>\n"
                f"• <b>Emergency GTT:</b> ₹{hard_gtt:.2f} <i>(-{hard_gtt_pct:.1f}% safety | Hands-off: ₹{intra_ref:.2f})</i>\n"
            )
        else:
            sl_lines = f"• <b>Stop Loss (GTT):</b> ₹{ev['invalidation_stop']:.2f} <i>(-{ev['live_risk_pct']:.1f}%)</i>\n"

        return (
            f"{icon} <b>LIVE {label}: <a href='{tv}'>{sym}</a></b> ({name})\n"
            f"🕒 {html.escape(ev['timestamp_ist'])} | Day <b>{ev['day_of_pattern']}/20</b> | Grade <b>{ev['setup_grade']} ({ev['setup_score']:.0f})</b>\n"
            f"💹 LTP <b>₹{ev['live_price']:.2f}</b> ({chg_sign}{ev['day_chg_pct']:.2f}%) | RVOL <b>{ev['projected_rvol_vs_base']:.1f}x</b> | CLV <b>{ev['daily_clv']:.0%}</b>\n\n"
            f"🎯 <b>V2.1 TRADE PLAN:</b>\n"
            f"{entry_line}\n"
            f"{sl_lines}"
            f"{_fmt_tp('TP1', ev['tp1'], ev['tp1_pct'], 'Book 45% → BE')}\n"
            f"{_fmt_tp('TP2', ev['tp2'], ev['tp2_pct'], 'Book 30% → TP1')}\n"
            f"{_fmt_tp('TP3', ev['tp3'], ev['tp3_pct'], 'Ride 25%')}\n\n"
            f"🧠 <i>{html.escape(ev['reason'])}</i>"
        )
