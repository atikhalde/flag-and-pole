"""Rich HTML Telegram Bot Notifier with Annotated Chart Images & Retry Handling."""

from __future__ import annotations

import html
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        enabled: bool = True,
        tv_url_template: str = "https://www.tradingview.com/chart/?symbol=NSE%3A{symbol}",
    ) -> None:
        self.bot_token = bot_token.strip()
        self.chat_id = chat_id.strip()
        self.enabled = enabled and bool(self.bot_token) and bool(self.chat_id)
        self.tv_url_template = tv_url_template
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

        if enabled and not self.enabled:
            logger.warning(
                "Telegram credentials (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID) not set — running in DRY-RUN mode."
            )

    def _tv_link(self, symbol: str) -> str:
        return self.tv_url_template.format(symbol=symbol)

    def send_message(self, text_html: str, disable_preview: bool = True) -> bool:
        if not self.enabled:
            logger.info("[TELEGRAM DRY-RUN MESSAGE]\n%s", text_html)
            return False

        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text_html[:4096],
            "parse_mode": "HTML",
            "disable_web_page_preview": disable_preview,
        }
        return self._post_with_retry(url, data=payload)

    def send_photo(self, image_path: str | Path, caption_html: str) -> bool:
        if not self.enabled:
            logger.info("[TELEGRAM DRY-RUN PHOTO: %s]\n%s", image_path, caption_html)
            return False

        p = Path(image_path)
        if not p.exists():
            logger.warning("Chart image missing (%s); falling back to text message.", p)
            return self.send_message(caption_html)

        url = f"{self.base_url}/sendPhoto"
        # Telegram sendPhoto caption limit is 1024 characters
        safe_caption = caption_html[:1020]
        for attempt in range(3):
            try:
                with open(p, "rb") as img_fh:
                    resp = requests.post(
                        url,
                        data={
                            "chat_id": self.chat_id,
                            "caption": safe_caption,
                            "parse_mode": "HTML",
                        },
                        files={"photo": img_fh},
                        timeout=30,
                    )
                if resp.status_code == 200:
                    time.sleep(0.4)  # Respect Telegram rate limits
                    return True
                if resp.status_code == 429:
                    retry_after = int(resp.json().get("parameters", {}).get("retry_after", 3))
                    time.sleep(retry_after + 1)
                    continue
                logger.warning("Telegram sendPhoto HTTP %d: %s", resp.status_code, resp.text[:300])
            except Exception as exc:
                logger.warning("Telegram sendPhoto attempt %d failed: %s", attempt + 1, exc)
            time.sleep(1.5 * (attempt + 1))
        return False

    def _post_with_retry(self, url: str, data: Dict[str, Any]) -> bool:
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
                logger.warning("Telegram API HTTP %d: %s", resp.status_code, resp.text[:300])
            except Exception as exc:
                logger.warning("Telegram API attempt %d failed: %s", attempt + 1, exc)
            time.sleep(1.5 * (attempt + 1))
        return False

    # ------------------------------------------------------------------
    # Message Formatters
    # ------------------------------------------------------------------
    def format_eod_digest(
        self,
        scan_date: str,
        universe_count: int,
        candidates: List[Dict[str, Any]],
    ) -> str:
        header = (
            f"🎯 <b>NSE DAILY COILED PRE-BREAKOUT SCAN (≤20d Pole-Flag)</b>\n"
            f"📅 <b>Date:</b> {html.escape(scan_date)} | 🔎 <b>Scanned:</b> {universe_count} NSE EQ stocks\n"
            f"📦 <b>Qualified Watchlist:</b> {len(candidates)} coiled bases saved for Live Stage-2\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
        )
        if not candidates:
            return header + "<i>No high-conviction ≤20d coiled flags met all criteria today.</i>"

        lines = [header]
        for idx, c in enumerate(candidates[:15], start=1):
            sym = html.escape(c["symbol"])
            tv = self._tv_link(c["symbol"])
            badge = "🔥" if c["grade"] == "A+" else ("⚡" if c["grade"] == "A" else "👀")
            lines.append(
                f"{idx}. {badge} <b><a href='{tv}'>{sym}</a></b> "
                f"[<b>{c['grade']} • {c['total_score']:.0f}</b>] — ₹{c['last_close']:.2f}\n"
                f"   • Day <b>{c['total_pattern_days']}/20</b> | Pole <b>+{c['pole_high_gain_pct']:.0f}%</b> ({c['pole_rvol']:.0f}x Vol)\n"
                f"   • Shelf <b>{c['shelf_close_spread_pct']:.1f}%</b> tight | Dry-Up <b>{c['dryup_pct']:.0f}%</b> | BO &gt; <b>₹{c['entry_breakout']:.2f}</b>"
            )
        lines.append("\n<i>Top setups with annotated Daily Charts + Volume Profile follow below 👇</i>")
        return "\n".join(lines)

    def format_eod_setup_caption(self, c: Dict[str, Any]) -> str:
        sym = html.escape(c["symbol"])
        name = html.escape(str(c.get("company_name", sym))[:36])
        tv = self._tv_link(c["symbol"])
        badge = "🔥" if c["grade"] == "A+" else "⚡"
        vah_str = f"₹{c['vah']:.1f}" if c.get("vah") else "N/A"
        poc_str = f"₹{c['poc']:.1f}" if c.get("poc") else "N/A"
        spring_tag = " | 🟢 Spring Bar" if c.get("has_spring_bar") else ""

        return (
            f"{badge} <b>PRE-BREAKOUT: <a href='{tv}'>{sym}</a></b> ({name})\n"
            f"🏆 <b>Grade {c['grade']} ({c['total_score']:.1f}/100)</b> — {html.escape(c['pattern_label'])}{spring_tag}\n"
            f"⏱ <b>Window:</b> Day <b>{c['total_pattern_days']}/20</b> ({c['pole_days']}d Pole + {c['base_days']}d Shelf)\n\n"
            f"📊 <b>Pattern DNA:</b>\n"
            f"• Pole: <b>+{c['pole_high_gain_pct']:.1f}%</b> ({c['pole_start_date']}→{c['pole_end_date']}) on <b>{c['pole_rvol']:.1f}x</b> RVOL (₹{c['pole_value_cr']:.1f}Cr)\n"
            f"• Shelf: ₹{c['shelf_low']:.2f}–₹{c['shelf_high']:.2f} (Close box <b>{c['shelf_close_spread_pct']:.1f}%</b>, upper {c['upper_pole_hold_pct']:.0f}%)\n"
            f"• Coil: 3d range <b>{c['recent_3d_range_pct']:.1f}%</b> | Dry-Up <b>{c['dryup_pct']:.1f}%</b> (Base/Pole {c['base_to_pole_vol_pct']:.1f}%)\n"
            f"• Daily VP: VAH <b>{vah_str}</b> | POC <b>{poc_str}</b>\n\n"
            f"🎯 <b>Trade Plan (Daily):</b>\n"
            f"• Close: <b>₹{c['last_close']:.2f}</b> | BO Trigger: <b>₹{c['entry_breakout']:.2f}</b>\n"
            f"• Invalidation SL: <b>₹{c['invalidation_stop']:.2f}</b> (-{c['risk_pct_anticipatory']:.1f}%)\n"
            f"• T1 (BOS): <b>₹{c['target_1_bos']:.2f}</b> ({c['rr_to_t1']:.1f}R) | T2: <b>₹{c['target_2_ext']:.2f}</b> ({c['rr_to_t2']:.1f}R)"
        )

    def format_live_trigger_caption(self, ev: Dict[str, Any]) -> str:
        sym = html.escape(ev["symbol"])
        name = html.escape(str(ev.get("company_name", sym))[:34])
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

        return (
            f"{icon} <b>LIVE {label}: <a href='{tv}'>{sym}</a></b> ({name})\n"
            f"🕒 {html.escape(ev['timestamp_ist'])} | Day <b>{ev['day_of_pattern']}/20</b> | Grade <b>{ev['setup_grade']} ({ev['setup_score']:.0f})</b>\n\n"
            f"💹 <b>Live Daily Candle:</b>\n"
            f"• LTP: <b>₹{ev['live_price']:.2f}</b> ({chg_sign}{ev['day_chg_pct']:.2f}%) | Gap: {ev['gap_pct']:+.2f}%\n"
            f"• O: ₹{ev['day_open']:.2f} | H: ₹{ev['day_high']:.2f} | L: ₹{ev['day_low']:.2f} | CLV: <b>{ev['daily_clv']:.0%}</b>\n"
            f"• Projected RVOL vs Base: <b>{ev['projected_rvol_vs_base']:.1f}x</b>\n\n"
            f"🧠 <i>{html.escape(ev['reason'])}</i>\n\n"
            f"🎯 <b>Key Levels:</b> Shelf ₹{ev['shelf_low']:.2f}–₹{ev['shelf_high']:.2f} | "
            f"SL <b>₹{ev['invalidation_stop']:.2f}</b> | T1 <b>₹{ev['target_1_bos']:.2f}</b> | T2 <b>₹{ev['target_2_ext']:.2f}</b>"
        )
