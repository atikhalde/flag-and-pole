"""
Strict Permanent Symbol-Level Alert Deduplication Ledger.

Guarantees that once any stock symbol has been alerted to Telegram (by EITHER Stage-1 EOD
or Stage-2 Live, on ANY scan or ANY date), that symbol will NEVER be alerted twice.
State persists in state/sent_alerts.json and is committed back to GitHub after every scan.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AlertLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.alerted_symbols: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self.alerted_symbols = {}
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to parse alert ledger %s (%s); starting fresh.", self.path, exc)
            self.alerted_symbols = {}
            return

        # New strict permanent schema
        stored = raw.get("alerted_symbols", {})
        if isinstance(stored, dict):
            for sym, meta in stored.items():
                norm = str(sym).strip().upper()
                if norm:
                    self.alerted_symbols[norm] = meta if isinstance(meta, dict) else {"meta": meta}

        # Auto-migrate any legacy "SYMBOL:TRIGGER_TYPE" keys so already-sent symbols never repeat
        legacy = raw.get("alerts", {})
        if isinstance(legacy, dict):
            for key, meta in legacy.items():
                sym = str(key).split(":")[0].strip().upper()
                if sym and sym not in self.alerted_symbols:
                    entry = dict(meta) if isinstance(meta, dict) else {}
                    entry.setdefault("stage", "LEGACY")
                    entry.setdefault("alert_type", str(key).split(":")[1] if ":" in str(key) else "ALERT")
                    self.alerted_symbols[sym] = entry

    def is_sent(self, symbol: str) -> bool:
        return symbol.strip().upper() in self.alerted_symbols

    def mark_sent(
        self,
        symbol: str,
        stage: str,
        alert_type: str,
        price: float,
        timestamp_ist: str,
        extra: Optional[Dict[str, Any]] = None,
        save_immediately: bool = True,
    ) -> None:
        norm = symbol.strip().upper()
        if not norm:
            return
        record: Dict[str, Any] = {
            "stage": stage,
            "alert_type": alert_type,
            "price": round(float(price), 2),
            "timestamp_ist": timestamp_ist,
        }
        if extra:
            record.update(extra)
        self.alerted_symbols[norm] = record
        if save_immediately:
            self.save()

    def merge_from(self, other_path: str | Path) -> int:
        """
        Losslessly union symbols from another alert ledger file into this ledger.
        Preserves existing first-alert records intact and appends any newly alerted symbols.
        Returns the count of newly added symbols.
        """
        other = AlertLedger(other_path)
        added = 0
        for sym, meta in other.alerted_symbols.items():
            if sym not in self.alerted_symbols:
                self.alerted_symbols[sym] = meta
                added += 1
        return added

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "policy": "strict_never_repeat_symbol",
            "total_alerted_symbols": len(self.alerted_symbols),
            "alerted_symbols": dict(sorted(self.alerted_symbols.items())),
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Losslessly union-merge alert deduplication ledgers")
    parser.add_argument("--merge-into", required=True, help="Target ledger path (e.g. state/sent_alerts.json)")
    parser.add_argument("--from", dest="src_path", required=True, help="Incoming ledger path to merge from")
    args = parser.parse_args()

    target = AlertLedger(args.merge_into)
    added_count = target.merge_from(args.src_path)
    if added_count > 0 or not Path(args.merge_into).exists():
        target.save()
    print(f"AlertLedger union-merge complete: added={added_count}, total={len(target.alerted_symbols)}")

