"""Configuration loader for settings.yaml and environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "settings.yaml"


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # Resolve relative paths against repository root
    cfg["_repo_root"] = str(REPO_ROOT)
    cfg["universe"]["fallback_csv_abs"] = str(REPO_ROOT / cfg["universe"]["fallback_csv_path"])
    cfg["universe"]["custom_watchlist_abs"] = str(REPO_ROOT / cfg["universe"]["custom_watchlist_path"])

    # Runtime secrets from environment (supports primary Bot #1 + optional Bot #2)
    bot1_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    bot1_chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    bot2_token = os.environ.get("TELEGRAM_BOT_TOKEN_2", "").strip()
    bot2_chat = os.environ.get("TELEGRAM_CHAT_ID_2", "").strip()

    extra_targets = []
    if bot2_chat:
        # If TELEGRAM_BOT_TOKEN_2 is omitted, cleanly fall back to primary bot token for 2nd chat
        eff_token2 = bot2_token or bot1_token
        if eff_token2:
            extra_targets.append((eff_token2, bot2_chat))

    cfg["runtime"] = {
        "telegram_bot_token": bot1_token,
        "telegram_chat_id": bot1_chat,
        "telegram_bot_token_2": bot2_token,
        "telegram_chat_id_2": bot2_chat,
        "telegram_extra_targets": extra_targets,
        "state_dir": str(REPO_ROOT / "state"),
        "charts_dir": str(REPO_ROOT / "output" / "charts"),
    }
    return cfg
