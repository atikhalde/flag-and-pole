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

    # Runtime secrets from environment
    cfg["runtime"] = {
        "telegram_bot_token": os.environ.get("TELEGRAM_BOT_TOKEN", "").strip(),
        "telegram_chat_id": os.environ.get("TELEGRAM_CHAT_ID", "").strip(),
        "state_dir": str(REPO_ROOT / "state"),
        "charts_dir": str(REPO_ROOT / "output" / "charts"),
    }
    return cfg
