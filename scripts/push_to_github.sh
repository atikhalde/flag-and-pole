#!/usr/bin/env bash
# Safe helper script to initialize & push this scanner repository to GitHub.
# Usage:
#   GITHUB_PAT="ghp_xxx" ./scripts/push_to_github.sh https://github.com/<user>/<repo>.git [main]

set -euo pipefail

REPO_URL="${1:-}"
BRANCH="${2:-main}"

if [ -z "$REPO_URL" ]; then
  echo "Usage: GITHUB_PAT=<token> $0 https://github.com/<username>/<repo>.git [branch]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

if [ ! -d ".git" ]; then
  git init -b "$BRANCH"
fi

git add .
if ! git diff --staged --quiet; then
  git commit -m "feat: NSE daily <=20d high-tight flag pre-breakout scanner & live Telegram workflows"
fi

if [ -n "${GITHUB_PAT:-}" ]; then
  # Inject token into temporary push URL without storing credentials in .git/config
  AUTH_URL="${REPO_URL/https:\/\//https:\/\/x-access-token:${GITHUB_PAT}@}"
  git remote remove origin 2>/dev/null || true
  git remote add origin "$REPO_URL"
  git push -u "$AUTH_URL" "$BRANCH"
else
  git remote remove origin 2>/dev/null || true
  git remote add origin "$REPO_URL"
  git push -u origin "$BRANCH"
fi

echo "✅ Pushed cleanly to $REPO_URL ($BRANCH)!"
echo "Next: Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in Repo Settings -> Secrets and variables -> Actions."
