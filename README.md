# NSE Daily Coiled Pre-Breakout Scanner (≤20-Day High-Tight Flag)

A production-ready, **Daily Timeframe (`1d`)** NSE equity scanner powered by `yfinance`, GitHub Actions, and Telegram Bot alerts (with annotated dark-theme Daily Chart PNGs + Daily Volume Profile).

Modeled directly after high-conviction institutional post-breakout consolidations like **BOROLTD (30 Oct → 13 Nov 2026, before the 16 Nov +9.5% breakout)**:
1. **Explosive First Breakout (Pole):** Strong 1–3 day vertical expansion clearing prior 20d/40d highs on ≥3× volume surge.
2. **Tight High-Flag Shelf (Pre-Breakout):** Tight horizontal consolidation holding in the upper 50%+ of the pole without breaking down.
3. **Strict ≤20-Day Window:** Entire pattern (**Pole + Tight Consolidation**) must occur within **20 trading sessions** of the first breakout day.
4. **Volatility Contraction (VCP / NR Coil):** Shrinking daily candle ranges, inside bars / tight dojis, and optional dry stop-sweep spring bars.
5. **Severe Volume Dry-Up:** ≥70–95% volume collapse vs pole peak while total base volume remains a small fraction of pole volume (proving pole buyers aren't distributing).
6. **Daily Volume Profile Confluence:** Acceptance at/above Daily `VAH` perched over an unfilled `LVN` gap vacuum.

---

## Architecture: Two-Stage GitHub Actions Workflow

| Stage | Workflow | Schedule (IST) | What It Does |
|---|---|---|---|
| **Stage 1 (EOD)** | `.github/workflows/eod_scanner.yml` | **15:45 IST** Mon–Fri (+ manual run) | Scans all **~2,400 NSE EQ stocks** on Daily `1d`, scores every ≤20d coiled flag `0–100`, saves top setups to `state/coiled_candidates.json`, commits state back to GitHub, and sends **Telegram EOD Digest + Annotated Chart PNGs**. |
| **Stage 2 (Live)** | `.github/workflows/live_scanner.yml` | **Every 15 mins**, 09:15–15:30 IST Mon–Fri | Monitors `state/coiled_candidates.json` (+ `config/custom_watchlist.txt`) on today's live forming **Daily candle** (`1d` with automatic NaN-today repair), firing deduplicated Telegram photo alerts for **Gap-and-Go**, **Breakout Trigger**, **Spring Retest**, and **Tight Coil Watch**. |

---

## 5-Pillar Daily Scoring System (100 Points)

- **Pole Strength (`25 pts`):** Pole gain%, volume explosion RVOL, 20d/40d breakout, gap-up strength.
- **High-Tight Shelf Geometry (`25 pts`):** Upper-pole hold %, tight closing spread, tight wick box, proximity to ceiling.
- **VCP / NR Volatility Contraction (`20 pts`):** Range shrinkage vs pole/early base, absolute 3-day range tightness, doji/inside-bar cluster, dry stop-sweep spring bar.
- **Volume Dry-Up (`20 pts`):** Recent base avg volume vs pole peak collapse %, total base/pole volume retention ratio.
- **Daily Volume Profile (`10 pts`):** Acceptance above Daily `VAH`, unfilled `LVN` gap below the shelf.

**Grades:** `A+` (82–100) · `A` (70–81) · `B` (60–69)

---

## Quick Setup (GitHub + Telegram)

### 1. Create a Telegram Bot & Get Your Chat ID
1. Message **[@BotFather](https://t.me/BotFather)** on Telegram → `/newbot` → copy your **`TELEGRAM_BOT_TOKEN`**.
2. Send any message to your new bot (or add it to your alerts channel/group).
3. Visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser and copy `"chat":{"id": ...}` as your **`TELEGRAM_CHAT_ID`**.

### 2. Configure GitHub Repository Secrets & Permissions
In your GitHub repository:
1. Go to **Settings → Secrets and variables → Actions → New repository secret**:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
2. Go to **Settings → Actions → General → Workflow permissions**:
   - Select **Read and write permissions** (needed so workflows can commit `state/coiled_candidates.json` and `state/sent_alerts.json`).
3. Go to the **Actions** tab → select **Stage 1 - NSE Daily EOD Coiled Scanner** → click **Run workflow** once to seed `state/coiled_candidates.json` immediately!

---

## Running Locally & Historical Validation

```bash
pip install -r requirements.txt

# 1. Prove the scanner catches BOROLTD on 13 Nov (before 16 Nov breakout) & triggers live on 16 Nov
python -m src.backtest_validate --symbol BOROLTD --pre-date 2026-11-13 --bo-date 2026-11-16

# 2. Run Stage-1 EOD scan locally (dry-run on first 200 stocks)
python -m src.run_eod_scan --force --limit 200 --dry-run

# 3. Run Stage-2 Live Trigger scan locally
python -m src.run_live_scan --force --dry-run

# 4. Run unit tests
python -m unittest discover -s tests -v
```

---

## Tuning Parameters
Edit `config/settings.yaml` to customize:
- `pattern.max_total_pattern_days` (default: `20` trading days from first breakout pole)
- `universe.min_price` / `universe.min_pole_value_cr` (liquidity filter)
- `pattern.min_pole_high_gain_pct` / `pattern.max_dryup_ratio` / `pattern.max_base_close_spread_pct`
- `scoring.min_eod_alert_score` / `scoring.max_eod_chart_alerts`
- Always-monitored symbols in `config/custom_watchlist.txt`

> **Note on GitHub Actions Scheduling:** GitHub Actions cron triggers can experience a 3–12 minute delay during peak UTC windows, and private repos include 2,000 free Actions minutes/month (public repos have unlimited standard runner minutes). Because Stage 2 only checks ~10–40 coiled stocks on Daily `1d`, each live run finishes in ~20–35 seconds.
