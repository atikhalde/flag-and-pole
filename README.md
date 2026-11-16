# NSE Daily Coiled Pre-Breakout Scanner V2 (≤20-Day High-Tight Flag — Option A)

A production-ready, **Daily Timeframe (`1d`)** NSE equity scanner & zero-look-ahead backtest engine powered by `yfinance`, GitHub Actions, and Telegram Bot alerts (multi-bot broadcast with annotated dark-theme Daily Chart PNGs + Daily Volume Profile).

Upgraded to **V2 Option A** using a full 5-year out-of-sample quantitative audit across all **~2,403 NSE EQ stocks (`2021–2026`)**:
- **Win Rate:** `33.8%` (V1) → **`66.9%` (`71.2%` TP1 Hit Rate)** across 619 Grade A+ trades
- **Profit Factor:** `1.36` (V1) → **`1.73` Gross / `1.58` Net of 0.25% NSE costs**
- **Max Drawdown:** `-15.5%` (V1) → **`-5.9%`** (2.6× smaller!)
- **Consistency:** **All 6 calendar years (`2021–2026`, including the tough 2025 bear market) profitable**

---

## Core V2 Institutional Rules (Strictly Zero-Look-Ahead)

1. **Strict ≤20-Day Pole + Tight Flag Window:** Entire pattern (1–3d explosive pole clearing 20d highs on ≥3× RVOL + ≥3d tight upper-pole shelf with ≥70% volume dry-up) must occur within **20 trading sessions**.
2. **5 Pre-Signal Institutional Edge Gates:**
   - **Primary Uptrend:** `Close > 200-day SMA`
   - **52-Week Leadership:** Close within **`-25%` of 52-Week High**
   - **NIFTY 50 Regime Guard:** Pauses new alerts automatically if `NIFTY 50 (^NSEI)` 20-day return `< -3.0%`
   - **Anti-Blowoff Pole Cap:** `Pole High Gain <= 42.0%` (blocks parabolic spikes while passing leaders like `BOROLTD`)
   - **Upper-Flag Coil Position:** Signal close in upper **75% of the flag box** (`pos >= 0.25`)
3. **V2 Recalibrated 5-Pillar Score (100 Points):**
   - `Pole Quality (25)` · `Upper-Flag Coil & Box (25)` · `Momentum / ATR / RS vs NIFTY (20)` · `Healthy Supply Dry-Up & CLV (20)` · `Liquidity Sweet Spot & 52wH Leadership (10)`
   - **Grades:** `A+` (`>= 78`, EOD Telegram Alert threshold) · `A` (`66–77`, Stage-2 watchlist) · `B` (`58–65`)
4. **V2 Option A Trade Plan (Displayed on Every Alert & Chart):**
   - **Entry:** Strictly **below the breakout level** inside the coiled flag shelf
   - **Daily-Close Stop-Loss (3:20 PM IST):** `-1.5%` below flag low (`4.0%–6.5%` risk) — dodges intraday stop-hunts; exit only if Daily Close breaks below
   - **Emergency Broker GTT:** Hard **`-9.0%`** intraday safety net placed immediately on Entry *(alerts also print a hands-off Intraday GTT reference at `-2.5%` below flag low)*
   - **High-Win-Rate 3-Tranche Profit Booking:**
     - **TP1 (`+4.5%`):** Book **45%** & move SL to **Breakeven**
     - **TP2 (`+10.0%`):** Book **30%** & trail SL up to **TP1**
     - **TP3 (`+18.0%`):** Ride final **25%** runner
5. **Strict Never-Repeat Deduplication:** Permanent symbol-level ledger (`state/sent_alerts.json`) ensures the same stock is never alerted twice across scans, stages, or days.

---

## Architecture: Three-Stage GitHub Actions Workflow

| Stage | Workflow | Schedule (IST) | What It Does |
|---|---|---|---|
| **Stage 1 (EOD)** | `.github/workflows/eod_scanner.yml` | **15:45 IST** Mon–Fri (+ manual run) | Checks NIFTY 50 regime, scans all **~2,403 NSE EQ stocks** (`15mo` Daily OHLCV), saves Grade A/A+ setups to `state/coiled_candidates.json`, commits state, and sends **Telegram EOD Digest + Annotated Chart PNGs** for brand-new Grade A+ setups. |
| **Stage 2 (Live)** | `.github/workflows/live_scanner.yml` | **Every 15 mins**, 09:15–15:30 IST Mon–Fri | Checks NIFTY 50 regime & monitors `state/coiled_candidates.json` (+ `config/custom_watchlist.txt`) on today's live forming Daily candle (`1d` with automatic NaN-today repair) for **Gap-and-Go**, **Breakout Trigger**, **Spring Retest**, and **Tight Coil Watch**. |
| **Stage 3 (Backtest)** | `.github/workflows/backtest_scanner.yml` | Manual (`workflow_dispatch`) | Runs zero-look-ahead walk-forward historical backtest (`6mo`–`5y`, automatic +1y warmup + `^NSEI`) across NSE equities, logs every **Pole, Entry, Breakout, SL, TP1, TP2 & TP3 Date**, generates a multi-page **Executive PDF Report** (Gross + Net of 0.25% NSE costs), uploads artifacts, and sends the PDF to Telegram. |

---

## Quick Setup (GitHub Secrets & Permissions)

1. Go to **Settings → Secrets and variables → Actions → New repository secret**:
   - **Bot #1 (Primary):** `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
   - **Bot #2 (Optional Broadcast):** `TELEGRAM_BOT_TOKEN_2`, `TELEGRAM_CHAT_ID_2`
2. Go to **Settings → Actions → General → Workflow permissions** → enable **Read and write permissions**.
3. Run **Stage 1 - NSE Daily EOD Coiled Scanner** once from the **Actions** tab to seed `state/coiled_candidates.json`.

---

## Running Locally & Testing

```bash
pip install -r requirements.txt

# 1. Run unit tests (7/7 covering V2 gates, Option A Daily-Close SL, multi-bot, dedup & PDF)
python -m unittest discover -s tests -v

# 2. Validate V2 Option A on BOROLTD (pre-breakout 13 Nov + live breakout 16 Nov)
python -m src.backtest_validate --symbol BOROLTD --pre-date 2026-11-13 --bo-date 2026-11-16

# 3. Run Stage-1 EOD scan locally (dry-run)
python -m src.run_eod_scan --force --limit 200 --dry-run

# 4. Run Stage-3 Historical Backtest & generate PDF Report
python -m src.run_backtest --period 2y --limit 500 --min-score 78
```
