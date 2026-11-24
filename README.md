# NSE Daily Coiled Pre-Breakout Scanner V2.1 Core + Coil (≤20-Day High-Tight Flag — Option A)

A production-ready, **Daily Timeframe (`1d`)** NSE equity scanner & zero-look-ahead backtest engine powered by `yfinance`, GitHub Actions, and Telegram Bot alerts (multi-bot broadcast with annotated dark-theme Daily Chart PNGs + Daily Volume Profile).

Upgraded to **V2.1 Core + Coil** using a strict In-Sample (`2022–2024`) → Blind Out-of-Sample (`2025–2026`) forensic audit across **2,381 unbiased 5-year full-NSE trades** (`min_score = 40`, all 2,404 NSE EQ stocks) and verified end-to-end across a **1,124-stock 5-year walk-forward backtest (`2021–2026`)**:
- **All V2.1 Core + Coil (`Score >= 66`, Grade A & A+, `n = 538` trades):**
  - **Win Rate:** **`68.6%`** (`72.1%` TP1 Hit Rate) | **Profit Factor:** **`1.90` Gross / `1.74` Net** (0.25% NSE cost)
  - **Blind OOS (`2025–2026`, `n = 123`):** **`2.06` Gross PF / `1.87` Net PF** (`70.7%` WR!)
  - **Consistency:** **All 6 calendar years (`2021–2026`, every single year `>= 1.56 PF`, including `2.10 PF` in the 2025 mid/small-cap bear market)**
  - **Drawdown & Equity:** `-7.9%` MaxDD (`₹1.00L → ₹2.50L` at 10% alloc; **`₹5.93L`** at 1% risk-parity)
- **V2.1 Grade A+ Elite (`Score >= 78`, `n = 455` trades):**
  - **Win Rate:** **`70.1%`** (`74.3%` TP1 Hit) | **Profit Factor:** **`2.11` Gross / `1.93` Net / `2.38` Blind OOS PF** | **MaxDD: `-5.7%` (`-6.6%` Net)**

---

## Core V2.1 Institutional Edge Rules (Strictly Zero-Look-Ahead)

1. **Strict ≤20-Day Pole + Tight Flag Window:** Entire pattern (1–3d explosive pole clearing 20d highs on ≥3× RVOL + ≥3d tight upper-pole shelf with ≥70% volume dry-up) must occur within **20 trading sessions**.
2. **11 Out-of-Sample Validated V2.1 Core + Coil Gates:**
   - **Dual Market Regime Guard:**
     1. **NIFTY 50 Guard:** Pauses alerts if `NIFTY 50 (^NSEI)` 20-day return `< -3.0%`
     2. **Live NSE Market Breadth50 Guard:** Computes `% of NSE stocks > 50-day SMA` directly from Stage-1 OHLCV (**0 extra API calls**); pauses breakouts when broad market breadth `< 40.0%` (historical `<40%` breadth = `0.89 PF` loss trap)
   - **Stage-2 Trend Stack:** `Close > 150-SMA` **AND** `Close > 200-SMA`
   - **Rising 50-Day SMA:** `SMA50[t] > SMA50[t - 10]` (eliminates falling-trend traps with `0.84` OOS PF)
   - **3-Month Relative Strength Leader:** `63-day return vs NIFTY 50 >= +15.0%` (eliminates `<0%` RS laggards at `0.67` OOS PF)
   - **Tight 52-Week High Proximity:** Close within **`-15.0%` of 52-Week High** (upgraded from `-25%`)
   - **Institutional Pole Band:** Pole High Gain between **`+18.0%` and `+42.0%`**
   - **Anti-Micro-Box Floor:** Flag wick box **`>= 5.5%`** (and `<= 15.0%`; blocks `<5.5%` stop-hunt magnets at `0.56` OOS PF)
   - **True High-Tight Upper-Pole Hold:** Every base close holds the **upper 35% of the pole (`>= 65%`)**
   - **Upper-Flag Coil Position:** Signal close in upper **80% of the flag box (`pos >= 0.20`)**
   - **Soft Climax Volume Penalty:** `-8.5 pt` Pillar-1 penalty for `>60×` news/operator spikes (keeps clean institutional leaders like `BOROLTD` eligible while downgrading weak climaxes)
3. **V2.1 Recalibrated 5-Pillar Score (100 Points):**
   - `Pole Quality (25)` · `High-Tight Flag & Upper Coil (25)` · `RS & 52wH Leadership (20)` · `Supply Dry-Up & Active Bid (15)` · `Breadth & Stage-2 Power (15)`
   - **Grades:** `🔥 A+` (`>= 78`, `2.11 PF / 2.38 OOS PF`) · `⚡ A` (`66–77.9`, e.g., `BOROLTD` at `70.5`) · `B` (`58–65.9`)
4. **Option A Trade Plan (Displayed on Every Alert & Chart):**
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
| **Stage 1 (EOD)** | `.github/workflows/eod_scanner.yml` | **15:47 IST** Mon–Fri (+ manual run) | Checks NIFTY 50 regime + computes live **NSE Breadth50** across all **~2,403 NSE EQ stocks** (`15mo` Daily OHLCV, 0 extra API calls), saves V2.1 Grade A/A+ setups to `state/coiled_candidates.json`, commits state, and sends **Telegram EOD Digest + Annotated Chart PNGs** for brand-new setups. |
| **Stage 2 (Live)** | `.github/workflows/live_scanner.yml` | **Every 10 mins**, 09:17–15:33 IST Mon–Fri | Checks NIFTY 50 regime & EOD Breadth50 guard, monitors `state/coiled_candidates.json` (+ `config/custom_watchlist.txt`) on today's live forming Daily candle (`1d` with automatic NaN-today repair) for **Gap-and-Go**, **Breakout Trigger**, **Spring Retest**, and **Tight Coil Watch**. |
| **Stage 3 (Backtest)** | `.github/workflows/backtest_scanner.yml` | Manual (`workflow_dispatch`) | Runs zero-look-ahead V2.1 walk-forward historical backtest (`6mo`–`5y`, automatic +1y warmup + `^NSEI` + point-in-time `Breadth50` series) across NSE equities, logs every **Pole, Entry, Breakout, SL, TP1, TP2 & TP3 Date**, generates a multi-page **Executive PDF Report** (Gross + Net of 0.25% NSE costs), uploads artifacts, and sends the PDF to Telegram. |

---

## Reliability Model

Three things used to make the automation *look* healthy while producing nothing. Each is now handled
explicitly.

### 1. Scheduling reliability

GitHub's `schedule` trigger is **best-effort**: runs are queued behind other work, frequently delayed
by minutes-to-hours, and dropped outright when a slot is missed (there is no catch-up). An audit of
this repo found Stage-2's `*/15` cron delivering only **~16%** of its ticks, none inside market hours.

Mitigations in place:

- Ticks are offset **off the `:00/:15/:30/:45` boundaries**, which are GitHub's busiest slots
  (`7,17,27,37,47,57 4-9 * * 1-5`).
- Stage-2 checks the NSE market state on every run, so a late tick is skipped rather than mis-firing.
- Each run writes the cron slot and actual dispatch time to the **job summary**, so drift is visible.
- For dependable intraday cadence, add an **external scheduler** (cron-job.org, Cloudflare Worker,
  AWS EventBridge) that calls `workflow_dispatch` at the exact times. Keep `schedule` as a fallback:

```bash
gh workflow run live_scanner.yml --ref main
```

### 2. Guards suppress alerts, not detection

*Previously:* tripping the breadth guard returned early and wrote `candidates: []`. That threw away
the day's watchlist **and** handed Stage-2 a `breadth_ok: false` which kept live scanning muted long
after the market recovered.

*Now:* Stage-1 always computes and persists the watchlist. The guard suppresses **alert dispatch only**,
and the state record carries `alerts_paused` / `pause_reason`. A tripped guard is recorded in the
Actions job summary and the log, so "paused" can never be mistaken for "no setups found" — without
posting anything to Telegram.

Stage-2 additionally ignores a **stale** breadth flag (> 4 days old, tolerating a long weekend) and
falls back to the fresh NIFTY 50 regime guard instead of staying muted forever.

### 3. Data-integrity circuit breaker

*Previously:* a Yahoo/NSE outage or throttled run logged `Fetched 0 / 2403 symbols` and then overwrote a
good `coiled_candidates.json` with an empty watchlist, committing it.

*Now:* the run **aborts, leaves state untouched, and fails the workflow red** when fewer than
`fetcher.min_fetch_success_pct` (default `50%`) or `fetcher.min_fetched_symbols` (default `100`) could be
fetched. Small `--limit` test runs are exempt. `python -m src.check_state` converts the soft abort into a
non-zero exit so the Actions UI shows a failure rather than a false green.

### 4. Telegram carries stock alerts only

A guard pause is an **operational state**, not a trading signal, so it is never sent to Telegram.
It appears in the GitHub Actions job summary and the log, where it belongs.

This replaced an earlier design that posted a `⏸ Scanner paused` message. Because the scheduler
dispatched Stage-1 every 15 minutes and a sub-40% breadth reading can persist for days, that message
repeated dozens of times a day. Deduplicating it was the wrong fix — the message should not have been
sent at all.

There are **three** enforcement points, so no future code path can leak a status message:

1. `publish_pause()` writes only to the job summary + log; it no longer takes a notifier.
2. `TelegramNotifier._is_stock_alert()` hard-blocks any message or caption containing an operational
   marker (`scanner paused`, `scanner aborted`, `no alerts were evaluated`, `scan paused`) — even if a
   caller tries.
3. There is no configuration switch to re-enable it.

What still reaches Telegram:

| message | sent? |
|---|---|
| Annotated chart + trade plan per stock setup | ✅ |
| Live breakout / gap-and-go / spring trigger | ✅ |
| EOD digest of new setups | ✅ |
| Backtest PDF (manual Stage-3 run) | ✅ |
| Guard pause / no-setups / data outage | ❌ job summary + logs only |

**Stage-1 also no longer re-scans the universe every 15 minutes.** `src/skip_scan.py` gates the EOD
scan to once per trading session:

- a completed post-close scan for today already exists → skip
- still inside the session (before 15:30 IST) → wait for the post-close scan
- otherwise → proceed

Scheduled runs are always allowed, so the settled close is never missed; the setting
`STAGE1_ONCE_PER_SESSION=false` (repository variable) disables the gate if you want a scan per
dispatch.

### 5. Verifying without network access

`scripts/dryrun.py` patches only the network boundary and runs the real detection, scoring, chart,
ledger and PDF code against synthetic data:

```bash
python scripts/dryrun.py            # 19 checks
python scripts/dryrun.py --quick    # skip the Stage-3 PDF build
```

It is the fastest way to confirm a change did not break the pipeline when Yahoo/NSE are
unreachable, and it is what caught the "tripped breadth guard still wipes the watchlist" bug.

### 6. Repeat-alert policy

`deduplication.strict_never_repeat_symbol: true` together with `repeat_cooldown_days: 20` gives a
**cooldown** policy: a symbol is suppressed for 20 days after alerting, then becomes eligible again for a
genuinely new setup. Set `repeat_cooldown_days: 0` for the old once-per-lifetime behaviour, or
`strict_never_repeat_symbol: false` to deduplicate per `(symbol, trigger)`.

> The lifetime behaviour is what previously left Stage-2 permanently muted: `BOROLTD` alerted on
> 2026-11-17 and was then skipped on every subsequent run, forever.

Dependencies are pinned with `~=` compatible-release bounds so an upstream major version cannot silently
break all three workflows.

---

## Quick Setup (GitHub Secrets & Permissions)

1. Go to **Settings → Secrets and variables → Actions → New repository secret**:
   - **Bot #1 (Primary):** `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
   - **Bot #2 (Optional Broadcast):** `TELEGRAM_BOT_TOKEN_2`, `TELEGRAM_CHAT_ID_2`
2. Go to **Settings → Actions → General → Workflow permissions** → enable **Read and write permissions**.
3. Run **Stage 1 - NSE Daily EOD Flag Scanner** once from the **Actions** tab to seed `state/coiled_candidates.json`.

---

## Running Locally & Testing

```bash
pip install -r requirements.txt

# 1. Run unit tests (54/54 covering all V2.1 gates, Breadth50, Option A Daily-Close SL, multi-bot, dedup, data-integrity & PDF)
python -m unittest discover -s tests -v

# 2. Validate V2.1 Core + Coil on BOROLTD (pre-breakout 13 Nov + live breakout 16 Nov)
python -m src.backtest_validate --symbol BOROLTD --pre-date 2026-11-13 --bo-date 2026-11-16

# 3. Run Stage-1 EOD scan locally (dry-run)
python -m src.run_eod_scan --force --limit 200 --dry-run

# 4. Run Stage-3 Historical Backtest & generate PDF Report (default min-score 66 = V2.1 Grade A & A+)
python -m src.run_backtest --period 2y --limit 500 --min-score 66

# 5. Offline end-to-end dry run (no network needed) — drives Stage 1+2+3 with
#    synthetic OHLCV and asserts 19 invariants. Useful in sandboxes/CI where
#    Yahoo & NSE are unreachable.
python scripts/dryrun.py
```
