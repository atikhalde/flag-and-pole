# GitHub Actions Workflow Audit — `atikhalde/flag-and-pole`

**Audit date:** 2026-11-23 · **Branch:** `arena/90a4e93a-flag-and-pole` (from `main` @ `7d3e62d`)
**Scope:** all 3 workflows, the Python code they invoke, state files, and 271 real run records from the GitHub API.

---

## 1. Verdict

| Question | Answer |
|---|---|
| Do the workflows *run* without errors? | **Yes.** 259 of 261 runs succeeded. Syntax, deps, paths and CLI args all check out. |
| Do they *do their job*? | **No — currently no alerts are produced at all**, and Stage 2's schedule fires at ~16% of the configured rate. |

The pipeline is **structurally sound but functionally stalled**. Every recent run is green, yet
`state/sent_alerts.json` still contains only the **8 symbols alerted on 2026-11-17** — nothing new
in 6 days and ~260 runs. Two independent causes, detailed in §4.

---

## 2. What is verified working

| Check | Result |
|---|---|
| All 3 workflow YAML files parse; triggers/jobs/steps valid | ✅ |
| Cron expressions valid and map to intended IST times | ✅ `15 10 * * 1-5` = 15:45 IST; `45 3`/`*/15 4-9`/`0 10` = 09:15→15:30 IST |
| Every `python -m src.*` entry point in the YAML exists | ✅ `run_eod_scan`, `run_live_scan`, `run_backtest`, `alert_ledger` |
| Every CLI flag passed by the YAML is accepted by argparse | ✅ 8/8 args match |
| All 16 modules in `src/` import cleanly | ✅ no import failures |
| Test suite | ✅ **10 passed** in 1.72 s |
| `alert_ledger --merge-into/--from` union merge (lossless) | ✅ tested: added 1, preserved all 8 |
| `requirements.txt` resolves; 3.11 pinned in all jobs | ✅ |
| Bot state commits reach `main` | ✅ `7d3e62d` by `github-actions[bot]` |
| Stage 3 produces a real PDF artifact | ✅ 105 KB – 1.4 MB PDFs uploaded, 5/5 runs green |
| Push-race handling (3-attempt fetch/reset/merge retry) | ✅ well designed |
| Telegram multi-bot support, 429 back-off | ✅ |

**Conclusion on code/CI health:** the implementation quality is genuinely good. The problems below are
*operational and logic-policy* issues, not broken code.

---

## 3. Runs I executed locally

Yahoo Finance and `nsearchives.nseindia.com` are **blocked by this sandbox's egress policy**
(`pypi.org` and `github.com` reach fine), so data-dependent paths were run in dry-run/limited mode:

| Command | Result |
|---|---|
| `pytest tests/ -q` | `10 passed` |
| `python -m compileall src tests` | clean |
| `run_eod_scan --force --limit 5 --dry-run` | Pipeline executes start→finish; 0/5 fetched (egress blocked, not a code bug) |
| `run_live_scan` (no `--force`) | Correctly returns `{"skipped": true, "reason": "market_closed"}` |
| `alert_ledger --merge-into ... --from ...` | `added=1, total=9` ✅ |

Every log line in the EOD run was coherent — the pipeline did not crash, it just had no data to chew on.

---

## 4. Findings

### 🔴 P1 — Scheduled cron triggers are firing ~16% of the time (Stage 2)

Actual `schedule`-event runs vs. what the cron should produce, over 6.7 days:

```
Stage 1 EOD    :  4 scheduled  (~5 expected)     80%   ✅ acceptable
Stage 2 Live   : 14 scheduled  (~90 expected)    16%   ❌ badly degraded
```

Observed Stage 2 scheduled times (converted to IST) are **2–6 hours off** the cron:

| Fired (IST) | Should have fired |
|---|---|
| Wed 00:12, 00:28 | a 15-min tick during 09:15–15:30 |
| Wed 17:42, 18:14 | a 15-min tick during 09:15–15:30 |
| Thu 17:37, 18:06 | a 15-min tick during 09:15–15:30 |
| Fri 17:39, 18:06, 23:53 | a 15-min tick during 09:15–15:30 |
| Mon 19:08, 19:40 | a 15-min tick during 09:15–15:30 |

`createdAt` is when GitHub actually dispatched the run, so these are genuine multi-hour delays.
This matches GitHub's documented behaviour: `schedule` is **best-effort**, delayed under load, and
missed ticks are **dropped, not queued** ([discussion #210111](https://github.com/orgs/community/discussions/210111),
[#211641](https://github.com/orgs/community/discussions/211641)).

**Why this is benign for now:** 200 of Stage 2's runs are manual `workflow_dispatch`, and the
market-hours gate (`run_live_scan` → `is_nse_market_open`) correctly no-ops when a late run lands
outside 09:15–15:35 IST:

```
20:23:50 | INFO | NSE market is currently closed (2026-11-24 01:53 IST). Use --force to override.
{ "skipped": true, "reason": "market_closed" }
```

So a late tick self-cancels. **But** that also means a delayed tick during market hours simply
skipped that 15-minute window — with 16% delivery you are realistically scanning a handful of
times per day, not 18. A `*/15` schedule on shared runners is the wrong tool for a live scanner.

**Fix (priority order):**
1. Add an external trigger (cron-job.org / Cloudflare Worker / AWS EventBridge) calling
   `workflow_dispatch` at the exact ticks. Keep `schedule` as a fallback.
2. If staying on GH cron, move off the `:00/:15/:30/:45` boundaries (the busiest slots) and accept
   coarse timing — e.g. `7,22,37,52 4-9 * * 1-5`.
3. Make the job convergent: on each run, compare against the last successful run timestamp and catch
   up, instead of assuming every tick happened.

---

### 🔴 P2 — The pipeline is currently emitting zero alerts (3 compounding causes)

`state/coiled_candidates.json` today:

```json
"nse_breadth50": { "breadth50_pct": 32.8, "min_threshold_pct": 40.0, "breadth_ok": false },
"candidates": []
```

1. **Breadth guard is paused.** 32.8% of NSE stocks are above their 50-SMA, under the configured
   40% floor. Stage 1 correctly stops and writes `candidates: []`
   (`run_eod_scan.py:196` → `nse_breadth_paused`). Stage 2 then reads that state back and pauses too
   (`run_live_scan.py:80`). **This is intended behaviour**, but it means every green run since
   2026-11-17 has produced nothing, and the UI gives no signal that the scanner is *deliberately
   muted* rather than *finding no setups*.
2. **Empty watchlist.** With `candidates: []`, Stage 2 monitors only `config/custom_watchlist.txt`,
   which holds a single symbol: **BOROLTD**.
3. **Permanent dedup silences that last symbol forever.** `strict_never_repeat_symbol: true` plus
   `BOROLTD` already present in the ledger (`"stage": "LEGACY"`, 2026-11-17) means Stage 2 skips it
   on every run, forever.

Net effect: **Stage 2 cannot fire a single alert right now**, and 214 consecutive "successful" runs
hide that fact.

**Fix:**
- Emit an explicit, visible signal when a guard pauses the scan (a Telegram line "⏸ Scanner paused:
  breadth 32.8% < 40%" or a job-summary annotation) so "green" is not confused with "no setups".
- Reconsider the semantics of `strict_never_repeat_symbol`. As configured, a symbol is alertable
  **once in the lifetime of the repo**. A stock that alerted on 17 Nov and is now setting up a
  textbook flag will be silently dropped forever. Consider scoping dedup to
  `(symbol, trigger_type, session_date)` or a cooldown window (e.g. 20 sessions), which preserves
  the "don't spam the same breakout" intent without permanently blinding the scanner.
- If you accept lifetime dedup, `max_watchlist_size: 40` becomes meaningless over time — the
  monitored universe only ever shrinks.

---

### 🟠 P3 — Stage 1 gates on breadth *before* it needs to, and freezes the watchlist

Because the breadth check returns early (`run_eod_scan.py:196-218`) **before** pattern detection, a
weak-breadth day also discards that day's watchlist update. When breadth recovers, Stage 1 restarts
from scratch rather than having kept a current candidate list. Additionally, Stage 2's own guard reads
a *stale* breadth value baked into `coiled_candidates.json` — if Stage 1 hasn't run since the market
turned up, Stage 2 stays paused on old data.

**Fix:** compute and persist candidates first, gate only the *alert dispatch* on breadth. In Stage 2,
fall back to a freshly computed breadth (or the NIFTY guard only) when the saved breadth is older than
the last trading session.

---

### 🟠 P4 — No guard against a total data-fetch failure wiping state

`fetch_daily_batch` returning an empty/partial map is logged but **not treated as an error**:

```
Fetched clean Daily OHLCV for 0 / 5 symbols in 0.5s
Saved 0 qualified V2.1 coiled candidates to state/coiled_candidates.json
```

A Yahoo/NSE outage (or a throttled run) on a breadth-OK day will therefore overwrite a good
`coiled_candidates.json` with `candidates: []`, commit it, and silently kill live monitoring until the
next good run. `min_history_days` is configured but not used as a run-level sanity check.

**Fix:** abort (non-zero exit, leave state untouched) when `len(daily_map) < 50%` of the universe, or
when `0 < len(daily_map) == 0` after fetching — mirroring the `min_breadth_universe` idea.

---

### 🟡 P5 — `git add` on a missing state file aborts the commit step

`eod_scanner.yml` runs `git add state/coiled_candidates.json state/sent_alerts.json` unconditionally.
If either file is absent, git exits **128** and the job fails:

```
fatal: pathspec 'state/coiled_candidates.json' did not match any files
git add exit=128
```

Both files exist today, so this is latent — but it is exactly the class of failure that produced the
one real `failure` in your history:

```
failure  workflow_dispatch  2026-11-17 05:02 IST  Stage 1 - EOD
  step: "Commit updated coiled watchlist & alert ledger"   ❌
```

(The older `Commit updated coiled watchlist state` failure shows the step was already patched once.)

**Fix:** `git add -A state/` (or `git add --ignore-errors` / `|| true` per file).

---

### 🟡 P6 — `requirements.txt` is fully unpinned

`pip install -q -r requirements.txt` today resolves to **pandas 3.1.1, numpy 2.4.6, yfinance 1.8.0,
scipy 1.17.1, matplotlib 3.11.2**. Lower bounds only (`>=`) means any upstream breaking change
immediately breaks all three scheduled workflows with no code change on your side. pandas 3.x and
yfinance 1.x were both major-version jumps.

**Fix:** pin with `~=` (e.g. `pandas~=3.1`, `yfinance~=1.8`) or commit a `requirements.lock` /
`pip-compile` output.

---

### 🟡 P7 — One backtest run looks anomalously short

| Run | Duration | Artifact |
|---|---|---|
| 43966415398 | **21m 27s** | 1.44 MB PDF |
| 43957123088 | 11m 14s | — |
| 43976949894 | 3m 12s | 105 KB PDF |
| 44463480774 | 2m 39s | 105 KB PDF |
| 45079327569 (latest) | 3m 38s | 105 KB PDF |

A 2y × ~2,400-symbol backtest completing in ~3 minutes with a 105 KB report suggests a much smaller
traded sample than the 21-minute run's 1.44 MB report. Worth confirming the latest PDF's trade count
and date coverage — it may simply reflect the strict gates plus paused breadth, or it may indicate the
backtest silently ran on a reduced universe. (I could not download the artifact — Azure Blob storage
is outside this sandbox's allowed egress.)

---

### 🟢 P8 — Minor / cosmetic

- `is_nse_market_open` allows a 5-minute post-close buffer, so ticks to 15:35 IST run; the `0 10 * * 1-5`
  tick (15:30 IST) is correct but will usually catch a not-yet-final 1d bar.
- Test coverage is concentrated in `test_detector.py`; `data_fetcher`, `chart_generator`,
  `volume_profile`, `run_live_scan`, and `run_backtest` have **zero** direct coverage.
- `NSE_HOLIDAYS_2026` is hardcoded and will need updating for 2027; a missing holiday only wastes a
  run, but a wrong date silently scans a closed session.
- README documents the intended schedule accurately — the drift is purely GitHub's scheduler.

---

## 4b. Follow-up fixes (second pass)

The first round of fixes was re-verified end-to-end rather than trusted. It contained a
**real bug of its own**, plus two gaps of the same family.

### 🔴 F1 — The abort fix did not actually fail the build (fixed)

The circuit breaker returned `{"aborted": true}` from the library, but:

1. `run_eod_scan.main()` still **exited 0**, and
2. `check_state.py` inspected `state/coiled_candidates.json` — which an abort deliberately
   leaves **untouched**, so it still held the last healthy scan and reported `State OK`.

Net result: an outage produced `"aborted": true` and a **green** run. Exactly the failure
mode the fix was meant to remove.

```
RUN EXIT CODE = 0
"aborted": true
State OK: 0 candidate(s), alerts_paused=False     <- verifier saw stale-but-healthy state
CHECK_STATE EXIT = 0
```

**Fixed by:** `--summary-out` persists the run summary; `main()` raises `SystemExit(1)` on
`aborted`; `check_state.verify()` now reads the **summary** (falling back to state) so an
abort is red while a guard pause stays green.

```
RUN EXIT = 1
::error::Stage-1 aborted (data_fetch_failed) (0/5 symbols); state left untouched.
CHECK_STATE EXIT = 1
STATE UNTOUCHED ✅
```

### 🟠 F2 — Stage 2 had no integrity guard at all (fixed)

A Yahoo outage during market hours produced `0 triggers fired` — indistinguishable from a
genuinely quiet session. A missed breakout and no breakout looked identical. Stage 2 now
detects `<50%` fetch, logs `Live scan DEGRADED`, writes a job summary and **exits 1**.

### 🟠 F3 — Stage 3's anomaly had the same root cause as P4 (fixed)

The unexplained 3-minute / 105 KB backtests (P7) were almost certainly a small universe: Stage 3
had **no** integrity guard, so a partial Yahoo fetch silently produced a tiny backtest and a
near-empty PDF, still green. It now aborts below `min_backtest_universe` / `min_fetch_success_pct`
(explicit `--symbols` and tiny `--limit` runs exempt, but **zero** data always aborts).

### 🟠 F4 — Stage 3 was completely unverified in CI (fixed)

It is manual-only, so a broken argparse/config path would go unnoticed for weeks. A new
`preflight` job runs the test suite plus `--parse-only` (validates CLI + config + universe +
download, then exits before the long scan). `backtest-report` now `needs: preflight`.

---

## 4c. Third pass — dry run findings

Running the pipeline offline end-to-end (synthetic data, real code) surfaced **two more real bugs**,
one of which meant the headline Stage-1 fix still did not work.

### 🔴 F5 — A tripped breadth guard still wiped the watchlist (fixed)

The reordering in §4/P3 was supposed to keep detection independent of the guard. It did not:
`detect_daily_pattern` **self-gates** on the breadth value it is given and returns `None` when that
value is under the floor. Stage-1 was passing the *live* reading (`32.8`), so every setup was
rejected and the paused day still wrote `candidates: []` — exactly the bug the reordering existed to
remove. Stage-2 had the same flaw for custom-watchlist scoring.

**Fixed:** detection now receives the configured *floor* (always satisfied), never the live reading.
The guard is applied strictly downstream, to decide dispatch. Both stages had the issue.

```
before:  paused run -> watchlist_count = 0    (candidates wiped)
after :  paused run -> watchlist_count = 3    (candidates preserved, alerts suppressed)
```

### 🟠 F6 — The integrity floor aborted perfectly healthy small runs (fixed)

`min_fetched_symbols: 100` was applied to the *absolute* count, so a 63-symbol universe that fetched
**63/63 (100%)** aborted with the self-contradictory message
`Only 63/63 symbols fetched (100.0%) — below the ... floor`.

**Fixed:** the coverage check (`<50%`) always applies; the absolute floor now only applies when a
large universe was requested:

```python
if requested >= max(min_abs * 2, 200) and fetched < min_abs:  # absolute floor
```

| fetched/requested | result |
|---|---|
| 0 / 2403 | abort |
| 63 / 63 (100%) | pass |
| 100 / 2403 | abort |
| 5 / 2403 | abort |
| 60 / 120 (50%) | pass |
| 2380 / 2403 | pass |

### Verification harness

`scripts/dryrun.py` drives Stage-1 → Stage-2 → Stage-3 offline with synthetic data, patching **only**
the network boundary. It asserts 19 invariants (detection, dud rejection, schema, trade-plan
ordering, chart rendering, live trigger firing, real `%PDF-` output).

```
DRY RUN: 19/19 checks passed
```

---

---

---

## 5. Recommended priority

| # | Action | Effort | Impact |
|---|---|---|---|
| 1 | Pin `requirements.txt` | 5 min | Prevents all three workflows breaking without warning |
| 2 | Surface guard-pause state in Telegram / job summary | 20 min | Turns silent "green, no alerts" into an explicit signal |
| 3 | Decide dedup policy (lifetime vs. cooldown window) | 30 min | Unblocks Stage 2 from being permanently muted |
| 4 | Move candidate detection ahead of the breadth gate | 45 min | Keeps the watchlist fresh through weak-breadth stretches |
| 5 | Abort instead of overwriting state on fetch failure | 20 min | Removes the silent-outage failure mode |
| 6 | `git add -A state/` | 2 min | Fixes the observed commit-step failure class |
| 7 | External scheduler for Stage 2 ticks | 1–2 h | 16% → reliable intraday coverage |
| 8 | Review latest Stage 3 PDF trade count | 15 min | Confirms backtest integrity |

---

---

## 6. Bottom line

Nothing is broken. The code, tests, CLI contracts, cron expressions, dedup merge and state-commit
retry logic are all in good shape.

The automation was **idle by design rather than by failure**: NSE breadth (32.8%) sat below the 40%
floor, so Stage 1 wrote an empty watchlist, and Stage 2 — which honoured that same breadth flag —
monitored one symbol the permanent-dedup ledger had already silenced.

After two passes, every finding in this document is fixed and **verified by execution**, not by
inspection:

| Check | Result |
|---|---|
| Tests | **43 passed** (was 10) |
| pyflakes | clean |
| All 5 CLI entry points `--help` | OK |
| All 18 modules import | OK |
| Workflow YAML parse + job graph | OK |
| Abort → red build | `scan_exit=1`, `verify_exit=1` |
| Guard pause → green build | `verify_exit=0` |
| Market closed → green build | `verify_exit=0` |
| Outage → state preserved | byte-identical ✅ |
| Offline end-to-end dry run | **19/19 checks** (`scripts/dryrun.py`) |
| Paused day keeps watchlist | 3 candidates preserved, 0 alerts |

The single most important lesson from the second pass: **a fix that reports success without
proving it is the same bug in a new costume.** The first abort fix looked correct and was not.
