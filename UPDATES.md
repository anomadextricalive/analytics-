# Cricket Analytics — Change Log

Reverse-chronological log of all changes. Full thesis and methodology in [README.md](README.md).

Format: `sha | files changed | +added / -deleted`

---

## 2026-05-16

### `529f4cf` — Cosmic UI theme + UPDATES.md changelog
`2 files | +315 / -212`

- **Modified** `src/dashboard/app.py` — full CSS variable rewrite: neobrutalist (light/yellow/Space Mono) → Cosmic dark theme
  - New surface scale: `--cosmic-void-950` → `--cosmic-void-600` (deep navy/void backgrounds)
  - New ink scale: `--cosmic-ink-100` → `--cosmic-ink-700`
  - New accent palette: mint (`#2DD4BF`), amber (`#F5C842`), magenta (`#EC4899`)
  - Nebula bloom: two layered radial gradients over void background (teal top-left, magenta bottom-right)
  - Fonts: Oswald (display) · Inter (body) · JetBrains Mono (code)
  - Compat aliases (`--bg`, `--surface`, `--accent`, etc.) preserved — no downstream component changes needed
  - No logic changes — visual redesign only
- **Added** `UPDATES.md` — this file

---

## 2026-04-15

### `5cbe848` — Cricket GPT insight engine, chat history, admin page
`8 files | +1365 / -17`

- **Added** Cricket GPT chat interface — natural language → SQL → answer pipeline against 26-table schema
- **Added** persistent chat history across session
- **Added** admin page to dashboard
- Multiple supporting files added/modified

---

## 2026-04-12

### `6d21520` — Restore cricket.db.gz with enrich tables
`1 file | binary delta`

- **Modified** `data/cricket.db.gz` — rebuilt compressed DB including `player_similarity` and `player_form` tables

### `c1f9cbb` — By Opponent sub-filters, fix similarity/form stale cache
`3 files | +95 / -17`

- **Added** By Opponent sub-filter dropdowns to player drill-down
- **Fixed** similarity and form panels serving stale cached data

### `404581e` — Player similarity, form analytics, UI overhaul
`4 files | +1398 / -206`

- **Added** `player_similarity` table — cosine similarity across rating/phase vectors
- **Added** `player_form` table — rolling-window recent-form scoring
- **Modified** dashboard UI — significant layout and component updates

---

## 2026-04-05

### `230fcc6` — Fix stale cached engine
`1 file | +3 / -3`

- **Fixed** `get_db_engine()` caching bug — DB_PATH now passed as cache key so engine refreshes on path change

---

## 2026-04-04

### `5d2991f` — Always overwrite /tmp/cricket.db on startup
`1 file | +4 / -5`

- **Fixed** stale DB on hosted deployments — `/tmp` persists across redeploys; now always overwrites on startup

### `18cb14b` — Fix ROOT path, derive gz path from DB_PATH.parent
`1 file | +10 / -8`

- **Fixed** ROOT resolution for hosted environments; gz path derived from `DB_PATH.parent` instead of hardcoded

### `52dbae5` — Surface _exec_sql exceptions in debug panel
`1 file | +28 / -23`

- **Fixed** silent exception swallowing in `_exec_sql` — errors now surfaced in debug panel

### `76740f1` — Deep debug: trace merge steps in all_players()
`1 file | +14 / -8`

- **Added** step-by-step merge tracing to `all_players()` for hosted deployment diagnosis

### `b1c4b82` — Fix _exec_sql: use mappings().fetchall()
`1 file | +11 / -12`

- **Fixed** `_exec_sql` returning unreliable row types — switched to `mappings().fetchall()` for consistent dict rows

### `6a12fad` — Fix all_players() merge: use _exec_sql + cast IDs to Int64
`1 file | +11 / -12`

- **Fixed** DataFrame merge failure in `all_players()` — player IDs cast to `Int64` before join

### `8591bcf` — Fix SQL execution for Python 3.14
`1 file | +53 / -66`

- **Fixed** `pd.read_sql` broken on Python 3.14 — replaced with `conn.execute()` + manual DataFrame construction

### `2f400b6` — Enhance debug panel
`1 file | +31 / -15`

- **Added** JOIN query test and table listing to debug panel

### `f992f2b` — Add debug panel to Player Explorer
`1 file | +26`

- **Added** collapsible debug panel to Player Explorer for diagnosing hosted deployment issues

### `726d2b8` — Fix all_players() for hosted deployment
`1 file | +57 / -2`

- **Fixed** player list not loading on hosted Streamlit — switched to native MongoDB merge path with label fix

---

## 2026-04-03

### `84359f5` — Fix SQLAlchemy 2.x incompatibility
`1 file | +15 / -17`

- **Fixed** `session.bind` removed in SQLAlchemy 2.x — replaced with `engine.connect()` context manager

### `ab35835` — Fix hosted Streamlit: SQLite fallback via /tmp
`1 file | +40 / -13`

- **Fixed** read-only filesystem on hosted Streamlit — SQLite decompressed to `/tmp`; MongoDB used as primary layer

### `91c9157` — Wire dashboard to MongoDB
`1 file | +120 / -2`

- **Added** MongoDB data layer for hosted deployments — player queries routed to MongoDB when available

### `9d43e2c` — Fix hosted deployment: decompress DB to /tmp
`1 file | +11 / -4`

- **Fixed** DB write failure on read-only hosted filesystems — decompress target changed to `/tmp/cricket.db`

### `4e92e42` — Add player classification data files
`2 files | +7474`

- **Added** `data/all_players_full.csv` — full player list with roles
- **Added** `data/unclassified_bowlers.csv` — bowlers missing style classification

### `0421330` — Add compressed DB + auto-decompress on startup
`2 files | +7`

- **Added** `data/cricket.db.gz` — bundled compressed database
- **Added** decompress-on-startup logic to `app.py`

### `57ae46c` — Matchup Lab: replace tabs with selectbox
`1 file | +10 / -10`

- **Changed** Matchup Lab sub-page navigation from tabs → selectbox (better mobile/small-screen support)

### `02d9445` — Fix Predicted Matchup tab not showing
`1 file | +1 / -2`

- **Fixed** button state logic preventing Predicted Matchup tab from rendering

### `0f3f998` — Matchup Lab: Predicted Matchup tab
`1 file | +228 / -2`

- **Added** Predicted Matchup tab — model-predicted outcomes for any batter vs any bowler

### `77d4cbb` — Fix NameError: hoist comparison helpers to module level
`1 file | +57 / -53`

- **Fixed** `NameError` on comparison helpers — hoisted to module scope from inside page function

### `d6af77a` — Matchup Lab: role-filtered dropdowns + career comparison
`1 file | +58 / -5`

- **Added** role-filtered batter/bowler dropdowns (filter by position/style)
- **Added** career comparison panel alongside ball-by-ball matchup data

### `147341a` — Pitch Intelligence: search + venue metadata card
`1 file | +78 / -15`

- **Added** country/city/pitch-type search to Pitch Intelligence page
- **Added** venue metadata card (boundaries, capacity, soil, pitch type)

### `5010489` — Replace venue metadata CSV (218 venues corrected)
`2 files | +436 / -466`

- **Updated** `data/venues_meta.csv` — 218 venue records corrected/replaced

### `4fdfd7c` — Add venue metadata + venues export CSVs
`2 files | +498`

- **Added** `data/venues_meta.csv` — boundaries, capacity, pitch type, soil per venue
- **Added** `data/venues_full.csv` — full venues export

### `3124c39` — Add venue metadata, H2H comparison panel
`3 files | +401 / -9`

- **Added** venue metadata table (boundaries, capacity, pitch type, soil)
- **Added** Head-to-Head venue comparison panel

### `562075b` — Full bowling style + country import (99.8% delivery coverage)
`1 file | +36 / -25`

- **Added** final batch of bowling style classifications — coverage 72% → 99.8% of deliveries

---

## 2026-04-02

### `921aa0a` — Batch 4: 150 bowling style classifications (63% → 72%)
`1 file | +152`

- **Added** 150 user-provided bowling style entries raising delivery coverage from 63% to 72%

### `e322479` — Extend bowling style coverage to 63%, add player role classification
`1 file | +111`

- **Added** additional bowling style entries
- **Added** player role classification data

---

## 2026-04-01

### `2e4902c` — Bowling style matchups, venue dedup, top nav, expanded thesis
`4 files | +1252 / -41`

- **Added** `player_style_matchup` analysis — batter SR/avg/dot%/boundary rate vs 8 bowling style buckets
- **Added** venue deduplication (363 → 248 venues via 90-group redirect map)
- **Added** top navigation bar to dashboard
- **Updated** README thesis with matchup and venue dedup methodology

---

## 2026-03-31

### `fb6e3ab` — Fix identical scores: additive net-edge formula
`1 file | +12 / -14`

- **Fixed** win probability stuck at 50/50 for identical scores — replaced multiplicative with additive net-edge formula

### `28e8a33` — Fix ValueError: guard NaN in _bar, _grade, _xi_rating
`1 file | +15 / -5`

- **Fixed** `ValueError` when NaN values passed to `_bar`, `_grade`, `_xi_rating` display helpers

### `7a95f4d` — Fix 50/50 win prob: factor in bowling quality
`1 file | +66 / -36`

- **Fixed** win probability ignoring opposition bowling quality — now factors in bowler ratings

### `4cfce3e` — Fix team score predictions: anchor to venue average
`1 file | +50 / -16`

- **Fixed** score predictions not anchored to venue baseline — quality-scaled from venue average

### `b7288f4` — Fix inflated team score predictions
`1 file | +12 / -5`

- **Fixed** team score predictions inflated — corrected with position batting probability weights

### `c502494` — Fix predict_bat NameError on Match Predictor
`1 file | +8 / -5`

- **Fixed** `NameError` on `predict_bat` call in Match Predictor page

### `071fff3` — Add XI strength rating cards to Match Predictor
`1 file | +148`

- **Added** per-XI strength rating cards showing batting/bowling composite scores

### `3852e5b` — Fix Match Predictor bugs: bowler slice, chase, win prob, button
`1 file | +61 / -57`

- **Fixed** bowler slice index error
- **Fixed** chase flag not set correctly
- **Fixed** win probability calculation
- **Fixed** predict button state

### `cb39e47` — Enrich GROUND_INFO with pitch/surface/boundary data
`1 file | +277 / -54`

- **Updated** `GROUND_INFO` dict — researched pitch types, surface conditions, boundary dimensions for major venues

### `446370a` — Add GBM score predictions to Match Predictor
`1 file | +98`

- **Added** GBM model predictions integrated into Match Predictor page

### `2f0833d` — Fix SyntaxError: move GROUND_INFO before page routing
`1 file | +99 / -102`

- **Fixed** `SyntaxError` caused by `GROUND_INFO` and pitch helpers defined inside routing block — hoisted to module level

### `4dd2caf` — Add Matchup Lab, XI vs XI Match Predictor, pitch labels
`1 file | +334 / -5`

- **Added** Matchup Lab page — ball-by-ball batter vs bowler analysis
- **Added** XI vs XI Match Predictor page
- **Added** pitch condition labels to Pitch Intelligence

### `9f9bcf1` — Add Matchup Lab — ball-by-ball batter vs bowler
`1 file | +231`

- **Added** initial Matchup Lab implementation

### `8987fe0` — Update bundled DB with 7-league dataset + position-stratified models
`1 file | binary delta`

- **Updated** `data/cricket.db.gz` — rebuilt with 7 leagues and position-stratified GBM models

### `d5fac42` — Position-stratified batting models + expanded README
`13 files | +121 / -27`

- **Added** 4 position-group GBM models (openers 1–2, top order 3–5, lower order 6–8, tail 9–11)
- **Updated** `README.md` with position-model methodology and backtest results

### `293759b` — Optuna hyperparameter tuning + 2024 backtest
`5 files | +524`

- **Added** `scripts/tune_models.py` — Optuna 30-trial search, GBM vs XGBoost on 2024 holdout
- **Added** `scripts/backtest_2024.py` — leave-one-season-out backtest (train ≤2023, test 2024)
- Results: batting R²=0.834, MAE=5.36 runs; bowling R²=0.002 (economy too volatile for career features)

### `a764d84` — Fix venue search excluding low-match grounds
`2 files | +319 / -2`

- **Fixed** venue search dropping grounds with few matches (e.g. VRA Cricket Ground)
- **Added** `data/venues_export.csv`

### `51fd3d3` — Expand to 7 T20 leagues + UUID-based deduplication
`10 files | +216 / -22`

- **Added** BBL, CPL, LPL, MSL to existing T20I/WC/IPL/PSL coverage (7 total)
- **Added** UUID-based player deduplication across tournaments
- **Updated** parser to reject non-T20 match types

---

## 2026-03-30

### `6820c04` — Fix Streamlit 1.55 compatibility
`3 files | +18 / -19`

- **Fixed** API breakages from Streamlit 1.55 upgrade

### `b09d6cd` — Bundle compressed database for instant deploy
`4 files | +32 / -88`

- **Added** compressed DB bundling strategy — no pipeline run needed on cold deploy
- **Removed** auto-bootstrap download logic from entry point

### `ac99aae` — Add comprehensive README with project thesis
`1 file | +270`

- **Added** full `README.md` — thesis, methodology, schema, architecture, deployment docs

### `f0f10c4` — Add Dockerfile
`1 file | +24`

- **Added** `Dockerfile` for Railway/Render/Fly.io deployment

---

## 2026-03-29

### `c360c30` — Initial commit: T20 cricket analytics system
`29 files | +5729`

- **Added** full project skeleton: ingest pipeline, SQLite schema (26 tables), analytics engine, Streamlit dashboard, health monitor, CLI scripts
- Core tables: `players`, `teams`, `venues`, `matches`, `innings`, `deliveries`, `partnerships`
- Analytics: `venue_difficulty` (Bayesian shrinkage), `player_ratings` (z-score → sigmoid), phase splits, chase splits
- Dashboard: Player Explorer, Head-to-Head radar, Pitch Intelligence scatter, Prediction Engine

---

## Project Snapshot

**Dataset (last full ingest):** 5,811 matches · 1,335,728 deliveries · 5,128 players · 248 venues · 8 tournaments · 2004–2026

**Bowling style coverage:** 99.8% of deliveries classified across 8 style buckets (RAF, RAFM, OB, LBG, LAF, LAFM, SLA, LWS)

**Models:** Position-stratified GBM — openers R²=0.925, top order R²=0.867, lower order R²=0.760, tail R²=0.808 (in-sample); 2024 holdout R²=0.834, MAE=5.36 runs

**Active branch:** `concept`
