# Cricket Analytics — Project Updates

Reverse-chronological log of significant changes. Full thesis and methodology in [README.md](README.md).

---

## 2026-05-16 — Cosmic UI Theme

**Branch:** `concept`

Dashboard theme overhauled from neobrutalist (light, Space Mono, #FFE500) to **Cosmic** (dark void, Inter/Oswald/JetBrains Mono, mint/amber/magenta palette).

**What changed in `src/dashboard/app.py`:**
- CSS variables rewritten: `--cosmic-void-*` surface scale, `--cosmic-ink-*` text scale, `--cosmic-mint-*` / `--cosmic-amber-*` / `--cosmic-magenta-*` accent palette
- Background: deep `#0A0E1C` void with nebula bloom (two layered radial gradients — teal top-left, magenta bottom-right)
- Fonts: Oswald (display/headings) · Inter (body) · JetBrains Mono (code/numbers)
- Border/elevation tokens replacing hard box-shadows
- Compat aliases (`--bg`, `--surface`, `--accent`, etc.) kept so downstream component CSS requires no changes
- No functional/logic changes — visual redesign only

---

## 2026-05-?? — Cricket GPT Insight Engine + Admin Page

**Branch:** `concept`
**Commit:** `5cbe848`

- Added Cricket GPT chat interface with persistent history
- Admin page added to dashboard
- Chat backed by SQL query generation against the 26-table SQLite schema

---

## 2026-05-?? — Player Similarity & Form Analytics

**Commit:** `404581e` → `c1f9cbb`

- `player_similarity` table: cosine similarity across rating/phase vectors
- `player_form` table: rolling-window recent-form scoring
- By Opponent sub-filters added to player drill-down
- Similarity and form panels fixed for stale cache issue

---

## Earlier — Core System

See [README.md § Thesis](README.md#thesis) for full methodology. Summary:

| Layer | What it does |
|---|---|
| Ingest | Cricsheet ball-by-ball JSON → SQLite (26 tables, ~130 matches/sec) |
| Venue model | Bayesian shrinkage `bat_factor` per venue; dedup 363→248 grounds |
| Phase splits | PP / Middle / Death SR for every batter and bowler |
| Chase splits | Separate chase/first-innings avg + SR |
| Style matchups | 8 bowling style buckets, min 24 balls, for every batter |
| Rating engine | Z-score → sigmoid → 0–100 composite + 6 specialisation scores |
| Prediction | Position-stratified GBM (4 groups); 2024 holdout R²=0.834, MAE=5.36 runs |
| Dashboard | Streamlit: Player Explorer · Head-to-Head · Pitch Intelligence · Prediction Engine |

**Dataset as of last full ingest:** 5,811 matches · 1,335,728 deliveries · 5,128 players · 248 venues · 8 tournaments (2004–2026)
