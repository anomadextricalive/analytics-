# 🏏 Cricket Analytics

A comprehensive T20 cricket analytics and player rating system covering international and domestic leagues from 2004–2026. Built on ball-by-ball data from [cricsheet.org](https://cricsheet.org), with a full data pipeline, pitch regression engine, machine learning prediction models, and two interactive neobrutalist-themed dashboards.

---

## Live Dashboard

Deploy to [Railway](https://railway.app) or [Streamlit Cloud](https://share.streamlit.io) — see [Deployment](#deployment).

---

## Thesis

### The Problem with Traditional Cricket Statistics

Cricket statistics have long suffered from a fundamental bias: they do not account for the conditions under which performances are recorded. A batter averaging 45 at Wankhede Stadium in Mumbai — a flat, high-scoring ground — is not necessarily superior to one averaging 38 at Headingley in Leeds, where seam movement and variable bounce routinely suppress scoring. Traditional averages treat all venues equally. They do not.

T20 cricket compounds this problem. The format's compressed nature means a single explosive innings can distort a career average upward, while a string of low scores on difficult pitches can make a technically excellent player look mediocre in raw numbers. Phase-of-play performance (powerplay vs. middle overs vs. death), batting position, and match situation (chasing a target vs. setting one) all materially affect outcomes — yet none of these are reflected in a batting average or strike rate.

This system is an attempt to build a more honest picture of T20 player quality.

### Methodology

#### 1. Data Foundation

All data is sourced from cricsheet.org, which publishes free, open, ball-by-ball JSON scorecards for T20 matches dating back to 2004. Every delivery in the database contains: runs scored, extras, wicket type, fielders involved, the over and ball number, batting and bowling team, and the match situation at that point (required run rate, current run rate).

The current dataset covers **3,111 matches**, **701,637 deliveries**, **4,256 players**, and **325 venues** across T20 Internationals and the ICC Men's T20 World Cup.

#### 2. Pitch Regression (Venue Difficulty Factor)

The core of the system is a Bayesian shrinkage model that estimates each venue's inherent scoring difficulty — its `bat_factor`.

For each venue with sufficient match data, we compute the raw average first-innings score. We then apply Bayesian shrinkage toward the global mean:

```
shrunk_avg  = (N × raw_avg + W × global_avg) / (N + W)
bat_factor  = shrunk_avg / global_avg
```

Where `N` is the number of innings at that venue and `W = 20` is the prior weight (equivalent to 20 innings of evidence). A `bat_factor > 1.0` indicates a batter-friendly venue; `< 1.0` is bowler-friendly. Bootstrap confidence intervals (1,000 samples) quantify uncertainty — venues with fewer than 5 innings are flagged as low-confidence.

The venue model also tracks:
- **Pace index** — proportion of wickets taken by pace bowlers (proxy for seam/bounce conditions)
- **Spin index** — proportion taken by spinners (proxy for turn and slowness)
- **Boundary rate** — boundaries per ball faced (proxy for ground size and outfield pace)

Player batting and bowling averages are then adjusted by the venue factor of every match in their career, producing `adj_average` and `adj_strike_rate` — statistics that measure performance independent of where matches were played.

#### 3. Phase-Split Analysis

T20 innings are divided into three phases:

| Phase | Overs | Character |
|---|---|---|
| Powerplay | 1–6 | Only 2 fielders outside 30-yard circle; batting-friendly |
| Middle | 7–15 | Field restrictions lifted; dot-ball pressure builds |
| Death | 16–20 | Maximum aggression; specialist finishers and death bowlers |

Each batter's strike rate is computed independently per phase. A player with a modest overall SR of 130 but a death-overs SR of 195 is a specialist finisher — a profile that raw numbers obscure entirely. The system captures this through `pp_sr`, `mid_sr`, and `death_sr` fields, and uses them to produce specialisation scores.

#### 4. Chase vs. First-Innings Splits

Chasing a target introduces a dimension absent from first-innings batting: the required run rate. A batter chasing 185 who scores 60 off 35 balls while the required rate is 11 is under qualitatively different pressure than one posting 60 off 35 in the first innings with no constraint. The system records `required_rr_start` per innings and computes separate `chase_avg` and `first_avg` for every player. These feed directly into the prediction model as features.

#### 5. Player Rating Engine

Player ratings normalise raw performance metrics onto a 0–100 scale using a two-step process:

1. **Z-score** each metric across the population of qualifying players (minimum 10 innings)
2. **Sigmoid transform**: `score = 100 / (1 + exp(−0.8 × z))`

The sigmoid prevents extreme outliers from dominating and produces a scale where 50 represents the population mean, ~75 represents genuinely elite, and 90+ represents all-time great territory.

**Batting rating sub-components:**

| Component | Weight | Metric |
|---|---|---|
| Adjusted average | 30% | Venue-corrected batting average |
| Adjusted strike rate | 20% | Venue-corrected SR |
| Powerplay SR | 10% | SR in overs 1–6 |
| Middle-overs SR | 10% | SR in overs 7–15 |
| Death-overs SR | 10% | SR in overs 16–20 |
| Chase performance | 20% | Chase average relative to first-innings average |

**Bowling rating sub-components:**

| Component | Weight | Metric |
|---|---|---|
| Adjusted economy | 25% | Venue-corrected economy rate |
| Adjusted average | 20% | Venue-corrected bowling average |
| Strike rate | 15% | Balls per wicket |
| Dot ball % | 15% | Percentage of deliveries that are dot balls |
| Powerplay economy | 8% | Economy in overs 1–6 |
| Middle economy | 8% | Economy in overs 7–15 |
| Death economy | 9% | Economy in overs 16–20 |

Beyond the composite rating, specialisation scores are computed for: **opener**, **finisher**, **anchor**, **chase specialist**, **powerplay bowler**, and **death bowler** — enabling nuanced role-based comparisons.

#### 6. Prediction Model

A `GradientBoostingRegressor` (scikit-learn) is trained on all individual innings in the database to predict how many runs a batter will score, or what economy a bowler will concede, in a given match context.

**Batting model features (16):**

- Venue bat factor, boundary rate, pace index
- Batting position (1–11)
- Is-chase flag, required run rate at innings start
- Career adjusted average, adjusted SR, total innings
- Chase average, first-innings average, chase SR
- Phase-split SRs (powerplay, middle, death)
- Tournament encoding

**Model performance on T20I data:**
- Batting: R² = 0.877, MAE = 4.7 runs/innings
- Bowling: R² = 0.205, MAE = 2.17 economy (cricket bowling is inherently high-variance)

The prediction engine also generates 80% confidence intervals via bootstrap sampling (300 iterations), which reflect the natural randomness of T20 cricket — even the best model cannot predict a specific innings with precision, and the CI makes that uncertainty explicit.

#### 7. Stored Player Metadata

Beyond aggregated statistics, the system stores granular per-player breakdowns modelled on professional cricket statistics databases:

- **By season** — year-by-year progression
- **By opponent** — head-to-head records vs. each country/team
- **By batting position** — performance at each position in the order
- **By match result** — stats in wins vs. losses
- **Dismissal analysis** — breakdown of how each batter gets out (caught, bowled, LBW, run out, etc.)
- **Bowling dismissal analysis** — wicket types per bowler
- **Fielding stats** — catches, most catches in a single innings
- **Milestones** — every half-century, century, 30, duck, and 4-wicket haul with venue and opposition
- **Player of the Match awards**
- **Partnership data** — ball-by-ball partnership tracking

This gives the system the depth of a dedicated cricket statistics platform while remaining entirely derived from open-source ball-by-ball data.

---

## Architecture

```
cricket_analytics/
├── app.py                      # Streamlit Cloud / Railway entry point
├── config.py                   # Central config: paths, phases, thresholds
├── requirements.txt
├── Dockerfile                  # For Railway/Render/Fly.io
│
├── src/
│   ├── db/
│   │   └── schema.py           # 26-table SQLAlchemy ORM (SQLite)
│   ├── ingest/
│   │   ├── downloader.py       # Cricsheet zip downloader
│   │   └── parser.py           # Fast bulk parser (~130 matches/sec)
│   └── analytics/
│       ├── pitch.py            # Bayesian venue difficulty estimation
│       ├── metrics.py          # Rebuilds all 12 aggregate tables
│       ├── rating.py           # Z-score → sigmoid player ratings
│       └── model.py            # GBM prediction models
│
├── src/dashboard/
│   ├── app.py                  # Main analytics dashboard (4 pages)
│   └── health.py               # Backend health monitor dashboard
│
├── scripts/
│   ├── pipeline.py             # CLI: download / ingest / venue / metrics / ratings / all
│   ├── query.py                # CLI: search / profile / compare / leaderboard
│   └── inspect_db.py           # Print all 26 tables with row counts
│
└── data/
    ├── cricket.db              # SQLite database (not in git — auto-built)
    ├── raw/                    # Downloaded JSON files (not in git)
    └── models/                 # Trained GBM models (committed to git)
```

### Database Schema (26 tables)

| Layer | Tables |
|---|---|
| Core | `players`, `teams`, `venues`, `matches`, `innings`, `deliveries`, `partnerships` |
| Per-innings | `player_innings`, `player_bowling_innings` |
| Career aggregates | `player_career_bat`, `player_career_bowl` |
| Breakdowns | `player_position_bat`, `player_chase_bat`, `player_venue_bat`, `player_venue_bowl` |
| Metadata | `player_perf_by_opponent`, `player_perf_by_season`, `player_perf_by_team`, `player_perf_by_result` |
| Analysis | `player_dismissal_analysis`, `player_bowling_dismissal_analysis`, `player_fielding_stats` |
| Events | `player_milestones`, `player_of_match_awards` |
| Model output | `venue_difficulty`, `player_ratings` |

### Parser Performance

The ingestion pipeline is optimised for bulk loading:

- **In-memory registry cache** — all player/team/venue ID lookups are resolved from a Python dict after the first scan, eliminating per-delivery database queries
- **Bulk inserts** — `session.bulk_insert_mappings()` batches all deliveries per match into a single INSERT rather than one ORM call per row
- **SQLite PRAGMAs** — WAL mode, `synchronous=NORMAL`, 64MB cache, 256MB mmap
- **Result**: ~130 matches/second (vs. ~1.2 matches/second with naive ORM inserts — a 100× improvement)

---

## Dashboards

### Analytics Dashboard (`src/dashboard/app.py`)

Four pages, all neobrutalist-themed (Space Mono + Space Grotesk, #FFE500 yellow, hard box shadows):

**Player Explorer** — Searchable, filterable table of all 4,256 players with rating bars. Click any player for a drill-down showing: season-by-season trend chart, phase SR bars, by-opponent breakdown, milestone log, and venue performance map.

**Head-to-Head** — Select 2–8 players for a radar chart across 6 dimensions (bat rating, bowl rating, opener score, finisher score, chase score, death bat score), phase SR comparison bars, chase vs. first-innings split, and shared-venue performance comparison.

**Pitch Intelligence** — Scatter plot of all 325 venues (bat_factor vs. boundary_rate, sized by match count). Venue deep-dive with top 15 batters and bowlers at that ground.

**Prediction Engine** — Train/retrain the GBM models with a single button. Feature importance charts. Live prediction: select any player + venue → predicted runs (first innings and chasing), 80% CI, predicted economy, comparison against historical actuals at that venue. Multi-player venue comparison table.

### Health Dashboard (`src/dashboard/health.py`)

Backend monitoring dashboard showing: 8 overview stat cards (players, matches, deliveries, venues, ratings, model status), fill-bar audit of all 26 tables, tournament coverage, matches-per-year bar chart, pipeline stage checklist, and command reference.

---

## Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download and ingest T20I data (~60 seconds)
python scripts/pipeline.py all --tournaments t20i_male

# 3. Or run the full pipeline (all leagues, ~10 minutes)
python scripts/pipeline.py all

# 4. Launch the analytics dashboard
streamlit run src/dashboard/app.py --server.port 8502

# 5. Launch the health monitor (separate terminal)
streamlit run src/dashboard/health.py --server.port 8501
```

---

## Deployment

### Railway (recommended)

1. Push this repo to GitHub
2. [railway.app](https://railway.app) → New Project → Deploy from GitHub repo
3. Select this repo — Railway auto-detects the Dockerfile
4. Live URL in ~3 minutes

The app auto-bootstraps the database on first run (downloads T20I data from cricsheet.org, ~2 minutes).

### Streamlit Cloud

1. [share.streamlit.io](https://share.streamlit.io) → New app
2. Repo: `anomadextricalive/analytics-`, Branch: `main`, Main file: `app.py`
3. Deploy

---

## Data Sources

All data is sourced from **[cricsheet.org](https://cricsheet.org)** — a free, open-licence ball-by-ball cricket data repository maintained by Stephen Rushe. No scraping is involved; data is downloaded as bulk ZIP archives via their public downloads API.

Tournaments currently covered: T20 Internationals (male), ICC Men's T20 World Cup, IPL, PSL, BBL, CPL, LPL, MSL, and others via `python scripts/pipeline.py download`.

---

## License

MIT
