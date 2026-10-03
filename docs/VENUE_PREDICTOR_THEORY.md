# Venue Predictor: theory, evidence and limits

Written 2026-10-03. Code: `src/analytics/venue_model.py` (model), `scripts/build_venue_model.py` (backtest and build),
`src/dashboard/venue_predictor.py` (tab), `scripts/geocode_venues.py` (altitude and coordinates).

## 1. What question this answers

> "If this player bats (or bowls) at this stadium, what score (or economy) should we expect, and how much of that is the ground?"

Output for a batter: expected runs, his own baseline, the **venue lift** (expected minus baseline) with an uncertainty band, a typical
range (10th to 90th percentile of an innings), and the chance of 30+ and 50+. For a bowler: expected economy, baseline, lift, range.

**The honest headline.** A ground is a real but small effect. A ground that scores 10% above average moves a typical batter's expected runs by
about 4%, roughly one or two runs. One T20 innings has a spread of about 19 runs, so the number is a tilt on a very wide distribution, not a forecast.

## 2. Why not just use raw venue averages

A venue's raw average score mixes the ground with who played on it. Raw venue scoring and the adjusted ground effect correlate at only
**0.62** over 221 grounds (at least 20 innings), so about 38% of what looks like "ground" is the teams, league and era. Our first attempt used raw
averages and ranked small associate-nation grounds (Malkerns, Jimmy Powell Oval) as best for stars who had never played there: weak bowling, not flat pitches.

## 3. The model, step by step

All features use only information available before the innings predicted (point-in-time). Every layer is backtested before it is used.

### Step 1: the ground effect, adjusted for who plays
Each team innings (23,142 men's T20 innings, T10 and Hundred excluded) is modelled as

    runs per 120 balls  =  batting side + bowling side + league + year trend + innings number + GROUND + noise

fitted as a ridge regression on one-hot effects (ridge so small teams and grounds do not get extreme coefficients). The GROUND coefficient is the
part of scoring left after removing the two teams, the league and the year. The same is done for wickets, boundary share, powerplay rate and death-over rate.
Using runs per 120 balls (not per innings) removes the effect of shortened innings.

**Check at team level, where the signal is cleaner** (test: 5,176 innings from 2025): predicting a team's total from teams + league + year gives R² 0.153;
adding the ground effect gives **0.166** (MAE 28.42 to 28.25 runs per 120 balls). Small, real.

### Step 2: a physical prior (altitude, boundary size, and friends)
The ground effects are regressed on physical traits: altitude, mean boundary size, straight-vs-square asymmetry, latitude, capacity, floodlights and
pitch type (drop-in or synthetic), with indicators for missing values. Weighted ridge, regularisation chosen by cross-validation grouped by ground,
so the score is on grounds the model has not seen.

| Trait | Correlation with ground effect | Effect per +1 sd of the trait, runs per 120 balls |
|---|---|---|
| Altitude | **+0.30** (195 grounds) | **+3.1** |
| Boundary size | **-0.22** (139 grounds) | **-2.2** |
| Boundary asymmetry | -0.16 (139 grounds) | -0.4 |
| Synthetic pitch, latitude, floodlights, capacity | | +1.0, +0.9, +0.9, +0.7 |

- **Bigger boundaries go with lower scoring, higher altitude with higher scoring.** Both match cricket intuition: thin air carries the ball further, and long boundaries cost sixes.
- Of the grounds with at least 60 innings, the highest adjusted scoring effects are Kathmandu (+19.1, 1,343 m), Centurion (+18.7, 1,432 m) and Kimberley (+17.6, 1,224 m); the lowest are Brian Lara Stadium (-16.5), Providence (-15.5), Melbourne (-14.1) and Mirpur (-13.1).
- **How much it explains:** cross-validated R² on unseen grounds is **0.10**. These traits explain about a tenth of how grounds differ. The rest is pitch behaviour we cannot measure from outside (bounce, grip, outfield speed).

### Step 3: partial pooling (empirical Bayes)
Few grounds have many innings. Each ground's final effect blends what happened there with what its physical traits predict:

    final = prior + w x (observed - prior),     w = tau^2 / (tau^2 + se^2)

`tau` (about 9.7 runs per 120 balls) is the real spread between grounds around the prior; `se` is the uncertainty of the observed effect. Grounds with
many innings keep their own number (e.g. weight 0.94 at Wankhede); grounds with few lean on the prior; grounds with no matches use the prior alone
(30 of 472). `se` is the larger of `sd/sqrt(n)` and a **match-level bootstrap** standard deviation. The bootstrap matters because a tiny team that only plays at one
ground cannot be separated from that ground, which `sd/sqrt(n)` ignores.

### Step 4: from a ground effect to a player expectation
1. **Baseline.** Gradient boosting on the player's own history (expanding, only earlier innings): shrunk career average and strike rate, phase strike rates, batting position, innings, required rate,
   league and year. No venue information. For bowlers: shrunk economy, dot-ball share, phase economy, league, year.
2. **Venue index** `x = ground effect / league mean team score`.
3. **Expectation** `= baseline x (1 + elasticity x x)`.

**Elasticity** is estimated from data (closed-form least squares of actual on baseline x index, using out-of-fold values): **0.44 for batters, 0.33 for bowlers**.
A batter's runs move less than the team's: a ground 10% above average lifts a batter by about 4.4%. This is the single most important number to explain to anyone reading the output.

### Step 5: range and probabilities
The spread of one innings is large and skewed, so it is not modelled with a normal curve. We take the **empirical distribution of real innings by players with the nearest expectation**
(test period, 15 bins with finer bins at the top) and rescale it to this player's level (multiplicatively for runs, additively for economy). Percentiles and P(30+), P(50+) are read from it.

## 4. Evidence: backtest

Train before 2025-01-01, test on 2025 onwards (batters 42,580 innings, bowlers 30,831 spells). Paired bootstrap, 1,000 resamples, 95% intervals. "Better" means lower error.

| | Baseline (no venue) | Baseline x venue index | Change (95% interval) |
|---|---|---|---|
| Batters, RMSE (runs) | 18.942 | 18.934 | -0.008 (-0.012 to -0.005), significant |
| Batters, MAE (runs) | 13.448 | 13.455 | +0.007 (+0.004 to +0.011), slightly worse |
| Bowlers, RMSE (economy) | 3.367 | 3.361 | -0.006 (-0.007 to -0.004), significant |
| Bowlers, MAE (economy) | 2.568 | 2.566 | -0.002 (-0.003 to -0.001), significant |
| Bowlers, grounds never seen in training, RMSE | | | -0.004 (-0.006 to -0.002), significant |

How to read this: the venue adjustment is **statistically real but tiny**. For bowlers it helps in both error measures; for batters it helps rare big innings
(RMSE) and slightly hurts the typical innings (MAE), so we cannot claim better batter accuracy. We tested richer alternatives first (gradient boosting and CatBoost with venue columns,
metadata-only imputers, player-by-venue history and personal sensitivity to ground type) and none beat this simple form, so it is what ships.

## 5. What this is not

- **Not a forecast.** The 10 to 90 percent range for a good batter is roughly 2 to 90 runs.
- **Not pitch behaviour.** Bounce, seam, spin, outfield speed and dew are not in our data. The ground effect is their combined, observed footprint on scoring.
- **Past form at one ground is shown for context only.** In our test, a player's history at a specific ground did not improve predictions.
- **League and year drive baselines.** Scoring rose in 2025, so a bowler's baseline in the IPL can be well above his career figure. The league selector exists for this reason.
- **Small grounds are guesses.** 267 of 472 grounds have fewer than 30 innings (30 of them none) and lean heavily on the prior; the page says so.

## 6. Data and provenance

- Matches and balls: Cricsheet, plus the project's own enrichment. Men's T20 only (T10, Hundred and Legends excluded).
- Altitude and coordinates: Open-Meteo's free geocoder (GeoNames-derived), by city name. 418 of 472 grounds resolved; `data/venue_geo.csv` records the match quality for each (`confidence`),
  `data/venue_geo_raw.csv` is the unprocessed result. Altitude is city-level, so accurate to roughly 100 m.
- Boundary sizes, capacity, pitch type and floodlights: project venue records, filled for 248 of 472 grounds.
- Tables written to the main database: `venue_geo`, `venue_profile` (per-ground effects and uncertainty), `venue_model_meta` (elasticities, prior, backtest, distribution tables).
  Models: `data/models/venue_models.joblib`. Backtest detail: `data/venue_model_backtest.json`.

## 7. Rebuild and update

    python scripts/geocode_venues.py            # only when new grounds appear
    python scripts/build_venue_model.py          # backtest only
    python scripts/build_venue_model.py --build  # refit on all data and write tables and models

Then checkpoint the WAL and rebuild `data/cricket.db.gz` so the hosted app sees the new tables.

## 8. Next candidates (each must pass the same backtest before it is shown)

1. **Match-day weather** (temperature, humidity, dew point, cloud, wind) and day or night, from a free historical weather API (tested). Physically motivated, especially for swing, dew and chasing.
2. **Pitch fingerprint from ball-by-ball:** bounce and carry (caught-behind and slip share), seam and swing (early pace wickets), spin turn, first-versus-second-innings behaviour.
3. **Test and ODI data as ground knowledge.** Cricsheet has Tests and ODIs. Use format-relative indices, not raw scores, and first test whether a Test and ODI profile predicts the T20 effect on held-out grounds.
4. Stadium-level (not city-level) coordinates for altitude and orientation.
