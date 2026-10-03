"""Venue-aware expectation: how much a ground moves a player's expected runs (batters) or economy (bowlers).

Theory in one paragraph (full write-up: docs/VENUE_PREDICTOR_THEORY.md):
  1. Venue effect. Team innings (runs per 120 balls) are modelled as batting side + bowling side + league + year + innings number + VENUE
     (ridge, crossed effects). The venue coefficient is what remains after removing who played, i.e. the ground's own effect.
  2. Physics prior. Venue effects are regressed on altitude, boundary size/asymmetry, latitude, pitch type, floodlights, capacity.
  3. Partial pooling (empirical Bayes). Each venue's final effect = prior + w * (observed - prior), w = tau^2 / (tau^2 + se^2).
     Grounds with many innings keep their own number; grounds with few (or none) lean on the prior. se shrinks with sqrt(n).
  4. Player expectation = the player's own baseline (gradient boosting on his expanding history, no venue information)
     x (1 + elasticity * venue_effect / league_mean). The elasticity is estimated from data, not assumed.
  5. Every layer is backtested on a time split with paired bootstrap intervals before it is allowed into the app.
All features use only information available before the innings they predict (point-in-time)."""
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingRegressor as HGR, HistGradientBoostingClassifier as HGC
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder

import src.analytics.model as M

FILTER = ("m.tournament NOT LIKE 't10%' AND m.tournament NOT LIKE 'legends%' AND m.tournament != 'hundred_male' "
          "AND m.venue_id IS NOT NULL")
CATS = ["bt", "wt", "tournament", "venue_id"]
TARGETS = ("runs", "wkts", "bnd", "pp_rr", "death_rr")
PRIOR_FEATURES = ["elevation_m", "boundary_mean_m", "boundary_asym_m", "abs_lat", "capacity", "floodlights", "pt_drop_in", "pt_synthetic"]
BAT_BASE = ["prior_innings", "prior_runs_per_inn", "prior_sr", "prior_pp_sr", "prior_mid_sr", "prior_death_sr",
            "batting_position", "is_chase", "req_rr", "year", "innings_number", "tournament"]
BOWL_BASE = ["prior_spells", "prior_econ", "prior_dot_pct", "prior_pp_econ", "prior_mid_econ", "prior_death_econ",
             "year", "innings_number", "tournament"]
RIDGE_ALPHA = 1.0


# ───────────────────────── data ─────────────────────────
def load_team_innings(q):
    """One row per team innings with what is needed to estimate venue effects. `q(sql)` returns a DataFrame."""
    inn = q(f"""SELECT i.id AS innings_id, i.match_id, i.innings_number, i.batting_team_id AS bt, i.bowling_team_id AS wt,
        i.total_runs, i.total_wickets, i.total_balls, m.match_date AS date, m.tournament, m.venue_id,
        (SELECT SUM(fours+sixes) FROM player_innings p WHERE p.innings_id=i.id) AS bnd,
        (SELECT SUM(pp_runs) FROM player_innings p WHERE p.innings_id=i.id) AS ppr,
        (SELECT SUM(pp_balls) FROM player_innings p WHERE p.innings_id=i.id) AS ppb,
        (SELECT SUM(death_runs) FROM player_innings p WHERE p.innings_id=i.id) AS dr,
        (SELECT SUM(death_balls) FROM player_innings p WHERE p.innings_id=i.id) AS db
        FROM innings i JOIN matches m ON m.id=i.match_id WHERE {FILTER} AND i.total_balls>=30 AND i.total_runs IS NOT NULL""")
    inn["date"] = pd.to_datetime(inn["date"]); inn = inn.sort_values(["date", "innings_id"]).reset_index(drop=True)
    inn["year"] = inn.date.dt.year + inn.date.dt.dayofyear / 366 - 2015
    inn["t_runs"] = inn.total_runs / inn.total_balls * 120
    inn["t_wkts"] = inn.total_wickets / inn.total_balls * 120
    inn["t_bnd"] = inn.bnd / inn.total_balls * 100
    inn["t_pp_rr"] = inn.ppr / inn.ppb.clip(lower=1) * 6
    inn["t_death_rr"] = inn.dr / inn.db.clip(lower=1) * 6
    return inn


def load_venue_meta(q, geo: pd.DataFrame | None):
    meta = q("SELECT id AS venue_id, name, city, country, pitch_type, boundary_straight_m, boundary_square_m, capacity, floodlights FROM venues")
    if geo is not None and len(geo):
        g = geo[["venue_id", "lat", "lon", "elevation_m", "confidence"]].copy()
        g.loc[g.confidence.astype(str).str.startswith(("none", "error", "weak")), ["lat", "lon", "elevation_m"]] = np.nan
        meta = meta.merge(g[["venue_id", "lat", "lon", "elevation_m"]], on="venue_id", how="left")
    else:
        meta["lat"] = meta["lon"] = meta["elevation_m"] = np.nan
    meta["boundary_mean_m"] = (meta.boundary_straight_m + meta.boundary_square_m) / 2
    meta["boundary_asym_m"] = (meta.boundary_straight_m - meta.boundary_square_m).abs()
    meta["abs_lat"] = meta.lat.abs()
    meta["pt_drop_in"] = (meta.pitch_type == "Drop-in").astype(float)
    meta["pt_synthetic"] = (meta.pitch_type == "Synthetic").astype(float)
    return meta


# ───────────────────────── step 1: venue effect adjusted for who plays ─────────────────────────
def _enc(df): return {c: OneHotEncoder(handle_unknown="ignore").fit(df[[c]].astype(str)) for c in CATS}


def _design(df, enc):
    return sp.hstack([enc[c].transform(df[[c]].astype(str)) for c in CATS] +
                     [sp.csr_matrix(df[["year", "innings_number"]].values.astype(float))]).tocsr()


def _venue_coefs(enc, coef):
    off = sum(len(enc[c].categories_[0]) for c in CATS if c != "venue_id")
    cats = enc["venue_id"].categories_[0]
    return dict(zip([int(x) for x in cats], coef[off:off + len(cats)]))


def fit_venue_effects(inn: pd.DataFrame) -> tuple[dict, float]:
    """Ridge on crossed effects for each target; returns {target: {venue_id: coef}} and the residual sd of the runs model."""
    enc = _enc(inn); X = _design(inn, enc); out, sd = {}, None
    for t in TARGETS:
        r = Ridge(alpha=RIDGE_ALPHA).fit(X, inn["t_" + t].values); out[t] = _venue_coefs(enc, r.coef_)
        if t == "runs": sd = float(np.std(inn["t_runs"].values - r.predict(X)))
    return out, sd


def bootstrap_se(inn: pd.DataFrame, B=40, seed=0) -> pd.Series:
    """Empirical sd of each venue's run effect over match-level bootstrap refits. Unlike sd/sqrt(n) it captures collinearity
    (a tiny team that only plays at one ground cannot be told apart from that ground)."""
    rng = np.random.default_rng(seed); mids = inn.match_id.unique(); by = {m: g.index.values for m, g in inn.groupby("match_id")}; coefs = []
    for _ in range(B):
        pick = rng.choice(mids, len(mids)); idx = np.concatenate([by[m] for m in pick]); sample = inn.loc[idx].reset_index(drop=True)
        enc = _enc(sample); r = Ridge(alpha=RIDGE_ALPHA).fit(_design(sample, enc), sample["t_runs"].values); coefs.append(pd.Series(_venue_coefs(enc, r.coef_)))
    D = pd.concat(coefs, axis=1); return D.std(axis=1, skipna=True).where(D.notna().sum(axis=1) >= B * 0.6)


def cross_fit_effects(inn: pd.DataFrame, split: pd.Timestamp) -> pd.DataFrame:
    """Per-innings venue effects with no look-ahead: train innings get out-of-fold (by match) values, test innings the train-fit values."""
    trn = (inn.date < split).values; res = {t: np.zeros(len(inn)) for t in TARGETS}
    tr_idx = np.where(trn)[0]
    for a, b in GroupKFold(5).split(inn.iloc[tr_idx], groups=inn.iloc[tr_idx].match_id):
        fit_i, ho_i = tr_idx[a], tr_idx[b]; eff, _ = fit_venue_effects(inn.iloc[fit_i])
        for t in TARGETS: res[t][ho_i] = inn.iloc[ho_i].venue_id.map(eff[t]).fillna(0).values
    eff, _ = fit_venue_effects(inn[trn])
    for t in TARGETS: res[t][~trn] = inn[~trn].venue_id.map(eff[t]).fillna(0).values
    return pd.DataFrame({"av_" + t: v for t, v in res.items()}, index=inn.index)


# ───────────────────────── step 2: physics prior ─────────────────────────
def _prior_matrix(meta, med=None):
    X = meta[PRIOR_FEATURES].copy(); med = med if med is not None else X.median()
    for c in ["elevation_m", "boundary_mean_m", "boundary_asym_m", "abs_lat", "capacity"]:
        X[c + "_missing"] = X[c].isna().astype(float)
    return X.fillna(med), med


def fit_prior(meta: pd.DataFrame, eff: dict, n: pd.Series, min_n=20):
    """Weighted ridge of venue effect on physical traits. Returns predictor, CV R2 on unseen venues, and standardized coefficients."""
    v = pd.DataFrame({"venue_id": list(eff["runs"]), "y": list(eff["runs"].values())}).merge(meta, on="venue_id")
    v["n"] = v.venue_id.map(n).fillna(0); v = v[v.n >= min_n]
    X, med = _prior_matrix(v); mu, sd = X.mean(), X.std().replace(0, 1); Z = (X - mu) / sd; w = v.n.values / (v.n.values + 25.0)
    best = None
    for a in (3, 10, 30, 100, 300):
        oof = np.zeros(len(v))
        for tr_i, te_i in GroupKFold(5).split(Z, groups=v.venue_id):
            oof[te_i] = Ridge(alpha=a).fit(Z.iloc[tr_i], v.y.iloc[tr_i], sample_weight=w[tr_i]).predict(Z.iloc[te_i])
        ss = np.average((v.y - oof) ** 2, weights=w); st = np.average((v.y - np.average(v.y, weights=w)) ** 2, weights=w); r2 = 1 - ss / st
        if best is None or r2 > best[1]: best = (a, r2)
    a, r2 = best; mdl = Ridge(alpha=a).fit(Z, v.y, sample_weight=w)
    coefs = pd.Series(mdl.coef_, index=Z.columns).sort_values(key=abs, ascending=False)
    state = dict(cols=list(Z.columns), mu=mu.to_dict(), sd=sd.to_dict(), med=med.to_dict(), coef=coefs.to_dict(), intercept=float(mdl.intercept_), alpha=a)
    predict = lambda m: prior_from_state(state, m)
    return predict, float(r2), coefs, dict(alpha=a, n_venues=int(len(v)), intercept=float(mdl.intercept_), state=state)


def _z(state, meta):
    Xm, _ = _prior_matrix(meta, pd.Series(state["med"]))
    return (Xm[state["cols"]] - pd.Series(state["mu"])[state["cols"]]) / pd.Series(state["sd"])[state["cols"]]


def prior_from_state(state, meta) -> np.ndarray:
    """Physics-prior venue effect (runs per 120 balls above the league average) from a saved prior state."""
    return _z(state, meta).values @ np.array([state["coef"][c] for c in state["cols"]]) + state["intercept"]


def prior_breakdown(state, meta_row: pd.DataFrame) -> dict:
    """How much each physical trait moves this venue's prior, in runs per 120 balls (one-row frame in)."""
    z = _z(state, meta_row).iloc[0]; return {c: float(z[c] * state["coef"][c]) for c in state["cols"] if not c.endswith("_missing")}


# ───────────────────────── step 3: partial pooling ─────────────────────────
def pool(meta: pd.DataFrame, eff: dict, n: pd.Series, sd_resid: float, prior_fn, se_boot: pd.Series | None = None) -> tuple[pd.DataFrame, float]:
    """Empirical-Bayes shrinkage of every venue toward its physics prior. Returns the profile table and tau (between-venue sd)."""
    t = meta[["venue_id"]].copy(); t["n"] = t.venue_id.map(n).fillna(0).astype(int)
    t["prior"] = prior_fn(meta); t["obs"] = t.venue_id.map(eff["runs"])
    t["se"] = sd_resid / np.sqrt(t.n.clip(lower=1) + RIDGE_ALPHA)
    if se_boot is not None: t["se"] = np.maximum(t["se"], t.venue_id.map(se_boot).fillna(0))     # whichever is more cautious
    seen = t[(t.n >= 10) & t.obs.notna()]
    tau2 = max(float(np.var(seen.obs - seen.prior) - np.mean(seen.se ** 2)), 1.0)
    w = np.where(t.n > 0, tau2 / (tau2 + t.se ** 2), 0.0)
    t["final"] = t.prior + w * (t.obs.fillna(t.prior) - t.prior); t["weight_obs"] = w
    t["se_final"] = np.sqrt(np.where(t.n > 0, 1.0 / (1.0 / tau2 + 1.0 / t.se ** 2), tau2))
    for k in ("wkts", "bnd", "pp_rr", "death_rr"): t["eff_" + k] = t.venue_id.map(eff[k])
    return t, float(np.sqrt(tau2))


# ───────────────────────── step 4: player baseline x venue index ─────────────────────────
def _hgr(): return HGR(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0,
                       random_state=0, categorical_features="from_dtype")


def bat_frame(q, pid=None):
    bat = q(f"""SELECT pi.batter_id AS player_id, pi.match_id, pi.innings_id, m.match_date, m.tournament, m.venue_id, i.innings_number,
        pi.runs, pi.balls_faced, COALESCE(pi.batting_position,5) AS batting_position, CAST(pi.is_chase AS INTEGER) AS is_chase,
        COALESCE(pi.required_rr_start,0) AS req_rr, pi.pp_runs, pi.pp_balls, pi.mid_runs, pi.mid_balls, pi.death_runs, pi.death_balls
        FROM player_innings pi JOIN matches m ON m.id=pi.match_id JOIN innings i ON i.id=pi.innings_id
        WHERE pi.balls_faced>=1 AND {FILTER} {"AND pi.batter_id = :pid" if pid is not None else ""}
        ORDER BY pi.batter_id, m.match_date, pi.innings_id""", **({"pid": int(pid)} if pid is not None else {}))
    bat["match_date"] = pd.to_datetime(bat["match_date"]); return bat.reset_index(drop=True)


def bowl_frame(q, pid=None):
    bw = q(f"""SELECT pb.bowler_id AS player_id, pb.match_id, pb.innings_id, m.match_date, m.tournament, m.venue_id, i.innings_number,
        pb.balls_bowled, pb.runs_conceded, pb.dot_balls, pb.pp_balls, pb.pp_runs, pb.mid_balls, pb.mid_runs, pb.death_balls, pb.death_runs
        FROM player_bowling_innings pb JOIN matches m ON m.id=pb.match_id JOIN innings i ON i.id=pb.innings_id
        WHERE pb.balls_bowled>=1 AND {FILTER} {"AND pb.bowler_id = :pid" if pid is not None else ""}
        ORDER BY pb.bowler_id, m.match_date, pb.innings_id""", **({"pid": int(pid)} if pid is not None else {}))
    bw["match_date"] = pd.to_datetime(bw["match_date"]); return bw.reset_index(drop=True)


_PLACE = {"venue_bat_factor": 1.0, "boundary_rate": 0.12, "pace_index": 0.5}


def bat_matrix(bat, pri):
    d = bat.assign(**_PLACE)
    X = M._bat_matrix(d, pri).drop(columns=list(_PLACE)).reset_index(drop=True)
    X["year"] = (bat.match_date.dt.year + bat.match_date.dt.dayofyear / 366).values
    X["innings_number"] = bat.innings_number.values; X["tournament"] = bat.tournament.astype("category").values
    return X


def bowl_matrix(bw, pri):
    d = bw.assign(**_PLACE)
    X = M._bowl_matrix(d, pri).drop(columns=list(_PLACE)).reset_index(drop=True)
    X["year"] = (bw.match_date.dt.year + bw.match_date.dt.dayofyear / 366).values
    X["innings_number"] = bw.innings_number.values; X["tournament"] = bw.tournament.astype("category").values
    return X


def bat_priors(bat, upto):
    dummy = pd.DataFrame({c: [0] for c in ["runs_conceded", "balls_bowled", "dot_balls", "pp_runs", "pp_balls", "mid_runs", "mid_balls", "death_runs", "death_balls"]})
    return M._fit_priors(bat[bat.match_date < upto], dummy)


def bowl_priors(bw, upto):
    d = pd.DataFrame({"runs": [1.0], "balls_faced": [1], "pp_runs": [0], "pp_balls": [1], "mid_runs": [0], "mid_balls": [1], "death_runs": [0], "death_balls": [1]})
    return M._fit_priors(d, bw[bw.match_date < upto])


def fit_elasticity(y, base, x):
    """argmin_e sum (y - base*(1 + e*x))^2, closed form."""
    z = base * x; return float((z * (y - base)).sum() / max((z * z).sum(), 1e-9))


def paired_boot(y, pa, pb, mask, B=1000, seed=0):
    """Bootstrap of (model B - model A) in MAE and RMSE; negative = B better."""
    rng = np.random.default_rng(seed); idx = np.where(mask)[0]
    ea, eb = np.abs(y[idx] - pa[idx]), np.abs(y[idx] - pb[idx]); sa, sb = (y[idx] - pa[idx]) ** 2, (y[idx] - pb[idx]) ** 2
    dm, ds = [], []
    for _ in range(B):
        j = rng.integers(0, len(idx), len(idx)); dm.append(eb[j].mean() - ea[j].mean()); ds.append(np.sqrt(sb[j].mean()) - np.sqrt(sa[j].mean()))
    f = lambda a: dict(mean=float(np.mean(a)), lo=float(np.percentile(a, 2.5)), hi=float(np.percentile(a, 97.5)))
    return dict(dMAE=f(dm), dRMSE=f(ds))


def score(y, p, m):
    from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE, r2_score as R2
    return dict(n=int(m.sum()), mae=float(MAE(y[m], p[m])), rmse=float(np.sqrt(MSE(y[m], p[m]))), r2=float(R2(y[m], p[m])))


# ───────────────────────── prediction (used by the dashboard) ─────────────────────────
from pathlib import Path
MODELS_PATH = Path(__file__).parents[2] / "data" / "models" / "venue_models.joblib"
FINGERPRINT = [("final_effect", "Scoring (runs/120 balls vs league)"), ("eff_wkts", "Wickets"), ("eff_bnd", "Boundary hitting"),
               ("eff_pp_rr", "Powerplay run rate"), ("eff_death_rr", "Death-over run rate"), ("boundary_mean_m", "Boundary size (m)"), ("elevation_m", "Altitude (m)")]


def load_artifacts(q):
    """Venue profile table, model metadata and fitted models; None when the build has not been run."""
    try:
        kv = q("SELECT key, value FROM venue_model_meta")
        if kv is None or kv.empty: return None
        meta = {r.key: json.loads(r.value) for r in kv.itertuples()}
        prof = q("""SELECT v.id AS venue_id, v.name, v.city, v.country, v.pitch_type, v.capacity, v.floodlights, p.* , g.lat, g.lon
                    FROM venues v JOIN venue_profile p ON p.venue_id = v.id LEFT JOIN venue_geo g ON g.venue_id = v.id""")
        if prof is None or prof.empty: return None
        prof = prof.loc[:, ~prof.columns.duplicated()]
        import joblib; return prof, meta, joblib.load(MODELS_PATH)
    except Exception:
        return None


def _next_row(hist: pd.DataFrame, **over) -> pd.DataFrame:
    """History plus one pseudo innings whose own values never enter its features (they are cumulative sums up to the previous row)."""
    nxt = hist.iloc[[-1]].copy()
    for k, v in over.items(): nxt[k] = v
    return pd.concat([hist, nxt], ignore_index=True)


def venue_explainer(prof: pd.DataFrame, meta: dict, vid: int) -> dict:
    """The ground's numbers, how much of its effect is physics vs observed, and what each trait contributes."""
    r = prof[prof.venue_id == vid].iloc[0]; G = meta["G"]
    row = pd.DataFrame([{c: r.get(c) for c in ["elevation_m", "boundary_mean_m", "boundary_asym_m", "abs_lat", "capacity", "floodlights", "pt_drop_in", "pt_synthetic"]}])
    row["abs_lat"] = abs(r["lat"]) if pd.notna(r.get("lat")) else np.nan
    row["boundary_asym_m"] = abs(r["boundary_straight_m"] - r["boundary_square_m"]) if pd.notna(r.get("boundary_straight_m")) and pd.notna(r.get("boundary_square_m")) else np.nan
    row["pt_drop_in"] = float(r.get("pitch_type") == "Drop-in"); row["pt_synthetic"] = float(r.get("pitch_type") == "Synthetic")
    row["floodlights"] = r.get("floodlights"); row["capacity"] = r.get("capacity")
    parts = prior_breakdown(meta["prior_state"], row)
    return dict(name=r["name"], n=int(r["n_innings"]), final=float(r["final_effect"]), prior=float(r["prior_effect"]), observed=None if pd.isna(r["observed"]) else float(r["observed"]),
                se=float(r["se_effect"]), weight_obs=float(r["weight_obs"]), pct_vs_league=float(r["final_effect"] / G * 100), parts=parts,
                boundary=None if pd.isna(r.get("boundary_mean_m")) else float(r["boundary_mean_m"]), elevation=None if pd.isna(r.get("elevation_m")) else float(r["elevation_m"]))


def similar_grounds(prof: pd.DataFrame, vid: int, k=6, min_n=15) -> pd.DataFrame:
    """Grounds with the nearest fingerprint (standardized), among those with enough innings to trust."""
    cols = [c for c, _ in FINGERPRINT]; P = prof.copy()
    Z = (P[cols] - P[cols].median()) / P[cols].std().replace(0, 1); Z = Z.fillna(0); P["_d"] = np.sqrt(((Z - Z[P.venue_id == vid].iloc[0]) ** 2).sum(axis=1))
    return P[(P.venue_id != vid) & (P.n_innings >= min_n)].nsmallest(k, "_d")[["venue_id", "name", "country", "n_innings", "_d"]]


def _dist(bins, level, additive=False):
    """Distribution of one innings for a player whose expected value is `level`: the empirical innings distribution of players with the
    nearest expectation, rescaled to `level` (multiplicative for runs, additive for economy). Returns percentiles and P(runs >= 30 / 50)."""
    b = min(bins, key=lambda r: abs(r["actual"] - level)); g = np.array(b["qgrid"])
    g = g + (level - b["actual"]) if additive else g * (level / max(b["actual"], 1e-6))
    pct = lambda p: float(np.percentile(g, p)); out = dict(q10=pct(10), q25=pct(25), q50=pct(50), q75=pct(75), q90=pct(90))
    out["p30"] = float((g >= 30).mean()); out["p50"] = float((g >= 50).mean()); return out


def predict_batter(q, art, pid: int, vid: int, position=None, chasing=False, tournament=None):
    prof, meta, mdl = art; hist = bat_frame(q, pid)
    if hist.empty: return None
    pos = int(position or round(hist.batting_position.tail(30).median())); tour = tournament or hist.tournament.iloc[-1]
    rows = _next_row(hist, runs=0, balls_faced=1, pp_runs=0, pp_balls=0, mid_runs=0, mid_balls=0, death_runs=0, death_balls=0, batting_position=pos,
                     is_chase=int(chasing), req_rr=8.5 if chasing else 0.0, innings_number=2 if chasing else 1, tournament=tour, match_date=pd.Timestamp.today().normalize())
    X = bat_matrix(rows, meta["priors_bat"]).iloc[[-1]][BAT_BASE].copy(); X["tournament"] = pd.Categorical([tour], categories=mdl["bat_cats"])
    base = float(mdl["bat_base"].predict(X)[0]); G, e = meta["G"], meta["elasticity_bat"]; v = venue_explainer(prof, meta, vid)
    x, se = v["final"] / G, v["se"] / G; exp = base * (1 + e * x)
    dist = _dist(meta["bins_bat"], exp)
    here = hist[hist.venue_id == vid]
    return dict(role="bat", baseline=base, expected=exp, lift=exp - base, lift_lo=base * e * (x - se), lift_hi=base * e * (x + se), elasticity=e, pos=pos, tournament=tour,
                range=(dist["q10"], dist["q90"]), median=dist["q50"], p30=dist["p30"], p50=dist["p50"], venue=v, innings_total=int(len(hist)),
                by_venue=hist.groupby("venue_id").agg(inns=("runs", "size"), avg=("runs", "mean"), runs=("runs", "sum"), balls=("balls_faced", "sum")).reset_index(),
                record=dict(inns=int(len(here)), runs=int(here.runs.sum()), avg=float(here.runs.mean()) if len(here) else None,
                            sr=float(here.runs.sum() / max(here.balls_faced.sum(), 1) * 100) if len(here) else None, best=int(here.runs.max()) if len(here) else None))


def predict_bowler(q, art, pid: int, vid: int, tournament=None):
    prof, meta, mdl = art; hist = bowl_frame(q, pid)
    if hist.empty: return None
    tour = tournament or hist.tournament.iloc[-1]
    rows = _next_row(hist, balls_bowled=24, runs_conceded=0, dot_balls=0, pp_balls=0, pp_runs=0, mid_balls=0, mid_runs=0, death_balls=0, death_runs=0,
                     innings_number=1, tournament=tour, match_date=pd.Timestamp.today().normalize())
    X = bowl_matrix(rows, meta["priors_bowl"]).iloc[[-1]][BOWL_BASE].copy(); X["tournament"] = pd.Categorical([tour], categories=mdl["bowl_cats"])
    base = float(mdl["bowl_base"].predict(X)[0]); G, e = meta["G"], meta["elasticity_bowl"]; v = venue_explainer(prof, meta, vid)
    x, se = v["final"] / G, v["se"] / G; exp = base * (1 + e * x); dist = _dist(meta["bins_bowl"], exp, additive=True)
    here = hist[hist.venue_id == vid]; balls = float(here.balls_bowled.sum()) if len(here) else 0
    return dict(role="bowl", baseline=base, expected=exp, lift=exp - base, lift_lo=base * e * (x - se), lift_hi=base * e * (x + se), elasticity=e, tournament=tour,
                range=(dist["q10"], dist["q90"]), venue=v, innings_total=int(len(hist)),
                by_venue=hist.groupby("venue_id").agg(spells=("balls_bowled", "size"), balls=("balls_bowled", "sum"), runs=("runs_conceded", "sum")).reset_index(),
                record=dict(spells=int(len(here)), econ=float(here.runs_conceded.sum() * 6 / balls) if balls else None, overs=balls / 6 if balls else 0.0))
