"""
Player performance prediction models (v2: no look-ahead).

What changed from v1 (old file kept in ~/etpl2026/backups/model.py.pre_v2):
  * v1 fed each innings' OWN powerplay/middle/death strike rates to the model. Those numbers give away the runs
    being predicted, so training R² looked great (0.84) while real predictions were far too high.
  * v1 scored itself on the data it trained on, and its career averages included the innings being predicted.

v2 rules:
  * Every feature for an innings is computed from that player's EARLIER innings only (expanding career stats),
    shrunk toward the league average so players with few innings are not extreme.
  * Models are evaluated on matches from SPLIT_DATE onward, which the model never saw.
  * Prediction ranges come from the model's real errors on those held-out matches (10th/90th percentile).
  * Innings with fewer than 4 balls are kept (v1 dropped them, which pushed every prediction up).

Batting: position-stratified GradientBoosting on runs per innings.  Bowling: GradientBoosting on spell economy.
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from sqlalchemy.orm import Session

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parents[2]))

MODEL_DIR = Path(__file__).parents[2] / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BAT_MODEL_PATH   = MODEL_DIR / "bat_model.joblib"      # global fallback
BOWL_MODEL_PATH  = MODEL_DIR / "bowl_model.joblib"
BAT_SCALER_PATH  = MODEL_DIR / "bat_scaler.joblib"
BOWL_SCALER_PATH = MODEL_DIR / "bowl_scaler.joblib"
META_PATH        = MODEL_DIR / "model_meta.joblib"

SPLIT_DATE = "2025-01-01"     # models are tested on matches on/after this date

# Position groups: 0 openers (1-2), 1 top order (3-5), 2 lower order (6-8), 3 tail (9-11)
POS_GROUPS = {0: (1, 2), 1: (3, 5), 2: (6, 8), 3: (9, 11)}
POS_LABELS = {0: "openers", 1: "top_order", 2: "lower_order", 3: "tail"}


def _pos_group(position: int) -> int:
    for g, (lo, hi) in POS_GROUPS.items():
        if lo <= position <= hi:
            return g
    return 1


def _bat_model_path(group: int) -> Path:
    return MODEL_DIR / f"bat_model_g{group}.joblib"


def _bat_scaler_path(group: int) -> Path:
    return MODEL_DIR / f"bat_scaler_g{group}.joblib"


def pos_models_exist() -> bool:
    return all(_bat_model_path(g).exists() for g in POS_GROUPS)


TOURNAMENT_MAP = {          # kept for the tuning/backtest scripts
    "t20i_male": 0, "t20_wc_male": 0, "ipl": 1, "psl": 2, "bbl": 3, "cpl": 4,
    "t20_blast": 5, "sa20": 6, "lpl": 7, "ilt20": 8, "hundred_male": 9,
}

# Shrinkage: how much a player's own history counts against the league average
M_INN = 12        # innings of prior weight (runs per innings)
W_BALLS = 80      # balls of prior weight (overall strike rate / economy / dot %)
W_PHASE = 40      # balls of prior weight (phase strike rates / economies)
BAT_PHASE_SHARE = (0.30, 0.45, 0.25)    # share of a T20 innings faced in powerplay / middle / death (36/54/30 balls)
BOWL_PHASE_SHARE = (0.30, 0.45, 0.25)

BAT_FEATURES = [
    "venue_bat_factor", "boundary_rate", "pace_index",
    "batting_position", "is_chase", "req_rr",
    "prior_innings", "prior_runs_per_inn", "prior_sr",
    "prior_pp_sr", "prior_mid_sr", "prior_death_sr",
]
BOWL_FEATURES = [
    "venue_bat_factor", "boundary_rate", "pace_index",
    "prior_spells", "prior_econ", "prior_dot_pct",
    "prior_pp_econ", "prior_mid_econ", "prior_death_econ",
]

_EXCLUDE = "m.tournament NOT LIKE 't10%' AND m.tournament NOT LIKE 'legends%' AND m.tournament != 'hundred_male'"


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def _bat_raw(session: Session) -> pd.DataFrame:
    return pd.read_sql(text(f"""
        SELECT pi.batter_id AS player_id, m.match_date, pi.innings_id, pi.runs, pi.balls_faced,
               COALESCE(pi.batting_position, 5) AS batting_position,
               CAST(pi.is_chase AS INTEGER) AS is_chase,
               COALESCE(pi.required_rr_start, 0) AS req_rr,
               pi.pp_runs, pi.pp_balls, pi.mid_runs, pi.mid_balls, pi.death_runs, pi.death_balls,
               COALESCE(vd.bat_factor, 1.0) AS venue_bat_factor,
               COALESCE(vd.boundary_rate, 0.12) AS boundary_rate,
               COALESCE(vd.pace_index, 0.5) AS pace_index
        FROM player_innings pi
        JOIN matches m ON m.id = pi.match_id
        LEFT JOIN venue_difficulty vd ON vd.venue_id = m.venue_id
        WHERE pi.balls_faced >= 1 AND {_EXCLUDE}
        ORDER BY pi.batter_id, m.match_date, pi.innings_id
    """), session.bind)


def _bowl_raw(session: Session) -> pd.DataFrame:
    return pd.read_sql(text(f"""
        SELECT pbi.bowler_id AS player_id, m.match_date, pbi.innings_id, pbi.balls_bowled, pbi.runs_conceded,
               pbi.dot_balls, pbi.pp_balls, pbi.pp_runs, pbi.mid_balls, pbi.mid_runs, pbi.death_balls, pbi.death_runs,
               COALESCE(vd.bat_factor, 1.0) AS venue_bat_factor,
               COALESCE(vd.boundary_rate, 0.12) AS boundary_rate,
               COALESCE(vd.pace_index, 0.5) AS pace_index
        FROM player_bowling_innings pbi
        JOIN matches m ON m.id = pbi.match_id
        LEFT JOIN venue_difficulty vd ON vd.venue_id = m.venue_id
        WHERE pbi.balls_bowled >= 1 AND {_EXCLUDE}
        ORDER BY pbi.bowler_id, m.match_date, pbi.innings_id
    """), session.bind)


# ─────────────────────────────────────────────
# FEATURES (shared by training and prediction)
# ─────────────────────────────────────────────

def _shrink(total, weight_n, n, prior):
    """(total + n_prior*prior) / (weight_n + n_prior) style blend; callers pass the right pieces."""
    return (total + weight_n * prior) / (n + weight_n)


def _fit_priors(bat: pd.DataFrame, bowl: pd.DataFrame) -> dict:
    sr = lambda r, b: float(r.sum() / max(b.sum(), 1) * 100)
    return {
        "rpi": float(bat["runs"].mean()),
        "sr": sr(bat["runs"], bat["balls_faced"]),
        "pp_sr": sr(bat["pp_runs"], bat["pp_balls"]),
        "mid_sr": sr(bat["mid_runs"], bat["mid_balls"]),
        "death_sr": sr(bat["death_runs"], bat["death_balls"]),
        "econ": float(bowl["runs_conceded"].sum() * 6 / max(bowl["balls_bowled"].sum(), 1)),
        "dot_pct": float(bowl["dot_balls"].sum() / max(bowl["balls_bowled"].sum(), 1) * 100),
        "pp_econ": float(bowl["pp_runs"].sum() * 6 / max(bowl["pp_balls"].sum(), 1)),
        "mid_econ": float(bowl["mid_runs"].sum() * 6 / max(bowl["mid_balls"].sum(), 1)),
        "death_econ": float(bowl["death_runs"].sum() * 6 / max(bowl["death_balls"].sum(), 1)),
    }


def _bat_matrix(df: pd.DataFrame, pri: dict) -> pd.DataFrame:
    """Expanding (earlier-innings-only) batting features. df must be sorted by player, date."""
    g = df.groupby("player_id", sort=False)
    prior_inn = g.cumcount()
    cs = lambda col: g[col].cumsum() - df[col]
    p_runs, p_balls = cs("runs"), cs("balls_faced")
    out = pd.DataFrame({
        "venue_bat_factor": df["venue_bat_factor"].clip(0.5, 2.0),
        "boundary_rate": df["boundary_rate"].clip(0.05, 0.25),
        "pace_index": df["pace_index"].clip(0, 1),
        "batting_position": df["batting_position"].clip(1, 11),
        "is_chase": df["is_chase"].astype(float),
        "req_rr": df["req_rr"].clip(0, 20),
        "prior_innings": prior_inn.clip(0, 300),
        "prior_runs_per_inn": (p_runs + M_INN * pri["rpi"]) / (prior_inn + M_INN),
        "prior_sr": (p_runs + W_BALLS * pri["sr"] / 100) / (p_balls + W_BALLS) * 100,
    })
    for ph, key in (("pp", "pp_sr"), ("mid", "mid_sr"), ("death", "death_sr")):
        pr, pb = cs(f"{ph}_runs"), cs(f"{ph}_balls")
        out[f"prior_{key}"] = (pr + W_PHASE * pri[key] / 100) / (pb + W_PHASE) * 100
    return out[BAT_FEATURES]


def _bowl_matrix(df: pd.DataFrame, pri: dict) -> pd.DataFrame:
    g = df.groupby("player_id", sort=False)
    prior_n = g.cumcount()
    cs = lambda col: g[col].cumsum() - df[col]
    p_balls, p_runs, p_dots = cs("balls_bowled"), cs("runs_conceded"), cs("dot_balls")
    out = pd.DataFrame({
        "venue_bat_factor": df["venue_bat_factor"].clip(0.5, 2.0),
        "boundary_rate": df["boundary_rate"].clip(0.05, 0.25),
        "pace_index": df["pace_index"].clip(0, 1),
        "prior_spells": prior_n.clip(0, 300),
        "prior_econ": (p_runs + W_BALLS * pri["econ"] / 6) / (p_balls + W_BALLS) * 6,
        "prior_dot_pct": (p_dots + W_BALLS * pri["dot_pct"] / 100) / (p_balls + W_BALLS) * 100,
    })
    for ph, key in (("pp", "pp_econ"), ("mid", "mid_econ"), ("death", "death_econ")):
        pr, pb = cs(f"{ph}_runs"), cs(f"{ph}_balls")
        out[f"prior_{key}"] = (pr + W_PHASE * pri[key] / 6) / (pb + W_PHASE) * 6
    return out[BOWL_FEATURES]


def _gbr(depth=3, n=220):
    return GradientBoostingRegressor(n_estimators=n, learning_rate=0.05, max_depth=depth, subsample=0.8,
                                     min_samples_leaf=150, random_state=42)


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────

def train(session: Session, verbose: bool = True) -> dict:
    say = print if verbose else (lambda *a, **k: None)
    say("Loading data…")
    bat, bowl = _bat_raw(session), _bowl_raw(session)
    bat["match_date"], bowl["match_date"] = pd.to_datetime(bat["match_date"]), pd.to_datetime(bowl["match_date"])
    bat = bat.sort_values(["player_id", "match_date", "innings_id"]).reset_index(drop=True)
    bowl = bowl.sort_values(["player_id", "match_date", "innings_id"]).reset_index(drop=True)
    say(f"  Batting : {len(bat):,} innings   Bowling: {len(bowl):,} spells")

    split = pd.Timestamp(SPLIT_DATE)
    pri = _fit_priors(bat[bat["match_date"] < split], bowl[bowl["match_date"] < split])   # priors from the training period only

    # ── batting ──
    Xb = _bat_matrix(bat, pri)
    yb = bat["runs"].values.astype(float)
    is_test = (bat["match_date"] >= split).values
    pos_group = Xb["batting_position"].astype(int).apply(_pos_group).values
    pred_test = np.full(len(bat), np.nan)
    resid_q = {}
    say(f"Batting: train {int((~is_test).sum()):,} innings, held-out test {int(is_test.sum()):,} (from {SPLIT_DATE})")
    for g, label in POS_LABELS.items():
        tr, te = (pos_group == g) & ~is_test, (pos_group == g) & is_test
        if tr.sum() < 500 or te.sum() < 50:
            say(f"  {label}: too few rows, skipped"); continue
        sc = StandardScaler().fit(Xb[tr])
        m = _gbr(depth=3 if g >= 2 else 4).fit(sc.transform(Xb[tr]), yb[tr])
        pred_test[te] = m.predict(sc.transform(Xb[te]))
        res = yb[te] - pred_test[te]
        resid_q[g] = (float(np.percentile(res, 10)), float(np.percentile(res, 90)))
        say(f"  {label}: test R²={r2_score(yb[te], pred_test[te]):.3f}  MAE={mean_absolute_error(yb[te], pred_test[te]):.2f}"
            f"  (baseline MAE {mean_absolute_error(yb[te], Xb.loc[te, 'prior_runs_per_inn']):.2f})")
    ok = ~np.isnan(pred_test)
    bat_r2, bat_mae = float(r2_score(yb[ok], pred_test[ok])), float(mean_absolute_error(yb[ok], pred_test[ok]))
    bat_base_mae = float(mean_absolute_error(yb[ok], Xb.loc[ok, "prior_runs_per_inn"]))
    say(f"  ALL held-out: R²={bat_r2:.3f} MAE={bat_mae:.2f} runs  vs 'shrunk average' baseline MAE {bat_base_mae:.2f}")

    # final models on ALL data
    for g in POS_GROUPS:
        sel = pos_group == g
        if sel.sum() < 500: continue
        sc = StandardScaler().fit(Xb[sel])
        joblib.dump(_gbr(depth=3 if g >= 2 else 4).fit(sc.transform(Xb[sel]), yb[sel]), _bat_model_path(g))
        joblib.dump(sc, _bat_scaler_path(g))
    bat_sc = StandardScaler().fit(Xb)
    bat_model = _gbr(depth=4).fit(bat_sc.transform(Xb), yb)
    joblib.dump(bat_model, BAT_MODEL_PATH); joblib.dump(bat_sc, BAT_SCALER_PATH)

    # ── bowling ──
    Xw = _bowl_matrix(bowl, pri)
    econ = (bowl["runs_conceded"] * 6 / bowl["balls_bowled"].clip(lower=1)).values
    valid = (bowl["balls_bowled"] >= 6).values & (econ > 2) & (econ < 24)
    wte = (bowl["match_date"] >= split).values
    wtr_m, wte_m = valid & ~wte, valid & wte
    say(f"Bowling: train {int(wtr_m.sum()):,} spells, held-out test {int(wte_m.sum()):,}")
    wsc = StandardScaler().fit(Xw[wtr_m])
    wm = _gbr(depth=3, n=260).fit(wsc.transform(Xw[wtr_m]), econ[wtr_m])
    wp = wm.predict(wsc.transform(Xw[wte_m]))
    bowl_r2, bowl_mae = float(r2_score(econ[wte_m], wp)), float(mean_absolute_error(econ[wte_m], wp))
    bowl_base_mae = float(mean_absolute_error(econ[wte_m], Xw.loc[wte_m, "prior_econ"]))
    wres = econ[wte_m] - wp
    bowl_q = (float(np.percentile(wres, 10)), float(np.percentile(wres, 90)))
    say(f"  held-out: R²={bowl_r2:.3f} MAE={bowl_mae:.2f}  vs 'shrunk economy' baseline MAE {bowl_base_mae:.2f}")
    wsc_all = StandardScaler().fit(Xw[valid])
    bowl_model = _gbr(depth=3, n=260).fit(wsc_all.transform(Xw[valid]), econ[valid])
    joblib.dump(bowl_model, BOWL_MODEL_PATH); joblib.dump(wsc_all, BOWL_SCALER_PATH)

    meta = {
        "version": 2, "split_date": SPLIT_DATE, "priors": pri,
        "bat_features": BAT_FEATURES, "bowl_features": BOWL_FEATURES,
        "bat_importances": bat_model.feature_importances_.tolist(),
        "bowl_importances": bowl_model.feature_importances_.tolist(),
        "bat_r2": round(bat_r2, 3), "bat_mae": round(bat_mae, 2), "bat_cv_mae": round(bat_base_mae, 2),   # cv key = baseline MAE
        "bowl_r2": round(bowl_r2, 3), "bowl_mae": round(bowl_mae, 3), "bowl_base_mae": round(bowl_base_mae, 3),
        "bat_resid_q": resid_q, "bowl_resid_q": bowl_q,
        "n_bat": int(len(bat)), "n_bowl": int(valid.sum()),
        "n_bat_test": int(ok.sum()), "n_bowl_test": int(wte_m.sum()),
    }
    joblib.dump(meta, META_PATH)
    return meta


# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────

def models_exist() -> bool:
    if not (BAT_MODEL_PATH.exists() and META_PATH.exists()):
        return False
    return joblib.load(META_PATH).get("version") == 2     # old v1 files are ignored (they cannot predict correctly)


def _meta() -> dict:
    return joblib.load(META_PATH)


def predict_bat(player: dict, venue: dict, n_boot: int = 0) -> dict:
    """player: career_runs, career_balls, career_innings (+ optional career pp_sr/mid_sr/death_sr, batting_position).
    Old v1 keys (career_adj_avg, career_adj_sr) are still understood as a fallback. n_boot is ignored (kept for callers)."""
    meta = _meta(); pri = meta["priors"]
    inn = float(player.get("career_innings") or 0)
    runs = player.get("career_runs")
    if runs is None:
        runs = float(player.get("career_adj_avg") or pri["rpi"]) * 0.85 * inn
    balls = player.get("career_balls")
    if balls is None:
        balls = float(runs) * 100 / max(float(player.get("career_adj_sr") or pri["sr"]), 50)
    runs, balls = float(runs), float(balls)

    def phase(key, share):
        sr = float(player.get(key) or pri[key]); pb = share * balls
        return (sr * pb / 100 + W_PHASE * pri[key] / 100) / (pb + W_PHASE) * 100

    pos = int(player.get("batting_position", 4))
    base = {
        "venue_bat_factor": min(max(venue.get("bat_factor", 1.0), 0.5), 2.0),
        "boundary_rate": min(max(venue.get("boundary_rate", 0.12), 0.05), 0.25),
        "pace_index": min(max(venue.get("pace_index", 0.5), 0), 1),
        "batting_position": min(max(pos, 1), 11),
        "prior_innings": min(inn, 300),
        "prior_runs_per_inn": (runs + M_INN * pri["rpi"]) / (inn + M_INN),
        "prior_sr": (runs + W_BALLS * pri["sr"] / 100) / (balls + W_BALLS) * 100,
        "prior_pp_sr": phase("pp_sr", BAT_PHASE_SHARE[0]),
        "prior_mid_sr": phase("mid_sr", BAT_PHASE_SHARE[1]),
        "prior_death_sr": phase("death_sr", BAT_PHASE_SHARE[2]),
    }
    group = _pos_group(pos)
    if _bat_model_path(group).exists():
        model, sc = joblib.load(_bat_model_path(group)), joblib.load(_bat_scaler_path(group))
    else:
        model, sc = joblib.load(BAT_MODEL_PATH), joblib.load(BAT_SCALER_PATH)

    def run(is_chase, rr):
        row = pd.DataFrame([{**base, "is_chase": float(is_chase), "req_rr": float(rr)}])[BAT_FEATURES]
        return max(0.0, float(model.predict(sc.transform(row))[0]))

    first, chase = run(0, 0), run(1, 8.5)
    q_lo, q_hi = meta["bat_resid_q"].get(group, (-15.0, 20.0))
    mid = (first + chase) / 2
    return {"first_innings": round(first, 1), "chasing": round(chase, 1),
            "ci_lo": round(max(0.0, mid + q_lo), 1), "ci_hi": round(mid + q_hi, 1),
            "venue_factor": round(venue.get("bat_factor", 1.0), 3)}


def predict_bowl(player: dict, venue: dict, n_boot: int = 0) -> dict:
    """player: career_bowl_balls, career_bowl_runs or career_econ, career_dot_pct, career_bowl_inn, pp/mid/death_economy."""
    meta = _meta(); pri = meta["priors"]
    n = float(player.get("career_bowl_inn") or 0)
    balls = player.get("career_bowl_balls")
    balls = float(balls) if balls is not None else n * 18.0
    econ = float(player.get("career_econ") or player.get("career_adj_econ") or pri["econ"])
    runs = float(player["career_bowl_runs"]) if player.get("career_bowl_runs") is not None else econ * balls / 6
    dot_pct = float(player.get("career_dot_pct") or pri["dot_pct"])

    def phase(key, share):
        e = float(player.get(f"{key.split('_')[0]}_economy") or pri[key]); pb = share * balls
        return (e * pb / 6 + W_PHASE * pri[key] / 6) / (pb + W_PHASE) * 6

    row = pd.DataFrame([{
        "venue_bat_factor": min(max(venue.get("bat_factor", 1.0), 0.5), 2.0),
        "boundary_rate": min(max(venue.get("boundary_rate", 0.12), 0.05), 0.25),
        "pace_index": min(max(venue.get("pace_index", 0.5), 0), 1),
        "prior_spells": min(n, 300),
        "prior_econ": (runs + W_BALLS * pri["econ"] / 6) / (balls + W_BALLS) * 6,
        "prior_dot_pct": (dot_pct / 100 * balls + W_BALLS * pri["dot_pct"] / 100) / (balls + W_BALLS) * 100,
        "prior_pp_econ": phase("pp_econ", BOWL_PHASE_SHARE[0]),
        "prior_mid_econ": phase("mid_econ", BOWL_PHASE_SHARE[1]),
        "prior_death_econ": phase("death_econ", BOWL_PHASE_SHARE[2]),
    }])[BOWL_FEATURES]
    pred = max(0.0, float(joblib.load(BOWL_MODEL_PATH).predict(joblib.load(BOWL_SCALER_PATH).transform(row))[0]))
    q_lo, q_hi = meta["bowl_resid_q"]
    return {"predicted_economy": round(pred, 2), "ci_lo": round(max(0.0, pred + q_lo), 2), "ci_hi": round(pred + q_hi, 2)}


def feature_importance_df(kind: str = "bat") -> pd.DataFrame:
    meta = joblib.load(META_PATH)
    return pd.DataFrame({
        "feature": meta[f"{kind}_features"], "importance": meta[f"{kind}_importances"],
    }).sort_values("importance", ascending=False)


def model_metrics() -> dict:
    return joblib.load(META_PATH) if META_PATH.exists() else {}
