"""
Player performance prediction model.

Stage 1 — Global GradientBoosting trained on all innings
  Features: venue_bat_factor, batting_position, is_chase, career_adj_avg,
            career_adj_sr, chase_avg, first_avg, pp_sr, mid_sr, death_sr
  Target: runs scored per innings / economy per spell

Stage 2 — Per-player bias correction
  player_offset = player_career_adj_avg - global_mean_predicted_for_their_features
  Final prediction = stage1_pred + offset
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from sqlalchemy.orm import Session

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parents[2]))

MODEL_DIR = Path(__file__).parents[2] / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BAT_MODEL_PATH   = MODEL_DIR / "bat_model.joblib"      # legacy / fallback
BOWL_MODEL_PATH  = MODEL_DIR / "bowl_model.joblib"
BAT_SCALER_PATH  = MODEL_DIR / "bat_scaler.joblib"     # legacy / fallback
BOWL_SCALER_PATH = MODEL_DIR / "bowl_scaler.joblib"
META_PATH        = MODEL_DIR / "model_meta.joblib"

# Position-stratified batting models
# Group 0: openers (1-2), Group 1: top-order (3-5),
# Group 2: lower-order (6-8), Group 3: tail (9-11)
POS_GROUPS = {0: (1, 2), 1: (3, 5), 2: (6, 8), 3: (9, 11)}
POS_LABELS = {0: "openers", 1: "top_order", 2: "lower_order", 3: "tail"}

def _pos_group(position: int) -> int:
    for g, (lo, hi) in POS_GROUPS.items():
        if lo <= position <= hi:
            return g
    return 1  # default to top-order

def _bat_model_path(group: int) -> Path:
    return MODEL_DIR / f"bat_model_g{group}.joblib"

def _bat_scaler_path(group: int) -> Path:
    return MODEL_DIR / f"bat_scaler_g{group}.joblib"

def pos_models_exist() -> bool:
    return all(_bat_model_path(g).exists() for g in POS_GROUPS)

TOURNAMENT_MAP = {
    "t20i_male": 0, "t20_wc_male": 0,
    "ipl": 1, "psl": 2, "bbl": 3, "cpl": 4,
    "t20_blast": 5, "sa20": 6, "lpl": 7,
    "ilt20": 8, "hundred_male": 9,
}


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def _bat_raw(session: Session) -> pd.DataFrame:
    return pd.read_sql(text("""
        SELECT
            pi.batter_id                             AS player_id,
            pi.runs,
            pi.balls_faced,
            COALESCE(pi.batting_position, 5)         AS batting_position,
            CAST(pi.is_chase AS INTEGER)             AS is_chase,
            COALESCE(pi.required_rr_start, 0)        AS req_rr,
            pi.pp_runs,  pi.pp_balls,
            pi.mid_runs, pi.mid_balls,
            pi.death_runs, pi.death_balls,
            COALESCE(vd.bat_factor,    1.0)          AS venue_bat_factor,
            COALESCE(vd.boundary_rate, 0.12)         AS boundary_rate,
            COALESCE(vd.pace_index,    0.5)          AS pace_index,
            COALESCE(m.tournament, 'other')          AS tournament
        FROM player_innings pi
        JOIN matches m ON m.id = pi.match_id
        LEFT JOIN venue_difficulty vd ON vd.venue_id = m.venue_id
        WHERE pi.balls_faced >= 4
    """), session.bind)


def _bowl_raw(session: Session) -> pd.DataFrame:
    return pd.read_sql(text("""
        SELECT
            pbi.bowler_id                            AS player_id,
            pbi.balls_bowled,
            pbi.runs_conceded,
            pbi.wickets,
            pbi.dot_balls,
            pbi.pp_balls,    pbi.pp_runs,
            pbi.mid_balls,   pbi.mid_runs,
            pbi.death_balls, pbi.death_runs,
            COALESCE(vd.bat_factor,    1.0)          AS venue_bat_factor,
            COALESCE(vd.boundary_rate, 0.12)         AS boundary_rate,
            COALESCE(vd.pace_index,    0.5)          AS pace_index,
            COALESCE(m.tournament, 'other')          AS tournament
        FROM player_bowling_innings pbi
        JOIN matches m ON m.id = pbi.match_id
        LEFT JOIN venue_difficulty vd ON vd.venue_id = m.venue_id
        WHERE pbi.balls_bowled >= 6
    """), session.bind)


def _player_bat_career(session: Session) -> pd.DataFrame:
    return pd.read_sql(text("""
        SELECT player_id,
               COALESCE(adj_average,    20.0) AS career_adj_avg,
               COALESCE(adj_strike_rate,120.0) AS career_adj_sr,
               COALESCE(innings,        1)    AS career_innings
        FROM player_career_bat WHERE tournament = 'ALL'
    """), session.bind)


def _player_chase(session: Session) -> pd.DataFrame:
    return pd.read_sql(text("""
        SELECT player_id,
            MAX(CASE WHEN innings_type='chase' THEN COALESCE(average,20)      END) AS chase_avg,
            MAX(CASE WHEN innings_type='first' THEN COALESCE(average,20)      END) AS first_avg,
            MAX(CASE WHEN innings_type='chase' THEN COALESCE(strike_rate,120) END) AS chase_sr
        FROM player_chase_bat GROUP BY player_id
    """), session.bind)


def _player_bowl_career(session: Session) -> pd.DataFrame:
    return pd.read_sql(text("""
        SELECT player_id,
               COALESCE(adj_economy, 8.5) AS career_adj_econ,
               COALESCE(dot_pct,     33)  AS career_dot_pct,
               COALESCE(innings,     1)   AS career_bowl_inn,
               COALESCE(pp_economy,  8.0) AS pp_economy,
               COALESCE(mid_economy, 8.0) AS mid_economy,
               COALESCE(death_economy,9.5)AS death_economy
        FROM player_career_bowl WHERE tournament = 'ALL'
    """), session.bind)


# ─────────────────────────────────────────────
# FEATURE MATRICES
# ─────────────────────────────────────────────

BAT_FEATURES = [
    "venue_bat_factor", "boundary_rate", "pace_index",
    "batting_position", "is_chase", "req_rr",
    "career_adj_avg", "career_adj_sr", "career_innings",
    "chase_avg", "first_avg", "chase_sr",
    "pp_sr", "mid_sr", "death_sr", "tournament_enc",
]

BOWL_FEATURES = [
    "venue_bat_factor", "boundary_rate", "pace_index",
    "career_adj_econ", "career_dot_pct", "career_bowl_inn",
    "pp_economy", "mid_economy", "death_economy",
    "tournament_enc",
]


def _make_bat_matrix(df: pd.DataFrame) -> np.ndarray:
    pp_sr    = (df["pp_runs"]    * 100 / df["pp_balls"].clip(lower=1)).clip(upper=300).fillna(130)
    mid_sr   = (df["mid_runs"]   * 100 / df["mid_balls"].clip(lower=1)).clip(upper=300).fillna(125)
    death_sr = (df["death_runs"] * 100 / df["death_balls"].clip(lower=1)).clip(upper=300).fillna(130)
    t_enc    = df["tournament"].map(TOURNAMENT_MAP).fillna(5)

    return np.column_stack([
        df["venue_bat_factor"].clip(0.5, 2.0),
        df["boundary_rate"].clip(0.05, 0.25),
        df["pace_index"].clip(0, 1),
        df["batting_position"].clip(1, 11),
        df["is_chase"].astype(float),
        df["req_rr"].clip(0, 20),
        df["career_adj_avg"].clip(0, 80),
        df["career_adj_sr"].clip(50, 250),
        df["career_innings"].clip(1, 300),
        df["chase_avg"].clip(0, 80),
        df["first_avg"].clip(0, 80),
        df["chase_sr"].clip(50, 250),
        pp_sr, mid_sr, death_sr, t_enc,
    ])


def _make_bowl_matrix(df: pd.DataFrame) -> np.ndarray:
    t_enc = df["tournament"].map(TOURNAMENT_MAP).fillna(5)
    return np.column_stack([
        df["venue_bat_factor"].clip(0.5, 2.0),
        df["boundary_rate"].clip(0.05, 0.25),
        df["pace_index"].clip(0, 1),
        df["career_adj_econ"].clip(4, 18),
        df["career_dot_pct"].clip(10, 70),
        df["career_bowl_inn"].clip(1, 300),
        df["pp_economy"].clip(4, 18),
        df["mid_economy"].clip(4, 18),
        df["death_economy"].clip(4, 20),
        t_enc,
    ])


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────

def train(session: Session, verbose: bool = True) -> dict:
    print("Loading data…") if verbose else None

    bat_df   = _bat_raw(session)
    bowl_df  = _bowl_raw(session)
    career   = _player_bat_career(session)
    chase_df = _player_chase(session)
    bowl_car = _player_bowl_career(session)

    # ── merge player features in ──
    bat_df = (bat_df
              .merge(career,   on="player_id", how="left")
              .merge(chase_df, on="player_id", how="left"))
    bat_df[["career_adj_avg","career_adj_sr","career_innings",
            "chase_avg","first_avg","chase_sr"]] = \
        bat_df[["career_adj_avg","career_adj_sr","career_innings",
                "chase_avg","first_avg","chase_sr"]].fillna({
            "career_adj_avg": 20, "career_adj_sr": 120, "career_innings": 10,
            "chase_avg": 20, "first_avg": 20, "chase_sr": 120,
        })

    bowl_df = bowl_df.merge(bowl_car, on="player_id", how="left")
    bowl_df[["career_adj_econ","career_dot_pct","career_bowl_inn",
             "pp_economy","mid_economy","death_economy"]] = \
        bowl_df[["career_adj_econ","career_dot_pct","career_bowl_inn",
                 "pp_economy","mid_economy","death_economy"]].fillna({
            "career_adj_econ":8.5,"career_dot_pct":33,"career_bowl_inn":10,
            "pp_economy":8.0,"mid_economy":8.0,"death_economy":9.5,
        })

    print(f"  Batting  : {len(bat_df):,} innings") if verbose else None
    print(f"  Bowling  : {len(bowl_df):,} spells") if verbose else None

    # ── Batting — position-stratified models ──
    bat_df["pos_group"] = bat_df["batting_position"].clip(1, 11).apply(_pos_group)

    all_bat_pred = np.zeros(len(bat_df))
    bat_r2_list, bat_mae_list = [], []

    print("Training position-stratified batting models (GBM)…") if verbose else None
    for g, label in POS_LABELS.items():
        mask_g = bat_df["pos_group"] == g
        df_g   = bat_df[mask_g]
        if len(df_g) < 100:
            if verbose: print(f"  {label}: too few samples ({len(df_g)}), skipping")
            continue

        X_g = _make_bat_matrix(df_g)
        y_g = df_g["runs"].values.astype(float)
        sc_g = StandardScaler().fit(X_g)
        X_gs = sc_g.transform(X_g)

        # Tuned params for openers/top-order; lighter for tail
        n_est = 196 if g <= 1 else 150
        model_g = GradientBoostingRegressor(
            n_estimators=n_est, learning_rate=0.1357 if g <= 1 else 0.1,
            max_depth=6 if g <= 1 else 4,
            subsample=0.7745, min_samples_leaf=41 if g <= 1 else 20,
            max_features=0.7645, random_state=42,
        )
        model_g.fit(X_gs, y_g)
        pred_g = model_g.predict(X_gs)
        all_bat_pred[mask_g] = pred_g

        r2_g  = r2_score(y_g, pred_g)
        mae_g = mean_absolute_error(y_g, pred_g)
        bat_r2_list.append(r2_g); bat_mae_list.append(mae_g)

        joblib.dump(model_g, _bat_model_path(g))
        joblib.dump(sc_g,    _bat_scaler_path(g))
        if verbose:
            print(f"  {label} (n={len(df_g):,}): R²={r2_g:.3f}  MAE={mae_g:.1f}")

    # Also save a global fallback model for predict_bat compatibility
    y_bat  = bat_df["runs"].values.astype(float)
    X_bat  = _make_bat_matrix(bat_df)
    bat_sc = StandardScaler().fit(X_bat)
    bat_model = GradientBoostingRegressor(
        n_estimators=196, learning_rate=0.1357, max_depth=6,
        subsample=0.7745, min_samples_leaf=41, max_features=0.7645, random_state=42,
    )
    bat_model.fit(bat_sc.transform(X_bat), y_bat)
    joblib.dump(bat_model, BAT_MODEL_PATH)
    joblib.dump(bat_sc,    BAT_SCALER_PATH)

    bat_r2  = float(np.mean(bat_r2_list))
    bat_mae = float(np.mean(bat_mae_list))
    bat_cv  = -cross_val_score(bat_model, bat_sc.transform(X_bat), y_bat,
                                cv=3, scoring="neg_mean_absolute_error")

    # ── Bowling ──
    econ_y  = (bowl_df["runs_conceded"] * 6 / bowl_df["balls_bowled"].clip(lower=1)).values
    mask    = (econ_y > 2) & (econ_y < 24)
    X_bowl  = _make_bowl_matrix(bowl_df)

    bowl_sc = StandardScaler().fit(X_bowl[mask])
    X_bowl_s = bowl_sc.transform(X_bowl)

    print("Training bowling model (GBM)…") if verbose else None
    bowl_model = GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=4,
        subsample=0.8, min_samples_leaf=20, random_state=42,
    )
    bowl_model.fit(X_bowl_s[mask], econ_y[mask])
    bowl_pred = bowl_model.predict(X_bowl_s[mask])
    bowl_r2   = r2_score(econ_y[mask], bowl_pred)
    bowl_mae  = mean_absolute_error(econ_y[mask], bowl_pred)

    # ── Save ──
    joblib.dump(bat_model,  BAT_MODEL_PATH)
    joblib.dump(bowl_model, BOWL_MODEL_PATH)
    joblib.dump(bat_sc,     BAT_SCALER_PATH)
    joblib.dump(bowl_sc,    BOWL_SCALER_PATH)

    meta = {
        "bat_features":      BAT_FEATURES,
        "bowl_features":     BOWL_FEATURES,
        "bat_importances":   bat_model.feature_importances_.tolist(),
        "bowl_importances":  bowl_model.feature_importances_.tolist(),
        "bat_r2":   round(bat_r2,  3),
        "bat_mae":  round(bat_mae, 2),
        "bat_cv_mae": round(bat_cv.mean(), 2),
        "bowl_r2":  round(bowl_r2,  3),
        "bowl_mae": round(bowl_mae, 3),
        "n_bat":    int(len(bat_df)),
        "n_bowl":   int(mask.sum()),
    }
    joblib.dump(meta, META_PATH)

    if verbose:
        print(f"\n  Batting  R²={bat_r2:.3f}  MAE={bat_mae:.1f} runs  CV-MAE={bat_cv.mean():.1f}")
        print(f"  Bowling  R²={bowl_r2:.3f}  MAE={bowl_mae:.3f} econ")
    return meta


# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────

def models_exist() -> bool:
    return BAT_MODEL_PATH.exists() and META_PATH.exists()


def predict_bat(player: dict, venue: dict, n_boot: int = 300) -> dict:
    """
    player keys: career_adj_avg, career_adj_sr, career_innings,
                 chase_avg, first_avg, chase_sr,
                 batting_position, pp_sr, mid_sr, death_sr
    venue keys:  bat_factor, boundary_rate, pace_index
    """
    # Route to position-stratified model if available
    pos   = int(player.get("batting_position", 4))
    group = _pos_group(pos)
    if _bat_model_path(group).exists():
        bat_model = joblib.load(_bat_model_path(group))
        bat_sc    = joblib.load(_bat_scaler_path(group))
    else:
        bat_model = joblib.load(BAT_MODEL_PATH)
        bat_sc    = joblib.load(BAT_SCALER_PATH)

    def _row(is_chase, rr):
        return np.array([[
            venue.get("bat_factor", 1.0),
            venue.get("boundary_rate", 0.12),
            venue.get("pace_index", 0.5),
            player.get("batting_position", 4),
            is_chase, rr,
            player.get("career_adj_avg", 20),
            player.get("career_adj_sr", 120),
            player.get("career_innings", 30),
            player.get("chase_avg", 20),
            player.get("first_avg", 20),
            player.get("chase_sr", 120),
            player.get("pp_sr", 130),
            player.get("mid_sr", 125),
            player.get("death_sr", 135),
            1,
        ]])

    x_first = bat_sc.transform(_row(0, 0))
    x_chase = bat_sc.transform(_row(1, 9))
    pred_first = float(bat_model.predict(x_first)[0])
    pred_chase = float(bat_model.predict(x_chase)[0])

    rng   = np.random.default_rng(0)
    boots = []
    base  = _row(0.5, 5)
    for _ in range(n_boot):
        nb = base * rng.normal(1.0, 0.07, base.shape)
        boots.append(float(bat_model.predict(bat_sc.transform(nb))[0]))
    boots = np.clip(boots, 0, None)

    return {
        "first_innings": round(max(0, pred_first), 1),
        "chasing":       round(max(0, pred_chase), 1),
        "ci_lo":         round(float(np.percentile(boots, 10)), 1),
        "ci_hi":         round(float(np.percentile(boots, 90)), 1),
        "venue_factor":  round(venue.get("bat_factor", 1.0), 3),
    }


def predict_bowl(player: dict, venue: dict, n_boot: int = 300) -> dict:
    bowl_model = joblib.load(BOWL_MODEL_PATH)
    bowl_sc    = joblib.load(BOWL_SCALER_PATH)

    base = np.array([[
        venue.get("bat_factor", 1.0),
        venue.get("boundary_rate", 0.12),
        venue.get("pace_index", 0.5),
        player.get("career_adj_econ", 8.5),
        player.get("career_dot_pct", 33),
        player.get("career_bowl_inn", 30),
        player.get("pp_economy", 8.0),
        player.get("mid_economy", 8.0),
        player.get("death_economy", 9.5),
        1,
    ]])
    pred = float(bowl_model.predict(bowl_sc.transform(base))[0])

    rng   = np.random.default_rng(0)
    boots = []
    for _ in range(n_boot):
        nb = base * rng.normal(1.0, 0.06, base.shape)
        boots.append(float(bowl_model.predict(bowl_sc.transform(nb))[0]))
    boots = np.clip(boots, 0, None)

    return {
        "predicted_economy": round(max(0, pred), 2),
        "ci_lo": round(float(np.percentile(boots, 10)), 2),
        "ci_hi": round(float(np.percentile(boots, 90)), 2),
    }


def feature_importance_df(kind: str = "bat") -> pd.DataFrame:
    meta = joblib.load(META_PATH)
    if kind == "bat":
        return pd.DataFrame({
            "feature":    meta["bat_features"],
            "importance": meta["bat_importances"],
        }).sort_values("importance", ascending=False)
    return pd.DataFrame({
        "feature":    meta["bowl_features"],
        "importance": meta["bowl_importances"],
    }).sort_values("importance", ascending=False)


def model_metrics() -> dict:
    return joblib.load(META_PATH) if META_PATH.exists() else {}
