"""
Player Rating System — normalised composite scores (0-100).

Batting rating components:
  1. Volume score        — innings count, consistency proxy
  2. Average score       — venue-adjusted batting average
  3. Strike rate score   — venue-adjusted SR (vs position peers)
  4. Phase scores        — PP / middle / death SR
  5. Chase score         — performance in chases, weighted by pressure
  6. Opener score        — if positions 1-2, additional contribution
  7. Finisher score      — if positions 5-8, death SR + chase finishes

Bowling rating components:
  1. Economy score       — venue-adjusted economy
  2. Wicket score        — strike rate
  3. Dot ball score      — dot %
  4. Phase scores        — PP / middle / death economy
  5. Overall control     — wickets + economy combined

All scores are normalised via:
  z-score within the population → sigmoid → 0-100 scale

This means ratings are relative to the population of players in the DB,
which is what you want for a comparison tool.
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import MIN_BAT_INNINGS, MIN_BOWL_INNINGS
from src.db.schema import (
    PlayerCareerBat, PlayerCareerBowl, PlayerPositionBat,
    PlayerChaseBat, PlayerRating, Player,
)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _z_to_score(z: np.ndarray) -> np.ndarray:
    """Sigmoid-scaled z-score → [0, 100]. Mean=50, ±2σ ≈ 88/12."""
    return 100 / (1 + np.exp(-z * 0.8))


def _normalise(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Compute 0-100 normalised score for a pandas Series."""
    s = series.fillna(series.median())
    mean, std = s.mean(), s.std()
    if std < 1e-6:
        return pd.Series(50.0, index=series.index)
    z = (s - mean) / std
    if not higher_is_better:
        z = -z
    return pd.Series(_z_to_score(z.values), index=series.index)


# ---------------------------------------------------------------------------
# Load aggregates
# ---------------------------------------------------------------------------

def _load_career_bat(session: Session, tournament: str = "ALL") -> pd.DataFrame:
    sql = text("""
        SELECT
            pcb.*,
            pcd_c.innings           AS chase_innings,
            pcd_c.average           AS chase_avg,
            pcd_c.strike_rate       AS chase_sr,
            pcd_c.high_pressure_sr  AS hp_sr,
            pcd_c.finishes          AS chase_finishes,
            pcd_f.innings           AS first_innings,
            pcd_f.average           AS first_avg,
            pcd_f.strike_rate       AS first_sr
        FROM player_career_bat pcb
        LEFT JOIN player_chase_bat pcd_c ON pcd_c.player_id = pcb.player_id
                                         AND pcd_c.innings_type = 'chase'
        LEFT JOIN player_chase_bat pcd_f ON pcd_f.player_id = pcb.player_id
                                         AND pcd_f.innings_type = 'first'
        WHERE pcb.tournament = :t
    """)
    return pd.read_sql(sql, session.bind, params={"t": tournament})


def _load_career_bowl(session: Session, tournament: str = "ALL") -> pd.DataFrame:
    sql = text("""
        SELECT * FROM player_career_bowl WHERE tournament = :t
    """)
    return pd.read_sql(sql, session.bind, params={"t": tournament})


def _load_position_bat(session: Session) -> pd.DataFrame:
    sql = text("SELECT * FROM player_position_bat")
    return pd.read_sql(sql, session.bind)


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def _bat_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute batting sub-scores for all players. Input: career_bat DataFrame."""
    d = df[df["innings"] >= MIN_BAT_INNINGS].copy()

    # Core
    d["s_avg"]      = _normalise(d["adj_average"])
    d["s_sr"]       = _normalise(d["adj_strike_rate"])
    d["s_pp_sr"]    = _normalise(d["pp_sr"])
    d["s_mid_sr"]   = _normalise(d["mid_sr"])
    d["s_death_sr"] = _normalise(d["death_sr"])

    # Chase vs first innings
    d["s_chase"]    = _normalise(
        d["chase_avg"].fillna(0) * 0.5 +
        d["chase_sr"].fillna(0)  * 0.3 +
        d["hp_sr"].fillna(0)     * 0.2
    )
    d["s_first"]    = _normalise(
        d["first_avg"].fillna(0) * 0.6 +
        d["first_sr"].fillna(0)  * 0.4
    )

    # Opener: average + PP SR + first innings
    d["s_opener"]   = (d["s_first"] * 0.4 + d["s_pp_sr"] * 0.4 + d["s_avg"] * 0.2)

    # Finisher: death SR + chase finishes + chase SR
    finishes_norm = _normalise(d["chase_finishes"].fillna(0))
    d["s_finisher"] = (d["s_death_sr"] * 0.5 + d["s_chase"] * 0.3 + finishes_norm * 0.2)

    # Anchor: high average, moderate SR, strong chase
    d["s_anchor"]   = (d["s_avg"] * 0.5 + d["s_chase"] * 0.3 + d["s_mid_sr"] * 0.2)

    # Overall batting rating (weighted harmonic-like blend)
    d["bat_rating"] = (
        d["s_avg"]      * 0.30 +
        d["s_sr"]       * 0.20 +
        d["s_pp_sr"]    * 0.10 +
        d["s_mid_sr"]   * 0.10 +
        d["s_death_sr"] * 0.10 +
        d["s_chase"]    * 0.20
    ).round(1)

    return d


def _bowl_scores(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["innings"] >= MIN_BOWL_INNINGS].copy()

    d["s_economy"]       = _normalise(d["adj_economy"],    higher_is_better=False)
    d["s_avg"]           = _normalise(d["average"],        higher_is_better=False)
    d["s_sr"]            = _normalise(d["strike_rate"],    higher_is_better=False)
    d["s_dot"]           = _normalise(d["dot_pct"],        higher_is_better=True)
    d["s_pp_bowl"]       = _normalise(d["pp_economy"],     higher_is_better=False)
    d["s_mid_bowl"]      = _normalise(d["mid_economy"],    higher_is_better=False)
    d["s_death_bowl"]    = _normalise(d["death_economy"],  higher_is_better=False)

    d["bowl_rating"] = (
        d["s_economy"]    * 0.25 +
        d["s_avg"]        * 0.20 +
        d["s_sr"]         * 0.15 +
        d["s_dot"]        * 0.15 +
        d["s_pp_bowl"]    * 0.08 +
        d["s_mid_bowl"]   * 0.08 +
        d["s_death_bowl"] * 0.09
    ).round(1)

    return d


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def rebuild_ratings(session: Session, tournament: str = "ALL"):
    print(f"Building ratings for tournament={tournament}…")

    bat_df  = _load_career_bat(session, tournament)
    bowl_df = _load_career_bowl(session, tournament)

    bat_scores  = _bat_scores(bat_df)
    bowl_scores = _bowl_scores(bowl_df)

    # Merge
    merged = bat_scores[["player_id", "bat_rating",
                          "s_opener", "s_finisher", "s_anchor",
                          "s_pp_sr", "s_death_sr", "s_chase",
                          "adj_average", "adj_strike_rate"]].merge(
        bowl_scores[["player_id", "bowl_rating",
                     "s_pp_bowl", "s_mid_bowl", "s_death_bowl",
                     "adj_economy"]],
        on="player_id", how="outer",
    )
    merged["bat_rating"]  = merged["bat_rating"].fillna(0)
    merged["bowl_rating"] = merged["bowl_rating"].fillna(0)

    # Simple all-rounder: only if enough innings in both
    merged["overall_rating"] = (
        merged["bat_rating"]  * 0.6 +
        merged["bowl_rating"] * 0.4
    ).round(1)

    # Persist
    session.query(PlayerRating).filter_by(tournament=tournament).delete()
    today = date.today()

    for _, row in merged.iterrows():
        r = row.to_dict()
        session.add(PlayerRating(
            player_id        = int(r["player_id"]),
            tournament       = tournament,
            bat_rating       = r.get("bat_rating"),
            bowl_rating      = r.get("bowl_rating"),
            overall_rating   = r.get("overall_rating"),
            opener_score     = r.get("s_opener"),
            finisher_score   = r.get("s_finisher"),
            anchor_score     = r.get("s_anchor"),
            pp_bat_score     = r.get("s_pp_sr"),
            death_bat_score  = r.get("s_death_sr"),
            chase_score      = r.get("s_chase"),
            pp_bowl_score    = r.get("s_pp_bowl"),
            mid_bowl_score   = r.get("s_mid_bowl"),
            death_bowl_score = r.get("s_death_bowl"),
            adj_bat_rating   = r.get("adj_average"),
            adj_bowl_rating  = r.get("adj_economy"),
            updated_at       = today,
        ))

    session.commit()
    print(f"Ratings built for {len(merged)} players.")
    return merged


# ---------------------------------------------------------------------------
# Player comparison helper (used by dashboard)
# ---------------------------------------------------------------------------

def compare_players(session: Session,
                    player_a_id: int, player_b_id: int,
                    tournament: str = "ALL") -> dict:
    """
    Return a side-by-side comparison dict for the dashboard.
    """
    def _fetch(pid):
        r = session.query(PlayerRating).filter_by(
            player_id=pid, tournament=tournament
        ).first()
        cb = session.query(PlayerCareerBat).filter_by(
            player_id=pid, tournament=tournament
        ).first()
        cbowl = session.query(PlayerCareerBowl).filter_by(
            player_id=pid, tournament=tournament
        ).first()
        p = session.query(Player).filter_by(id=pid).first()
        return {"player": p, "rating": r, "bat": cb, "bowl": cbowl}

    return {
        "player_a": _fetch(player_a_id),
        "player_b": _fetch(player_b_id),
    }
