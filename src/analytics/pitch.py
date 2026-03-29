"""
Venue difficulty estimation via Bayesian shrinkage.

The core idea:
  - Compute raw average first-innings score at each venue.
  - Shrink it toward the global mean using a prior equivalent to N_PRIOR innings.
  - bat_factor = shrunk_venue_avg / global_avg
  - Adjust player stats: adj_avg = raw_avg / bat_factor

This removes venue bias from player comparisons: a player averaging 35 at
Chepauk (spin-friendly, low-scoring) is rated higher than one averaging 35
at Wankhede (flat pitch, high-scoring).

Additionally we compute:
  - pace_index: fraction of bowler wickets taken by pace bowlers
  - spin_index: complementary
  - boundary_rate: boundaries per ball
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session
from scipy import stats

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import VENUE_PRIOR_WEIGHT, MIN_VENUE_INNINGS
from src.db.schema import VenueDifficulty, Venue


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def _load_innings_data(session: Session) -> pd.DataFrame:
    sql = text("""
        SELECT
            i.id            AS innings_id,
            i.innings_number,
            i.total_runs,
            i.total_wickets,
            i.total_balls,
            m.venue_id
        FROM innings i
        JOIN matches m ON m.id = i.match_id
        WHERE i.total_balls > 0
    """)
    return pd.read_sql(sql, session.bind)


def _load_delivery_data(session: Session) -> pd.DataFrame:
    """Aggregate boundary rate and wicket type by venue."""
    sql = text("""
        SELECT
            m.venue_id,
            SUM(d.is_boundary_4 + d.is_boundary_6) AS boundaries,
            COUNT(d.id)                              AS total_balls,
            SUM(CASE WHEN d.is_wicket = 1 THEN 1 ELSE 0 END) AS wickets,
            SUM(CASE WHEN d.wicket_kind IN ('bowled','lbw','caught',
                'caught and bowled','hit wicket') THEN 1 ELSE 0 END) AS pace_like_wkts
        FROM deliveries d
        JOIN innings i  ON i.id  = d.innings_id
        JOIN matches m  ON m.id  = i.match_id
        GROUP BY m.venue_id
    """)
    return pd.read_sql(sql, session.bind)


# ---------------------------------------------------------------------------
# Bayesian shrinkage
# ---------------------------------------------------------------------------

def _shrink_toward_mean(values: np.ndarray, weights: np.ndarray,
                        prior_weight: float) -> np.ndarray:
    """
    Bayesian shrinkage: each estimate is pulled toward the global mean
    proportional to (prior_weight / (weight + prior_weight)).
    """
    global_mean = np.average(values, weights=weights)
    shrunk = (weights * values + prior_weight * global_mean) / (weights + prior_weight)
    return shrunk


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_venue_factors(session: Session) -> pd.DataFrame:
    """
    Compute and persist venue difficulty factors. Returns summary DataFrame.
    """
    df_inn  = _load_innings_data(session)
    df_del  = _load_delivery_data(session)

    if df_inn.empty:
        print("No innings data found — ingest data first.")
        return pd.DataFrame()

    # Only first innings for "batter friendliness" measure
    first_inn = df_inn[df_inn["innings_number"] == 1].copy()

    venue_agg = (
        first_inn.groupby("venue_id")
        .agg(
            matches       = ("innings_id",  "count"),
            avg_runs      = ("total_runs",  "mean"),
            avg_wickets   = ("total_wickets", "mean"),
            sum_runs      = ("total_runs",  "sum"),
        )
        .reset_index()
    )

    # Only venues with enough data
    venue_agg = venue_agg[venue_agg["matches"] >= MIN_VENUE_INNINGS].copy()

    # Bayesian shrinkage using match count as weight
    shrunk_avg = _shrink_toward_mean(
        venue_agg["avg_runs"].values,
        venue_agg["matches"].values,
        prior_weight=VENUE_PRIOR_WEIGHT,
    )
    global_mean = np.average(
        venue_agg["avg_runs"].values,
        weights=venue_agg["matches"].values,
    )

    venue_agg["shrunk_avg"]  = shrunk_avg
    venue_agg["bat_factor"]  = shrunk_avg / global_mean    # >1 = easy batting
    venue_agg["bowl_factor"] = global_mean / shrunk_avg    # >1 = easy bowling

    # Credible interval via bootstrap (1000 resamples per venue is cheap)
    ci_lo, ci_hi = [], []
    for _, row in venue_agg.iterrows():
        inn_subset = first_inn[first_inn["venue_id"] == row["venue_id"]]["total_runs"]
        if len(inn_subset) < 3:
            ci_lo.append(np.nan); ci_hi.append(np.nan)
            continue
        boots = [inn_subset.sample(len(inn_subset), replace=True).mean()
                 for _ in range(500)]
        lo, hi = np.percentile(boots, [5, 95])
        ci_lo.append(lo / global_mean)
        ci_hi.append(hi / global_mean)
    venue_agg["bat_factor_lo"] = ci_lo
    venue_agg["bat_factor_hi"] = ci_hi

    # Merge delivery stats
    df_del = df_del.copy()
    df_del["boundary_rate"] = df_del["boundaries"] / df_del["total_balls"].clip(lower=1)
    df_del["pace_index"]    = df_del["pace_like_wkts"] / df_del["wickets"].clip(lower=1)
    df_del["spin_index"]    = 1 - df_del["pace_index"]

    venue_agg = venue_agg.merge(
        df_del[["venue_id", "boundary_rate", "pace_index", "spin_index"]],
        on="venue_id", how="left",
    )

    # Second innings
    second_inn_avg = (
        df_inn[df_inn["innings_number"] == 2]
        .groupby("venue_id")["total_runs"].mean()
        .rename("avg_second_inn_runs")
    )
    venue_agg = venue_agg.merge(second_inn_avg, on="venue_id", how="left")
    venue_agg = venue_agg.rename(columns={"avg_runs": "avg_first_inn_runs"})

    # Persist to DB
    _upsert_venue_difficulty(session, venue_agg)

    return venue_agg


def _upsert_venue_difficulty(session: Session, df: pd.DataFrame):
    for _, row in df.iterrows():
        vid = int(row["venue_id"])
        existing = session.query(VenueDifficulty).filter_by(venue_id=vid).first()
        if existing is None:
            existing = VenueDifficulty(venue_id=vid)
            session.add(existing)

        existing.total_matches        = int(row["matches"])
        existing.avg_first_inn_runs   = float(row["avg_first_inn_runs"])
        existing.avg_second_inn_runs  = float(row.get("avg_second_inn_runs") or 0)
        existing.avg_total_runs       = float(row["avg_first_inn_runs"])
        existing.avg_wickets_per_inn  = float(row.get("avg_wickets") or 0)
        existing.bat_factor           = float(row["bat_factor"])
        existing.bowl_factor          = float(row["bowl_factor"])
        existing.pace_index           = float(row.get("pace_index") or 0.5)
        existing.spin_index           = float(row.get("spin_index") or 0.5)
        existing.boundary_rate        = float(row.get("boundary_rate") or 0)
        existing.bat_factor_lo        = float(row.get("bat_factor_lo") or row["bat_factor"])
        existing.bat_factor_hi        = float(row.get("bat_factor_hi") or row["bat_factor"])

    session.commit()
    print(f"Venue difficulty computed for {len(df)} venues.")


# ---------------------------------------------------------------------------
# Venue-adjusted player stat helper
# ---------------------------------------------------------------------------

def venue_adjusted_average(raw_avg: float, venue_bat_factor: float) -> float:
    """
    Normalise a batting average to a neutral venue.
    adj_avg = raw_avg / bat_factor
    (if bat_factor > 1, venue is easy — adjust down)
    """
    if venue_bat_factor and venue_bat_factor > 0:
        return raw_avg / venue_bat_factor
    return raw_avg


def venue_adjusted_economy(raw_econ: float, venue_bat_factor: float) -> float:
    """
    Normalise bowling economy: at a high-scoring venue the same economy
    is more impressive, so we divide by bat_factor.
    """
    return venue_adjusted_average(raw_econ, venue_bat_factor)
