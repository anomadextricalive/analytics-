"""
Player similarity and form analytics.

  build_similarity() — cosine similarity between player role profiles
  build_form()       — rolling avg/SR + breakout detection per player
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.db.schema import PlayerSimilarity, PlayerForm


ROLE_COLS = [
    "opener_score", "finisher_score", "anchor_score",
    "pp_bat_score", "death_bat_score", "chase_score",
    "pp_bowl_score", "mid_bowl_score", "death_bowl_score",
]


def _load_ratings(session: Session, tournament: str) -> pd.DataFrame:
    sql = text("""
        SELECT pr.player_id,
               pr.opener_score, pr.finisher_score, pr.anchor_score,
               pr.pp_bat_score, pr.death_bat_score, pr.chase_score,
               pr.pp_bowl_score, pr.mid_bowl_score, pr.death_bowl_score
        FROM player_ratings pr
        WHERE pr.tournament = :t

    """)
    return pd.read_sql(sql, session.bind, params={"t": tournament})


def build_similarity(session: Session, tournament: str = "ALL", top_k: int = 20):
    """Compute and persist top-K cosine-similar players for each player."""
    print(f"  Similarity: tournament={tournament}…")
    df = _load_ratings(session, tournament)
    if df.empty:
        print("  No ratings — skipping.")
        return

    mat = df[ROLE_COLS].fillna(0).values.astype(float)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1e-9, norms)
    mat_n = mat / norms
    sim_mat = mat_n @ mat_n.T

    pids = df["player_id"].values
    n = len(pids)

    session.query(PlayerSimilarity).filter_by(tournament=tournament).delete()

    rows = []
    for i in range(n):
        sims = sim_mat[i]
        top_idx = [j for j in np.argsort(sims)[::-1] if j != i][:top_k]
        for j in top_idx:
            rows.append(PlayerSimilarity(
                player_id_a=int(pids[i]),
                player_id_b=int(pids[j]),
                tournament=tournament,
                similarity=float(round(float(sims[j]), 4)),
            ))

    session.bulk_save_objects(rows)
    session.commit()
    print(f"  Done: {len(rows)} pairs for {n} players.")


def build_form(session: Session):
    """Compute rolling form metrics from all player innings."""
    print("  Form metrics…")
    sql = text("""
        SELECT pi.batter_id AS player_id,
               pi.runs, pi.balls_faced,
               m.match_date
        FROM player_innings pi
        JOIN matches m ON m.id = pi.match_id
        WHERE pi.balls_faced > 0
        ORDER BY pi.batter_id, m.match_date ASC
    """)
    raw = pd.read_sql(sql, session.bind)
    if raw.empty:
        print("  No innings data.")
        return

    session.query(PlayerForm).delete()
    today = date.today()
    rows = []

    for pid, grp in raw.groupby("player_id"):
        runs  = grp["runs"].values.astype(float)
        balls = grp["balls_faced"].values.astype(float)
        n = len(runs)

        career_avg = float(np.mean(runs)) if n > 0 else None
        career_sr  = float(np.sum(runs) / np.sum(balls) * 100) if np.sum(balls) > 0 else None
        cv = float(np.std(runs) / career_avg * 100) if career_avg and career_avg > 0 else None

        def _avg(w):
            if n < 3:
                return None
            r = runs[-w:] if n >= w else runs
            return float(np.mean(r))

        def _sr(w):
            if n < 3:
                return None
            r = runs[-w:]  if n >= w else runs
            b = balls[-w:] if n >= w else balls
            return float(np.sum(r) / np.sum(b) * 100) if np.sum(b) > 0 else None

        avg_10 = _avg(10)
        breakout_flag  = bool(avg_10 and career_avg and avg_10 > career_avg * 1.2 and n >= 10)
        breakout_delta = float(avg_10 - career_avg) if avg_10 and career_avg else None

        rows.append(PlayerForm(
            player_id     =int(pid),
            avg_5         =_avg(5),
            avg_10        =avg_10,
            avg_20        =_avg(20),
            sr_5          =_sr(5),
            sr_10         =_sr(10),
            sr_20         =_sr(20),
            career_avg    =career_avg,
            career_sr     =career_sr,
            cv            =cv,
            breakout_flag =breakout_flag,
            breakout_delta=breakout_delta,
            innings_total =n,
            updated_at    =today,
        ))

    session.bulk_save_objects(rows)
    session.commit()
    print(f"  Done: {len(rows)} players.")
