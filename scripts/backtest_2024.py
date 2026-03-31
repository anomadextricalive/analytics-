"""
Leave-one-season-out backtest: train on ≤2023, predict 2024.

This gives honest out-of-sample metrics — no data leakage.
Career features (avg, SR, economy) are computed from pre-2024 data only,
so the model never sees future information.

Usage:
  python scripts/backtest_2024.py
  python scripts/backtest_2024.py --test-season 2023   # test on 2023 instead
  python scripts/backtest_2024.py --plot               # save residual plots
"""

import sys
import warnings
from pathlib import Path

import click
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from sqlalchemy.orm import Session

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH
from src.db.schema import get_engine
from src.analytics.model import (
    TOURNAMENT_MAP, BAT_FEATURES, BOWL_FEATURES,
    _make_bat_matrix, _make_bowl_matrix,
)

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (season-aware — no leakage)
# ─────────────────────────────────────────────────────────────────────────────

def _load_bat(session: Session, test_season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (train_df, test_df).
    Career features are computed from train data only and joined in.
    """
    df = pd.read_sql(text("""
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
            COALESCE(m.tournament, 'other')          AS tournament,
            m.season                                 AS season
        FROM player_innings pi
        JOIN matches m ON m.id = pi.match_id
        LEFT JOIN venue_difficulty vd ON vd.venue_id = m.venue_id
        WHERE pi.balls_faced >= 4
    """), session.bind)

    train = df[df["season"] != test_season].copy()
    test  = df[df["season"] == test_season].copy()

    # Career features from train only — no leakage
    career = (train.groupby("player_id")
                   .agg(career_adj_avg=("runs", "mean"),
                        career_adj_sr=("runs", lambda x: (x.sum() / train.loc[x.index, "balls_faced"].sum() * 100)),
                        career_innings=("runs", "count"))
                   .reset_index())

    chase = (train.groupby("player_id")
                  .apply(lambda g: pd.Series({
                      "chase_avg": g.loc[g["is_chase"]==1, "runs"].mean() if (g["is_chase"]==1).any() else 20,
                      "first_avg": g.loc[g["is_chase"]==0, "runs"].mean() if (g["is_chase"]==0).any() else 20,
                      "chase_sr":  120.0,
                  }), include_groups=False)
                  .reset_index())

    for split in [train, test]:
        split = split.merge(career, on="player_id", how="left")
        split = split.merge(chase,  on="player_id", how="left")
        for col, default in [("career_adj_avg",20),("career_adj_sr",120),
                              ("career_innings",10),("chase_avg",20),
                              ("first_avg",20),("chase_sr",120)]:
            split[col] = split[col].fillna(default)

    train = train.merge(career, on="player_id", how="left").merge(chase, on="player_id", how="left")
    test  = test.merge(career,  on="player_id", how="left").merge(chase, on="player_id", how="left")

    for col, default in [("career_adj_avg",20),("career_adj_sr",120),
                          ("career_innings",10),("chase_avg",20),
                          ("first_avg",20),("chase_sr",120)]:
        train[col] = train[col].fillna(default)
        test[col]  = test[col].fillna(default)

    return train, test


def _load_bowl(session: Session, test_season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_sql(text("""
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
            COALESCE(m.tournament, 'other')          AS tournament,
            m.season                                 AS season
        FROM player_bowling_innings pbi
        JOIN matches m ON m.id = pbi.match_id
        LEFT JOIN venue_difficulty vd ON vd.venue_id = m.venue_id
        WHERE pbi.balls_bowled >= 6
    """), session.bind)

    df["econ"] = (df["runs_conceded"] * 6 / df["balls_bowled"].clip(lower=1)).clip(2, 24)
    df = df[(df["econ"] > 2) & (df["econ"] < 24)]

    train = df[df["season"] != test_season].copy()
    test  = df[df["season"] == test_season].copy()

    # Career features from train only
    career = (train.groupby("player_id")
                   .agg(career_adj_econ=("econ", "mean"),
                        career_dot_pct=("dot_balls", lambda x: x.sum() / train.loc[x.index, "balls_bowled"].sum() * 100),
                        career_bowl_inn=("econ", "count"),
                        pp_economy=("pp_runs", lambda x: x.sum() / train.loc[x.index, "pp_balls"].clip(lower=1).sum() * 6),
                        mid_economy=("mid_runs", lambda x: x.sum() / train.loc[x.index, "mid_balls"].clip(lower=1).sum() * 6),
                        death_economy=("death_runs", lambda x: x.sum() / train.loc[x.index, "death_balls"].clip(lower=1).sum() * 6))
                   .reset_index())

    train = train.merge(career, on="player_id", how="left")
    test  = test.merge(career,  on="player_id", how="left")

    for col, default in [("career_adj_econ",8.5),("career_dot_pct",33),
                          ("career_bowl_inn",10),("pp_economy",8.0),
                          ("mid_economy",8.0),("death_economy",9.5)]:
        train[col] = train[col].fillna(default)
        test[col]  = test[col].fillna(default)

    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--test-season", default="2024", show_default=True,
              help="Season to hold out as test set")
@click.option("--plot", is_flag=True, default=False,
              help="Save residual plots to data/models/")
def main(test_season: str, plot: bool):
    """Leave-one-season-out backtest."""

    if not DB_PATH.exists():
        console.print(f"[red]DB not found: {DB_PATH}[/red]")
        sys.exit(1)

    engine = get_engine()
    session = Session(engine)

    console.print(f"[bold]Backtest: train on <{test_season}, test on {test_season}[/bold]\n")

    # ── BATTING ──────────────────────────────────────────────────────────────
    console.print("Loading batting data…")
    bat_train, bat_test = _load_bat(session, test_season)
    console.print(f"  Train: {len(bat_train):,} innings  |  Test: {len(bat_test):,} innings")

    X_train = _make_bat_matrix(bat_train)
    X_test  = _make_bat_matrix(bat_test)
    y_train = bat_train["runs"].values.astype(float)
    y_test  = bat_test["runs"].values.astype(float)

    sc = StandardScaler().fit(X_train)
    bat_model = GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=5,
        subsample=0.8, min_samples_leaf=30, random_state=42,
    )
    console.print("Training batting model on pre-2024 data…")
    bat_model.fit(sc.transform(X_train), y_train)

    pred_train = bat_model.predict(sc.transform(X_train))
    pred_test  = bat_model.predict(sc.transform(X_test))

    bat_results = {
        "train_r2":  round(r2_score(y_train, pred_train), 3),
        "train_mae": round(mean_absolute_error(y_train, pred_train), 2),
        "test_r2":   round(r2_score(y_test, pred_test), 3),
        "test_mae":  round(mean_absolute_error(y_test, pred_test), 2),
        "n_train":   len(y_train),
        "n_test":    len(y_test),
    }

    # Per-tournament breakdown
    bat_test["pred"] = pred_test
    bat_test["error"] = bat_test["pred"] - bat_test["runs"]

    # ── BOWLING ──────────────────────────────────────────────────────────────
    console.print("Loading bowling data…")
    bowl_train, bowl_test = _load_bowl(session, test_season)
    console.print(f"  Train: {len(bowl_train):,} spells  |  Test: {len(bowl_test):,} spells")

    X_btr = _make_bowl_matrix(bowl_train)
    X_bte = _make_bowl_matrix(bowl_test)
    y_btr = bowl_train["econ"].values
    y_bte = bowl_test["econ"].values

    bsc = StandardScaler().fit(X_btr)
    bowl_model = GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=4,
        subsample=0.8, min_samples_leaf=20, random_state=42,
    )
    console.print("Training bowling model on pre-2024 data…")
    bowl_model.fit(bsc.transform(X_btr), y_btr)

    pred_btr = bowl_model.predict(bsc.transform(X_btr))
    pred_bte = bowl_model.predict(bsc.transform(X_bte))

    bowl_results = {
        "train_r2":  round(r2_score(y_btr, pred_btr), 3),
        "train_mae": round(mean_absolute_error(y_btr, pred_btr), 3),
        "test_r2":   round(r2_score(y_bte, pred_bte), 3),
        "test_mae":  round(mean_absolute_error(y_bte, pred_bte), 3),
        "n_train":   len(y_btr),
        "n_test":    len(y_bte),
    }

    bowl_test["pred"] = pred_bte
    bowl_test["error"] = bowl_test["pred"] - bowl_test["econ"]

    session.close()

    # ── PRINT RESULTS ────────────────────────────────────────────────────────
    console.print()
    t = Table(title=f"Backtest Results — Test Season: {test_season}", show_header=True,
              header_style="bold cyan")
    t.add_column("Model",      style="bold")
    t.add_column("Train R²",   justify="right")
    t.add_column("Test R²",    justify="right")
    t.add_column("Train MAE",  justify="right")
    t.add_column("Test MAE",   justify="right")
    t.add_column("Overfit Gap",justify="right")

    bat_gap  = round(bat_results["train_r2"]  - bat_results["test_r2"],  3)
    bowl_gap = round(bowl_results["train_r2"] - bowl_results["test_r2"], 3)

    t.add_row("Batting (runs)",
              str(bat_results["train_r2"]),
              f"[{'green' if bat_results['test_r2'] > 0.5 else 'yellow'}]{bat_results['test_r2']}[/]",
              f"{bat_results['train_mae']} runs",
              f"{bat_results['test_mae']} runs",
              f"[{'red' if bat_gap > 0.1 else 'green'}]{bat_gap}[/]")

    t.add_row("Bowling (econ)",
              str(bowl_results["train_r2"]),
              f"[{'green' if bowl_results['test_r2'] > 0.05 else 'yellow'}]{bowl_results['test_r2']}[/]",
              f"{bowl_results['train_mae']} rpm",
              f"{bowl_results['test_mae']} rpm",
              f"[{'red' if bowl_gap > 0.1 else 'green'}]{bowl_gap}[/]")

    console.print(t)

    # ── PER-TOURNAMENT BREAKDOWN ─────────────────────────────────────────────
    console.print("\n[bold]Batting MAE by tournament (test set):[/bold]")
    by_t = (bat_test.groupby("tournament")
                    .apply(lambda g: pd.Series({
                        "innings": len(g),
                        "mae":     round(mean_absolute_error(g["runs"], g["pred"]), 2),
                        "bias":    round(g["error"].mean(), 2),
                    }), include_groups=False)
                    .reset_index()
                    .sort_values("mae"))
    console.print(by_t.to_string(index=False))

    # ── BIAS CHECK ───────────────────────────────────────────────────────────
    console.print(f"\n[bold]Batting bias (mean error):[/bold] {round(bat_test['error'].mean(), 2)} runs")
    console.print(f"[bold]Bowling bias (mean error):[/bold] {round(bowl_test['error'].mean(), 3)} econ")

    pct_within_10 = (abs(bat_test["error"]) <= 10).mean() * 100
    pct_within_20 = (abs(bat_test["error"]) <= 20).mean() * 100
    console.print(f"\n[bold]Batting predictions within:[/bold]")
    console.print(f"  ±10 runs: {pct_within_10:.1f}% of innings")
    console.print(f"  ±20 runs: {pct_within_20:.1f}% of innings")

    # ── PLOTS ────────────────────────────────────────────────────────────────
    if plot:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].scatter(y_test, pred_test, alpha=0.2, s=8, color="steelblue")
            axes[0].plot([0, y_test.max()], [0, y_test.max()], "r--", linewidth=1)
            axes[0].set_xlabel("Actual runs"); axes[0].set_ylabel("Predicted runs")
            axes[0].set_title(f"Batting — Test R²={bat_results['test_r2']}")

            axes[1].scatter(y_bte, pred_bte, alpha=0.2, s=8, color="coral")
            axes[1].plot([2, 20], [2, 20], "r--", linewidth=1)
            axes[1].set_xlabel("Actual economy"); axes[1].set_ylabel("Predicted economy")
            axes[1].set_title(f"Bowling — Test R²={bowl_results['test_r2']}")

            out = Path("data/models/backtest_2024.png")
            plt.tight_layout()
            plt.savefig(out, dpi=150)
            console.print(f"\n[green]Plot saved → {out}[/green]")
        except Exception as e:
            console.print(f"[yellow]Plot skipped: {e}[/yellow]")


if __name__ == "__main__":
    main()
