"""
Optuna hyperparameter search for batting model.
Compares GBM (sklearn) vs XGBoost on 2024 leave-one-season-out CV.

Usage:
  python scripts/tune_models.py               # 60 trials, batting only
  python scripts/tune_models.py --trials 100  # more trials
"""

import sys
import warnings
from pathlib import Path

import click
import numpy as np
import optuna
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session
import joblib

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DB_PATH
from src.db.schema import get_engine
from src.analytics.model import (
    _make_bat_matrix, BAT_MODEL_PATH, BAT_SCALER_PATH, META_PATH,
)
from scripts.backtest_2024 import _load_bat

console = Console()


def _get_xy(session: Session, test_season: str = "2024"):
    """Load train/test matrices with no leakage."""
    train, test = _load_bat(session, test_season)
    X_tr = _make_bat_matrix(train);  y_tr = train["runs"].values.astype(float)
    X_te = _make_bat_matrix(test);   y_te = test["runs"].values.astype(float)
    sc = StandardScaler().fit(X_tr)
    return sc.transform(X_tr), y_tr, sc.transform(X_te), y_te, sc


# ─────────────────────────────────────────────────────────────────────────────
# OPTUNA OBJECTIVES
# ─────────────────────────────────────────────────────────────────────────────

def _objective_gbm(trial, X_tr, y_tr):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators",    100, 400),
        "learning_rate":    trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "max_depth":        trial.suggest_int("max_depth",       3, 6),
        "subsample":        trial.suggest_float("subsample",     0.6, 1.0),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf",10, 60),
        "max_features":     trial.suggest_float("max_features",  0.5, 1.0),
        "random_state": 42,
    }
    model = GradientBoostingRegressor(**params)
    cv = -cross_val_score(model, X_tr, y_tr, cv=3,   # 3-fold to keep it fast
                          scoring="neg_mean_absolute_error", n_jobs=1)
    return cv.mean()


def _objective_xgb(trial, X_tr, y_tr):
    from xgboost import XGBRegressor
    params = {
        "n_estimators":     trial.suggest_int("n_estimators",    100, 600),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth":        trial.suggest_int("max_depth",       3, 7),
        "subsample":        trial.suggest_float("subsample",     0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
        "reg_alpha":        trial.suggest_float("reg_alpha",     0.0, 1.0),
        "reg_lambda":       trial.suggest_float("reg_lambda",    0.5, 5.0),
        "n_jobs": -1, "random_state": 42, "verbosity": 0,
    }
    model = XGBRegressor(**params)
    cv = -cross_val_score(model, X_tr, y_tr, cv=5,
                          scoring="neg_mean_absolute_error", n_jobs=1)
    return cv.mean()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--trials", default=60, show_default=True, help="Optuna trials per model")
@click.option("--test-season", default="2024", show_default=True)
def main(trials: int, test_season: str):
    engine  = get_engine()
    session = Session(engine)

    console.print(f"[bold]Loading data (test season: {test_season})…[/bold]")
    X_tr, y_tr, X_te, y_te, scaler = _get_xy(session, test_season)
    session.close()
    console.print(f"  Train: {len(y_tr):,}  |  Test: {len(y_te):,}\n")

    results = []

    # ── Baseline (current model) ──────────────────────────────────────────
    console.print("[cyan]Baseline GBM (current hyperparams)…[/cyan]")
    base = GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=5,
        subsample=0.8, min_samples_leaf=30, random_state=42,
    ).fit(X_tr, y_tr)
    results.append({
        "model": "GBM baseline",
        "test_r2":  round(r2_score(y_te,  base.predict(X_te)),  3),
        "test_mae": round(mean_absolute_error(y_te, base.predict(X_te)), 3),
        "params":   "current defaults",
    })
    console.print(f"  MAE={results[-1]['test_mae']}  R²={results[-1]['test_r2']}")

    # ── Optuna GBM ───────────────────────────────────────────────────────
    console.print(f"\n[cyan]Tuning GBM ({trials} trials)…[/cyan]")
    gbm_study = optuna.create_study(direction="minimize")
    with console.status("Searching…"):
        gbm_study.optimize(lambda t: _objective_gbm(t, X_tr, y_tr),
                           n_trials=trials, show_progress_bar=False)

    best_gbm_params = {**gbm_study.best_params, "random_state": 42}
    best_gbm = GradientBoostingRegressor(**best_gbm_params).fit(X_tr, y_tr)
    results.append({
        "model": "GBM tuned",
        "test_r2":  round(r2_score(y_te,  best_gbm.predict(X_te)),  3),
        "test_mae": round(mean_absolute_error(y_te, best_gbm.predict(X_te)), 3),
        "params":   str(gbm_study.best_params),
    })
    console.print(f"  MAE={results[-1]['test_mae']}  R²={results[-1]['test_r2']}")
    console.print(f"  Best params: {gbm_study.best_params}")

    # ── Optuna XGBoost ───────────────────────────────────────────────────
    console.print(f"\n[cyan]Tuning XGBoost ({trials} trials)…[/cyan]")
    xgb_study = optuna.create_study(direction="minimize")
    with console.status("Searching…"):
        xgb_study.optimize(lambda t: _objective_xgb(t, X_tr, y_tr),
                           n_trials=trials, show_progress_bar=False)

    from xgboost import XGBRegressor
    best_xgb_params = {**xgb_study.best_params, "random_state": 42, "verbosity": 0}
    best_xgb = XGBRegressor(**best_xgb_params).fit(X_tr, y_tr)
    results.append({
        "model": "XGBoost tuned",
        "test_r2":  round(r2_score(y_te,  best_xgb.predict(X_te)),  3),
        "test_mae": round(mean_absolute_error(y_te, best_xgb.predict(X_te)), 3),
        "params":   str(xgb_study.best_params),
    })
    console.print(f"  MAE={results[-1]['test_mae']}  R²={results[-1]['test_r2']}")
    console.print(f"  Best params: {xgb_study.best_params}")

    # ── Results table ────────────────────────────────────────────────────
    console.print()
    t = Table(title="Tuning Results — Batting Model", header_style="bold cyan")
    t.add_column("Model",    style="bold")
    t.add_column("Test R²",  justify="right")
    t.add_column("Test MAE", justify="right")
    t.add_column("vs Baseline", justify="right")

    base_mae = results[0]["test_mae"]
    for r in results:
        diff = round(r["test_mae"] - base_mae, 3)
        diff_str = f"[green]{diff}[/green]" if diff <= 0 else f"[red]+{diff}[/red]"
        t.add_row(r["model"], str(r["test_r2"]), f"{r['test_mae']} runs", diff_str)
    console.print(t)

    # ── Save best model ──────────────────────────────────────────────────
    best = min(results[1:], key=lambda x: x["test_mae"])
    console.print(f"\n[bold]Best model: {best['model']} (MAE={best['test_mae']})[/bold]")

    if best["model"] == "GBM tuned":
        winner = best_gbm
    else:
        winner = best_xgb

    save = click.confirm("Save best model as production model?", default=True)
    if save:
        joblib.dump(winner, BAT_MODEL_PATH)
        joblib.dump(scaler, BAT_SCALER_PATH)
        # update meta
        meta = joblib.load(META_PATH) if META_PATH.exists() else {}
        meta["bat_r2"]  = best["test_r2"]
        meta["bat_mae"] = best["test_mae"]
        meta["bat_model_type"] = best["model"]
        meta["bat_best_params"] = best["params"]
        joblib.dump(meta, META_PATH)
        console.print(f"[green]✓ Saved to {BAT_MODEL_PATH}[/green]")


if __name__ == "__main__":
    main()
