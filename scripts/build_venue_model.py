"""Backtest and (optionally) build the venue-aware expectation model. See src/analytics/venue_model.py for the theory.

  python scripts/build_venue_model.py                # backtest only, prints results, writes nothing
  python scripts/build_venue_model.py --build        # backtest, then refit on all data and write venue_profile / venue_model_meta / models
Needs data/venue_geo.csv (altitude and coordinates; see scripts/geocode_venues.py)."""
import json
import sqlite3
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parents[1]; sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss

import src.analytics.model as M
import src.analytics.venue_model as V
from config import DB_PATH

BUILD = "--build" in sys.argv; T0 = time.time()
def log(*a): print(f"[{time.time()-T0:4.0f}s]", *a, flush=True)
con = sqlite3.connect(DB_PATH); q = lambda sql, **kw: pd.read_sql(sql, con, params=kw or None)
SPLIT = pd.Timestamp(M.SPLIT_DATE)
geo = pd.read_csv(ROOT / "data" / "venue_geo.csv"); meta = V.load_venue_meta(q, geo)
inn = V.load_team_innings(q); trn = (inn.date < SPLIT).values
log(f"team innings {len(inn):,} (train {int(trn.sum()):,}, test {int((~trn).sum()):,}); venues with elevation {int(meta.elevation_m.notna().sum())}/{len(meta)}; with boundary size {int(meta.boundary_mean_m.notna().sum())}")

# ───────── venue profile as it would have looked on the split date ─────────
eff_tr, sd_tr = V.fit_venue_effects(inn[trn]); n_tr = inn[trn].groupby("venue_id").size()
prior_fn, prior_r2, coefs, prior_info = V.fit_prior(meta, eff_tr, n_tr)
prof_tr, tau_tr = V.pool(meta, eff_tr, n_tr, sd_tr, prior_fn, V.bootstrap_se(inn[trn]))
G = float(inn.loc[trn, "t_runs"].mean())
log(f"physics prior (unseen venues, cross-validated): R2 = {prior_r2:.3f} over {prior_info['n_venues']} venues | between-venue sd tau = {tau_tr:.1f} runs/120 balls | league mean {G:.1f}")
log("prior coefficients (runs per 120 balls for +1 sd of the trait):", {k: round(float(v), 2) for k, v in coefs.head(6).items()})
for k in ("boundary_mean_m", "elevation_m", "boundary_asym_m"):
    d = meta.merge(pd.DataFrame({"venue_id": list(eff_tr["runs"]), "eff": list(eff_tr["runs"].values())}), on="venue_id"); d = d[d.venue_id.map(n_tr).fillna(0) >= 20].dropna(subset=[k])
    log(f"   corr({k}, adjusted run effect) = {np.corrcoef(d[k], d.eff)[0, 1]:+.2f} over {len(d)} venues")
oof_eff = V.cross_fit_effects(inn, SPLIT)                       # raw ridge effects, no look-ahead; used for training rows
inn_x = inn.set_index("innings_id")
fin = prof_tr.set_index("venue_id")["final"]; pri_only = prof_tr.set_index("venue_id")["prior"]

def x_for(frame, train_mask, mode):
    """Venue index for each player row: effect / league mean. Train rows use out-of-fold raw effects; test rows use the pooled (or raw) train-fit value."""
    oof = frame.innings_id.map(pd.Series(oof_eff["av_runs"].values, index=inn.innings_id.values)).fillna(0).values
    if mode == "raw": test_v = frame.venue_id.map(eff_tr["runs"]).fillna(0).values
    elif mode == "prior": test_v = frame.venue_id.map(pri_only).fillna(0).values
    else: test_v = frame.venue_id.map(fin).fillna(0).values
    return np.where(train_mask, oof, test_v) / G

RESULTS = {}
# ───────── batters ─────────
bat = V.bat_frame(q); testb = (bat.match_date >= SPLIT).values; pri = V.bat_priors(bat, SPLIT); Xb = V.bat_matrix(bat, pri); yb = bat.runs.values.astype(float)
nv = bat.venue_id.map(n_tr).fillna(0).values; pv_n = bat.groupby(["player_id", "venue_id"]).cumcount().values
subs = {"all test": np.ones(len(bat), bool), "first time at venue": pv_n == 0, "data-poor venue (<30 train innings)": nv < 30, "venue never seen in training": nv == 0}
base = np.zeros(len(bat))
for a, b in GroupKFold(5).split(Xb[~testb], groups=bat.match_id[~testb]):
    ia, ib = np.where(~testb)[0][a], np.where(~testb)[0][b]; base[ib] = V._hgr().fit(Xb.loc[ia, V.BAT_BASE], yb[ia]).predict(Xb.loc[ib, V.BAT_BASE])
base_model = V._hgr().fit(Xb.loc[~testb, V.BAT_BASE], yb[~testb]); base[testb] = base_model.predict(Xb.loc[testb, V.BAT_BASE])
log("batting baseline models fitted")
out = {"baseline: player only (no venue)": {k: V.score(yb, base, testb & m) for k, m in subs.items()}}
for mode, label in (("raw", "baseline x venue index (raw ground effect)"), ("pooled", "baseline x venue index (pooled with physics prior)")):
    x = x_for(bat, ~testb, mode); e = V.fit_elasticity(yb[~testb], base[~testb], x[~testb]); p = base * (1 + e * x)
    out[label] = {k: V.score(yb, p, testb & m) for k, m in subs.items()}; out[label]["elasticity"] = e
    out[label]["boot"] = {k: V.paired_boot(yb, base, p, testb & subs[k]) for k in ("all test", "first time at venue", "venue never seen in training")}
    if mode == "pooled": E_BAT, XB, PB = e, x, p
xp = x_for(bat, ~testb, "prior"); lab = "baseline x venue index (physics prior only; for venues never seen)"
mask = testb & subs["venue never seen in training"]; p = base * (1 + E_BAT * xp)
out[lab] = {"venue never seen in training": V.score(yb, p, mask), "boot": {"venue never seen in training": V.paired_boot(yb, base, p, mask)}}
RESULTS["batting"] = out
for k, v in out.items():
    s = v.get("all test") or v["venue never seen in training"]; log(f"BAT {k:70s} MAE {s['mae']:.3f} RMSE {s['rmse']:.3f} R2 {s['r2']:.3f}" + (f" | elasticity {v['elasticity']:.2f}" if "elasticity" in v else ""))
# probability of 30+ with and without the venue index
yc = (yb >= 30).astype(int); Xc = Xb[V.BAT_BASE].copy(); Xc["vx"] = XB
c0 = V.HGC(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0, random_state=0, categorical_features="from_dtype").fit(Xb.loc[~testb, V.BAT_BASE], yc[~testb])
c1 = V.HGC(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0, random_state=0, categorical_features="from_dtype").fit(Xc[~testb], yc[~testb])
p0, p1 = c0.predict_proba(Xb.loc[testb, V.BAT_BASE])[:, 1], c1.predict_proba(Xc[testb])[:, 1]
RESULTS["p30"] = {"without venue": dict(auc=float(roc_auc_score(yc[testb], p0)), logloss=float(log_loss(yc[testb], p0))), "with venue": dict(auc=float(roc_auc_score(yc[testb], p1)), logloss=float(log_loss(yc[testb], p1)))}
log("P(30+) without venue:", {k: round(v, 4) for k, v in RESULTS["p30"]["without venue"].items()}, "| with:", {k: round(v, 4) for k, v in RESULTS["p30"]["with venue"].items()})

# ───────── bowlers ─────────
bw = V.bowl_frame(q); prw = V.bowl_priors(bw, SPLIT); Xw = V.bowl_matrix(bw, prw)
econ = (bw.runs_conceded * 6 / bw.balls_bowled.clip(lower=1)).values; valid = (bw.balls_bowled >= 6).values & (econ > 2) & (econ < 24); testw = (bw.match_date >= SPLIT).values
nvw = bw.venue_id.map(n_tr).fillna(0).values; pvw = bw.groupby(["player_id", "venue_id"]).cumcount().values
subw = {"all test": valid, "first time at venue": valid & (pvw == 0), "data-poor venue (<30 train innings)": valid & (nvw < 30), "venue never seen in training": valid & (nvw == 0)}
trw = valid & ~testw; basew = np.zeros(len(bw)); idx_tr = np.where(trw)[0]
for a, b in GroupKFold(5).split(idx_tr, groups=bw.match_id.values[idx_tr]):
    ia, ib = idx_tr[a], idx_tr[b]; basew[ib] = V._hgr().fit(Xw.loc[ia, V.BOWL_BASE], econ[ia]).predict(Xw.loc[ib, V.BOWL_BASE])
bwm = V._hgr().fit(Xw.loc[trw, V.BOWL_BASE], econ[trw]); basew[testw] = bwm.predict(Xw.loc[testw, V.BOWL_BASE])
outw = {"baseline: bowler only (no venue)": {k: V.score(econ, basew, testw & m) for k, m in subw.items()}}
for mode, label in (("raw", "baseline x venue index (raw ground effect)"), ("pooled", "baseline x venue index (pooled with physics prior)")):
    x = x_for(bw, ~testw, mode); e = V.fit_elasticity(econ[trw], basew[trw], x[trw]); p = basew * (1 + e * x)
    outw[label] = {k: V.score(econ, p, testw & m) for k, m in subw.items()}; outw[label]["elasticity"] = e
    outw[label]["boot"] = {k: V.paired_boot(econ, basew, p, testw & subw[k]) for k in ("all test", "first time at venue", "venue never seen in training")}
    if mode == "pooled": E_BOWL = e; PW = p
RESULTS["bowling"] = outw
for k, v in outw.items():
    s = v["all test"]; log(f"BOWL {k:68s} MAE {s['mae']:.3f} RMSE {s['rmse']:.3f} R2 {s['r2']:.3f}" + (f" | elasticity {v['elasticity']:.2f}" if "elasticity" in v else ""))
RESULTS["prior"] = dict(cv_r2=prior_r2, tau=tau_tr, coefs={k: float(v) for k, v in coefs.items()}, info=prior_info, G=G, sd_resid=sd_tr)
json.dump(RESULTS, open(ROOT / "data" / "venue_model_backtest.json", "w"), indent=1, default=float) if BUILD else json.dump(RESULTS, open("/tmp/venue_model_backtest.json", "w"), indent=1, default=float)
log("backtest done")
if not BUILD: sys.exit(0)


# ═════════════ production fit on ALL data ═════════════
import shutil, datetime
shutil.copy(DB_PATH, Path.home() / "etpl2026" / "backups" / "cricket.db.pre_venue_model")
FAR = pd.Timestamp("2100-01-01")
eff, sd_all = V.fit_venue_effects(inn); n_all = inn.groupby("venue_id").size()
prior_fn, prior_r2_all, coefs_all, info_all = V.fit_prior(meta, eff, n_all)
se_boot = V.bootstrap_se(inn); prof, tau_all = V.pool(meta, eff, n_all, sd_all, prior_fn, se_boot)
G_all = float(inn.t_runs.mean())
oof_all = V.cross_fit_effects(inn, FAR)
oof_map = pd.Series(oof_all["av_runs"].values, index=inn.innings_id.values)
log(f"production profile: {len(prof)} venues, {int((prof.n > 0).sum())} with matches | tau {tau_all:.1f} | prior CV R2 {prior_r2_all:.3f} | league mean {G_all:.1f}")
# batters
pri_b = V.bat_priors(bat, FAR); Xb2 = V.bat_matrix(bat, pri_b); xb = bat.innings_id.map(oof_map).fillna(0).values / G_all
base2 = np.zeros(len(bat))
for a, b in GroupKFold(5).split(Xb2, groups=bat.match_id):
    base2[b] = V._hgr().fit(Xb2.iloc[a][V.BAT_BASE], yb[a]).predict(Xb2.iloc[b][V.BAT_BASE])
E_BAT2 = V.fit_elasticity(yb, base2, xb)
bat_base_final = V._hgr().fit(Xb2[V.BAT_BASE], yb)
# bowlers
pri_w = V.bowl_priors(bw, FAR); Xw2 = V.bowl_matrix(bw, pri_w); xw = bw.innings_id.map(oof_map).fillna(0).values / G_all
idx = np.where(valid)[0]; basew2 = np.zeros(len(bw))
for a, b in GroupKFold(5).split(idx, groups=bw.match_id.values[idx]):
    basew2[idx[b]] = V._hgr().fit(Xw2.iloc[idx[a]][V.BOWL_BASE], econ[idx[a]]).predict(Xw2.iloc[idx[b]][V.BOWL_BASE])
E_BOWL2 = V.fit_elasticity(econ[valid], basew2[valid], xw[valid])
bowl_base_final = V._hgr().fit(Xw2.loc[valid, V.BOWL_BASE], econ[valid])
log(f"elasticities on all data: batters {E_BAT2:.2f}, bowlers {E_BOWL2:.2f} (backtest {E_BAT:.2f} / {E_BOWL:.2f})")
# likely-range bands from the backtest's out-of-sample errors
pg = Xb["batting_position"].astype(int).apply(M._pos_group).values; bands_bat = {}
for g in sorted(set(pg)):
    m = testb & (pg == g) & (PB > 3)
    if m.sum() > 200: r = yb[m] / PB[m]; bands_bat[str(g)] = [float(np.percentile(r, 10)), float(np.percentile(r, 90))]
bands_bat["all"] = [float(np.percentile(yb[testb & (PB > 3)] / PB[testb & (PB > 3)], 10)), float(np.percentile(yb[testb & (PB > 3)] / PB[testb & (PB > 3)], 90))]
resw = (econ - PW)[testw & valid]; bands_bowl = [float(np.percentile(resw, 10)), float(np.percentile(resw, 90))]
# empirical bins on the out-of-sample test period: calibrated likely range and P(30+/50+) for any expected value
def bins(pred, y, k=16, probs=True):
    m = testb if probs else (testw & valid); pr, yy = pred[m], y[m]; edges = np.unique(np.quantile(pr, [0, .1, .2, .3, .4, .5, .6, .7, .8, .88, .93, .96, .98, .99, 1.0])); out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (pr >= lo) & (pr <= hi) if hi == edges[-1] else (pr >= lo) & (pr < hi)
        if sel.sum() < 50: continue
        r = dict(pred=float(pr[sel].mean()), actual=float(yy[sel].mean()), n=int(sel.sum()), **{f"q{int(a*100)}": float(np.quantile(yy[sel], a)) for a in (.1, .25, .5, .75, .9)})
        r["qgrid"] = [float(v) for v in np.quantile(yy[sel], np.linspace(0, 1, 101))]     # empirical distribution of the bin, 0..100th percentile
        if probs: r.update(p30=float((yy[sel] >= 30).mean()), p50=float((yy[sel] >= 50).mean()))
        out.append(r)
    return out
bins_bat, bins_bowl = bins(PB, yb), bins(PW, econ, probs=False)
log("calibration of the batting expectation (predicted vs actual mean runs, test period):", [(round(r["pred"], 1), round(r["actual"], 1)) for r in bins_bat[::3]])
log("bowling:", [(round(r["pred"], 2), round(r["actual"], 2)) for r in bins_bowl[::3]])
# persist
geo.to_sql("venue_geo", con, if_exists="replace", index=False)
pf = prof.merge(meta[["venue_id", "boundary_mean_m", "elevation_m", "boundary_straight_m", "boundary_square_m"]], on="venue_id", how="left")
pf = pf.rename(columns={"n": "n_innings", "obs": "observed", "final": "final_effect", "se_final": "se_effect", "prior": "prior_effect", "se": "se_observed"})
pf.to_sql("venue_profile", con, if_exists="replace", index=False)
def num(d): return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in d.items()}
state = info_all.pop("state"); kv = {
    "as_of": datetime.date.today().isoformat(), "version": 1, "G": G_all, "tau": tau_all, "sd_resid": sd_all, "ridge_alpha": V.RIDGE_ALPHA,
    "elasticity_bat": E_BAT2, "elasticity_bowl": E_BOWL2, "prior_state": state, "prior_cv_r2": prior_r2_all, "prior_info": info_all,
    "priors_bat": num(pri_b), "priors_bowl": num(pri_w), "bands_bat": bands_bat, "bands_bowl": bands_bowl, "bins_bat": bins_bat, "bins_bowl": bins_bowl,
    "backtest": {"split": str(SPLIT.date()), "results": RESULTS, "n_train_innings": int((~testb).sum()), "n_test_innings": int(testb.sum())},
    "counts": {"team_innings": int(len(inn)), "venues": int(len(prof)), "venues_with_matches": int((prof.n > 0).sum()), "bat_rows": int(len(bat)), "bowl_spells": int(valid.sum())}}
pd.DataFrame({"key": list(kv), "value": [json.dumps(v, default=float) for v in kv.values()]}).to_sql("venue_model_meta", con, if_exists="replace", index=False)
joblib.dump({"bat_base": bat_base_final, "bowl_base": bowl_base_final,
             "bat_cats": list(Xb2["tournament"].cat.categories), "bowl_cats": list(Xw2["tournament"].cat.categories)}, ROOT / "data" / "models" / "venue_models.joblib")
con.commit(); con.execute("PRAGMA wal_checkpoint(TRUNCATE)"); con.close()
log("built: tables venue_geo, venue_profile, venue_model_meta; models data/models/venue_models.joblib; backtest data/venue_model_backtest.json")
