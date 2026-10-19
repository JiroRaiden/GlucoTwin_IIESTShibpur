"""Phase 2: calibrate one physiological twin per patient and prove it learns.

Outputs
  reports/twins.json               calibrated parameters per patient (used by the dashboard)
  reports/ehr_prior.pkl            EHR -> twin-parameter model (cold start for new patients)
  reports/phase2_summary.json      every number quoted in the README / slides
  reports/figures/p2_*.png
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from glucotwin import config
from glucotwin.twin import calibrate as C
from glucotwin.twin import model as M
from glucotwin.analysis import figures_twin as F

DAYS_GRID = [0, 1, 2, 3, 5, 99]   # 99 = all training days


def load_processed():
    grid_all = pd.read_parquet(config.PROCESSED_DIR / "grid_5min.parquet")
    grids = {pid: g.droplevel(0) for pid, g in grid_all.groupby(level=0)}
    meals = pd.read_parquet(config.PROCESSED_DIR / "meals.parquet")
    bio = pd.read_parquet(config.PROCESSED_DIR / "bio.parquet")
    return bio, grids, meals


def person_context(pid, bio, grids):
    row = bio.loc[pid]
    g = grids[pid]
    train = g[g.index < C.split_cutoff(g)]
    return C.basal_glucose(train, row.get("fasting_glucose", 100)), C.body_weight_kg(row)


_S: dict = {}          # shared state for worker processes (inherited via fork)


def _fit_train_sse(p):
    W = _S["wins"][p].subset(~_S["wins"][p].test)
    if len(W) < 3:
        return 0.0, 0
    Gb, BW = _S["ctx"][p]
    lt, _ = C.fit_twin(W, Gb, BW, M.DEFAULT, _S["glob"], prior_weight=0.3)
    e = (C.predict_windows(lt, W, Gb, BW, _S["glob"]) - W.obs)[:, C.PRE_STEPS + 1:]
    return float(np.nansum(e ** 2)), int(np.sum(~np.isnan(e)))


def choose_globals(bio, grids, meals, pids):
    """Pooled grid search for the two population constants, on TRAINING windows only."""
    best, results = None, []
    _S["ctx"] = {p: person_context(p, bio, grids) for p in pids}
    for beta in [0.0, 0.5]:
        _S["wins"] = {p: C.build_windows(p, grids[p], meals, beta) for p in pids}
        for alpha in [0.0, 0.3]:
            _S["glob"] = M.Globals(alpha=alpha, beta=beta)
            with mp.Pool(os.cpu_count() or 2) as pool:      # fork after the state is set
                out = pool.map(_fit_train_sse, pids)
            rmse = float(np.sqrt(sum(o[0] for o in out) / sum(o[1] for o in out)))
            results.append({"alpha": alpha, "beta": beta, "train_rmse": rmse})
            print(f"      alpha={alpha:<5} beta={beta:<4} train RMSE {rmse:6.2f}", flush=True)
            if best is None or rmse < best[0]:
                best = (rmse, _S["glob"])
    return best[1], results


def _stage_a(p):
    W = _S["wins"][p].subset(~_S["wins"][p].test)
    return C.fit_twin(W, *_S["ctx"][p], M.DEFAULT, _S["glob"], prior_weight=0.3)[0]


def _evaluate(p):
    """Leave-one-patient-out EHR prior, baselines and learning curve for one patient."""
    bio, wins, glob, theta_A, pop_median = _S["bio"], _S["wins"], _S["glob"], _S["theta_A"], _S["pop_median"]
    others = {q: theta_A[q] for q in theta_A if q != p}
    prior = C.EHRPrior().fit(bio, others).predict(bio.loc[p])
    Wtr, Wte = wins[p].subset(~wins[p].test), wins[p].subset(wins[p].test)
    Gb, BW = _S["ctx"][p]
    rows, examples, twin = [], [], None
    persist = np.repeat(Wte.obs[:, [C.PRE_STEPS]], C.T_WIN, axis=1)
    persist[:, :C.PRE_STEPS + 1] = Wte.obs[:, :C.PRE_STEPS + 1]
    r_persist, pk_persist = C.score(np.where(np.isnan(persist), Wte.G0[:, None], persist), Wte)
    r_pop, pk_pop = C.score(C.predict_windows(pop_median, Wte, Gb, BW, glob), Wte)
    rows.append({"pid": p, "model": "No-change (glucose stays flat)", "days": np.nan, "rmse": r_persist, "peak_err": pk_persist, "n_test": len(Wte)})
    rows.append({"pid": p, "model": "Average twin (no personalisation)", "days": np.nan, "rmse": r_pop, "peak_err": pk_pop, "n_test": len(Wte)})
    first_day = Wtr.day.min()
    for k in DAYS_GRID:
        Wk = Wtr.subset(Wtr.day < first_day + k)
        lt, _ = C.fit_twin(Wk, Gb, BW, prior, glob)
        r, pk = C.score(C.predict_windows(lt, Wte, Gb, BW, glob), Wte)
        rows.append({"pid": p, "model": "Personal twin", "days": k, "rmse": r, "peak_err": pk,
                     "n_test": len(Wte), "n_train_meals": len(Wk)})
        if k == 99:
            twin = M.TwinParams.from_log(lt, Gb, BW).to_dict() | {
                "status": bio.loc[p, "status"], "ehr_prior": M.TwinParams.from_log(prior, Gb, BW).to_dict(),
                "n_train_meals": int(len(Wtr)), "holdout_rmse": r}
            sim = C.predict_windows(lt, Wte, Gb, BW, glob)
            sim0 = C.predict_windows(prior, Wte, Gb, BW, glob)
            for i in range(len(Wte)):
                examples.append({"pid": p, "status": bio.loc[p, "status"], "obs": Wte.obs[i],
                                 "twin": sim[i], "ehr_only": sim0[i], "carbs": Wte.carbs[i].sum(),
                                 "rmse": float(np.sqrt(np.nanmean((sim[i] - Wte.obs[i])[C.PRE_STEPS + 1:C.PRE_STEPS + 25] ** 2)))})
    line = (f"      P{p:03d} [{bio.loc[p,'status']:<11}] train meals {len(Wtr):>3} | held-out {len(Wte):>2} "
            f"| no-change {r_persist:5.1f} | avg twin {r_pop:5.1f} | personal twin {twin['holdout_rmse']:5.1f} mg/dL")
    return p, rows, twin, examples, line


def group_separation(twins: dict) -> dict:
    """Do calibrated parameters differ between Healthy / Prediabetes / T2D? (Kruskal-Wallis on log values)"""
    from scipy.stats import kruskal
    d = pd.DataFrame(twins).T
    out = {}
    for c in ["S", "SG", "ka", "f"]:
        groups = [np.log(d.loc[d["status"] == s, c].astype(float)) for s in ["Healthy", "Prediabetes", "T2D"]]
        out[c] = float(f"{kruskal(*groups).pvalue:.2g}")
    return out


def main():
    t_start = time.time()
    bio, grids, meals = load_processed()
    pids = [p for p in grids if p in bio.index]
    print(f"[1/5] {len(pids)} patients")

    print("[2/5] Choosing population constants (exercise effect alpha, fat/protein slowing beta)", flush=True)
    glob, glob_results = choose_globals(bio, grids, meals, pids)
    print(f"      -> alpha={glob.alpha}, beta={glob.beta}", flush=True)

    _S.update(bio=bio, glob=glob, wins={p: C.build_windows(p, grids[p], meals, glob.beta) for p in pids})
    ctx = _S["ctx"]
    wins = _S["wins"]
    pids = [p for p in pids if (~wins[p].test).sum() >= 3 and wins[p].test.sum() >= 1]
    print(f"      {len(pids)} patients have >=3 training meals and >=1 held-out meal", flush=True)

    print("[3/5] Stage A: data-driven twin per patient (weak prior) -> targets for the EHR prior", flush=True)
    pool = mp.Pool(os.cpu_count() or 2)      # fork after the state is set, so workers see it
    theta_A = dict(zip(pids, pool.map(_stage_a, pids)))

    print("[4/5] Leave-one-patient-out EHR prior + learning curve on held-out days", flush=True)
    _S.update(theta_A=theta_A, pop_median=np.median(np.vstack(list(theta_A.values())), axis=0))
    pop_median = _S["pop_median"]
    pool.close(); pool = mp.Pool(os.cpu_count() or 2)
    rows, twins, examples = [], {}, []
    for p, r_, tw_, ex_, line in pool.imap(_evaluate, pids):
        rows += r_; twins[int(p)] = tw_; examples += ex_
        print(line, flush=True)
    pool.close()

    res = pd.DataFrame(rows)
    res.to_csv(config.REPO_ROOT / "reports" / "phase2_holdout_scores.csv", index=False)

    print("[5/5] Final EHR prior on all patients, figures, summary")
    final_prior = C.EHRPrior().fit(bio, theta_A)
    with open(config.REPO_ROOT / "reports" / "ehr_prior.pkl", "wb") as fh:
        pickle.dump({"prior": final_prior, "globals": glob}, fh)
    (config.REPO_ROOT / "reports" / "twins.json").write_text(json.dumps(
        {"globals": {"alpha": glob.alpha, "beta": glob.beta, "P2": M.P2, "VG": M.VG}, "patients": twins}, indent=1))

    # weighted (by number of held-out meals) mean across patients
    def agg(d):
        w = d["n_test"]
        return pd.Series({"rmse": np.average(d["rmse"], weights=w), "peak_err": np.average(d["peak_err"], weights=w)})
    lc = res[res["model"] == "Personal twin"].groupby("days").apply(agg)
    base = res[res["model"] != "Personal twin"].groupby("model").apply(agg)

    F.fig_learning_curve(lc, base, config.FIG_DIR / "p2_learning_curve.png")
    F.fig_params_by_status(twins, config.FIG_DIR / "p2_params_by_status.png")
    F.fig_example_forecasts(examples, config.FIG_DIR / "p2_example_forecasts.png", C.PRE_STEPS)

    summary = {
        "globals": {"alpha": glob.alpha, "beta": glob.beta},
        "globals_search": glob_results,
        "n_patients": len(pids),
        "holdout_rmse_2h_mgdl": {
            "no_change_baseline": round(float(base.loc["No-change (glucose stays flat)", "rmse"]), 1),
            "average_twin": round(float(base.loc["Average twin (no personalisation)", "rmse"]), 1),
            **{f"personal_twin_{'all' if k == 99 else int(k)}_days": round(float(lc.loc[k, "rmse"]), 1) for k in lc.index},
        },
        "holdout_peak_error_mgdl": {
            "no_change_baseline": round(float(base.loc["No-change (glucose stays flat)", "peak_err"]), 1),
            "average_twin": round(float(base.loc["Average twin (no personalisation)", "peak_err"]), 1),
            **{f"personal_twin_{'all' if k == 99 else int(k)}_days": round(float(lc.loc[k, "peak_err"]), 1) for k in lc.index},
        },
        "median_params_by_status": pd.DataFrame(twins).T.groupby("status")[["S", "SG", "ka", "f"]]
            .median().astype(float).map(lambda v: float(f"{v:.3g}")).to_dict(orient="index"),
        "group_separation_kruskal_p": group_separation(twins),
        "runtime_s": round(time.time() - t_start, 1),
    }
    (config.REPO_ROOT / "reports" / "phase2_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["holdout_rmse_2h_mgdl"], indent=2))
    print(json.dumps(summary["median_params_by_status"], indent=2))
    print(f"done in {summary['runtime_s']} s")


if __name__ == "__main__":
    main()
