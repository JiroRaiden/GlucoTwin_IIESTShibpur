"""Phase 3: hybrid forecaster, evaluated on unseen patients AND unseen days.

Protocol
  5-fold GroupKFold over patients. In each fold:
    train = every origin of the training patients
    test  = origins of the test patients that fall in their held-out last 3 days
  (twin parameters of a test patient were calibrated only on their own earlier days)

Outputs
  reports/phase3_summary.json, reports/phase3_test_predictions.parquet
  models/*.txt (final LightGBM models trained on everyone), models/meta.json
  reports/figures/p3_*.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import GroupKFold

from glucotwin import config
from glucotwin.forecast import clarke, features as FT, models as MD
from glucotwin.twin import model as M
from glucotwin.analysis import figures_forecast as FF

REPORTS = config.REPO_ROOT / "reports"
MODELS_DIR = config.REPO_ROOT / "models"


def rmse(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def mae(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return float(np.mean(np.abs(a[m] - b[m])))


def main():
    t0 = time.time()
    grid_all = pd.read_parquet(config.PROCESSED_DIR / "grid_5min.parquet")
    grids = {pid: g.droplevel(0) for pid, g in grid_all.groupby(level=0)}
    bio = pd.read_parquet(config.PROCESSED_DIR / "bio.parquet")
    ehr = pd.read_json(config.PROCESSED_DIR / "ehr_synthetic.json", orient="index")
    ehr.index = ehr.index.astype(int)
    tw = json.loads((REPORTS / "twins.json").read_text())
    glob = M.Globals(alpha=tw["globals"]["alpha"], beta=tw["globals"]["beta"])
    twins = {int(k): v for k, v in tw["patients"].items()}

    print(f"[1/5] Building forecast origins (every 15 min) for {len(twins)} patients")
    parts = []
    for pid, twin in twins.items():
        parts.append(FT.build_origins(pid, grids[pid], twin, glob, bio.loc[pid],
                                      ehr.loc[pid] if pid in ehr.index else None))
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(config.PROCESSED_DIR / "origins.parquet")
    print(f"      {len(df):,} origins | {df['test_period'].sum():,} in held-out days | "
          f"spike base rate {df['spike'].mean():.1%}")

    print("[2/5] Cross-validated training (unseen patients x unseen days)")
    pids = np.array(sorted(df["pid"].unique()))
    gkf = GroupKFold(n_splits=min(5, len(pids)))
    preds, abl_rows, thresholds = [], [], []
    for fold, (tr_i, te_i) in enumerate(gkf.split(pids, groups=pids)):
        tr_p, te_p = pids[tr_i], pids[te_i]
        train = df[df["pid"].isin(tr_p)]
        test = df[df["pid"].isin(te_p) & df["test_period"]].copy()
        if test.empty:
            continue
        test["fold"] = fold
        for kind, feats in MD.FEATURE_SETS.items():
            for h in FT.HORIZONS:
                b = MD.base_col(kind, h)
                m = MD.fit_reg(train, feats, h, b)
                test[f"pred_{kind}_{h}"] = MD.predict_reg(m, test, feats, b)
            clf = MD.fit_clf(train, feats)
            test[f"p_{kind}"] = clf.predict_proba(test[feats])[:, 1]
            thr = MD.choose_threshold(train, feats)
            test[f"thr_{kind}"] = thr
            if kind == "hybrid":
                thresholds.append(thr)
        for name, feats in MD.ABLATIONS.items():
            clf = MD.fit_clf(train, feats)
            p = clf.predict_proba(test[feats])[:, 1]
            reg = MD.fit_reg(train, feats, 60, "twin_60" if "twin_60" in feats else "g_now")
            pr = MD.predict_reg(reg, test, feats, "twin_60" if "twin_60" in feats else "g_now")
            abl_rows.append({"fold": fold, "set": name, "n": len(test),
                             "auroc": MD.clf_metrics(test["spike"].to_numpy(), p)["auroc"],
                             "sse60": np.nansum((pr - test["y_60"]) ** 2),
                             "n60": int(test["y_60"].notna().sum())})
        preds.append(test)
        print(f"      fold {fold}: train {len(train):,} | test {len(test):,} origins from {len(te_p)} patients | thr {thresholds[-1]:.2f}")
    P = pd.concat(preds, ignore_index=True)
    P.to_parquet(REPORTS / "phase3_test_predictions.parquet")

    print("[3/5] Metrics")
    reg_table = []
    for h in FT.HORIZONS:
        y = P[f"y_{h}"].to_numpy()
        for name, col in [("No change", "g_now"), ("Twin only (physics)", f"twin_{h}"),
                          ("Pure ML (LightGBM)", f"pred_pure_ml_{h}"), ("Hybrid twin + ML", f"pred_hybrid_{h}")]:
            p = P[col].to_numpy()
            z = clarke.zone_percentages(y, p)
            reg_table.append({"horizon_min": h, "model": name, "rmse": rmse(p, y), "mae": mae(p, y),
                              "clarke_A": z["A"], "clarke_AB": z["A"] + z["B"], "clarke_CDE": z["C"] + z["D"] + z["E"]})
    reg_table = pd.DataFrame(reg_table)
    print(reg_table.round(2).to_string(index=False))

    y = P["spike"].to_numpy()
    cls = {
        "Hybrid twin + ML": MD.clf_metrics(y, P["p_hybrid"].to_numpy()),
        "Pure ML (LightGBM)": MD.clf_metrics(y, P["p_pure_ml"].to_numpy()),
        "Twin only (simulated peak)": MD.clf_metrics(y, P["twin_max120"].to_numpy()),
        "Current glucose only": MD.clf_metrics(y, P["g_now"].to_numpy()),
    }
    meal = P[P["is_meal_now"] == 1]
    cls_meal = {
        "Hybrid twin + ML": MD.clf_metrics(meal["spike"].to_numpy(), meal["p_hybrid"].to_numpy()),
        "Pure ML (LightGBM)": MD.clf_metrics(meal["spike"].to_numpy(), meal["p_pure_ml"].to_numpy()),
        "Twin only (simulated peak)": MD.clf_metrics(meal["spike"].to_numpy(), meal["twin_max120"].to_numpy()),
        "n_meal_origins": int(meal["spike"].notna().sum()),
    }
    P["alert_hybrid"] = (P["p_hybrid"] >= P["thr_hybrid"]).astype(float)
    P["alert_pure"] = (P["p_pure_ml"] >= P["thr_pure_ml"]).astype(float)
    alerts = {
        "Hybrid twin + ML": MD.alert_events(P, grids, "alert_hybrid", 0.5),
        "Pure ML (LightGBM)": MD.alert_events(P, grids, "alert_pure", 0.5),
        "Twin only (peak >= 180)": MD.alert_events(P, grids, "twin_max120", config.HYPERGLYCEMIA_MGDL),
    }
    for k, v in cls.items():
        print(f"      spike AUROC {k:<28} {v['auroc']:.3f}  AUPRC {v['auprc']:.3f}")
    for k, v in alerts.items():
        print(f"      alerts {k:<26} {v}")

    abl = pd.DataFrame(abl_rows)
    abl_sum = abl.groupby("set", sort=False).apply(lambda d: pd.Series({
        "auroc": np.average(d["auroc"], weights=d["n"]), "rmse60": np.sqrt(d["sse60"].sum() / d["n60"].sum())}))
    print(abl_sum.round(3).to_string())

    print("[4/5] Final models on all patients + SHAP")
    MODELS_DIR.mkdir(exist_ok=True)
    feats = MD.FEATURE_SETS["hybrid"]
    final_clf = MD.fit_clf(df, feats)
    final_clf.booster_.save_model(str(MODELS_DIR / "spike_classifier.txt"))
    for h in FT.HORIZONS:
        MD.fit_reg(df, feats, h, f"twin_{h}").booster_.save_model(str(MODELS_DIR / f"residual_h{h}.txt"))
    sample = df.dropna(subset=["spike"]).sample(min(4000, len(df)), random_state=config.SEED)
    sv = shap.TreeExplainer(final_clf).shap_values(sample[feats])
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.Series(np.abs(sv).mean(0), index=feats)
    # merge features that share one human label (hour_sin + hour_cos = "time of day")
    imp = imp.groupby(lambda f: FT.FEATURE_LABELS.get(f, f)).sum().sort_values(ascending=False)
    thr_final = float(np.median(thresholds))
    (MODELS_DIR / "meta.json").write_text(json.dumps({
        "features": feats, "feature_labels": FT.FEATURE_LABELS, "alert_threshold": thr_final,
        "horizons": list(FT.HORIZONS), "globals": {"alpha": glob.alpha, "beta": glob.beta}}, indent=1))

    print("[5/5] Figures + summary")
    FF.fig_rmse_by_horizon(reg_table, config.FIG_DIR / "p3_rmse_by_horizon.png")
    FF.fig_clarke(P["y_60"].to_numpy(), P["pred_hybrid_60"].to_numpy(),
                  reg_table.query("horizon_min == 60 and model == 'Hybrid twin + ML'").iloc[0],
                  config.FIG_DIR / "p3_clarke_60min.png")
    FF.fig_shap_importance(imp, FT.FEATURE_LABELS, config.FIG_DIR / "p3_shap_importance.png")
    FF.fig_ablation(abl_sum, config.FIG_DIR / "p3_ablation.png")
    FF.fig_alert_timeline(P, grids, config.FIG_DIR / "p3_alert_timeline.png")

    summary = {
        "n_origins": int(len(df)), "n_test_origins": int(len(P)), "n_patients": int(len(pids)),
        "spike_base_rate_test": round(float(np.nanmean(y)), 4),
        "regression": reg_table.round(2).to_dict(orient="records"),
        "spike_classification_all_times": {k: {m: round(v, 3) for m, v in d.items()} for k, d in cls.items()},
        "spike_classification_at_meal_start": {k: ({m: round(v, 3) for m, v in d.items()} if isinstance(d, dict) else d)
                                               for k, d in cls_meal.items()},
        "alerts": {k: {m: (round(v, 3) if isinstance(v, float) else v) for m, v in d.items()} for k, d in alerts.items()},
        "alert_threshold_per_fold": [round(t, 2) for t in thresholds],
        "ablation": abl_sum.round(3).reset_index().to_dict(orient="records"),
        "top_shap_features": [{"feature": k, "mean_abs_shap": round(float(v), 4)}
                              for k, v in imp.head(12).items()],
        "runtime_s": round(time.time() - t0, 1),
    }
    (REPORTS / "phase3_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"done in {summary['runtime_s']} s")


if __name__ == "__main__":
    main()
