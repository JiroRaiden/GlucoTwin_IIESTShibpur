"""Export everything the doctor dashboard needs into dashboard/data.js.

The dashboard is a static site (no server): it reads window.GT_DATA, re-runs the
twin in the browser for the live forecast and the what-if simulator, and shows
the hybrid model's out-of-sample forecasts and alert reasons for a replayed
held-out day.

Usage:  python scripts/export_dashboard.py            (real CGMacros results)
        python scripts/export_dashboard.py --preview  (labels the data as a UI preview)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from glucotwin import config
from glucotwin.forecast import features as FT
from glucotwin.twin import calibrate as C
from glucotwin.twin import model as M

REPORTS = config.REPO_ROOT / "reports"
OUT = config.REPO_ROOT / "dashboard" / "data.js"

UNITS = {
    "g_now": ("{:.0f} mg/dL", 1), "d5": ("{:+.0f} mg/dL", 1), "d15": ("{:+.0f} mg/dL", 1), "d30": ("{:+.0f} mg/dL", 1),
    "g_mean_2h": ("{:.0f} mg/dL", 1), "g_std_2h": ("{:.0f} mg/dL", 1), "g_min_2h": ("{:.0f} mg/dL", 1), "g_max_2h": ("{:.0f} mg/dL", 1),
    "hr_30": ("{:.0f} bpm", 1), "hr_rel": ("{:+.0f} bpm", 1), "mets_30": ("{:.1f} METs", 1), "mets_60": ("{:.1f} METs", 1),
    "min_since_meal": ("{:.0f} min", 1), "carbs_last_meal": ("{:.0f} g", 1), "fat_last_meal": ("{:.0f} g", 1),
    "protein_last_meal": ("{:.0f} g", 1), "fiber_last_meal": ("{:.0f} g", 1), "carbs_3h": ("{:.0f} g", 1),
    "is_meal_now": ("{:.0f}", 1), "hba1c": ("{:.1f}%", 1), "homa_ir": ("{:.1f}", 1), "bmi": ("{:.1f}", 1),
    "fasting_glucose": ("{:.0f} mg/dL", 1), "age": ("{:.0f} y", 1),
    "twin_30": ("{:.0f} mg/dL", 1), "twin_60": ("{:.0f} mg/dL", 1), "twin_120": ("{:.0f} mg/dL", 1),
    "twin_max120": ("{:.0f} mg/dL", 1), "twin_rise": ("{:+.0f} mg/dL", 1), "tw_Gb": ("{:.0f} mg/dL", 1),
}


def fmt(feat, val):
    if pd.isna(val):
        return "n/a"
    if feat in ("hour_sin", "hour_cos"):
        return ""
    if feat.startswith("tw_log"):
        return f"{np.exp(val):.3g}"
    f, _ = UNITS.get(feat, ("{:.2g}", 1))
    return f.format(val)


def r(x, n=1):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


def agp(glucose: pd.Series) -> dict:
    g = glucose.dropna()
    if g.empty:
        return {}
    slot = (g.index.hour * 2 + g.index.minute // 30)
    q = g.groupby(slot).quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack()
    q = q.reindex(range(48)).interpolate(limit_direction="both")
    mean = g.mean()
    bands = {
        "very_low": (g < 54).mean(), "low": ((g >= 54) & (g < 70)).mean(),
        "in_range": ((g >= 70) & (g <= 180)).mean(), "high": ((g > 180) & (g <= 250)).mean(),
        "very_high": (g > 250).mean(),
    }
    return {
        "p05": [r(v) for v in q[0.05]], "p25": [r(v) for v in q[0.25]], "p50": [r(v) for v in q[0.5]],
        "p75": [r(v) for v in q[0.75]], "p95": [r(v) for v in q[0.95]],
        "tir": {k: r(100 * v, 1) for k, v in bands.items()},
        "mean": r(mean), "gmi": r(3.31 + 0.02392 * mean, 1), "cv": r(100 * g.std() / mean, 1),
        "days": r((g.index.max() - g.index.min()).total_seconds() / 86400, 1),
    }


def main(preview: bool):
    grid_all = pd.read_parquet(config.PROCESSED_DIR / "grid_5min.parquet")
    grids = {pid: g.droplevel(0) for pid, g in grid_all.groupby(level=0)}
    bio = pd.read_parquet(config.PROCESSED_DIR / "bio.parquet")
    ehr = pd.read_json(config.PROCESSED_DIR / "ehr_synthetic.json", orient="index")
    ehr.index = ehr.index.astype(int)
    tw = json.loads((REPORTS / "twins.json").read_text())
    twins = {int(k): v for k, v in tw["patients"].items()}
    glob = M.Globals(alpha=tw["globals"]["alpha"], beta=tw["globals"]["beta"])
    P = pd.read_parquet(REPORTS / "phase3_test_predictions.parquet")
    meta = json.loads((config.REPO_ROOT / "models" / "meta.json").read_text())
    s3 = json.loads((REPORTS / "phase3_summary.json").read_text())
    booster = lgb.Booster(model_file=str(config.REPO_ROOT / "models" / "spike_classifier.txt"))
    explainer = shap.TreeExplainer(booster)
    feats = meta["features"]
    thr = meta["alert_threshold"]

    reg = pd.DataFrame(s3["regression"])
    hyb = reg[reg["model"] == "Hybrid twin + ML"].set_index("horizon_min")

    tdf = pd.DataFrame(twins).T
    group_medians = {s: {k: float(tdf.loc[tdf["status"] == s, k].astype(float).median()) for k in ["S", "SG", "ka", "f"]}
                     for s in tdf["status"].unique()}

    patients = []
    for pid, twin in twins.items():
        b, e, g = bio.loc[pid], (ehr.loc[pid] if pid in ehr.index else None), grids[pid]
        d = P[P["pid"] == pid].copy()
        if d.empty:
            continue
        d["day"] = d["ts"].dt.normalize()
        days = d.groupby("day").agg(n=("ts", "size"), spikes=("spike", "sum"), gmax=("y_60", "max"))
        days = days[days["n"] >= 40].sort_values(["spikes", "gmax"], ascending=False)
        if days.empty:
            continue
        day = days.index[0]
        dd = d[d["day"] == day].sort_values("ts")

        # SHAP reasons for each replay origin
        sv = explainer.shap_values(dd[feats])
        sv = sv[1] if isinstance(sv, list) else sv
        origins = []
        for k, (_, row) in enumerate(dd.iterrows()):
            contrib = pd.Series(sv[k], index=feats)
            contrib = contrib.drop(labels=[c for c in ["hour_sin", "hour_cos"] if c in contrib.index])
            top = contrib.sort_values(ascending=False).head(3)
            reasons = [{"label": FT.FEATURE_LABELS.get(f, f), "value": fmt(f, row[f]), "push": r(v, 3)}
                       for f, v in top.items() if v > 0]
            origins.append({
                "t": int((row["ts"] - day).total_seconds() // 60), "p": r(row["p_hybrid"], 3),
                "pred": {"30": r(row["pred_hybrid_30"]), "60": r(row["pred_hybrid_60"]), "120": r(row["pred_hybrid_120"])},
                "reasons": reasons,
            })

        # signals on the 5-min grid from day-4h to day+26h (twin needs 3 h of history to sync)
        lo, hi = day - pd.Timedelta(hours=4), day + pd.Timedelta(hours=26)
        inp = C.prepare_inputs(g, glob.beta).loc[lo:hi]
        hr = g.loc[lo:hi, "hr"] if "hr" in g else pd.Series(np.nan, index=inp.index)
        t0 = int((inp.index[0] - day).total_seconds() // 60)
        meal_rows = g.loc[lo:hi]
        meal_rows = meal_rows[meal_rows["carbs"] > 0]
        meals = [{"t": int((t - day).total_seconds() // 60), "carbs": r(m["carbs"], 0), "protein": r(m.get("protein"), 0),
                  "fat": r(m.get("fat"), 0), "fiber": r(m.get("fiber"), 0)} for t, m in meal_rows.iterrows()]

        train = g[g.index < C.split_cutoff(g)]
        patients.append({
            "id": f"P{pid:03d}", "pid": int(pid), "age": r(b.get("age"), 0), "sex": b.get("sex"),
            "status": b.get("status"),
            "labs": {k: r(b.get(k), 1) for k in ["hba1c", "fasting_glucose", "fasting_insulin", "homa_ir", "bmi",
                                                 "triglycerides", "ldl", "hdl"]},
            "ehr": None if e is None else {
                "diagnoses": list(e["diagnoses"]), "medications": list(e["medications"]),
                "family_history_t2d": int(e["family_history_t2d"]), "tcf7l2": e["tcf7l2_rs7903146"],
                "prs_percentile": r(e["prs_t2d_percentile"], 0),
                "hba1c_history": [r(e["hba1c_minus_24m"]), r(e["hba1c_minus_12m"]), r(e["hba1c_minus_6m"]), r(b.get("hba1c"))],
                "bp": f"{e['systolic_bp']:.0f}/{e['diastolic_bp']:.0f}",
            },
            "twin": {k: twin[k] for k in ["S", "SG", "ka", "f", "Gb", "BW"]} | {
                "n_train_meals": twin["n_train_meals"], "holdout_rmse": r(twin["holdout_rmse"])},
            "agp": agp(train["glucose"]),
            "replay": {
                "date": day.strftime("%Y-%m-%d"), "grid_t0": t0,
                "glucose": [r(v) for v in inp["glucose"]], "carbs": [r(v, 1) for v in inp["carbs_avail"]],
                "kmult": [r(v, 3) for v in inp["kmult"]], "mets": [r(v, 2) for v in inp["mets"]],
                "hr": [r(v, 0) for v in hr.reindex(inp.index)],
                "meals": meals, "origins": origins,
            },
            "risk_now": None,
        })

    data = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "preview": preview,
        "source": "Mock data in the CGMacros format (UI preview only)" if preview else "CGMacros (PhysioNet), held-out days",
        "globals": {"alpha": glob.alpha, "beta": glob.beta, "P2": M.P2, "VG": M.VG},
        "alert_threshold": thr,
        "hyperglycemia": config.HYPERGLYCEMIA_MGDL,
        "rmse": {str(h): r(hyb.loc[h, "rmse"]) for h in hyb.index},
        "metrics": {
            "clarke_A_60": r(hyb.loc[60, "clarke_A"]),
            "auroc": s3["spike_classification_all_times"]["Hybrid twin + ML"]["auroc"],
            "auroc_meal": s3["spike_classification_at_meal_start"]["Hybrid twin + ML"]["auroc"],
            "sensitivity": s3["alerts"]["Hybrid twin + ML"]["sensitivity"],
            "lead_min": s3["alerts"]["Hybrid twin + ML"]["median_lead_min"],
            "false_alarms_day": s3["alerts"]["Hybrid twin + ML"]["false_alarms_per_day"],
        },
        "group_medians": group_medians,
        "patients": patients,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("window.GT_DATA = " + json.dumps(data, separators=(",", ":"), default=lambda o: None) + ";\n")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB, {len(patients)} patients)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    main(ap.parse_args().preview)
