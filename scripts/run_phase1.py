"""Phase 1: load CGMacros -> 5-min grids -> meal responses -> synthetic EHR -> evidence.

Usage:
    python scripts/run_phase1.py                 # expects data under data/raw/
    CGMACROS_DIR=/path/to/CGMacros python scripts/run_phase1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from glucotwin import config
from glucotwin.analysis import figures, personalization as pers
from glucotwin.data import load, meals as meals_mod, synthetic_ehr


def main():
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading CGMacros from {config.RAW_DIR}")
    bio, grids, meal_events = load.load_all()
    print(f"      {len(grids)} participants | status: {bio['status'].value_counts().to_dict()}")

    # Sensor agreement (Dexcom vs Libre) - a sanity check worth quoting
    both = pd.concat([g[["glucose", "glucose_libre"]] for g in grids.values()]).dropna()
    mard = (abs(both["glucose_libre"] - both["glucose"]) / both["glucose"]).mean() * 100
    print(f"      Dexcom vs Libre mean absolute relative difference: {mard:.1f}%")

    print("[2/5] Saving 5-min grids")
    grid_all = pd.concat({pid: g for pid, g in grids.items()}, names=["pid", "ts"])
    grid_all.to_parquet(config.PROCESSED_DIR / "grid_5min.parquet")

    print("[3/5] Measuring every meal's glucose response")
    mt = meals_mod.build_meal_table(bio, grids, meal_events)
    mt.to_parquet(config.PROCESSED_DIR / "meals.parquet")
    clean = mt[mt["clean"]]
    print(f"      {len(mt)} meals logged, {len(clean)} clean responses "
          f"({clean['hyper'].mean():.1%} crossed 180 mg/dL)")

    print("[4/5] Synthesising EHR history (anchored on real labs) + FHIR R4 export")
    ehr = synthetic_ehr.synthesize_history(bio)
    ehr.to_json(config.PROCESSED_DIR / "ehr_synthetic.json", orient="index", indent=1)
    bio.to_parquet(config.PROCESSED_DIR / "bio.parquet")
    n = synthetic_ehr.write_fhir(bio, ehr, config.PROCESSED_DIR / "fhir")
    print(f"      {n} FHIR bundles written")

    print("[5/5] Personalisation evidence")
    ve = pers.variance_explained(mt)
    print(ve.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    cs = pers.carb_sensitivity(mt)
    spread = pers.similar_meal_spread(mt)
    figures.fig_same_meal_spread(spread, config.FIG_DIR / "p1_same_meal_different_spike.png")
    figures.fig_variance_explained(ve, config.FIG_DIR / "p1_variance_explained.png")
    figures.fig_carb_sensitivity(cs, config.FIG_DIR / "p1_carb_sensitivity.png")
    figures.fig_example_curves(grids, mt, config.FIG_DIR / "p1_example_curves.png")

    summary = {
        "participants": len(grids),
        "status_counts": bio["status"].value_counts().to_dict(),
        "meals_logged": int(len(mt)),
        "meals_clean": int(len(clean)),
        "pct_meals_hyper_180": round(float(clean["hyper"].mean()) * 100, 1),
        "dexcom_vs_libre_mard_pct": round(float(mard), 1),
        "variance_explained": ve.round(3).to_dict(orient="records"),
        "carb_sensitivity_mgdl_per_10g": {
            "min": round(float(cs["slope_per_10g"].min()), 1),
            "median": round(float(cs["slope_per_10g"].median()), 1),
            "max": round(float(cs["slope_per_10g"].max()), 1),
            "n_people": int(len(cs)),
        },
        "similar_meal_band_40_70g": {
            "n_meals": int(len(spread)),
            "rise_p10": round(float(spread["rise"].quantile(0.1)), 1),
            "rise_p90": round(float(spread["rise"].quantile(0.9)), 1),
        },
    }
    (config.REPO_ROOT / "reports" / "phase1_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
