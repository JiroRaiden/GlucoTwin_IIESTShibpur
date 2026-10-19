"""Fill docs/README.template.md with the real numbers from reports/*.json -> README.md.

    python scripts/write_readme.py [--dashboard URL] [--video URL]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"


def main(dashboard_url: str | None, video_url: str | None):
    s1 = json.loads((REP / "phase1_summary.json").read_text())
    s2 = json.loads((REP / "phase2_summary.json").read_text())
    s3 = json.loads((REP / "phase3_summary.json").read_text())
    tw = json.loads((REP / "twins.json").read_text())

    ve = {r["model"].split()[0]: r for r in s1["variance_explained"]}
    lc = s2["holdout_rmse_2h_mgdl"]
    med = s2["median_params_by_status"]
    kp = s2["group_separation_kruskal_p"]
    reg = {(r["horizon_min"], r["model"]): r for r in s3["regression"]}
    cls, meal, al = s3["spike_classification_all_times"], s3["spike_classification_at_meal_start"], s3["alerts"]

    def reg_row(name, bold=False):
        b = "**" if bold else ""
        r = lambda h, k: reg[(h, name)][k]
        return (f"| {b}{name}{b} | {b}{r(30, 'rmse'):.1f}{b} | {b}{r(60, 'rmse'):.1f}{b} | {b}{r(120, 'rmse'):.1f}{b} | "
                f"{b}{r(60, 'clarke_A'):.1f}%{b} | {b}{r(60, 'clarke_AB'):.1f}%{b} |")

    def alert_row(name, alert_key, bold=False):
        b = "**" if bold else ""
        a = al[alert_key]
        m = meal.get(name, {}) if isinstance(meal.get(name), dict) else {}
        return (f"| {b}{name}{b} | {b}{cls[name]['auroc']:.3f}{b} | {b}{m.get('auroc', float('nan')):.3f}{b} | "
                f"{b}{a['detected']}/{a['events']} ({100 * a['sensitivity']:.0f}%){b} | {b}{a['median_lead_min']:.0f} min{b} | "
                f"{b}{a['false_alarms_per_day']:.1f}{b} |")

    fmt_p = lambda p: "< 0.0001" if p < 1e-4 else f"{p:.2g}"
    v = {
        "dashboard_url": f"[{dashboard_url}]({dashboard_url})" if dashboard_url else "*(deploy link added after Vercel deploy)*",
        "video_url": f"[Watch on YouTube]({video_url})" if video_url else "*(YouTube link added after upload)*",
        "n_participants": s1["participants"], "n_healthy": s1["status_counts"].get("Healthy", 0),
        "n_pre": s1["status_counts"].get("Prediabetes", 0), "n_t2d": s1["status_counts"].get("T2D", 0),
        "meals_logged": f"{s1['meals_logged']:,}", "meals_clean": f"{s1['meals_clean']:,}", "mard": f"{s1['dexcom_vs_libre_mard_pct']:.0f}",
        **{f"r2_{k.lower()}": f"{ve[k]['cv_r2']:.0%}" if k == "M0" else f"{ve[k]['cv_r2']:.2f}" for k in ve},
        **{f"mae_{k.lower()}": f"{ve[k]['cv_mae']:.0f}" for k in ve},
        "rise_p10": f"{s1['similar_meal_band_40_70g']['rise_p10']:.0f}", "rise_p90": f"{s1['similar_meal_band_40_70g']['rise_p90']:.0f}",
        "alpha": s2["globals"]["alpha"], "beta": s2["globals"]["beta"],
        "p_sg": fmt_p(kp["SG"]), "p_f": fmt_p(kp["f"]), "p_s": fmt_p(kp["S"]),
        "sg_h": f"{med['Healthy']['SG']:.4f}", "sg_t": f"{med['T2D']['SG']:.4f}",
        "f_h": f"{med['Healthy']['f']:.2f}", "f_t": f"{med['T2D']['f']:.2f}",
        "lc_none": lc["no_change_baseline"], "lc_avg": lc["average_twin"], "lc_0": lc["personal_twin_0_days"],
        "lc_1": lc["personal_twin_1_days"], "lc_2": lc["personal_twin_2_days"], "lc_3": lc["personal_twin_3_days"],
        "lc_5": lc["personal_twin_5_days"], "lc_all": lc["personal_twin_all_days"],
        "n_test_origins": f"{s3['n_test_origins']:,}", "n_patients": s3["n_patients"],
        "reg_rows": "\n".join([reg_row("No change"), reg_row("Twin only (physics)"), reg_row("Pure ML (LightGBM)"),
                               reg_row("Hybrid twin + ML", bold=True)]),
        "alert_rows": "\n".join([alert_row("Twin only (simulated peak)", "Twin only (peak >= 180)"),
                                 alert_row("Pure ML (LightGBM)", "Pure ML (LightGBM)"),
                                 alert_row("Hybrid twin + ML", "Hybrid twin + ML", bold=True)]),
        "n_meal_origins": meal["n_meal_origins"],
        "shap_top": ", ".join(f["feature"] for f in s3["top_shap_features"][:6]),
        "abl_rows": "\n".join(f"| {r['set']} | {r['auroc']:.3f} | {r['rmse60']:.1f} |" for r in s3["ablation"]),
    }
    text = (ROOT / "docs" / "README.template.md").read_text()
    text = re.sub(r"\{\{(\w+)\}\}", lambda m: str(v[m.group(1)]), text)
    (ROOT / "README.md").write_text(text)
    left = re.findall(r"\{\{\w+\}\}", text)
    print("README.md written", "(unfilled: " + ", ".join(left) + ")" if left else "")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dashboard")
    ap.add_argument("--video")
    a = ap.parse_args()
    main(a.dashboard, a.video)
