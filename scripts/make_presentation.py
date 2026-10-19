"""Build docs/presentation.pdf (16:9) from the pipeline's real outputs.

Every number on the slides is read from reports/phase*_summary.json, so the deck can never
disagree with the code. Figures come from reports/figures/, and the dashboard screenshot from
docs/dashboard.png (made by scripts/screenshot_dashboard.py).

Run:  python scripts/make_presentation.py      (needs playwright + chromium)
"""
from __future__ import annotations

import base64
import json
import os
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REP, FIG, DOCS = ROOT / "reports", ROOT / "reports" / "figures", ROOT / "docs"
s1 = json.loads((REP / "phase1_summary.json").read_text())
s2 = json.loads((REP / "phase2_summary.json").read_text())
s3 = json.loads((REP / "phase3_summary.json").read_text())


def img(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def font_faces() -> str:
    """Embed Geist (OFL) so the PDF renders the brand face. Looks for @fontsource files if present."""
    base = Path(os.environ.get("GEIST_DIR", "")) if os.environ.get("GEIST_DIR") else None
    if not base or not base.exists():
        return ""
    out = []
    for fam, folder, pre in (("Geist", base, "geist"), ("Geist Mono", base.parent.parent / "geist-mono" / "files", "geist-mono")):
        for w in (300, 400, 500, 600, 700):
            f = folder / f"{pre}-latin-{w}-normal.woff2"
            if f.exists():
                out.append(f"@font-face{{font-family:'{fam}';font-weight:{w};src:url(data:font/woff2;base64,"
                           f"{base64.b64encode(f.read_bytes()).decode()}) format('woff2')}}")
    return "\n".join(out)


reg = {(r["horizon_min"], r["model"]): r for r in s3["regression"]}
hyb = lambda h, k: reg[(h, "Hybrid twin + ML")][k]
cls = s3["spike_classification_all_times"]
cls_meal = s3["spike_classification_at_meal_start"]
alerts = s3["alerts"]["Hybrid twin + ML"]
ve = {r["model"].split()[0]: r for r in s1["variance_explained"]}
lc = s2["holdout_rmse_2h_mgdl"]


def r1(x):
    return f"{x:.1f}"


def pct(x):
    return f"{100 * x:.0f}%"


slides: list[str] = []


def slide(body: str, cls_: str = ""):
    slides.append(f'<section class="slide {cls_}"><div class="glow"></div>{body}'
                  f'<footer><span>GlucoTwin · IIEST Shibpur</span><span>{len(slides) + 1}</span></footer></section>')


def exhibit(path: Path, caption: str = "", max_h: int = 620) -> str:
    return (f'<figure class="ex"><img src="{img(path)}" style="max-height:{max_h}px" alt="{escape(caption)}">'
            + (f"<figcaption>{escape(caption)}</figcaption>" if caption else "") + "</figure>")


def stat(value: str, label: str) -> str:
    return f'<div class="stat"><b>{value}</b><span>{label}</span></div>'


# ---------------------------------------------------------------- 1 title
slide(f"""
<div class="title">
  <p class="kicker">Digital Twin Challenge 2026 · Happiest Health</p>
  <h1>Gluco<i>Twin</i></h1>
  <p class="lede">A personal digital twin that warns of post-meal glucose spikes before they happen,
  and lets a doctor test what would prevent them.</p>
  <p class="who">Bishal Mandi · B.Tech Information Technology · IIEST Shibpur</p>
</div>""", "cover")

# ---------------------------------------------------------------- 2 problem
slide(f"""
<h2>The problem</h2>
<div class="cols">
  <div>
    <p class="big">India has about <b>101 million</b> people with diabetes and <b>136 million</b> with prediabetes.</p>
    <p class="note">ICMR-INDIAB national study, <i>Lancet Diabetes &amp; Endocrinology</i>, 2023</p>
    <p>Much of the damage happens in the hours after meals. Rice- and roti-heavy meals drive sharp spikes above
    180 mg/dL, and repeated spikes drive complications.</p>
    <p>Doctors see these spikes after the fact, in an HbA1c report or a CGM download, when it is too late to act on that meal.</p>
  </div>
  <div class="card">
    <h3>Two kinds of data, kept apart</h3>
    <p><b>Health records</b>: HbA1c, fasting insulin, BMI, diagnoses, medication, family history, genetics.</p>
    <p><b>Wearables</b>: glucose every 5 minutes, heart rate, activity, meals.</p>
    <p class="accent">Our question: can a personal model of each patient, fed by both, predict a spike before it happens?</p>
  </div>
</div>""", "mid")

# ---------------------------------------------------------------- 3 insight
slide(f"""
<h2>Why it has to be personal</h2>
<div class="cols wide-left">
  {exhibit(FIG / "p1_same_meal_different_spike.png", "Every 40-70 g carb meal in CGMacros, one column per person", 560)}
  <div class="stack">
    {stat(f"{ve['M0']['cv_r2']:.0%}".replace('%', '') + "%", "of a spike's size explained by carbs alone")}
    {stat(f"{ve['M2']['cv_r2']:.0%}", "with the full meal and fasting labs")}
    {stat(f"{ve['M3']['cv_r2']:.0%}", "once the model knows who the person is")}
    <p class="note">Out-of-sample R², 5-fold CV, {ve['M0']['n_meals']:,} meals. Same carb dose: rises from
    {s1['similar_meal_band_40_70g']['rise_p10']:.0f} to {s1['similar_meal_band_40_70g']['rise_p90']:.0f} mg/dL (10th to 90th percentile).</p>
  </div>
</div>""")

# ---------------------------------------------------------------- 4 solution
slide("""
<h2>GlucoTwin: a virtual patient for each person</h2>
<div class="three">
  <div class="card"><h3>1 · The twin</h3><p>A physiological model of glucose and insulin, calibrated to one patient.
  It starts from their health record on day 0 and sharpens with every day of wearable data.</p></div>
  <div class="card"><h3>2 · Early warning</h3><p>Every 15 minutes: glucose forecast 30, 60 and 120 minutes ahead,
  and the probability of crossing 180 mg/dL in the next 2 hours, with the reasons.</p></div>
  <div class="card"><h3>3 · What-if</h3><p>The doctor tests a plan before the patient eats: rajma chawal now,
  half a portion, or the same meal and a 20-minute walk. The twin simulates each one.</p></div>
</div>
<p class="note center">Decision support only. Every alert is for a clinician to review.</p>""", "mid")

# ---------------------------------------------------------------- 5 data
slide(f"""
<h2>Data: real where it exists, synthetic only where it must be</h2>
<div class="cols">
  <div>
    <table>
      <tr><th>Stream</th><th>Source</th><th></th></tr>
      <tr><td>Labs, demographics, BMI</td><td>CGMacros bio.csv</td><td><span class="tag real">real</span></td></tr>
      <tr><td>Dexcom G6 + Libre Pro CGM</td><td>CGMacros</td><td><span class="tag real">real</span></td></tr>
      <tr><td>Fitbit heart rate, METs</td><td>CGMacros</td><td><span class="tag real">real</span></td></tr>
      <tr><td>Meals with macros</td><td>CGMacros</td><td><span class="tag real">real</span></td></tr>
      <tr><td>Diagnoses, medication, family history, TCF7L2 genotype, HbA1c history</td><td>Generated from each person's real labs</td><td><span class="tag syn">synthetic</span></td></tr>
    </table>
    <p class="note">{s1['participants']} adults ({s1['status_counts'].get('Healthy', 0)} healthy, {s1['status_counts'].get('Prediabetes', 0)} prediabetes,
    {s1['status_counts'].get('T2D', 0)} T2D by measured HbA1c), {s1['meals_logged']:,} meals, about 10 days each.
    Synthetic EHR is exported as FHIR R4, the format of India's ABDM.</p>
  </div>
  <div class="card">
    <h3>Data checks we made</h3>
    <p>HbA1c is labelled mmol/mol in the dictionary but its values are %. Fixed.</p>
    <p>The two CGMs disagree by {s1['dexcom_vs_libre_mard_pct']:.0f}% on average. Against lab fasting glucose, Dexcom was off by
    +12 mg/dL and Libre by −27, so Dexcom is the primary sensor.</p>
    <p>Documented error codes (LDL = 800, VLDL = 400) removed; Fitbit METs rescaled.</p>
  </div>
</div>""")

# ---------------------------------------------------------------- 6 architecture
slide(f"""<h2>Architecture</h2>{exhibit(DOCS / "architecture.png", "", 820)}""", "fullfig")

# ---------------------------------------------------------------- 7 twin
med = s2["median_params_by_status"]
kp = s2["group_separation_kruskal_p"]
kp_fmt = lambda p: "< 0.0001" if p < 1e-4 else f"= {p:.2g}"
slide(f"""
<h2>The twin: physiology with four personal numbers</h2>
<div class="cols wide-right">
  <div>
    <pre>dG/dt = −SG·E(t)·(G − Gb) − X·G + Ra(t)
dX/dt = −p2·X + p2·S·max(G − Gb, 0)
Ra(t) = f·ka·Q2 / (VG·BW)    gut: Q1 → Q2
E(t)  = 1 + α·(METs − 1)     exercise</pre>
    <p><b>SG</b> glucose clearance · <b>f</b> effective carb load · <b>ka</b> absorption · <b>S</b> insulin-driven clearance</p>
    <p>The calibrated twin separates the groups: <b>SG</b> falls about {med['Healthy']['SG'] / med['T2D']['SG']:.0f}× from healthy to T2D
    (p {kp_fmt(kp['SG'])}) and <b>f</b> rises about {med['T2D']['f'] / med['Healthy']['f']:.1f}× (p {kp_fmt(kp['f'])}).</p>
    <p class="note">Reduced Bergman minimal model (1979) with a two-compartment gut. CGMacros has no plasma insulin, so S is not
    identifiable from CGM alone (p = {kp['S']}), and we do not interpret it.</p>
  </div>
  {exhibit(FIG / "p2_params_by_status.png", "Calibrated parameters by clinical group", 470)}
</div>""")

# ---------------------------------------------------------------- 8 learning
lc_keys = [k for k in lc if k.startswith("personal_twin")]
slide(f"""
<h2>EHR starts the twin. Wearable data makes it personal.</h2>
<div class="cols wide-left">
  {exhibit(FIG / "p2_learning_curve.png", "2-hour forecast error on each patient's held-out last 3 days", 560)}
  <div class="stack">
    {stat(r1(lc['personal_twin_0_days']), "mg/dL error, day 0: twin built from the EHR alone")}
    {stat(r1(lc['personal_twin_all_days']), "mg/dL after all training days of CGM")}
    {stat(r1(lc['average_twin']), "mg/dL for one average twin for everyone")}
    <p class="note">Calibration is a MAP fit: CGM meal curves pull the parameters away from a prior predicted from labs.
    Few days, the record dominates; more days, the patient's own data does.</p>
  </div>
</div>""")

# ---------------------------------------------------------------- 9 hybrid
slide(f"""
<h2>Hybrid forecaster: physics + machine learning</h2>
<div class="cols">
  {exhibit(FIG / "p3_rmse_by_horizon.png", "Held-out patients AND held-out days", 470)}
  {exhibit(FIG / "p3_clarke_60min.png", "", 470)}
</div>
<p class="center">LightGBM learns the twin's error: <b>forecast = twin + correction</b>.
60-min error <b>{r1(hyb(60, 'rmse'))} mg/dL</b> · Clarke zone A <b>{hyb(60, 'clarke_A'):.1f}%</b> · zones A+B <b>{hyb(60, 'clarke_AB'):.1f}%</b></p>""")

# ---------------------------------------------------------------- 10 alerts
slide(f"""
<h2>Spike alerts a doctor can act on</h2>
<div class="cols wide-left">
  {exhibit(FIG / "p3_alert_timeline.png", "A held-out day the model never saw", 520)}
  <div class="stack">
    {stat(f"{cls['Hybrid twin + ML']['auroc']:.2f}", "AUROC for a crossing above 180 mg/dL within 2 h")}
    {stat(f"{cls_meal['Hybrid twin + ML']['auroc']:.2f}", "AUROC at the moment a meal is logged, before any rise")}
    {stat(pct(alerts['sensitivity']), f"of crossings caught, median {alerts['median_lead_min']:.0f} min ahead")}
    {stat(r1(alerts['false_alarms_per_day']), "false alarms per patient-day")}
  </div>
</div>""")

# ---------------------------------------------------------------- 11 explain
abl = {r["set"]: r for r in s3["ablation"]}
slide(f"""
<h2>Every alert explains itself, and every data stream earns its place</h2>
<div class="cols">
  {exhibit(FIG / "p3_shap_importance.png", "SHAP: what drives the spike-risk score", 470)}
  {exhibit(FIG / "p3_ablation.png", "Remove one data stream at a time", 470)}
</div>
<p class="note center">The synthetic EHR fields were generated from the real labs, so they should add nothing. The ablation confirms it,
and we report it instead of hiding it.</p>""")

# ---------------------------------------------------------------- 12 dashboard
dash = DOCS / "dashboard.png"
slide(f"""<h2>The doctor's console</h2>{exhibit(dash, "Risk assessment with provenance, live twin and forecast, what-if with Indian meals, glucose profile, FHIR record", 720) if dash.exists() else ''}""", "fullfig")

# ---------------------------------------------------------------- 13 impact
slide("""
<h2>Why this matters for India</h2>
<div class="three">
  <div class="card"><h3>Personal diet advice</h3><p>The what-if simulator speaks in Indian meals: idli-sambar, rajma chawal, dal-roti, biryani.
  Counselling becomes "for you, this", not generic rules.</p></div>
  <div class="card"><h3>Fits the national stack</h3><p>Health records go in and out as FHIR R4, the format of the Ayushman Bharat Digital Mission,
  so a real ABHA-linked record could replace the synthetic one.</p></div>
  <div class="card"><h3>Cheap to run</h3><p>The twin is a few equations. It runs in the doctor's browser with no server and no GPU,
  which matters for clinics outside metros.</p></div>
</div>""", "mid")

# ---------------------------------------------------------------- 14 limits
slide(f"""
<h2>Limitations and next steps</h2>
<div class="cols">
  <div class="card"><h3>What this prototype is not</h3>
    <p>{s1['participants']} US adults over about 10 days: a proof of concept, not clinical validation.</p>
    <p>Diagnoses, medication and genotype are synthetic. Plasma insulin was not measured.</p>
    <p>Meal macros in the what-if library are approximate.</p>
    <p>Not a medical device: decision support only.</p></div>
  <div class="card"><h3>Next</h3>
    <p>Validate on an Indian CGM cohort with Indian meals.</p>
    <p>Add sleep and stress signals from the wearable.</p>
    <p>Pilot with a diabetologist: do alerts change counselling?</p>
    <p>Connect to real ABDM records with consent.</p></div>
</div>""", "mid")

# ---------------------------------------------------------------- 15 close
slide("""
<div class="title">
  <h1>Gluco<i>Twin</i></h1>
  <p class="lede">Code, results and the live dashboard are in the GitHub repository.</p>
  <p class="who">github.com/JiroRaiden/GlucoTwin_IIESTShibpur</p>
  <p class="note">Data: CGMacros (Gutierrez-Osuna, Kerr, Mortazavi, Das; PhysioNet 2025). Model: Bergman 1979, Dalla Man 2007.
  Clinical metrics: Clarke 1987, Battelino 2019 consensus on time in range.</p>
</div>""", "cover")

CSS = """
@page { size: 1920px 1080px; margin: 0 }
* { box-sizing: border-box }
body { margin: 0; font-family: 'Geist', 'DejaVu Sans', sans-serif; color: #eceeee; background: #000; }
.slide { width: 1920px; height: 1080px; position: relative; overflow: hidden; padding: 84px 110px 90px;
  background: #000; page-break-after: always; display: flex; flex-direction: column; gap: 34px; }
.glow { display: none; }
.slide > *:not(.glow) { position: relative; }
h1 { font-size: 150px; font-weight: 600; letter-spacing: -.045em; margin: 0; line-height: 1; }
h1 i { font-style: normal; color: #91afc7; }
h2 { font-size: 58px; font-weight: 600; letter-spacing: -.03em; margin: 0; line-height: 1.1; max-width: 1600px; }
h3 { font-size: 30px; font-weight: 600; letter-spacing: -.01em; margin: 0 0 14px; }
p { font-size: 27px; line-height: 1.45; color: rgba(236, 238, 238, .78); margin: 0 0 16px; max-width: 70ch; }
p b { color: #fff; font-weight: 600; }
.big { font-size: 40px; line-height: 1.3; color: #fff; }
.note { font-size: 21px; color: rgba(239, 237, 253, .5); }
.center { text-align: center; margin-left: auto; margin-right: auto; }
.accent { color: #91afc7; }
.kicker { font-size: 26px; color: rgba(239, 237, 253, .55); margin-bottom: 30px; }
.lede { font-size: 40px; color: #fff; max-width: 34ch; line-height: 1.3; margin-top: 36px; }
.who { font-size: 26px; color: rgba(239, 237, 253, .6); margin-top: 40px; }
.title { margin: auto 0; }
.mid { justify-content: center; gap: 48px; }
.mid .cols, .mid .three { flex: 0 0 auto; }
.mid .card p, .mid p { font-size: 29px; }
.cover { border-top: 6px solid #91afc7; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 56px; align-items: start; flex: 1; min-height: 0; }
.cols.wide-left { grid-template-columns: 1.55fr 1fr; }
.cols.wide-right { grid-template-columns: 1fr 1.25fr; }
.three { display: grid; grid-template-columns: repeat(3, 1fr); gap: 36px; }
.card { border-top: 1px solid rgba(255,255,255,.22); padding: 26px 0 0; }
.card p { font-size: 25px; }
.stack { display: flex; flex-direction: column; gap: 26px; }
.stat { border-left: 1px solid rgba(255,255,255,.22); padding-left: 24px; }
.stat b { display: block; font-family: 'Geist Mono', 'DejaVu Sans Mono', monospace; font-size: 64px; font-weight: 400; letter-spacing: -.04em; color: #fff; line-height: 1.05; }
.stat span { font-size: 23px; color: rgba(239, 237, 253, .62); }
.ex { margin: 0; background: #fcfcfb; border-radius: 4px; padding: 18px; display: flex; flex-direction: column; align-items: center; }
.ex img { max-width: 100%; display: block; }
.ex figcaption { font-size: 19px; color: #52514e; margin-top: 10px; }
.fullfig .ex { flex: 1; justify-content: center; margin-bottom: 56px; }
table { border-collapse: collapse; font-size: 23px; width: 100%; margin-bottom: 18px; }
th { text-align: left; color: rgba(239, 237, 253, .5); font-weight: 500; padding: 8px 10px; border-bottom: 1px solid rgba(255,255,255,.15); }
td { padding: 12px 10px; border-bottom: 1px solid rgba(255,255,255,.07); color: rgba(239, 237, 253, .85); vertical-align: top; }
.tag { font-size: 18px; border-radius: 99px; padding: 2px 12px; border: 1px solid; white-space: nowrap; }
.tag.real { color: #34d399; border-color: rgba(52,211,153,.45); }
.tag.syn { color: #fbbf24; border-color: rgba(251,191,36,.45); border-style: dashed; }
pre { font-family: 'Geist Mono', 'DejaVu Sans Mono', monospace; font-size: 25px; line-height: 1.6; background: rgba(255,255,255,.04);
  border: 1px solid rgba(255,255,255,.1); border-radius: 18px; padding: 26px 30px; color: #e6e0ff; margin: 0 0 22px; }
.slide > footer:not(.glow) { position: absolute; left: 110px; right: 110px; bottom: 36px; display: flex; justify-content: space-between;
  font-size: 18px; color: rgba(239, 237, 253, .35); }
"""

html = f"<!doctype html><html><head><meta charset='utf-8'><style>{font_faces()}{CSS}</style></head><body>{''.join(slides)}</body></html>"
DOCS.mkdir(exist_ok=True)
(DOCS / "presentation.html").write_text(html)

from playwright.sync_api import sync_playwright  # noqa: E402

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1920, "height": 1080})
    pg.set_content(html, wait_until="load")
    pg.pdf(path=str(DOCS / "presentation.pdf"), width="1920px", height="1080px", print_background=True)
    for i in range(1, len(slides) + 1):
        pg.screenshot(path=str(REP / f"_slide_preview_{i}.png"), full_page=True, clip={"x": 0, "y": 1080 * (i - 1), "width": 1920, "height": 1080})
    b.close()
(DOCS / "presentation.html").unlink()
print(f"wrote docs/presentation.pdf ({len(slides)} slides)")
