"""Render docs/architecture.pdf and docs/architecture.png from a hand-laid SVG.

Run:  python scripts/make_architecture.py      (needs `pip install playwright` + a Chromium)
"""
from __future__ import annotations

import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs"

W, H = 1600, 1000
INK, INK2, MUTED, RULE = "#15211d", "#485853", "#74847f", "#c9d3cf"
C_DATA, C_DATA_BG = "#2f7d4f", "#eef7f1"
C_SYN, C_SYN_BG = "#8a6a12", "#fbf4e2"
C_TWIN, C_TWIN_BG = "#3b4cca", "#eef0fd"
C_ML, C_ML_BG = "#b4531c", "#fdf1ea"
C_UI, C_UI_BG = "#15211d", "#f1f4f3"

svg: list[str] = []


def box(x, y, w, h, title, lines, color, bg, tag=None, dashed=False, title_size=15):
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{bg}" stroke="{color}" stroke-width="1.6"{dash}/>')
    svg.append(f'<text x="{x + 14}" y="{y + 25}" font-size="{title_size}" font-weight="700" fill="{color}">{escape(title)}</text>')
    if tag:
        tw = 8 * len(tag) + 14
        svg.append(f'<rect x="{x + w - tw - 10}" y="{y + 10}" width="{tw}" height="20" rx="10" fill="#fff" stroke="{color}" stroke-width="1"{dash}/>')
        svg.append(f'<text x="{x + w - tw / 2 - 10}" y="{y + 24}" font-size="11" text-anchor="middle" fill="{color}">{escape(tag)}</text>')
    for i, ln in enumerate(lines):
        svg.append(f'<text x="{x + 14}" y="{y + 48 + i * 19}" font-size="13" fill="{INK2}">{escape(ln)}</text>')


def arrow(x1, y1, x2, y2, color=INK2, label=None, curve=0, dashed=False):
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    if curve:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + curve
        d = f"M{x1},{y1} Q{mx},{my} {x2},{y2}"
    else:
        d = f"M{x1},{y1} L{x2},{y2}"
    svg.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.8" marker-end="url(#ah)"{dash}/>')
    if label:
        lx, ly = (x1 + x2) / 2, (y1 + y2) / 2 + (curve / 2 if curve else 0) - 7
        svg.append(f'<text x="{lx}" y="{ly}" font-size="11.5" text-anchor="middle" fill="{INK2}" '
                   f'paint-order="stroke" stroke="#fff" stroke-width="4">{escape(label)}</text>')


def column_head(x, n, title, sub):
    svg.append(f'<text x="{x}" y="118" font-size="13" font-weight="700" fill="{MUTED}">{n}</text>')
    svg.append(f'<text x="{x + 18}" y="118" font-size="17" font-weight="700" fill="{INK}">{escape(title)}</text>')
    svg.append(f'<text x="{x}" y="138" font-size="12.5" fill="{MUTED}">{escape(sub)}</text>')


# ---------------------------------------------------------------- title
svg.append(f'<text x="40" y="52" font-size="28" font-weight="700" fill="{INK}">GlucoTwin system architecture</text>')
svg.append(f'<text x="40" y="78" font-size="15" fill="{INK2}">A per-patient digital twin for Type 2 Diabetes that fuses EHR and wearable data '
           f'to forecast glucose and warn of post-meal spikes up to 2 hours ahead</text>')

# ---------------------------------------------------------------- columns
X1, X2, X3, X4, X5 = 40, 350, 640, 1050, 1350
W1, W2, W3, W4, W5 = 260, 230, 340, 250, 210
column_head(X1, "1", "Data sources", "static + dynamic streams")
column_head(X2, "2", "Ingest and align", "one clock for every stream")
column_head(X3, "3", "Digital twin core", "physiology, calibrated per patient")
column_head(X4, "4", "Hybrid forecaster", "physics + machine learning")
column_head(X5, "5", "Clinician", "decision support")

# 1. sources
box(X1, 160, W1, 132, "Static EHR", ["Demographics, BMI", "HbA1c, fasting glucose, insulin", "Lipid panel (TG, LDL, HDL)", "CGMacros bio.csv (PhysioNet)"], C_DATA, C_DATA_BG, "real")
box(X1, 308, W1, 190, "Synthetic EHR layer", ["Diagnoses (ICD-10), medications", "Family history, BP", "TCF7L2 rs7903146, polygenic risk", "HbA1c history (-24 to -6 months)",
                                           "Conditioned on the real labs", "Exported as FHIR R4 (ABDM-ready)", "Shown in the dashboard record"], C_SYN, C_SYN_BG, "synthetic", dashed=True)
box(X1, 514, W1, 132, "Wearables / IoT", ["Dexcom G6 CGM, every 5 min", "FreeStyle Libre Pro, every 15 min", "Fitbit Sense: heart rate, METs", "45 participants, ~10 days each"], C_DATA, C_DATA_BG, "real")
box(X1, 662, W1, 94, "Meal log", ["Time, carbs, protein, fat, fibre", "Breakfast / lunch / dinner"], C_DATA, C_DATA_BG, "real")

# 2. ingest
box(X2, 160, W2, 152, "Time alignment", ["Shared 5-min grid", "Gaps <= 15 min interpolated", "Unit fix: HbA1c in %", "Sentinel values removed", "HOMA-IR computed"], INK, "#fff")
box(X2, 330, W2, 132, "Meal responses", ["Baseline, peak, rise, iAUC", "Crossed 180 mg/dL?", "Clean = full CGM coverage", "and no overlapping meal"], INK, "#fff")
box(X2, 480, W2, 112, "Live CGM stream", ["A new reading every 5 min", "keeps the running twin", "in sync with the patient"], INK, "#fff")

# 3. twin core (container)
svg.append(f'<rect x="{X3 - 12}" y="150" width="{W3 + 24}" height="606" rx="14" fill="none" stroke="{C_TWIN}" stroke-width="1.2" stroke-dasharray="2 5"/>')
box(X3, 160, W3, 112, "EHR prior (day 0)", ["Ridge regression: labs -> twin parameters", "A new patient gets a twin from the", "health record alone"], C_TWIN, C_TWIN_BG)
box(X3, 290, W3, 132, "MAP calibration (days 1-7)", ["CGM meal curves pull parameters", "away from the prior", "Personal theta = { S, SG, ka, f }", "S = insulin disposition, SG = self-clearance"], C_TWIN, C_TWIN_BG)
box(X3, 440, W3, 170, "Physiological model", ["Gut: Q1 -> Q2 (meal absorption, ka)", "Ra = f ka Q2 / (VG BW)",
                                                 "dG/dt = -SG E(t)(G - Gb) - X G + Ra",
                                                 "dX/dt = -p2 X + p2 S (G - Gb)+",
                                                 "E(t) = 1 + alpha (METs - 1)   exercise", "Fat + protein slow absorption (beta)"], C_TWIN, C_TWIN_BG)
box(X3, 628, W3, 118, "Data assimilation", ["Every 5 min the CGM reading nudges", "the twin's state (gain 0.6), so each", "simulation starts from reality"], C_TWIN, C_TWIN_BG)

# 4. hybrid forecaster
box(X4, 160, W4, 150, "Feature builder (time t)", ["CGM level and trend", "Heart rate, activity", "Meals: carbs on board, timing", "Labs + twin parameters", "Twin's simulated next 2 h"], C_ML, C_ML_BG)
box(X4, 328, W4, 112, "Residual regressors", ["LightGBM, one per horizon", "forecast = twin + correction", "+30 / +60 / +120 min"], C_ML, C_ML_BG)
box(X4, 458, W4, 112, "Spike classifier", ["LightGBM: P(glucose > 180", "mg/dL within 2 h)", "Threshold set on training patients"], C_ML, C_ML_BG)
box(X4, 588, W4, 94, "SHAP explainer", ["Top reasons behind", "each alert"], C_ML, C_ML_BG)

# 5. clinician
box(X5, 160, W5, 262, "Doctor dashboard", ["Patients ranked by risk", "Live twin + forecast", "Spike alert with reasons", "What-if simulator:", "  Indian meal library,", "  portion, post-meal walk", "Glucose profile (AGP), TIR",
                                            "FHIR health record", "Static web app: the twin", "runs in the browser"], C_UI, C_UI_BG)
box(X5, 458, W5, 132, "Alert", ["Risk >= threshold", "-> clinician reviews", "Decision support only:", "never auto-prescribes"], "#c62f2f", "#fbe9e7")

# evaluation strip
svg.append(f'<rect x="{X2}" y="790" width="{X5 + W5 - X2}" height="150" rx="12" fill="#fff" stroke="{RULE}" stroke-width="1.4"/>')
svg.append(f'<text x="{X2 + 16}" y="818" font-size="16" font-weight="700" fill="{INK}">Evaluation protocol</text>')
ev = [
    ("Unseen patients", "5-fold split by patient"),
    ("Unseen days", "only each patient's held-out last 3 days are scored"),
    ("Baselines", "no change, twin only, ML only"),
    ("Clinical accuracy", "Clarke Error Grid zones A-E"),
    ("Alerts", "detection rate, lead time, false alarms per day"),
    ("Ablations", "what each data stream adds, synthetic fields included"),
]
for i, (k, v) in enumerate(ev):
    cx = X2 + 16 + (i % 3) * 400
    cy = 850 + (i // 3) * 46
    svg.append(f'<text x="{cx}" y="{cy}" font-size="13.5" font-weight="700" fill="{INK}">{escape(k)}</text>')
    svg.append(f'<text x="{cx}" y="{cy + 19}" font-size="12.5" fill="{INK2}">{escape(v)}</text>')

# legend
lg = [(C_DATA, "real data", False), (C_SYN, "synthetic, anchored on real labs", True), (C_TWIN, "physics twin", False), (C_ML, "machine learning", False)]
for i, (c, t, d) in enumerate(lg):
    dash = ' stroke-dasharray="4 3"' if d else ""
    yy = 800 + i * 26
    svg.append(f'<rect x="{X1}" y="{yy}" width="18" height="12" rx="3" fill="#fff" stroke="{c}" stroke-width="1.6"{dash}/>')
    svg.append(f'<text x="{X1 + 26}" y="{yy + 11}" font-size="12.5" fill="{INK2}">{escape(t)}</text>')


def route(points, label=None, at=None):
    d = "M" + " L".join(f"{x},{y}" for x, y in points)
    svg.append(f'<path d="{d}" fill="none" stroke="{INK2}" stroke-width="1.8" stroke-linejoin="round" marker-end="url(#ah)"/>')
    if label:
        (ax, ay), (bx, by) = points[at], points[at + 1]
        svg.append(f'<text x="{(ax + bx) / 2}" y="{min(ay, by) - 6}" font-size="11.5" text-anchor="middle" fill="{INK2}">{escape(label)}</text>')


# ---------------------------------------------------------------- arrows (orthogonal, through the gutters)
route([(X1 + W1, 226), (X2, 226)])                                            # static EHR -> alignment
route([(X1 + W1, 580), (X1 + W1 + 18, 580), (X1 + W1 + 18, 270), (X2, 270)])  # wearables -> alignment
route([(X1 + W1, 709), (X1 + W1 + 34, 709), (X1 + W1 + 34, 400), (X2, 400)])  # meals -> responses
route([(X2 + W2, 216), (X3, 216)], "labs", 0)                                 # alignment -> prior
route([(X2 + W2, 396), (X3, 396)], "meals", 0)                                # responses -> calibration
route([(X2 + W2, 536), (X2 + W2 + 24, 536), (X2 + W2 + 24, 690), (X3, 690)])  # live CGM -> assimilation
route([(X3 + W3 / 2, 272), (X3 + W3 / 2, 290)])
route([(X3 + W3 / 2, 422), (X3 + W3 / 2, 440)])
route([(X3 + W3 / 2, 628), (X3 + W3 / 2, 610)])
route([(X3 + W3, 330), (X3 + W3 + 26, 330), (X3 + W3 + 26, 214), (X4, 214)])
route([(X3 + W3, 525), (X3 + W3 + 48, 525), (X3 + W3 + 48, 286), (X4, 286)])
route([(X4 + W4 / 2, 310), (X4 + W4 / 2, 328)])
route([(X4 + W4 / 2, 440), (X4 + W4 / 2, 458)])
route([(X4 + W4 / 2, 570), (X4 + W4 / 2, 588)])
route([(X4 + W4, 384), (X4 + W4 + 25, 384), (X4 + W4 + 25, 300), (X5, 300)])
route([(X4 + W4, 514), (X5, 514)])
route([(X4 + W4, 635), (X4 + W4 + 25, 635), (X4 + W4 + 25, 560), (X5, 560)])

doc = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}"
 font-family="'Atkinson Hyperlegible Next', 'Segoe UI', 'DejaVu Sans', Arial, sans-serif">
<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>
<rect width="{W}" height="{H}" fill="#ffffff"/>
{''.join(svg)}
</svg>"""

OUT.mkdir(exist_ok=True)
(OUT / "architecture.svg").write_text(doc)
html = f"<!doctype html><html><head><meta charset='utf-8'><style>@page{{size:{W}px {H}px;margin:0}}body{{margin:0}}</style></head><body>{doc}</body></html>"
(OUT / "architecture.html").write_text(html)

try:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": W, "height": H}, device_scale_factor=2)
        pg.set_content(html)
        pg.screenshot(path=str(OUT / "architecture.png"), full_page=True)
        pg.pdf(path=str(OUT / "architecture.pdf"), width=f"{W}px", height=f"{H}px", print_background=True)
        b.close()
    (OUT / "architecture.html").unlink()
    print("wrote docs/architecture.pdf, .png, .svg")
except ImportError:
    print("playwright not installed: wrote docs/architecture.svg only", file=sys.stderr)
