"""Build docs/presentation.pdf (16:9) from the pipeline's real outputs.

Every number on the slides is read from reports/phase*_summary.json, so the deck can never
disagree with the code. Figures come from reports/figures/, and the dashboard screenshot from
docs/dashboard.png (made by scripts/screenshot_dashboard.py).

Run:  python scripts/make_presentation.py [--theme glass|aura] [--out PATH] [--dashboard URL] [--video URL]
      (needs playwright + chromium). Themes: "glass" (gradient glass cards), "aura" (charcoal, holographic glow,
      serif titles) and "clinic" (the dashboard's own look: light app chrome, graphite hero slides, real glucose curve). Fonts: GEIST_DIR points at @fontsource/geist/files; Instrument Serif is looked up next to it.
      The closing slide lists the live dashboard, demo video and repository; a link left out is not shown.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from html import escape
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--dashboard", default="https://gluco-twin-iiest-shibpur.vercel.app")
ap.add_argument("--video", default=None)
ap.add_argument("--theme", choices=["glass", "aura", "clinic"], default="glass")
ap.add_argument("--out", default=None, help="output PDF (default docs/presentation.pdf)")
ap.add_argument("--no-flatten", action="store_true", help="keep CSS effects live (large, slow PDF)")
ARGS = ap.parse_args()
REPO_URL = "https://github.com/JiroRaiden/GlucoTwin_IIESTShibpur"

ROOT = Path(__file__).resolve().parents[1]
REP, FIG, DOCS = ROOT / "reports", ROOT / "reports" / "figures", ROOT / "docs"
s1 = json.loads((REP / "phase1_summary.json").read_text())
s2 = json.loads((REP / "phase2_summary.json").read_text())
s3 = json.loads((REP / "phase3_summary.json").read_text())


def img(path: Path, max_w: int = 2000) -> str:
    """Embed a PNG, downscaled to at most max_w px wide (slides show figures at <=1700 css px)."""
    from PIL import Image
    im = Image.open(path)
    if im.width <= max_w:
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
    im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    return _uri(im, optimize=True)


def _uri(im, fmt: str = "PNG", **kw) -> str:
    import io
    buf = io.BytesIO()
    im.save(buf, fmt, **kw)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def dark_img(path: Path, max_w: int = 1800) -> str:
    """A light matplotlib figure turned dark for the aura theme: invert, then rotate hue back by 180° so
    series keep their colours; done here rather than with a CSS filter, which Chromium rasterises into a huge PDF."""
    import numpy as np
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    hsv = np.asarray(Image.fromarray(255 - np.asarray(im)).convert("HSV")).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + 128) % 256
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.2, 0, 255)
    rgb = np.asarray(Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")).astype(float)
    rgb = 26 + rgb * (236 - 26) / 255          # lift blacks to the slide charcoal, soften pure white
    return _uri(Image.fromarray(rgb.astype(np.uint8)), optimize=True)


def glow_bg(blobs, seed: int = 0) -> str:
    """Charcoal background with soft glossy pearl orbs (rendered once as a small JPEG)."""
    import numpy as np
    from PIL import Image, ImageFilter
    W, H = 960, 540
    pal_ = np.array([[255, 255, 255], [205, 196, 255], [160, 176, 255], [150, 236, 214], [255, 190, 226], [230, 224, 255], [255, 255, 255]], float) / 255

    def pal(t):
        t = (t % 1) * (len(pal_) - 1); i = np.floor(t).astype(int); f = (t - i)[..., None]
        return pal_[i] * (1 - f) + pal_[np.minimum(i + 1, len(pal_) - 1)] * f
    y, x = np.mgrid[0:H, 0:W].astype(float)
    r = np.hypot((x - W * .5) / W, (y - H * .45) / H)
    out = np.stack([0.165 - 0.05 * r, 0.165 - 0.05 * r, 0.178 - 0.055 * r], -1).clip(0, 1)
    for cx, cy, rx, ry, rot, a, ph, light in blobs:
        dx, dy = x - cx * W, y - cy * H
        c, s_ = np.cos(rot), np.sin(rot)
        u, v = (dx * c + dy * s_) / (rx * W), (-dx * s_ + dy * c) / (ry * W)
        d, th = np.hypot(u, v), np.arctan2(v, u)
        body = np.clip(1 - d, 0, 1) ** 1.6
        crest = np.exp(-((d - 0.78) / 0.13) ** 2) * np.clip(np.cos(th - light), 0, 1) ** 2
        col = np.array([.66, .65, .72]) * (0.7 + 0.3 * body[..., None]) * (1 - crest[..., None]) + pal(th / (2 * np.pi) + ph + 0.3 * d) * crest[..., None]
        alpha = (a * (0.7 * body + crest)).clip(0, 1)[..., None]
        out = out * (1 - alpha) + col * alpha
    im = Image.fromarray((out.clip(0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(9)).resize((1920, 1080), Image.BICUBIC)
    arr = np.asarray(im).astype(float) + np.random.default_rng(seed).normal(0, 1.6, (1080, 1920, 3))  # grain stops banding
    return _uri(Image.fromarray(arr.clip(0, 255).astype(np.uint8)), "JPEG", quality=86)


GLOWS = {
    "v1": [(.0, -.02, .42, .30, .45, .62, 0.0, 0.6), (1.0, 1.04, .40, .28, -.35, .55, 0.35, 3.6), (.74, -.08, .16, .10, .15, .30, .6, 1.8)],
    "v2": [(1.0, -.02, .42, .30, -.45, .62, 0.15, -0.8), (.0, 1.03, .40, .29, .35, .55, 0.55, -0.6), (.52, 1.1, .18, .11, 0, .28, .8, -1.6)],
    "v3": [(1.08, .48, .30, .42, 0.15, .62, 0.3, 0.2), (-.02, -.06, .30, .22, .55, .5, .7, 0.8), (.34, 1.08, .20, .12, .05, .30, .1, -1.5)],
    "v0": [(-.04, 1.02, .42, .30, -.3, .62, 0.45, -0.7), (1.0, -.05, .36, .27, .5, .55, 0.05, 2.3), (1.02, .82, .14, .16, 0, .32, .6, 3.1)],
    "vc": [(.5, .5, .34, .22, 0.12, .40, 0.2, -1.2), (-.04, -.07, .30, .22, .5, .5, .6, 0.7), (1.04, 1.06, .34, .25, -.4, .52, .9, 3.7)],
}


def font_faces() -> str:
    """Embed Geist (OFL) so the PDF renders the brand face. Looks for @fontsource files if present."""
    base = Path(os.environ.get("GEIST_DIR", "")) if os.environ.get("GEIST_DIR") else None
    if not base or not base.exists():
        return ""
    out = []
    fams = (("Geist", base, "geist"), ("Geist Mono", base.parent.parent / "geist-mono" / "files", "geist-mono"),
            ("Instrument Serif", base.parent.parent / "instrument-serif" / "files", "instrument-serif"),
            ("Inter", base.parent.parent / "inter" / "files", "inter"))
    for fam, folder, pre in fams:
        for w in (300, 400, 500, 600, 700, 800):
            for style in ("normal", "italic"):
                f = folder / f"{pre}-latin-{w}-{style}.woff2"
                if f.exists():
                    out.append(f"@font-face{{font-family:'{fam}';font-weight:{w};font-style:{style};src:url(data:font/woff2;base64,"
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


AURA = ARGS.theme == "aura"
CLINIC = ARGS.theme == "clinic"
SECTION = {2: "Problem", 3: "Problem", 4: "Solution", 5: "Solution", 6: "Solution", 7: "Twin", 8: "Twin",
           9: "Evidence", 10: "Evidence", 11: "Evidence", 12: "Evidence", 13: "Impact", 14: "Impact"}
PULSE = '<path d="M2 13h4l2.2-6 3.6 10 2.4-7 1.3 3H20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'


def demo_day() -> dict:
    """P041's held-out day from the dashboard's own data file: the 08:00 moment the demo opens on."""
    t = (ROOT / "dashboard" / "data.js").read_text()
    d = json.loads(t[t.find("{"):t.rfind("}") + 1])
    p = next(x for x in d["patients"] if x["id"] == "P041")
    r = p["replay"]
    o = next(x for x in r["origins"] if x["t"] == 480)
    return {"g": r["glucose"], "t0": r["grid_t0"], "origin": o, "rmse": d["rmse"], "meals": r["meals"]}


def hero_curve(w: int = 1720, h: int = 430) -> str:
    """Measured CGM 02:00–08:00, the hybrid forecast at 08:00 with its typical-error band, the 180 line,
    and (dotted) what actually happened. All from dashboard/data.js."""
    D = demo_day()
    g, t0, o = D["g"], D["t0"], D["origin"]
    ta, tb, ylo, yhi = 300, 600, 90, 300
    pad_l, pad_r, pad_t, pad_b = 70, 150, 20, 46
    X = lambda t: pad_l + (t - ta) / (tb - ta) * (w - pad_l - pad_r)
    Y = lambda v: pad_t + (yhi - v) / (yhi - ylo) * (h - pad_t - pad_b)
    val = lambda t: g[(t - t0) // 5]
    meas = " ".join(f"{X(t):.1f},{Y(val(t)):.1f}" for t in range(ta, 481, 5) if val(t) is not None)
    act = " ".join(f"{X(t):.1f},{Y(val(t)):.1f}" for t in range(480, tb + 1, 5) if val(t) is not None)
    now_v = val(480)
    pts = [(480, now_v)] + [(480 + int(k), v) for k, v in sorted(o["pred"].items(), key=lambda kv: int(kv[0]))]
    err = {0: 0, **{int(k): v for k, v in D["rmse"].items()}}
    band_top = " ".join(f"{X(t):.1f},{Y(v + err[t - 480]):.1f}" for t, v in pts)
    band_bot = " ".join(f"{X(t):.1f},{Y(v - err[t - 480]):.1f}" for t, v in reversed(pts))
    fc = " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in pts)
    grid = "".join(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="hc-grid"/>'
                   f'<text x="{pad_l - 14}" y="{Y(v) + 6:.1f}" class="hc-ax" text-anchor="end">{v}</text>' for v in (100, 140, 220, 260, 300))
    hours = "".join(f'<text x="{X(t):.1f}" y="{h - 12}" class="hc-ax" text-anchor="middle">{t // 60:02d}:00</text>' for t in range(300, 601, 60))
    place = {30: (-18, -16, "end"), 60: (0, -24, "middle"), 120: (18, -18, "start")}   # keep labels clear of the lines
    dots = "".join(f'<circle cx="{X(t):.1f}" cy="{Y(v):.1f}" r="9" class="hc-pt"/>'
                   f'<text x="{X(t) + place[t - 480][0]:.1f}" y="{Y(v) + place[t - 480][1]:.1f}" text-anchor="{place[t - 480][2]}" '
                   f'class="hc-lbl">+{t - 480} min · {v:.0f}</text>' for t, v in pts[1:])
    meal = next(m for m in D["meals"] if m["t"] == 475)
    return f"""<svg class="hero-curve" viewBox="0 0 {w} {h}" width="{w}" height="{h}" aria-label="P041 glucose and forecast">
  {grid}{hours}
  <rect x="{X(480):.1f}" y="{pad_t}" width="{w - pad_r - X(480):.1f}" height="{h - pad_t - pad_b}" class="hc-future"/>
  <line x1="{pad_l}" x2="{w - pad_r}" y1="{Y(180):.1f}" y2="{Y(180):.1f}" class="hc-thr"/>
  <text x="{w - pad_r + 14}" y="{Y(180) + 7:.1f}" class="hc-thr-l">180 mg/dL</text>
  <polygon points="{band_top} {band_bot}" class="hc-band"/>
  <polyline points="{act}" class="hc-act"/>
  <polyline points="{meas}" class="hc-meas"/>
  <polyline points="{fc}" class="hc-fc"/>
  <line x1="{X(480):.1f}" x2="{X(480):.1f}" y1="{pad_t}" y2="{h - pad_b}" class="hc-now"/>
  <text x="{X(480) - 12:.1f}" y="{pad_t + 26}" class="hc-lbl" text-anchor="end">Now 08:00 · {now_v:.0f} mg/dL</text>
  <line x1="{X(meal['t']):.1f}" x2="{X(meal['t']):.1f}" y1="{h - pad_b}" y2="{h - pad_b - 18}" class="hc-meal"/>
  <text x="{X(meal['t']) - 12:.1f}" y="{h - pad_b - 24}" class="hc-ax" text-anchor="end">meal {meal['carbs']:.0f} g</text>
  <circle cx="{X(480):.1f}" cy="{Y(now_v):.1f}" r="10" class="hc-nowdot"/>
  {dots}
</svg>"""


def clinic_header(n: int) -> str:
    cur = SECTION.get(n, "")
    nav = "".join(f'<span class="{"on" if x == cur else ""}">{x}</span>' for x in ("Problem", "Solution", "Twin", "Evidence", "Impact"))
    return (f'<header class="chrome"><div class="brand"><svg width="30" height="30" viewBox="0 0 22 22">{PULSE}</svg>GlucoTwin</div>'
            f'<nav>{nav}</nav><div class="ctx">Digital Twin Challenge 2026 · IIEST Shibpur<b>{n:02d}</b></div></header>')
STAR = "M0-1C.08-.08.08-.08 1 0C.08.08.08.08 0 1C-.08.08-.08.08-1 0C-.08-.08-.08-.08 0-1Z"


def star(x: float, y: float, r: float, ring: bool = False) -> str:
    g = f'<path d="{STAR}" transform="translate({x} {y}) scale({r})" fill="#fff"/>'
    if ring:
        g += f'<circle cx="{x}" cy="{y}" r="{r * .62:.1f}" fill="none" stroke="#fff" stroke-width="1.6" opacity=".9"/>'
    return g


def ornaments(n: int) -> str:
    """Thin lines, sparkle stars and a coil mark; corners alternate so slides don't look stamped."""
    coil = "".join(f'<ellipse cx="{1716 + 9 * i}" cy="1006" rx="15" ry="26" fill="none" stroke="#fff" stroke-width="1.2" opacity=".55"/>'
                   for i in range(7))
    if n % 2:  # odd slides: rule + ringed star top-right, star + rule bottom-left
        g = (f'<line x1="1330" y1="88" x2="1774" y2="88" stroke="#fff" stroke-width="1.2" opacity=".55"/>{star(1800, 88, 24, True)}'
             f'<line x1="150" y1="1000" x2="560" y2="1000" stroke="#fff" stroke-width="1.2" opacity=".55"/>{star(124, 1000, 22, True)}')
    else:      # even slides: sparkle cluster top-left, vertical rule + star on the right edge
        g = (f'{star(1830, 70, 18)}<line x1="1830" y1="100" x2="1830" y2="520" stroke="#fff" stroke-width="1.2" opacity=".5"/>'
             f'{star(110, 1010, 16)}<line x1="134" y1="1010" x2="460" y2="1010" stroke="#fff" stroke-width="1.2" opacity=".5"/>')
    if n == 1:
        g += star(118, 92, 26) + star(150, 64, 10) + star(140, 126, 8)
    return f'<svg class="orn" viewBox="0 0 1920 1080" width="1920" height="1080" aria-hidden="true">{g}{coil}</svg>'


def slide(body: str, cls_: str = ""):
    n = len(slides) + 1
    deco = ornaments(n) if AURA else ""
    if CLINIC:
        deco = '<div class="halo"></div>' if ("cover" in cls_ or "dark" in cls_) else clinic_header(n)
    if AURA:
        cls_ += " vc" if "cover" in cls_ else f" v{n % 4}"
    slides.append(f'<section class="slide {cls_}">{deco}<div class="glow"></div>{body}'
                  f'<footer><span>GlucoTwin · IIEST Shibpur</span><span>{n}</span></footer></section>')


def exhibit(path: Path, caption: str = "", max_h: int = 620, keep: bool = False) -> str:
    src = dark_img(path) if AURA and not keep else img(path)
    return (f'<figure class="ex"><img src="{src}" style="max-height:{max_h}px" alt="{escape(caption)}">'
            + (f"<figcaption>{escape(caption)}</figcaption>" if caption else "") + "</figure>")


def stat(value: str, label: str) -> str:
    return f'<div class="stat"><b>{value}</b><span>{label}</span></div>'


# ---------------------------------------------------------------- 1 title
def wordmark(size: int = 168) -> str:
    """'GlucoTwin' as SVG text so the gradient on 'Twin' stays vector in the PDF (CSS background-clip text does not)."""
    gid = f"wm{size}"
    return (f'<svg class="wordmark" width="{int(size * 5.1)}" height="{int(size * 1.08)}" aria-label="GlucoTwin">'
            f'<defs><linearGradient id="{gid}" x1="0" x2="1" y1="0" y2="0">'
            f'<stop offset="0" stop-color="#5eead4"/><stop offset=".45" stop-color="#7dd3fc"/>'
            f'<stop offset=".8" stop-color="#a78bfa"/><stop offset="1" stop-color="#f0abfc"/></linearGradient></defs>'
            f'<text x="0" y="{int(size * .86)}" font-size="{size}" font-weight="600" letter-spacing="{-0.05 * size:.1f}">'
            f'<tspan fill="#f4f6fb">Gluco</tspan><tspan fill="url(#{gid})">Twin</tspan></text></svg>')


if AURA:
    slide(f"""
<div class="vside l">IIEST Shibpur · B.Tech Information Technology</div>
<div class="vside r">github.com/JiroRaiden</div>
<div class="center-stack">
  <p class="track">Digital Twin Challenge 2026 · Happiest Health</p>
  <h1 class="serif">Gluco<em>Twin</em></h1>
  <p class="track sub">A personal digital twin that warns of post-meal glucose spikes before they happen,
  and lets a doctor test what would prevent them</p>
  <p class="track who">Bishal Mandi</p>
</div>""", "cover")
elif CLINIC:
    _o = demo_day()["origin"]
    slide(f"""
<div class="cover-c">
  <div class="row">
    <div>
      <p class="eyebrow">Digital Twin Challenge 2026 · Happiest Health</p>
      <h1><svg width="118" height="118" viewBox="0 0 22 22">{PULSE}</svg>GlucoTwin</h1>
      <p class="lede">A personal digital twin that warns of post-meal glucose spikes before they happen,
      and lets a doctor test what would prevent them.</p>
      <p class="who">Bishal Mandi · B.Tech Information Technology · IIEST Shibpur</p>
    </div>
    <div class="alert-card">
      <span class="chip crit"><i></i>High risk</span>
      <b>{round(100 * _o['p'])}<small> / 100</small></b>
      <p>Crossing 180 mg/dL in the next 2 hours is likely</p>
      <div class="kpis"><span><em>{pct(alerts['sensitivity'])}</em>spikes caught</span><span><em>{alerts['median_lead_min']:.0f} min</em>median warning</span><span><em>{r1(hyb(60, 'rmse'))}</em>mg/dL error, 60 min</span></div>
      <p class="src">Patient P041, a held-out day the model never saw</p>
    </div>
  </div>
  {hero_curve()}
  <div class="legend"><span class="l-meas">Measured CGM</span><span class="l-fc">Hybrid forecast ± typical error</span><span class="l-act">What happened</span></div>
</div>""", "cover")
else:
    slide(f"""
    <div class="hero">
      <div class="title">
        <p class="kicker">Digital Twin Challenge 2026 · Happiest Health</p>
        {wordmark()}
        <p class="lede">A personal digital twin that warns of post-meal glucose spikes before they happen,
        and lets a doctor test what would prevent them.</p>
        <p class="who">Bishal Mandi · B.Tech Information Technology · IIEST Shibpur</p>
      </div>
      <div class="tiles">
        <div class="pill"><i></i>High risk · crossing 180 mg/dL likely</div>
        <div class="tile t1"><b class="c1">{pct(alerts['sensitivity'])}</b><span>of spikes above 180 mg/dL caught</span></div>
        <div class="tile t2"><b class="c2">{alerts['median_lead_min']:.0f} min</b><span>median warning before the crossing</span></div>
        <div class="tile t3"><b class="c3">{r1(hyb(60, 'rmse'))}</b><span>mg/dL forecast error at 60 min, on patients never seen</span></div>
      </div>
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
  {exhibit(FIG / "p1_same_meal_different_spike.png", "", 560)}
  <div class="stack">
    {stat(f"{ve['M0']['cv_r2']:.0%}".replace('%', '') + "%", "of a spike's size explained by carbs alone")}
    {stat(f"{ve['M2']['cv_r2']:.0%}", "with the full meal and fasting labs")}
    {stat(f"{ve['M3']['cv_r2']:.0%}", "once the model knows who the person is")}
    <p class="note">Out-of-sample R², 5-fold CV, {ve['M0']['n_meals']:,} meals. Same carb dose: rises from
    {s1['similar_meal_band_40_70g']['rise_p10']:.0f} to {s1['similar_meal_band_40_70g']['rise_p90']:.0f} mg/dL (10th to 90th percentile).</p>
  </div>
</div>""")

# ---------------------------------------------------------------- 4 solution
N = ("", "", "") if AURA else ("1 · ", "2 · ", "3 · ")
slide(f"""
<h2>GlucoTwin: a virtual patient for each person</h2>
<div class="three">
  <div class="card"><h3>{N[0]}The twin</h3><p>A physiological model of glucose and insulin, calibrated to one patient.
  It starts from their health record on day 0 and sharpens with every day of wearable data.</p></div>
  <div class="card"><h3>{N[1]}Early warning</h3><p>Every 15 minutes: glucose forecast 30, 60 and 120 minutes ahead,
  and the probability of crossing 180 mg/dL in the next 2 hours, with the reasons.</p></div>
  <div class="card"><h3>{N[2]}What-if</h3><p>The doctor tests a plan before the patient eats: rajma chawal now,
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
  {exhibit(FIG / "p2_params_by_status.png", "", 600)}
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
slide(f"""<h2>The doctor's console</h2>{exhibit(dash, "Risk assessment with provenance, live twin and forecast, what-if with Indian meals, glucose profile, FHIR record", 720, keep=True) if dash.exists() else ''}""", "fullfig keep" + (" dark" if CLINIC else ""))

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
def link_row(label: str, url: str | None) -> str:
    if not url:
        return ""
    shown = url.removeprefix("https://").removeprefix("http://").removeprefix("www.").rstrip("/")
    return f'<div class="lrow"><span>{label}</span><a href="{escape(url)}">{escape(shown)}</a></div>'


shot = f'<div class="shot"><img src="{img(dash)}" alt="GlucoTwin dashboard"></div>' if dash.exists() else ""
if AURA:
    slide(f"""
<div class="vside l">GlucoTwin · Digital Twin Challenge 2026</div>
<div class="vside r">IIEST Shibpur</div>
<div class="center-stack">
  <p class="track">GlucoTwin</p>
  <h1 class="serif">Thank <em>you</em></h1>
  <p class="track sub">Try the doctor's console yourself: it runs in the browser, no login</p>
  <div class="links">
    {link_row("Live dashboard", ARGS.dashboard)}
    {link_row("Demo video", ARGS.video)}
    {link_row("Code and results", REPO_URL)}
  </div>
  <p class="note">Data: CGMacros (Gutierrez-Osuna, Kerr, Mortazavi, Das; PhysioNet 2025). Model: Bergman 1979, Dalla Man 2007.
  Clinical metrics: Clarke 1987, Battelino 2019 consensus on time in range.</p>
</div>""", "cover")
elif CLINIC:
    slide(f"""
<div class="close-c">
  <div>
    <h1><svg width="92" height="92" viewBox="0 0 22 22">{PULSE}</svg>GlucoTwin</h1>
    <p class="lede">Try the doctor's console yourself: it runs in the browser, no login.</p>
    <div class="links">
      {link_row("Live dashboard", ARGS.dashboard)}
      {link_row("Demo video", ARGS.video)}
      {link_row("Code and results", REPO_URL)}
    </div>
    <p class="note">Data: CGMacros (Gutierrez-Osuna, Kerr, Mortazavi, Das; PhysioNet 2025). Model: Bergman 1979, Dalla Man 2007.
    Clinical metrics: Clarke 1987, Battelino 2019 consensus on time in range.</p>
  </div>
  {shot}
</div>""", "cover")
else:
    slide(f"""
    <div class="hero close">
      <div class="title">
        {wordmark(132)}
        <p class="lede">Try the doctor's console yourself: it runs in the browser, no login.</p>
        <div class="links">
          {link_row("Live dashboard", ARGS.dashboard)}
          {link_row("Demo video", ARGS.video)}
          {link_row("Code and results", REPO_URL)}
        </div>
        <p class="note">Data: CGMacros (Gutierrez-Osuna, Kerr, Mortazavi, Das; PhysioNet 2025). Model: Bergman 1979, Dalla Man 2007.
        Clinical metrics: Clarke 1987, Battelino 2019 consensus on time in range.</p>
      </div>
      {shot}
    </div>""", "cover")

CSS_GLASS = """
@page { size: 1920px 1080px; margin: 0 }
:root {
  --ink: #f4f6fb;
  --ink-2: rgba(232, 236, 248, .80);
  --ink-3: rgba(222, 228, 245, .58);
  --teal: #2dd4bf; --sky: #60a5fa; --violet: #a78bfa; --pink: #f472b6;
  --grad: linear-gradient(100deg, #5eead4 0%, #7dd3fc 38%, #a78bfa 72%, #f0abfc 100%);
  --glass: linear-gradient(140deg, rgba(255,255,255,.15) 0%, rgba(255,255,255,.06) 55%, rgba(255,255,255,.10) 100%);
  --edge: rgba(255,255,255,.20);
}
* { box-sizing: border-box }
body { margin: 0; font-family: 'Geist', 'DejaVu Sans', sans-serif; color: var(--ink); background: #060914; }

/* ---- slide canvas: deep navy with a soft colour mesh; three arrangements rotate so slides don't look cloned */
.slide { width: 1920px; height: 1080px; position: relative; overflow: hidden; padding: 84px 110px 96px;
  page-break-after: always; display: flex; flex-direction: column; gap: 34px;
  background:
    radial-gradient(900px 700px at 8% 0%, rgba(45, 212, 191, .42), transparent 70%),
    radial-gradient(1000px 800px at 100% 100%, rgba(139, 92, 246, .48), transparent 70%),
    radial-gradient(700px 520px at 85% 8%, rgba(56, 189, 248, .22), transparent 70%),
    linear-gradient(160deg, #0a1024 0%, #070a18 55%, #0b0a1f 100%); }
.slide:nth-of-type(3n+2) { background:
    radial-gradient(1000px 760px at 100% 0%, rgba(99, 102, 241, .48), transparent 70%),
    radial-gradient(900px 700px at 0% 100%, rgba(20, 184, 166, .40), transparent 70%),
    radial-gradient(600px 460px at 55% 110%, rgba(236, 72, 153, .22), transparent 70%),
    linear-gradient(200deg, #0b0f26 0%, #070a17 60%, #081320 100%); }
.slide:nth-of-type(3n) { background:
    radial-gradient(900px 700px at 0% 0%, rgba(139, 92, 246, .44), transparent 70%),
    radial-gradient(1000px 780px at 100% 90%, rgba(45, 212, 191, .38), transparent 70%),
    radial-gradient(640px 480px at 70% 0%, rgba(244, 114, 182, .20), transparent 70%),
    linear-gradient(170deg, #0c0b22 0%, #070a18 55%, #071420 100%); }
/* faint grid texture for depth */
.glow { position: absolute; inset: 0; pointer-events: none;
  background-image: linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px);
  background-size: 96px 96px;
  -webkit-mask-image: radial-gradient(1200px 800px at 50% 40%, #000 30%, transparent 85%);
          mask-image: radial-gradient(1200px 800px at 50% 40%, #000 30%, transparent 85%); }
.slide > *:not(.glow) { position: relative; }

/* ---- type */
h1 { font-size: 168px; font-weight: 600; letter-spacing: -.05em; margin: 0; line-height: 1; }
h1 i { font-style: normal; color: #7dd3fc; }
.wordmark { display: block; overflow: visible; font-family: 'Geist', 'DejaVu Sans', sans-serif; }
h2 { font-size: 58px; font-weight: 600; letter-spacing: -.03em; margin: 0; line-height: 1.1; max-width: 1600px; }
h2::after { content: ""; display: block; width: 120px; height: 6px; border-radius: 6px; margin-top: 22px; background: var(--grad); }
h3 { font-size: 30px; font-weight: 600; letter-spacing: -.01em; margin: 0 0 14px; color: #fff; }
p { font-size: 27px; line-height: 1.45; color: var(--ink-2); margin: 0 0 16px; max-width: 70ch; }
p b { color: #fff; font-weight: 600; }
.big { font-size: 40px; line-height: 1.3; color: #fff; }
.big b { color: #5eead4; }
.note { font-size: 21px; color: var(--ink-3); }
.center { text-align: center; margin-left: auto; margin-right: auto; }
.accent { color: #99f6e4; }
.kicker { display: inline-block; font-size: 24px; color: #e6fffb; margin-bottom: 34px; padding: 10px 24px; border-radius: 99px;
  background: var(--glass); border: 1px solid var(--edge); box-shadow: inset 0 1px 0 rgba(255,255,255,.25); }
.lede { font-size: 42px; color: #fff; max-width: 34ch; line-height: 1.28; margin-top: 40px; }
.who { font-size: 26px; color: var(--ink-3); margin-top: 40px; }
.title { margin: auto 0; }
.mid { justify-content: center; gap: 48px; }
.mid .cols, .mid .three { flex: 0 0 auto; }
.mid .card p, .mid p { font-size: 28px; }

/* ---- glass surfaces */
.card, .stat, table, pre, .links, .ex {
  background: var(--glass); border: 1px solid var(--edge); border-radius: 28px;
  box-shadow: 0 24px 60px rgba(2, 6, 23, .45), inset 0 1px 0 rgba(255,255,255,.28), inset 0 -1px 0 rgba(255,255,255,.05); }
.card { padding: 34px 38px 22px; position: relative; overflow: hidden; }
.card::before { content: ""; position: absolute; left: 0; top: 0; right: 0; height: 4px; background: var(--grad); opacity: .9; }
.card p { font-size: 25px; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 48px; align-items: start; flex: 1; min-height: 0; }
.cols.wide-left { grid-template-columns: 1.55fr 1fr; }
.cols.wide-right { grid-template-columns: 1fr 1.25fr; }
.three { display: grid; grid-template-columns: repeat(3, 1fr); gap: 32px; }
.stack { display: flex; flex-direction: column; gap: 20px; }
.stat { padding: 20px 30px 18px; border-radius: 24px; }
.stat b { display: block; font-family: 'Geist Mono', 'DejaVu Sans Mono', monospace; font-size: 66px; font-weight: 400; letter-spacing: -.04em;
  line-height: 1.05; color: #5eead4; }
.stack .stat:nth-child(2) b, .c2 { color: #7dd3fc; }
.stack .stat:nth-child(3) b, .c3 { color: #c4b5fd; }
.stack .stat:nth-child(4) b { color: #f9a8d4; }
.c1 { color: #5eead4; }
.stat span { font-size: 22px; color: var(--ink-3); }

/* figures: white matplotlib plots sit on a bright pane inside a glass frame */
.ex { margin: 0; padding: 14px; display: flex; flex-direction: column; align-items: center; }
.ex img { max-width: 100%; display: block; background: #fff; }
.ex { overflow: hidden; }
.ex figcaption { font-size: 19px; color: var(--ink-3); margin-top: 12px; }
.fullfig .ex { flex: 1; justify-content: center; margin-bottom: 56px; }
.slide > p.center { align-self: center; margin-top: auto; margin-bottom: 40px; padding: 18px 34px; border-radius: 99px;
  background: var(--glass); border: 1px solid var(--edge); box-shadow: inset 0 1px 0 rgba(255,255,255,.25); max-width: none; }

table { border-collapse: separate; border-spacing: 0; font-size: 23px; width: 100%; margin-bottom: 18px; overflow: hidden; }
th { text-align: left; color: var(--ink-3); font-weight: 500; padding: 16px 18px 10px; border-bottom: 1px solid rgba(255,255,255,.14); }
td { padding: 13px 18px; border-bottom: 1px solid rgba(255,255,255,.07); color: var(--ink-2); vertical-align: top; }
tr:last-child td { border-bottom: 0; }
.tag { font-size: 18px; border-radius: 99px; padding: 3px 13px; border: 1px solid; white-space: nowrap; }
.tag.real { color: #6ee7b7; border-color: rgba(110,231,183,.5); background: rgba(16,185,129,.12); }
.tag.syn { color: #fcd34d; border-color: rgba(252,211,77,.5); border-style: dashed; background: rgba(245,158,11,.10); }
pre { font-family: 'Geist Mono', 'DejaVu Sans Mono', monospace; font-size: 25px; line-height: 1.6; padding: 28px 32px; color: #e0e7ff; margin: 0 0 22px; }

/* cover slides */
.cover .title { position: relative; }
.links { margin: 44px 0 40px; display: flex; flex-direction: column; max-width: 1300px; padding: 8px 34px; }
.lrow { display: grid; grid-template-columns: 300px 1fr; align-items: baseline; padding: 20px 0; border-top: 1px solid rgba(255,255,255,.12); }
.close .lrow { grid-template-columns: 1fr; gap: 4px; padding: 16px 0; }
.close .lrow a { font-size: 29px; }
.close .links { margin: 36px 0 32px; }

/* hero layout on the title and closing slides */
.hero { display: grid; grid-template-columns: 1.15fr 1fr; gap: 40px; align-items: center; margin: auto 0; }
.hero .title { margin: 0; }
.hero.close { grid-template-columns: 1fr 1.05fr; }
.tiles { position: relative; height: 640px; }
.tile, .pill { position: absolute; background: var(--glass); border: 1px solid var(--edge);
  box-shadow: 0 30px 70px rgba(2, 6, 23, .5), inset 0 1px 0 rgba(255,255,255,.3); }
.tile { width: 470px; padding: 30px 36px; border-radius: 30px; }
.tile b { display: block; font-family: 'Geist Mono', 'DejaVu Sans Mono', monospace; font-weight: 400; font-size: 84px; letter-spacing: -.04em; line-height: 1.05; }
.tile span { display: block; font-size: 23px; color: var(--ink-2); margin-top: 6px; }
.t1 { left: 40px; top: 70px; } .t2 { left: 300px; top: 250px; } .t3 { left: 90px; top: 440px; }
.pill { right: 10px; top: 0; padding: 14px 26px; border-radius: 99px; font-size: 22px; color: #fecdd3;
  background: linear-gradient(140deg, rgba(244,63,94,.28), rgba(244,63,94,.10)); border-color: rgba(251,113,133,.45); }
.pill i { display: inline-block; width: 12px; height: 12px; border-radius: 50%; background: #fb7185; margin-right: 12px;
  box-shadow: 0 0 0 6px rgba(251,113,133,.22); vertical-align: 1px; }
.shot { padding: 14px; border-radius: 30px; background: var(--glass); border: 1px solid var(--edge);
  box-shadow: 0 40px 90px rgba(2, 6, 23, .6), inset 0 1px 0 rgba(255,255,255,.3);
  transform: perspective(2200px) rotateY(-13deg) rotateX(3deg); transform-origin: left center; }
.shot img { display: block; width: 100%; border-radius: 18px; }
.lrow:first-child { border-top: 0; }
.lrow span { font-size: 24px; color: var(--ink-3); }
.lrow a { font-family: 'Geist Mono', 'DejaVu Sans Mono', monospace; font-size: 34px; color: #fff; text-decoration: none; letter-spacing: -.01em; }

.slide > footer:not(.glow) { position: absolute; left: 110px; right: 110px; bottom: 36px; display: flex; justify-content: space-between;
  font-size: 18px; color: rgba(226, 232, 240, .42); }
"""

CSS_AURA = """
@page { size: 1920px 1080px; margin: 0 }
:root { --ink: #f5f4f8; --ink-2: rgba(240, 239, 246, .80); --ink-3: rgba(232, 230, 242, .58); --pearl: #ddd6fe;
  --serif: 'Instrument Serif', 'DejaVu Serif', serif; --sans: 'Geist', 'DejaVu Sans', sans-serif; --mono: 'Geist Mono', 'DejaVu Sans Mono', monospace; }
* { box-sizing: border-box }
body { margin: 0; font-family: var(--sans); color: var(--ink); background: #1d1d20; }
.slide { width: 1920px; height: 1080px; position: relative; overflow: hidden; padding: 96px 150px 104px;
  page-break-after: always; display: flex; flex-direction: column; gap: 36px;
  background: radial-gradient(1400px 900px at 50% 45%, #2b2b2f 0%, #232326 55%, #1b1b1e 100%); }
.slide > *:not(.blobs):not(.orn):not(.glow) { position: relative; z-index: 2; }
.glow { display: none; }
.orn { position: absolute; inset: 0; z-index: 1; pointer-events: none; }

/* type */
h1.serif { font-family: var(--serif); font-weight: 400; font-size: 250px; line-height: .95; letter-spacing: -.01em; margin: 26px 0 30px; color: #fff; }
h1.serif em, h2 em { font-style: italic; color: var(--pearl); }
h2 { font-family: var(--serif); font-weight: 400; font-size: 84px; line-height: 1.02; letter-spacing: -.005em; margin: 0; max-width: 1500px; color: #fff; }
h3 { font-size: 28px; font-weight: 600; margin: 0 0 14px; color: #fff; }
p { font-size: 26px; line-height: 1.5; color: var(--ink-2); margin: 0 0 16px; max-width: 70ch; font-weight: 400; }
p b { color: #fff; font-weight: 600; }
.big { font-size: 42px; line-height: 1.3; color: #fff; font-weight: 300; }
.big b { color: var(--pearl); font-weight: 500; }
.note { font-size: 20px; color: var(--ink-3); letter-spacing: .02em; }
.accent { color: var(--pearl); }
.center { text-align: center; margin-left: auto; margin-right: auto; }
.track { font-size: 22px; letter-spacing: .34em; color: rgba(255,255,255,.72); font-weight: 300; margin: 0; max-width: none; }
.mid { justify-content: center; gap: 56px; }
.mid .cols, .mid .three { flex: 0 0 auto; }

/* white label bar, the template's signature caption */
.slide > p.center { align-self: center; margin-top: auto; margin-bottom: 30px; background: #f3f2f6; color: #1b1b1f; padding: 14px 30px;
  border-radius: 2px; font-size: 23px; font-weight: 500; max-width: none; }
.slide > p.center b { color: #000; }
.mid > p.center { margin-top: 0; }

/* columns: numbered, ruled, no boxes */
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 72px; align-items: start; flex: 1; min-height: 0; }
.cols.wide-left { grid-template-columns: 1.55fr 1fr; }
.cols.wide-right { grid-template-columns: 1fr 1.25fr; }
.three { display: grid; grid-template-columns: repeat(3, 1fr); gap: 64px; counter-reset: c; }
.card { border-top: 1px solid rgba(255,255,255,.38); padding: 24px 0 0; }
.three .card::before { counter-increment: c; content: counter(c, decimal-leading-zero); display: block; font-family: var(--serif);
  font-size: 64px; line-height: 1; color: #fff; margin-bottom: 18px; }
.card p { font-size: 24px; }
.stack { display: flex; flex-direction: column; gap: 26px; }
.stat { border-left: 1px solid rgba(255,255,255,.35); padding-left: 28px; }
.stat b { display: block; font-family: var(--serif); font-weight: 400; font-size: 92px; line-height: 1; color: #fff; }
.stat span { font-size: 21px; color: var(--ink-3); letter-spacing: .04em; }

/* figures: plots turned to the dark palette; the dashboard screenshot keeps its own colours */
.ex { margin: 0; display: flex; flex-direction: column; align-items: center; }
.ex img { max-width: 100%; display: block; border-radius: 6px; border: 1px solid rgba(255,255,255,.10); }
.keep .ex img { border-radius: 10px; box-shadow: 0 40px 90px rgba(0,0,0,.55); }
.slide:nth-of-type(11) .ex img { max-height: 400px !important; }
.ex figcaption { font-size: 18px; letter-spacing: .22em; color: var(--ink-3); margin-top: 14px; }
.fullfig .ex { flex: 1; justify-content: center; margin-bottom: 30px; }

table { border-collapse: separate; border-spacing: 0 6px; font-size: 22px; width: 100%; margin-bottom: 14px; }
th { text-align: left; background: #f3f2f6; color: #1b1b1f; font-weight: 600; font-size: 16px; letter-spacing: .16em; text-transform: uppercase; padding: 12px 16px; }
td { padding: 12px 16px; background: rgba(255,255,255,.06); color: var(--ink-2); vertical-align: top; }
.tag { font-size: 17px; border-radius: 99px; padding: 2px 12px; border: 1px solid; white-space: nowrap; letter-spacing: .06em; }
.tag.real { color: #a7f3d0; border-color: rgba(167,243,208,.5); }
.tag.syn { color: #fde68a; border-color: rgba(253,230,138,.5); border-style: dashed; }
pre { font-family: var(--mono); font-size: 25px; line-height: 1.6; background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.12);
  border-radius: 4px; padding: 26px 30px; color: #ede9fe; margin: 0 0 22px; }

/* cover and thank-you */
.center-stack { margin: auto; text-align: center; display: flex; flex-direction: column; align-items: center; }
.center-stack .sub { max-width: 1240px; line-height: 1.7; font-size: 21px; }
.center-stack .who { margin-top: 34px; font-size: 19px; color: rgba(255,255,255,.55); }
.center-stack .note { margin-top: 26px; max-width: 1200px; }
.vside { position: absolute !important; top: 50%; font-size: 17px; letter-spacing: .34em; color: rgba(255,255,255,.55); white-space: nowrap; }
.vside.l { left: 58px; writing-mode: vertical-rl; transform: translateY(-50%) rotate(180deg); }
.vside.r { right: 58px; writing-mode: vertical-rl; transform: translateY(-50%); }
.links { margin: 44px 0 6px; border-top: 1px solid rgba(255,255,255,.3); border-bottom: 1px solid rgba(255,255,255,.3); padding: 8px 0; }
.lrow { display: grid; grid-template-columns: 260px auto; gap: 24px; align-items: baseline; padding: 14px 10px; text-align: left; }
.lrow span { font-size: 17px; letter-spacing: .28em; color: var(--ink-3); text-transform: uppercase; }
.lrow a { font-family: var(--mono); font-size: 30px; color: #fff; text-decoration: none; }

.slide > footer { position: absolute; left: 0; right: 0; bottom: 40px; display: flex; justify-content: center; font-size: 16px;
  letter-spacing: .3em; color: rgba(255,255,255,.4); z-index: 2; }
.slide > footer span:first-child { display: none; }
.cover > footer { display: none; }
"""

if AURA:
    CSS_AURA += "".join(f".slide.{k} {{ background: #232326 url({glow_bg(b, i)}) center / cover no-repeat; }}\n"
                        for i, (k, b) in enumerate(GLOWS.items()))
CSS_CLINIC = """
@page { size: 1920px 1080px; margin: 0 }
:root { --bg: #f2f3f4; --surface: #fff; --surface-2: #f7f8f9; --text: #1e2023; --text-2: #555b63; --text-3: #62686f; --faint: #8b9198;
  --border: #e1e3e6; --accent: #2a2c31; --pred: #4f7493; --pred-ink: #3d6382; --pred-lite: #8fb3d4;
  --crit: #b0393f; --crit-mark: #d9575e; --ok: #3b7353; --ok-mark: #6fa585; --warn: #865410; --plan-b: #a8732f;
  --font: 'Inter', 'DejaVu Sans', sans-serif; --mono: 'Geist Mono', 'DejaVu Sans Mono', monospace;
  --sheet: 0 1px 2px rgba(30,32,35,.05), 0 14px 34px rgba(30,32,35,.07); }
* { box-sizing: border-box }
body { margin: 0; font-family: var(--font); color: var(--text); background: var(--bg); font-feature-settings: "tnum" 1, "cv11" 1; }

/* ---------- light content slides: the dashboard's surfaces, chrome and cards ---------- */
.slide { width: 1920px; height: 1080px; position: relative; overflow: hidden; padding: 156px 110px 84px;
  page-break-after: always; display: flex; flex-direction: column; gap: 34px;
  background:
    radial-gradient(900px 640px at 100% 0%, rgba(79,116,147,.13), transparent 70%),
    radial-gradient(800px 560px at 0% 100%, rgba(111,165,133,.10), transparent 70%),
    linear-gradient(rgba(30,32,35,.035) 1px, transparent 1px) 0 0 / 100% 54px,
    linear-gradient(90deg, rgba(30,32,35,.035) 1px, transparent 1px) 0 0 / 54px 100%,
    var(--bg); }
.slide > * { position: relative; }
.glow { display: none; }
.chrome { position: absolute !important; left: 0; right: 0; top: 0; height: 86px; display: flex; align-items: center; gap: 64px;
  padding: 0 110px; background: rgba(255,255,255,.92); border-bottom: 1px solid var(--border); z-index: 3; }
.chrome .brand { display: flex; align-items: center; gap: 12px; font-size: 26px; font-weight: 600; letter-spacing: -.01em; color: var(--text); }
.chrome nav { display: flex; gap: 42px; height: 100%; }
.chrome nav span { display: flex; align-items: center; height: 100%; font-size: 21px; color: var(--text-2); position: relative; }
.chrome nav span.on { color: var(--text); font-weight: 600; }
.chrome nav span.on::after { content: ""; position: absolute; left: 0; right: 0; bottom: -1px; height: 4px; border-radius: 4px 4px 0 0; background: var(--accent); }
.chrome .ctx { margin-left: auto; font-size: 18px; color: var(--faint); display: flex; align-items: center; gap: 18px; }
.chrome .ctx b { background: var(--accent); color: #f5f5f5; font-weight: 600; font-size: 17px; padding: 6px 13px; border-radius: 99px; }

h2 { font-size: 64px; font-weight: 700; letter-spacing: -.035em; line-height: 1.06; margin: 0; max-width: 1600px; color: var(--text); }
h3 { font-size: 28px; font-weight: 600; letter-spacing: -.015em; margin: 0 0 12px; color: var(--text); }
p { font-size: 26px; line-height: 1.5; color: var(--text-2); margin: 0 0 16px; max-width: 70ch; }
p b { color: var(--text); font-weight: 600; }
.big { font-size: 42px; line-height: 1.28; color: var(--text); font-weight: 500; letter-spacing: -.02em; }
.big b { color: var(--pred-ink); font-weight: 700; }
.note { font-size: 20px; color: var(--text-3); }
.accent { color: var(--pred-ink); font-weight: 500; }
.center { text-align: center; margin-left: auto; margin-right: auto; }
.mid { justify-content: center; gap: 46px; }
.mid .cols, .mid .three { flex: 0 0 auto; }

/* key line = the dashboard's charcoal primary button */
.slide > p.center { align-self: center; margin-top: auto; margin-bottom: 6px; background: var(--accent); color: #f2f3f4; border-radius: 99px;
  padding: 17px 36px; font-size: 23px; max-width: none; box-shadow: 0 12px 30px rgba(30,32,35,.18); }
.slide > p.center b { color: #fff; }
.mid > p.center { margin-top: 0; }

.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 48px; align-items: start; flex: 1; min-height: 0; }
.cols.wide-left { grid-template-columns: 1.55fr 1fr; }
.cols.wide-right { grid-template-columns: 1fr 1.25fr; }
.three { display: grid; grid-template-columns: repeat(3, 1fr); gap: 32px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 18px; box-shadow: var(--sheet); padding: 34px 38px 24px;
  border-top: 5px solid var(--pred); }
.three .card:nth-child(2) { border-top-color: var(--crit-mark); }
.three .card:nth-child(3) { border-top-color: var(--plan-b); }
.card p { font-size: 24px; }
.stack { display: flex; flex-direction: column; gap: 18px; }
.stat { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; box-shadow: var(--sheet); padding: 18px 28px 16px 30px;
  border-left: 5px solid var(--pred); }
.stat b { display: block; font-size: 62px; font-weight: 700; letter-spacing: -.04em; line-height: 1.05; color: var(--pred-ink); }
.stat span { font-size: 21px; color: var(--text-3); }

.ex { margin: 0; background: var(--surface); border: 1px solid var(--border); border-radius: 18px; box-shadow: var(--sheet); padding: 18px;
  display: flex; flex-direction: column; align-items: center; }
.ex img { max-width: 100%; display: block; }
.ex figcaption { font-size: 18px; color: var(--text-3); margin-top: 10px; }
.fullfig .ex { flex: 1; justify-content: center; margin-bottom: 10px; }

table { border-collapse: separate; border-spacing: 0; font-size: 22px; width: 100%; margin-bottom: 16px; background: var(--surface);
  border: 1px solid var(--border); border-radius: 16px; overflow: hidden; box-shadow: var(--sheet); }
th { text-align: left; background: var(--surface-2); color: var(--text-3); font-weight: 600; font-size: 15px; letter-spacing: .1em; text-transform: uppercase;
  padding: 14px 18px; border-bottom: 1px solid var(--border); }
td { padding: 13px 18px; border-bottom: 1px solid #eceef0; color: var(--text-2); vertical-align: top; }
tr:last-child td { border-bottom: 0; }
.tag { font-size: 16px; font-weight: 600; border-radius: 99px; padding: 4px 13px; white-space: nowrap; }
.tag.real { color: var(--ok); background: rgba(111,165,133,.16); }
.tag.syn { color: var(--warn); background: rgba(201,143,56,.16); border: 1px dashed rgba(201,143,56,.6); }
pre { font-family: var(--mono); font-size: 24px; line-height: 1.65; background: #17191c; color: #dbe7f2; border-radius: 16px; padding: 28px 32px; margin: 0 0 22px;
  box-shadow: 0 18px 40px rgba(30,32,35,.22); }
.slide > footer { display: none; }

/* ---------- graphite hero slides: cover, console, close ---------- */
.cover, .dark { color: #fff; background:
    radial-gradient(1100px 760px at 78% 18%, rgba(79,116,147,.55), transparent 62%),
    radial-gradient(900px 700px at 4% 104%, rgba(111,165,133,.20), transparent 62%),
    radial-gradient(760px 560px at 100% 104%, rgba(217,87,94,.16), transparent 62%),
    linear-gradient(160deg, #15181c 0%, #101214 60%, #0d0f12 100%); }
.halo { position: absolute !important; inset: 0; pointer-events: none;
  background: linear-gradient(rgba(255,255,255,.04) 1px, transparent 1px) 0 0 / 100% 54px,
              linear-gradient(90deg, rgba(255,255,255,.04) 1px, transparent 1px) 0 0 / 54px 100%;
  -webkit-mask-image: radial-gradient(1300px 820px at 60% 40%, #000 25%, transparent 80%); mask-image: radial-gradient(1300px 820px at 60% 40%, #000 25%, transparent 80%); }
.cover h1, .dark h2 { color: #fff; }
.cover p, .dark p { color: rgba(255,255,255,.78); }
.glassc { background: linear-gradient(140deg, rgba(255,255,255,.12), rgba(255,255,255,.04)); border: 1px solid rgba(255,255,255,.16);
  box-shadow: 0 30px 70px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.2); }

.cover { padding: 70px 100px 46px; }
.cover-c { display: flex; flex-direction: column; height: 100%; }
.cover-c .row { display: grid; grid-template-columns: 1fr 590px; gap: 70px; align-items: end; }
.eyebrow { display: inline-block; font-size: 21px; color: #dbe7f2 !important; padding: 9px 22px; border-radius: 99px; margin: 0 0 26px;
  background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.16); }
.cover h1 { font-size: 150px; font-weight: 800; letter-spacing: -.05em; line-height: .95; margin: 0 0 26px; display: flex; align-items: center; gap: 22px; }
.cover h1 svg { color: var(--pred-lite); stroke-width: 2.4; }
.cover .lede { font-size: 33px; line-height: 1.32; color: rgba(255,255,255,.86); max-width: 38ch; margin: 0 0 22px; }
.cover .who { font-size: 21px; color: rgba(255,255,255,.55) !important; margin: 0; }
.alert-card { border-radius: 28px; padding: 28px 34px 22px;
  background: linear-gradient(140deg, rgba(255,255,255,.13), rgba(255,255,255,.04)); border: 1px solid rgba(255,255,255,.18);
  box-shadow: 0 30px 70px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.22); }
.chip { display: inline-flex; align-items: center; gap: 10px; font-size: 19px; font-weight: 600; border-radius: 99px; padding: 6px 16px; }
.chip i { width: 11px; height: 11px; border-radius: 50%; display: inline-block; }
.chip.crit { color: #ffc2c5; background: rgba(217,87,94,.22); border: 1px solid rgba(217,87,94,.55); }
.chip.crit i { background: #ff7a81; box-shadow: 0 0 0 5px rgba(255,122,129,.22); }
.alert-card > b { display: block; font-size: 112px; font-weight: 700; letter-spacing: -.05em; line-height: 1; margin: 14px 0 4px; color: #ff9aa0; }
.alert-card > b small { font-size: 34px; font-weight: 500; color: rgba(255,255,255,.55); letter-spacing: 0; }
.alert-card > p { font-size: 23px; color: rgba(255,255,255,.88); margin: 0 0 18px; }
.kpis { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; border-top: 1px solid rgba(255,255,255,.14); padding-top: 16px; }
.kpis span { font-size: 15px; color: rgba(255,255,255,.6); line-height: 1.3; }
.kpis em { display: block; font-style: normal; font-size: 32px; font-weight: 700; letter-spacing: -.03em; color: #cfe0ef; }
.alert-card .src { font-size: 15px; color: rgba(255,255,255,.45) !important; margin: 14px 0 0; }
.hero-curve { margin-top: auto; display: block; overflow: visible; font-family: var(--font); }
.hc-grid { stroke: rgba(255,255,255,.07); stroke-width: 1; }
.hc-ax { fill: rgba(255,255,255,.45); font-size: 18px; }
.hc-future { fill: rgba(143,179,212,.07); }
.hc-thr { stroke: #e5646b; stroke-width: 2.5; stroke-dasharray: 10 8; }
.hc-thr-l { fill: #f4a3a8; font-size: 20px; font-weight: 600; }
.hc-band { fill: rgba(143,179,212,.24); }
.hc-act { fill: none; stroke: rgba(255,255,255,.6); stroke-width: 4; stroke-dasharray: 1 10; stroke-linecap: round; }
.hc-meas { fill: none; stroke: #fff; stroke-width: 4.5; stroke-linejoin: round; }
.hc-fc { fill: none; stroke: var(--pred-lite); stroke-width: 3; stroke-dasharray: 12 8; }
.hc-now { stroke: rgba(255,255,255,.35); stroke-width: 2; }
.hc-meal { stroke: rgba(255,255,255,.6); stroke-width: 3; }
.hc-lbl { fill: #fff; font-size: 20px; font-weight: 600; paint-order: stroke; stroke: #111316; stroke-width: 8px; stroke-linejoin: round; }
.hc-nowdot { fill: #fff; stroke: #101214; stroke-width: 4; }
.hc-pt { fill: var(--pred-lite); stroke: #101214; stroke-width: 4; }
.legend { display: flex; gap: 40px; justify-content: flex-end; font-size: 17px; color: rgba(255,255,255,.6); margin-top: 4px; padding-right: 150px; }
.legend span::before { content: ""; display: inline-block; width: 34px; height: 0; margin-right: 10px; vertical-align: middle; }
.l-meas::before { border-top: 4px solid #fff; }
.l-fc::before { border-top: 3px dashed var(--pred-lite); }
.l-act::before { border-top: 4px dotted rgba(255,255,255,.7); }

.dark { padding: 90px 110px 60px; }
.dark h2 { font-size: 64px; font-weight: 700; letter-spacing: -.035em; }
.dark .ex { background: none; border: 0; box-shadow: none; padding: 0; }
.dark .ex img { border-radius: 18px; box-shadow: 0 50px 120px rgba(0,0,0,.6), 0 0 0 1px rgba(255,255,255,.12), 0 0 120px rgba(79,116,147,.35); }
.dark .ex figcaption { color: rgba(255,255,255,.6); font-size: 19px; margin-top: 22px; }

.close-c { display: grid; grid-template-columns: 1fr 1.05fr; gap: 70px; align-items: center; margin: auto 0; }
.close-c h1 { font-size: 118px; font-weight: 800; letter-spacing: -.05em; line-height: 1; margin: 0 0 24px; display: flex; align-items: center; gap: 18px; color: #fff; }
.close-c h1 svg { color: var(--pred-lite); }
.close-c .lede { font-size: 34px; line-height: 1.3; color: rgba(255,255,255,.86); max-width: 26ch; }
.close-c .note { font-size: 17px; color: rgba(255,255,255,.45) !important; max-width: 60ch; }
.links { margin: 30px 0 30px; padding: 6px 30px; border-radius: 24px;
  background: linear-gradient(140deg, rgba(255,255,255,.12), rgba(255,255,255,.04)); border: 1px solid rgba(255,255,255,.16);
  box-shadow: 0 30px 70px rgba(0,0,0,.4), inset 0 1px 0 rgba(255,255,255,.2); }
.lrow { display: grid; gap: 4px; padding: 16px 0; border-top: 1px solid rgba(255,255,255,.12); }
.lrow:first-child { border-top: 0; }
.lrow span { font-size: 15px; font-weight: 600; letter-spacing: .14em; text-transform: uppercase; color: rgba(255,255,255,.5); }
.lrow a { font-size: 29px; font-weight: 600; color: #dbe7f2; text-decoration: none; letter-spacing: -.01em; }
.shot { transform: perspective(2200px) rotateY(-12deg) rotateX(3deg); transform-origin: left center; }
.shot img { display: block; width: 100%; border-radius: 16px;
  box-shadow: 0 50px 120px rgba(0,0,0,.6), 0 0 0 1px rgba(255,255,255,.14), 0 0 140px rgba(79,116,147,.45); }
"""

CSS = CSS_AURA if AURA else CSS_CLINIC if CLINIC else CSS_GLASS
html = f"<!doctype html><html><head><meta charset='utf-8'><style>{font_faces()}{CSS}</style></head><body>{''.join(slides)}</body></html>"
DOCS.mkdir(exist_ok=True)
OUT = Path(ARGS.out) if ARGS.out else DOCS / "presentation.pdf"
(DOCS / "presentation.html").write_text(html)

from playwright.sync_api import sync_playwright  # noqa: E402

# Chromium's print-to-PDF cannot express blurred box-shadows, masks or layered gradients as vectors: it
# writes each one as a large soft-masked image, and PDF viewers (Chrome's especially) re-composite all
# of them on every scroll. So the decorative layer of each slide is baked into one JPEG, and only text,
# figures, SVG and links are drawn on top as before. Pass --no-flatten to get the raw output.
BAKE_CSS = """
html.bake .slide *, html.bake .slide *::before, html.bake .slide *::after { color: transparent !important; text-shadow: none !important; }
html.bake .slide svg, html.bake .slide text { visibility: hidden !important; }
html.bake .slide img { object-position: -99999px 0 !important; }
html.flat .slide *, html.flat .slide *::before, html.flat .slide *::after { background: transparent !important; box-shadow: none !important;
  border-color: transparent !important; outline-color: transparent !important; -webkit-mask-image: none !important; mask-image: none !important;
  -webkit-backdrop-filter: none !important; backdrop-filter: none !important; }
html.flat .slide .halo, html.flat .slide .glow, html.flat .slide .blobs { visibility: hidden !important; }
"""


def flatten(pg, n: int, scale: float = 1.5) -> None:
    pg.add_style_tag(content=BAKE_CSS)
    pg.set_viewport_size({"width": 1920, "height": 1080})
    cdp = pg.context.new_cdp_session(pg)
    cdp.send("Emulation.setDeviceMetricsOverride", {"width": 1920, "height": 1080, "deviceScaleFactor": scale, "mobile": False})
    pg.evaluate("document.documentElement.classList.add('bake')")
    pg.evaluate("document.fonts.ready")
    uris = []
    for i in range(n):
        shot = pg.screenshot(type="jpeg", quality=88, full_page=True, clip={"x": 0, "y": 1080 * i, "width": 1920, "height": 1080})
        uris.append("data:image/jpeg;base64," + base64.b64encode(shot).decode())
    cdp.send("Emulation.clearDeviceMetricsOverride")
    pg.evaluate("""(u) => { const s = document.querySelectorAll('section.slide');
        s.forEach((el, i) => { el.style.setProperty('background', `#fff url(${u[i]}) 0 0 / 1920px 1080px no-repeat`, 'important'); });
        document.documentElement.classList.remove('bake'); document.documentElement.classList.add('flat'); }""", uris)


with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1920, "height": 1080})
    pg.set_content(html, wait_until="load")
    if not ARGS.no_flatten:
        flatten(pg, len(slides))
    pg.pdf(path=str(OUT), width="1920px", height="1080px", print_background=True)
    for i in range(1, len(slides) + 1):
        pg.screenshot(path=str(REP / f"_slide_preview_{i}.png"), full_page=True, clip={"x": 0, "y": 1080 * (i - 1), "width": 1920, "height": 1080})
    b.close()
(DOCS / "presentation.html").unlink()
print(f"wrote {OUT} ({len(slides)} slides, theme {ARGS.theme})")
