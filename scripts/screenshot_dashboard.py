"""Screenshot the dashboard at an informative moment, for the README and slides -> docs/dashboard.png.

Picks the patient/time with the strongest correct alert on a held-out day: risk above threshold and
glucose actually crossing 180 mg/dL afterwards, so the screenshot shows the product doing its job.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "dashboard"


def fonts_css() -> str:
    """Embed Inter so headless renders match the deployed page (which loads it from Google Fonts).
    FONTSOURCE_DIR points at node_modules/@fontsource with inter installed."""
    base = Path(os.environ.get("FONTSOURCE_DIR", "/nonexistent"))
    css = []
    for fam, pre, weights in (("Inter", "inter", (400, 500, 600)),):
        for w in weights:
            f = base / pre / "files" / f"{pre}-latin-{w}-normal.woff2"
            if f.exists():
                css.append(f"@font-face{{font-family:'{fam}';font-weight:{w};src:url(data:font/woff2;base64,"
                           f"{base64.b64encode(f.read_bytes()).decode()}) format('woff2')}}")
    return "\n".join(css)


def best_moment(data: dict):
    thr = data["alert_threshold"]
    best = None
    for p in data["patients"]:
        R = p["replay"]
        g = R["glucose"]
        for o in R["origins"]:
            if (o["p"] or 0) < thr or o["t"] < 360 or o["t"] > 1320:
                continue
            i = (o["t"] - R["grid_t0"]) // 5
            now = g[i] if i < len(g) else None
            fut = [v for v in g[i + 1:i + 25] if v is not None]
            if now is None or now >= 170 or not fut or max(fut) < 180:
                continue
            score = o["p"] + (max(fut) - now) / 200
            if best is None or score > best[0]:
                best = (score, p["id"], o["t"])
    return best


def main():
    data_js = (D / "data.js").read_text()
    data = json.loads(data_js[data_js.index("{"):data_js.rindex("}") + 1])
    pick = best_moment(data)
    html = (D / "index.html").read_text().replace("<style>", "<style>" + fonts_css(), 1)
    tmp = D / "_shot.html"
    tmp.write_text(html)
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={"width": 1440, "height": 960}, device_scale_factor=1.5)
        pg.goto(tmp.as_uri())
        pg.wait_for_timeout(600)
        if pick:
            _, pid, t = pick
            pg.evaluate("""([pid, t]) => {
                localStorage.setItem('gt-patient', pid);
            }""", [pid, t])
            pg.reload()
            pg.wait_for_timeout(500)
            pg.evaluate("""(t) => { const s = document.getElementById('time'); s.value = t; s.dispatchEvent(new Event('input')); }""", t)
            pg.evaluate("""() => { const r = document.getElementById('reveal'); r.checked = true; r.dispatchEvent(new Event('change')); }""")
            pg.wait_for_timeout(900)
            print("moment:", pick)
        pg.screenshot(path=str(ROOT / "docs" / "dashboard.png"), full_page=False)
        pg.screenshot(path=str(ROOT / "docs" / "dashboard_full.png"), full_page=True)
        b.close()
    tmp.unlink()
    print("wrote docs/dashboard.png")


if __name__ == "__main__":
    main()
