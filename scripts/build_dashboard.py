"""Wrap dashboard/app.html into a standalone dashboard/index.html (for Vercel / GitHub Pages / local)."""
from pathlib import Path

d = Path(__file__).resolve().parents[1] / "dashboard"
body = (d / "app.html").read_text()
head_end = body.index("</style>") + len("</style>")
html = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
        + body[:head_end] + "\n</head>\n<body>\n" + body[head_end:] + "\n</body>\n</html>\n")
(d / "index.html").write_text(html)
print("wrote", d / "index.html")
