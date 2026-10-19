"""Phase 3 figures: forecast accuracy, clinical safety, explanations, alerts."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from glucotwin import config
from glucotwin.analysis.figures import AXIS, INK, INK2, MUTED, SURFACE, _style

MODEL_COLOR = {"Hybrid twin + ML": "#2a78d6", "Pure ML (LightGBM)": "#eb6834",
               "Twin only (physics)": "#1baf7a", "No change": MUTED}


def fig_rmse_by_horizon(reg: pd.DataFrame, path):
    _style()
    fig, ax = plt.subplots(figsize=(8, 4.4))
    hs = sorted(reg["horizon_min"].unique())
    for name, c in MODEL_COLOR.items():
        d = reg[reg["model"] == name].set_index("horizon_min").loc[hs]
        lw = 2.6 if name.startswith("Hybrid") else 1.8
        ax.plot(hs, d["rmse"], color=c, lw=lw, marker="o", ms=7, markeredgecolor=SURFACE, markeredgewidth=1.4)
        ax.text(hs[-1] + 4, d["rmse"].iloc[-1], f"{name}  {d['rmse'].iloc[-1]:.1f}", va="center", fontsize=9, color=INK)
    ax.set_xticks(hs)
    ax.set_xticklabels([f"+{h} min" for h in hs])
    ax.set_xlim(hs[0] - 8, hs[-1] + 70)
    ax.set_ylabel("RMSE (mg/dL), held-out patients & days")
    ax.set_title("Forecast error by horizon: physics + ML beats either alone", pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_clarke(ref, pred, row, path):
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    m = ~(np.isnan(ref) | np.isnan(pred))
    ax.scatter(ref[m], pred[m], s=4, color="#2a78d6", alpha=0.25, linewidths=0)
    k = dict(color=INK2, lw=1)
    for xs, ys in [([0, 175 / 3], [70, 70]), ([175 / 3, 400 / 1.2], [70, 400]), ([70, 70], [84, 400]),
                   ([0, 70], [180, 180]), ([70, 290], [180, 400]), ([70, 70], [0, 56]), ([70, 400], [56, 320]),
                   ([180, 180], [0, 70]), ([180, 400], [70, 70]), ([240, 240], [70, 180]), ([240, 400], [180, 180]),
                   ([130, 180], [0, 70])]:
        ax.plot(xs, ys, **k)
    ax.plot([0, 400], [0, 400], color=AXIS, lw=1, ls=":")
    for txt, (x, y) in {"A": (30, 15), "B": (370, 260), "B ": (280, 370), "C": (160, 370), "C ": (160, 15),
                        "D": (30, 140), "D ": (370, 120), "E": (30, 370), "E ": (370, 15)}.items():
        ax.text(x, y, txt.strip(), fontsize=13, color=MUTED, fontweight="bold", ha="center")
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_xlabel("Actual glucose at +60 min (mg/dL)")
    ax.set_ylabel("Hybrid twin forecast (mg/dL)")
    ax.set_title("Clarke Error Grid, 60-min forecasts", pad=26)
    ax.text(0, 1.02, f"Zone A {row['clarke_A']:.1f}%  ·  A+B {row['clarke_AB']:.1f}%  ·  C/D/E {row['clarke_CDE']:.2f}%",
            transform=ax.transAxes, color=INK2, fontsize=9.5)
    ax.set_aspect("equal")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_shap_importance(imp: pd.Series, labels: dict, path, top=12):
    _style()
    imp = imp.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(range(len(imp)), imp.values, color="#2a78d6", height=0.6)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels([labels.get(k, k) for k in imp.index], color=INK)
    ax.set_xlabel("Mean |SHAP value| (impact on spike-risk log-odds)")
    ax.set_title("What drives a spike alert", pad=12)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_ablation(abl: pd.DataFrame, path):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    y = np.arange(len(abl))[::-1]
    for ax, col, lab, fmt in [(axes[0], "auroc", "Spike AUROC (higher is better)", "{:.3f}"),
                              (axes[1], "rmse60", "60-min RMSE, mg/dL (lower is better)", "{:.1f}")]:
        ax.barh(y, abl[col], color=["#2a78d6"] + ["#86b6ef"] * (len(abl) - 1), height=0.6)
        for yi, v in zip(y, abl[col]):
            ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=9, color=INK)
        lo = abl[col].min()
        ax.set_xlim(lo * 0.9 if col == "rmse60" else max(0.5, lo - 0.08), abl[col].max() * (1.12 if col == "rmse60" else 1.03))
        ax.set_xlabel(lab)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(abl.index, color=INK)
    fig.suptitle("Ablation: remove one data stream at a time", x=0.01, ha="left", fontweight="bold", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_alert_timeline(P: pd.DataFrame, grids: dict, path):
    """A held-out day with at least one hyperglycaemic crossing: CGM on top, risk below."""
    _style()
    P = P.copy()
    P["day"] = P["ts"].dt.normalize()
    cand = P.groupby(["pid", "day"]).agg(spikes=("spike", "sum"), status=("status", "first"), n=("ts", "size"))
    cand = cand[(cand["n"] > 60)].sort_values("spikes", ascending=False)
    if cand.empty:
        return
    pid, day = cand.index[0]
    d = P[(P["pid"] == pid) & (P["day"] == day)]
    g = grids[pid].loc[day: day + pd.Timedelta(days=1), ["glucose", "carbs"]]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(g.index, g["glucose"], color=INK, lw=1.6, label="CGM")
    a1.axhline(config.HYPERGLYCEMIA_MGDL, color=MUTED, ls=":", lw=1)
    for t, c in g.loc[g["carbs"] > 0, "carbs"].items():
        a1.axvline(t, color="#1baf7a", lw=1.2, alpha=0.8)
        a1.text(t, a1.get_ylim()[1] if False else g["glucose"].max() + 8, f" {c:.0f} g", color=INK2, fontsize=8.5)
    a1.set_ylabel("Glucose (mg/dL)")
    a1.set_title(f"P{pid} ({d['status'].iloc[0]}), a held-out day: the alert fires before the crossing", pad=10)
    thr = d["thr_hybrid"].iloc[0]
    a2.plot(d["ts"], d["p_hybrid"], color="#2a78d6", lw=2)
    a2.axhline(thr, color="#d03b3b", lw=1, ls="--")
    a2.text(d["ts"].iloc[0], thr + 0.03, f"alert threshold {thr:.2f}", color=INK2, fontsize=8.5)
    a2.fill_between(d["ts"], 0, d["p_hybrid"], where=d["p_hybrid"] >= thr, color="#2a78d6", alpha=0.15, linewidth=0)
    a2.set_ylim(0, 1)
    a2.set_ylabel("Spike risk (2 h)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
