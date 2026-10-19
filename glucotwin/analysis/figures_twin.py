"""Phase 2 figures: does the twin learn the patient?"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from glucotwin.analysis.figures import (AXIS, INK, INK2, MUTED, STATUS_COLOR, STATUS_ORDER, SURFACE, _style)

TWIN_C = "#2a78d6"     # categorical slot 1
EHR_C = "#eb6834"      # slot 2
OBS_C = INK


def fig_learning_curve(lc: pd.DataFrame, base: pd.DataFrame, path):
    _style()
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = [d if d != 99 else 7.5 for d in lc.index]
    ax.plot(x, lc["rmse"], color=TWIN_C, lw=2, marker="o", ms=8, markeredgecolor=SURFACE, markeredgewidth=1.5,
            zorder=3, label="Personal twin")
    for xi, yi, d in zip(x, lc["rmse"], lc.index):
        lab = "EHR only" if d == 0 else ("all" if d == 99 else f"{d}d")
        ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color=INK)
    for name, ls in zip(base.index, ["--", ":"]):
        ax.axhline(base.loc[name, "rmse"], color=MUTED, lw=1.2, ls=ls)
        ax.text(7.6, base.loc[name, "rmse"], f"{name}: {base.loc[name, 'rmse']:.1f}  ", va="bottom",
                fontsize=8.5, color=INK2, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(["0\n(EHR only)" if d == 0 else ("all\ntraining" if d == 99 else str(int(d))) for d in lc.index])
    ax.set_xlabel("Days of wearable data used to calibrate the twin")
    ax.set_ylabel("2-h forecast RMSE on held-out days (mg/dL)")
    ax.set_title("The twin learns its patient", pad=12)
    lo = min(lc["rmse"].min(), base["rmse"].min())
    hi = max(lc["rmse"].max(), base["rmse"].max())
    ax.set_ylim(lo * 0.85, hi * 1.12)
    ax.legend(frameon=False, loc="upper center", labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_params_by_status(twins: dict, path):
    _style()
    df = pd.DataFrame(twins).T
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, (col, title) in zip(axes, [("SG", "Glucose self-clearance SG (1/min)"),
                                       ("f", "Effective carb load f (glucose per gram)")]):
        for i, s in enumerate(STATUS_ORDER):
            v = df.loc[df["status"] == s, col].astype(float)
            if v.empty:
                continue
            j = np.random.default_rng(i).uniform(-0.15, 0.15, len(v))
            ax.scatter(np.full(len(v), i) + j, v, color=STATUS_COLOR[s], s=36, edgecolors=SURFACE, linewidths=1.2, zorder=3)
            ax.plot([i - 0.25, i + 0.25], [v.median()] * 2, color=STATUS_COLOR[s], lw=2.4, solid_capstyle="round")
        ax.set_yscale("log")
        ax.set_xticks(range(3))
        ax.set_xticklabels(STATUS_ORDER, color=INK)
        ax.set_title(title, fontsize=10.5, pad=8)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Calibrated twin parameters separate the clinical groups", x=0.02, ha="left",
                 fontweight="bold", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_example_forecasts(examples: list, path, pre_steps: int):
    """One held-out meal per group, chosen as the one closest to that group's median error (not cherry-picked)."""
    _style()
    picks = []
    for s in STATUS_ORDER:
        ex = [e for e in examples if e["status"] == s and np.isfinite(e["rmse"])]
        if not ex:
            continue
        med = np.median([e["rmse"] for e in ex])
        picks.append(min(ex, key=lambda e: abs(e["rmse"] - med)))
    fig, axes = plt.subplots(1, len(picks), figsize=(4.2 * len(picks), 3.8), sharey=False, squeeze=False)
    for ax, e in zip(axes[0], picks):
        t = (np.arange(len(e["obs"])) - pre_steps) * 5
        ax.axvspan(t[0], 0, color=AXIS, alpha=0.18, linewidth=0)
        ax.plot(t, e["obs"], color=OBS_C, lw=1.6, label="CGM (truth)")
        ax.plot(t[pre_steps:], e["ehr_only"][pre_steps:], color=EHR_C, lw=1.8, ls="--", label="Twin from EHR only")
        ax.plot(t[pre_steps:], e["twin"][pre_steps:], color=TWIN_C, lw=2.2, label="Calibrated twin")
        ax.axhline(180, color=MUTED, lw=1, ls=":")
        ax.set_title(f"P{e['pid']} · {e['status']} · {e['carbs']:.0f} g carbs", fontsize=10.5, pad=8)
        ax.set_xlabel("Minutes from meal (shaded = synced to CGM)")
    axes[0][0].set_ylabel("Glucose (mg/dL)")
    axes[0][0].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    fig.suptitle("Held-out meals the twin never saw: forecast from the moment of eating", x=0.01, ha="left",
                 fontweight="bold", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
