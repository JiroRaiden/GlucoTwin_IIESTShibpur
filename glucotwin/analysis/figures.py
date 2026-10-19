"""Static figures for the README / slides (matplotlib, consistent styling)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from glucotwin import config

# Validated categorical slots 1-3 (CVD-safe as a set), chart chrome tokens.
STATUS_COLOR = {"Healthy": "#2a78d6", "Prediabetes": "#eb6834", "T2D": "#1baf7a"}
STATUS_ORDER = ["Healthy", "Prediabetes", "T2D"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    })


def fig_same_meal_spread(spread: pd.DataFrame, path):
    """Each column = one person; dots = their spikes to 40-70 g carb meals."""
    _style()
    order = spread.groupby("pid")["rise"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for i, pid in enumerate(order):
        d = spread[spread["pid"] == pid]
        c = STATUS_COLOR.get(d["status"].iloc[0], MUTED)
        jitter = np.random.default_rng(pid).uniform(-0.18, 0.18, len(d))
        ax.scatter(np.full(len(d), i) + jitter, d["rise"], s=16, color=c, alpha=0.55, linewidths=0)
        ax.plot([i - 0.32, i + 0.32], [d["rise"].median()] * 2, color=c, lw=2.2, solid_capstyle="round")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{p}" for p in order], fontsize=7)
    ax.set_xlabel("Participant (sorted by median spike)")
    ax.set_ylabel("Glucose rise after meal (mg/dL)")
    ax.set_title("Same carbohydrate dose, very different spikes", pad=26)
    ax.text(0, 1.02, "Every meal with 40-70 g carbs; bar = person's median. "
            f"n = {len(spread)} meals, {spread['pid'].nunique()} people",
            transform=ax.transAxes, color=INK2, fontsize=9)
    handles = [plt.Line2D([], [], marker="o", ls="", color=STATUS_COLOR[s], label=s) for s in STATUS_ORDER]
    ax.legend(handles=handles, frameon=False, loc="upper left", ncols=3, labelcolor=INK2)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_variance_explained(ve: pd.DataFrame, path):
    _style()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(ve))[::-1]
    ax.barh(y, ve["cv_r2"].clip(lower=0), color="#2a78d6", height=0.55)
    for yi, (r2, mae) in zip(y, zip(ve["cv_r2"], ve["cv_mae"])):
        ax.text(max(r2, 0) + 0.01, yi, f"R² {r2:.2f}  ·  MAE {mae:.0f} mg/dL", va="center", color=INK, fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(ve["model"], color=INK)
    ax.set_xlim(0, max(0.05, ve["cv_r2"].max()) * 1.45)
    ax.set_xlabel("Out-of-sample R² for predicting a meal's glucose spike (5-fold CV)")
    ax.set_title("Knowing the meal is not enough: knowing the person is", pad=12)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_carb_sensitivity(cs: pd.DataFrame, path):
    _style()
    cs = cs.sort_values("slope_per_10g")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for s in STATUS_ORDER:
        d = cs[cs["status"] == s]
        ax.scatter(d["hba1c"], d["slope_per_10g"], s=46, color=STATUS_COLOR[s], label=s,
                   edgecolors=SURFACE, linewidths=1.5, zorder=3)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xlabel("HbA1c (%)")
    ax.set_ylabel("Spike per 10 g carbs (mg/dL)")
    ax.set_title("Each person has their own carbohydrate sensitivity", pad=12)
    ax.legend(frameon=False, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_example_curves(grids: dict, meals: pd.DataFrame, path, lo=40, hi=70):
    """Average post-meal curve for one representative person per status group."""
    _style()
    d = meals[meals["clean"] & meals["carbs"].between(lo, hi)]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    t = np.arange(-15, config.RESPONSE_MIN + 1, config.GRID_MIN)
    for s in STATUS_ORDER:
        ds = d[d["status"] == s]
        if ds.empty:
            continue
        pid = ds.groupby("pid")["rise"].count().idxmax()  # the person with most such meals
        curves = []
        for t0 in ds[ds["pid"] == pid]["ts"]:
            seg = grids[pid]["glucose"].reindex(t0.floor(config.GRID) + pd.to_timedelta(t, "min"))
            curves.append(seg.to_numpy() - np.nanmedian(seg.to_numpy()[:4]))
        c = np.nanmean(np.vstack(curves), axis=0)
        sd = np.nanstd(np.vstack(curves), axis=0)
        ax.fill_between(t, c - sd, c + sd, color=STATUS_COLOR[s], alpha=0.12, linewidth=0)
        ax.plot(t, c, color=STATUS_COLOR[s], lw=2, label=f"P{pid} · {s} (n={len(curves)})")
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_xlabel("Minutes after starting the meal")
    ax.set_ylabel("Glucose above baseline (mg/dL)")
    ax.set_title(f"Average response to a {lo}-{hi} g carb meal", pad=12)
    ax.legend(frameon=False, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
