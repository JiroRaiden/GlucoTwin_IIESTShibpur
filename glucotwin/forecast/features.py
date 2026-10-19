"""Build forecasting examples: one row per (patient, time) 'forecast origin'.

At each origin t (every 15 min) we pretend we are live at t and only use
information available at t:
  * CGM, heart rate, activity up to t
  * meals logged up to and including t (a meal logged at t counts: in the app,
    the patient logs lunch as they start eating)
  * EHR labs and the twin's calibrated parameters (fixed per patient)
  * the twin's own forecast: synced to CGM for the past 3 h, then run forward
    with no future meals and resting activity (the twin knows only the present)

Targets: glucose at t+30, t+60, t+120 min, and SPIKE = glucose crosses
180 mg/dL at any point in (t, t+120] while it is still below 180 at t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from glucotwin import config
from glucotwin.twin import calibrate as C
from glucotwin.twin import model as M

PRE = 36            # 3 h of twin synchronisation before the origin
H = 24              # 2 h forecast horizon (in 5-min steps)
STRIDE = 3          # an origin every 15 min
HORIZONS = {30: 6, 60: 12, 120: 24}

# feature groups (used for ablations and for human-readable explanations)
CGM_FEATS = ["g_now", "d5", "d15", "d30", "g_mean_2h", "g_std_2h", "g_min_2h", "g_max_2h"]
WEARABLE_FEATS = ["hr_30", "hr_rel", "mets_30", "mets_60"]
MEAL_FEATS = ["min_since_meal", "carbs_last_meal", "fat_last_meal", "protein_last_meal",
              "fiber_last_meal", "carbs_3h", "is_meal_now"]
TIME_FEATS = ["hour_sin", "hour_cos"]
LAB_FEATS = ["hba1c", "homa_ir", "bmi", "fasting_glucose", "age"]
TWIN_PARAM_FEATS = ["tw_logS", "tw_logSG", "tw_logka", "tw_logf", "tw_Gb"]
TWIN_FCST_FEATS = ["twin_30", "twin_60", "twin_120", "twin_max120", "twin_rise"]
SYNTH_EHR_FEATS = ["on_metformin", "tcf7l2_risk_alleles", "family_history_t2d",
                   "prs_t2d_percentile", "hba1c_trend_per_yr"]

FEATURE_LABELS = {
    "g_now": "current glucose", "d5": "5-min glucose change", "d15": "15-min glucose trend",
    "d30": "30-min glucose trend", "g_mean_2h": "2-h mean glucose", "g_std_2h": "2-h glucose variability",
    "g_min_2h": "2-h minimum glucose", "g_max_2h": "2-h maximum glucose",
    "hr_30": "heart rate (30 min)", "hr_rel": "heart rate vs personal baseline",
    "mets_30": "activity (last 30 min)", "mets_60": "activity (last 60 min)",
    "min_since_meal": "time since last meal", "carbs_last_meal": "carbs in last meal",
    "fat_last_meal": "fat in last meal", "protein_last_meal": "protein in last meal",
    "fiber_last_meal": "fibre in last meal", "carbs_3h": "carbs eaten in last 3 h", "is_meal_now": "meal starting now",
    "hour_sin": "time of day", "hour_cos": "time of day",
    "hba1c": "HbA1c", "homa_ir": "insulin resistance (HOMA-IR)", "bmi": "BMI",
    "fasting_glucose": "fasting glucose", "age": "age",
    "tw_logS": "twin: insulin-driven clearance", "tw_logSG": "twin: glucose clearance",
    "tw_logka": "twin: carb absorption rate", "tw_logf": "twin: effective carb load", "tw_Gb": "twin: basal glucose",
    "twin_30": "twin forecast +30 min", "twin_60": "twin forecast +60 min", "twin_120": "twin forecast +120 min",
    "twin_max120": "twin-simulated peak (2 h)", "twin_rise": "twin-simulated rise",
    "on_metformin": "on metformin (synthetic)", "tcf7l2_risk_alleles": "TCF7L2 risk alleles (synthetic)",
    "family_history_t2d": "family history of T2D (synthetic)", "prs_t2d_percentile": "polygenic risk (synthetic)",
    "hba1c_trend_per_yr": "HbA1c trend (synthetic)",
}

ALL_FEATS = (CGM_FEATS + WEARABLE_FEATS + MEAL_FEATS + TIME_FEATS + LAB_FEATS
             + TWIN_PARAM_FEATS + TWIN_FCST_FEATS + SYNTH_EHR_FEATS)


def _rolling_last(x: np.ndarray, i: int, n: int, fn):
    seg = x[max(0, i - n + 1): i + 1]
    seg = seg[~np.isnan(seg)]
    return fn(seg) if len(seg) else np.nan


def build_origins(pid, grid: pd.DataFrame, twin: dict, glob: M.Globals, bio_row, ehr_row,
                  stride: int = STRIDE) -> pd.DataFrame:
    inp = C.prepare_inputs(grid, glob.beta)
    g = inp["glucose"].to_numpy()
    carbs = inp["carbs_avail"].to_numpy()
    kmult = inp["kmult"].to_numpy()
    mets = inp["mets"].to_numpy()
    hr = grid["hr"].to_numpy() if "hr" in grid else np.full(len(g), np.nan)
    raw_carbs = grid.get("carbs", pd.Series(0, index=grid.index)).fillna(0).to_numpy()
    fat = grid.get("fat", pd.Series(0, index=grid.index)).fillna(0).to_numpy()
    prot = grid.get("protein", pd.Series(0, index=grid.index)).fillna(0).to_numpy()
    fib = grid.get("fiber", pd.Series(0, index=grid.index)).fillna(0).to_numpy()
    ts = grid.index
    N = len(g)
    cutoff = C.split_cutoff(grid)
    hr_base = np.nanmedian(hr) if np.isfinite(hr).any() else np.nan

    # every 15 min, plus every meal start (so "predict at the moment of eating" is scored on all meals)
    meal_idx = set(np.flatnonzero(raw_carbs > 0).tolist())
    cand = sorted(set(range(PRE, N - H, stride)) | {i for i in meal_idx if PRE <= i < N - H})
    idx = np.array([i for i in cand if not np.isnan(g[i])])
    if len(idx) == 0:
        return pd.DataFrame()

    # ---- twin forecasts, vectorised over all origins ------------------------
    lt = np.log([twin["S"], twin["SG"], twin["ka"], twin["f"]])
    T = PRE + H + 1
    win = idx[:, None] + np.arange(-PRE, H + 1)[None, :]
    W_carbs = carbs[win].copy()
    W_carbs[:, PRE + 1:] = 0.0                      # future meals unknown
    W_mets = mets[win].copy()
    W_mets[:, PRE + 1:] = 1.0                       # future activity unknown -> rest
    W_obs = g[win].copy()
    W_obs[:, PRE + 1:] = np.nan                     # future CGM unknown
    G0 = np.array([row[~np.isnan(row)][0] for row in W_obs])
    sim = M.simulate(lt, twin["Gb"], twin["BW"], G0, W_carbs, kmult[win], W_mets,
                     obs=W_obs, n_assim=PRE + 1, glob=glob)

    # ---- last-meal bookkeeping ------------------------------------------------
    meal_bins = np.flatnonzero(raw_carbs > 0)
    rows = []
    for k, i in enumerate(idx):
        prev = meal_bins[meal_bins <= i]
        j = prev[-1] if len(prev) else None
        fut = g[i + 1: i + H + 1]
        covered = np.mean(~np.isnan(fut)) >= 0.7
        cross = np.flatnonzero(fut >= config.HYPERGLYCEMIA_MGDL)
        r = {
            "pid": pid, "ts": ts[i], "test_period": ts[i] >= cutoff,
            # targets
            **{f"y_{h}": g[i + s] for h, s in HORIZONS.items()},
            "spike": float(covered and g[i] < config.HYPERGLYCEMIA_MGDL and len(cross) > 0) if covered else np.nan,
            "min_to_cross": float((cross[0] + 1) * 5) if len(cross) else np.nan,
            # CGM history
            "g_now": g[i], "d5": g[i] - g[i - 1], "d15": g[i] - g[i - 3], "d30": g[i] - g[i - 6],
            "g_mean_2h": _rolling_last(g, i, 24, np.mean), "g_std_2h": _rolling_last(g, i, 24, np.std),
            "g_min_2h": _rolling_last(g, i, 24, np.min), "g_max_2h": _rolling_last(g, i, 24, np.max),
            # wearable
            "hr_30": _rolling_last(hr, i, 6, np.mean),
            "mets_30": _rolling_last(mets, i, 6, np.mean), "mets_60": _rolling_last(mets, i, 12, np.mean),
            # meals
            "min_since_meal": min((i - j) * 5, 600) if j is not None else 600,
            "carbs_last_meal": raw_carbs[j] if j is not None else 0, "fat_last_meal": fat[j] if j is not None else 0,
            "protein_last_meal": prot[j] if j is not None else 0, "fiber_last_meal": fib[j] if j is not None else 0,
            "carbs_3h": carbs[max(0, i - 35): i + 1].sum(), "is_meal_now": float(raw_carbs[i] > 0),
            # time
            "hour_sin": np.sin(2 * np.pi * (ts[i].hour + ts[i].minute / 60) / 24),
            "hour_cos": np.cos(2 * np.pi * (ts[i].hour + ts[i].minute / 60) / 24),
            # twin forecast
            "twin_30": sim[k, PRE + 6], "twin_60": sim[k, PRE + 12], "twin_120": sim[k, PRE + 24],
            "twin_max120": sim[k, PRE + 1:].max(), "twin_rise": sim[k, PRE + 1:].max() - g[i],
        }
        r["hr_rel"] = r["hr_30"] - hr_base
        rows.append(r)
    df = pd.DataFrame(rows)

    # ---- static: labs, twin parameters, synthetic EHR ------------------------------
    for c in LAB_FEATS:
        df[c] = bio_row.get(c, np.nan)
    df["tw_logS"], df["tw_logSG"], df["tw_logka"], df["tw_logf"] = lt
    df["tw_Gb"] = twin["Gb"]
    for c in SYNTH_EHR_FEATS:
        df[c] = ehr_row.get(c, np.nan) if ehr_row is not None else np.nan
    df["status"] = bio_row.get("status", "unknown")
    return df
