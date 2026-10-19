"""Hybrid (physics + ML) forecasters and the clinical alert evaluation.

Regression: LightGBM learns the twin's ERROR at each horizon,
    forecast(t+h) = twin(t+h) + LGBM_h(features)
The pure-ML comparison learns the change from current glucose instead,
    forecast(t+h) = g(t) + LGBM_h(features without any twin information)
so the only difference between the two is whether the physics is in the loop.

Classification: LightGBM outputs P(glucose crosses 180 mg/dL in the next 2 h).
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import GroupShuffleSplit

from glucotwin import config
from glucotwin.forecast import features as FT

REG_PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=config.SEED, verbose=-1, n_jobs=2)
CLF_PARAMS = REG_PARAMS | dict(n_estimators=300)

FEATURE_SETS = {
    "hybrid": FT.ALL_FEATS,
    "pure_ml": [f for f in FT.ALL_FEATS if f not in FT.TWIN_PARAM_FEATS + FT.TWIN_FCST_FEATS],
}
ABLATIONS = {
    "Full hybrid": FT.ALL_FEATS,
    "- synthetic EHR fields": [f for f in FT.ALL_FEATS if f not in FT.SYNTH_EHR_FEATS],
    "- all EHR (labs + twin params + synthetic)": [f for f in FT.ALL_FEATS if f not in FT.LAB_FEATS + FT.TWIN_PARAM_FEATS + FT.SYNTH_EHR_FEATS],
    "- wearable (HR, activity)": [f for f in FT.ALL_FEATS if f not in FT.WEARABLE_FEATS],
    "- meal log": [f for f in FT.ALL_FEATS if f not in FT.MEAL_FEATS],
    "- twin (pure ML)": FEATURE_SETS["pure_ml"],
}


def base_col(kind: str, h: int) -> str:
    return f"twin_{h}" if kind == "hybrid" else "g_now"


def fit_reg(df, feats, h, base):
    d = df.dropna(subset=[f"y_{h}"])
    m = lgb.LGBMRegressor(**REG_PARAMS)
    m.fit(d[feats], d[f"y_{h}"] - d[base])
    return m


def predict_reg(m, df, feats, base):
    return m.predict(df[feats]) + df[base].to_numpy()


def fit_clf(df, feats):
    d = df.dropna(subset=["spike"])
    pos = d["spike"].sum()
    m = lgb.LGBMClassifier(**CLF_PARAMS, scale_pos_weight=max(1.0, (len(d) - pos) / max(pos, 1)))
    m.fit(d[feats], d["spike"].astype(int))
    return m


def choose_threshold(df_train, feats, seed=config.SEED) -> float:
    """Pick the alert threshold on an inner split of TRAINING patients (max F1)."""
    d = df_train.dropna(subset=["spike"])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    a, b = next(gss.split(d, groups=d["pid"]))
    m = fit_clf(d.iloc[a], feats)
    p = m.predict_proba(d.iloc[b][feats])[:, 1]
    y = d.iloc[b]["spike"].astype(int).to_numpy()
    grid = np.linspace(0.05, 0.95, 91)
    f1 = [f1_score(y, p >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1))])


def clf_metrics(y, p) -> dict:
    m = ~np.isnan(y)
    y, p = y[m].astype(int), p[m]
    if y.min() == y.max():
        return {"auroc": np.nan, "auprc": np.nan, "prevalence": float(y.mean())}
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "prevalence": float(y.mean())}


def alert_events(test: pd.DataFrame, grids: dict, score_col: str, thr: float) -> dict:
    """Event-level alert performance on held-out days.

    A hyperglycaemic EVENT = CGM crosses 180 mg/dL from below (after >= 60 min below).
    DETECTED if any origin in the 2 h before the crossing (while still < 180) alerted.
    LEAD TIME = crossing time - first such alert.
    FALSE ALARM = an alert episode (alerts merged within 60 min) with no crossing in the next 2 h.
    """
    detected, missed, leads, false_alarms, days = 0, 0, [], 0, 0.0
    thr_hit = test[score_col] >= thr
    for pid, d in test.groupby("pid"):
        g = grids[pid]["glucose"]
        g = g[(g.index >= d["ts"].min()) & (g.index <= d["ts"].max() + pd.Timedelta(hours=2))]
        days += (d["ts"].max() - d["ts"].min()).total_seconds() / 86400
        above = (g >= config.HYPERGLYCEMIA_MGDL).to_numpy()
        crossings = []
        last_above = -10_000
        for i in range(1, len(g)):
            if above[i] and not above[i - 1] and (i - last_above) >= 12:
                crossings.append(g.index[i])
            if above[i]:
                last_above = i
        alerts = d.loc[thr_hit.loc[d.index] & (d["g_now"] < config.HYPERGLYCEMIA_MGDL), "ts"].sort_values()
        for c in crossings:
            win = alerts[(alerts < c) & (alerts >= c - pd.Timedelta(hours=2))]
            if len(win):
                detected += 1
                # lead time = start of the continuous alert run that is still active nearest the crossing
                # (an older, unrelated alert in the window must not inflate the lead time)
                run = list(win)
                start = run[-1]
                for t in reversed(run[:-1]):
                    if (start - t) <= pd.Timedelta(minutes=15):
                        start = t
                    else:
                        break
                leads.append((c - start).total_seconds() / 60)
            else:
                missed += 1
        # alert episodes
        ep_start = None
        prev = None
        episodes = []
        for t in alerts:
            if prev is None or (t - prev) > pd.Timedelta(minutes=60):
                episodes.append(t)
            prev = t
        cr = pd.Series(crossings, dtype="datetime64[ns]")
        for t in episodes:
            if not ((cr > t) & (cr <= t + pd.Timedelta(hours=2))).any():
                false_alarms += 1
    n = detected + missed
    return {
        "events": n, "detected": detected,
        "sensitivity": detected / n if n else np.nan,
        "median_lead_min": float(np.median(leads)) if leads else np.nan,
        "false_alarms_per_day": false_alarms / days if days else np.nan,
        "test_patient_days": round(days, 1),
    }
