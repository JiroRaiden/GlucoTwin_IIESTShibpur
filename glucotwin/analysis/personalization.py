"""The evidence that a *personal* twin is needed.

Question: if you know exactly what someone ate, how well can you predict their
glucose spike? And how much better if you also know WHO they are?

We fit four nested linear models on clean meals and score them with 5-fold
cross-validated R^2 (out-of-sample, so adding person dummies can't just
memorise - each fold predicts meals it never saw):

  M0  carbs only                                   "the carb-counting app"
  M1  + protein, fat, fiber, baseline, meal type   "a better nutrition app"
  M2  + fasting labs (HbA1c, HOMA-IR, BMI)          "population model with EHR"
  M3  + participant identity                       "a personal twin"

The gap between M2 and M3 is variance that is *personal* - explained by the
individual's own physiology beyond what their labs say. That gap is the reason
the twin is calibrated per patient in Phase 2.

We also estimate each person's carb sensitivity (mg/dL of spike per 10 g carbs)
with a per-participant regression, which shows the spread directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from glucotwin import config

MODELS = {
    "M0  Carbs only": (["carbs"], []),
    "M1  + full meal & context": (["carbs", "protein", "fat", "fiber", "baseline"], ["meal_type"]),
    "M2  + fasting labs (EHR)": (["carbs", "protein", "fat", "fiber", "baseline", "hba1c", "homa_ir", "bmi"], ["meal_type"]),
    "M3  + who the person is": (["carbs", "protein", "fat", "fiber", "baseline", "hba1c", "homa_ir", "bmi"], ["meal_type", "pid"]),
}


def _pipe(num, cat):
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        *([("cat", OneHotEncoder(handle_unknown="ignore"), cat)] if cat else []),
    ])
    return make_pipeline(pre, RidgeCV(alphas=np.logspace(-2, 3, 20)))


def variance_explained(meals: pd.DataFrame, target: str = "rise", n_splits: int = 5) -> pd.DataFrame:
    d = meals[meals["clean"]].copy()
    d["pid"] = d["pid"].astype(str)
    y = d[target].to_numpy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=config.SEED)
    out = []
    for name, (num, cat) in MODELS.items():
        pred = cross_val_predict(_pipe(num, cat), d[num + cat], y, cv=kf)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        out.append({"model": name, "cv_r2": 1 - ss_res / ss_tot,
                    "cv_mae": np.mean(np.abs(y - pred)), "n_meals": len(d)})
    return pd.DataFrame(out)


def carb_sensitivity(meals: pd.DataFrame, min_meals: int = 6) -> pd.DataFrame:
    """Per-person slope: spike (mg/dL) per 10 g carbohydrate, adjusting for baseline."""
    rows = []
    for pid, d in meals[meals["clean"]].groupby("pid"):
        d = d.dropna(subset=["carbs", "rise", "baseline"])
        if len(d) < min_meals or d["carbs"].std() < 5:
            continue
        X = np.column_stack([d["carbs"] / 10, d["baseline"] - d["baseline"].mean()])
        lr = LinearRegression().fit(X, d["rise"])
        rows.append({"pid": pid, "slope_per_10g": lr.coef_[0], "n_meals": len(d),
                     "status": d["status"].iloc[0], "hba1c": d["hba1c"].iloc[0]})
    return pd.DataFrame(rows).set_index("pid")


def similar_meal_spread(meals: pd.DataFrame, lo: float = 40, hi: float = 70) -> pd.DataFrame:
    """Meals in a fixed carb band - same 'dose', how different is the response?"""
    d = meals[meals["clean"] & meals["carbs"].between(lo, hi)]
    return d[["pid", "status", "carbs", "rise", "peak", "hyper"]]
