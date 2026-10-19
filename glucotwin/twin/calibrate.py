"""Calibrating a twin to one patient: EHR prior first, wearable data second.

THE FUSION STORY
  1. EHR -> prior.   A ridge regression maps fasting labs (HbA1c, HOMA-IR, BMI,
     fasting glucose, age, triglycerides) to the twin's 4 personal parameters.
     A brand-new patient gets a twin on day 0 from their health record alone.
  2. Wearable -> posterior.  Each CGM meal curve is evidence. We minimise
         sum_k ((twin_k - cgm_k) / SIGMA)^2  +  sum_j ((theta_j - prior_j) / TAU_j)^2
     i.e. maximum-a-posteriori (MAP) estimation in log-parameter space.
     Few meals -> the EHR prior dominates; many meals -> the patient's own data does.

EVERY WINDOW starts 2 h before the meal: the twin is synced to the CGM over
those 2 h (data assimilation, so carry-over from earlier meals and current
insulin action are absorbed), then it must predict 3 h ahead with no further
CGM. Only the post-meal part is scored.

TRAIN / TEST: each patient's last 3 days are never used for calibration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from glucotwin import config
from glucotwin.twin import model as M

PRE_STEPS = 24                     # 2 h of sync before the meal
POST_STEPS = 36                    # 3 h forecast after the meal
T_WIN = PRE_STEPS + POST_STEPS + 1
# Likelihood width per CGM point. Twin residuals are ~30 mg/dL and strongly autocorrelated
# (about 30 min, i.e. ~6 samples), so a window carries far fewer independent points than samples.
# Effective sigma = 30 * sqrt(6) ~ 70 mg/dL keeps a handful of meals from overriding the EHR prior.
# Fixed a priori from the residual scale, not tuned on held-out days.
SIGMA = 70.0
TAU = np.array([0.9, 0.6, 0.5, 0.4])   # prior width per log-parameter
HOLDOUT_DAYS = 3
EHR_FEATURES = ["hba1c", "homa_ir", "bmi", "fasting_glucose", "age", "triglycerides"]


# ----------------------------------------------------------------- per-person inputs

def basal_glucose(grid: pd.DataFrame, fallback: float) -> float:
    """Overnight (03:00-06:00) median CGM = the person's basal glucose."""
    night = grid.loc[(grid.index.hour >= 3) & (grid.index.hour < 6), "glucose"].dropna()
    return float(night.median()) if len(night) > 20 else float(fallback)


def body_weight_kg(bio_row) -> float:
    w = bio_row.get("weight_lb", np.nan)
    return float(w * 0.4536) if pd.notna(w) else 80.0


def split_cutoff(grid: pd.DataFrame) -> pd.Timestamp:
    """Start of the held-out period (last HOLDOUT_DAYS days of the recording)."""
    return (grid["glucose"].last_valid_index() - pd.Timedelta(days=HOLDOUT_DAYS)).normalize()


def prepare_inputs(grid: pd.DataFrame, beta: float) -> pd.DataFrame:
    """Model drivers on the 5-min grid: available carbs, absorption multiplier, METs."""
    d = pd.DataFrame(index=grid.index)
    carbs = grid.get("carbs", 0).fillna(0)
    fiber = grid.get("fiber", 0).fillna(0)
    d["carbs_avail"] = (carbs - fiber).clip(lower=0)
    km = pd.Series(np.nan, index=grid.index)
    has_meal = carbs > 0
    km[has_meal] = M.composition_multiplier(carbs[has_meal], grid.get("fat", 0)[has_meal],
                                            grid.get("protein", 0)[has_meal], beta)
    d["kmult"] = km.ffill().fillna(1.0)
    d["mets"] = grid["mets"].ffill(limit=6).fillna(1.0) if "mets" in grid else 1.0
    d["glucose"] = grid["glucose"]
    return d


@dataclass
class Windows:
    pid: int
    t0: np.ndarray          # meal start times
    day: np.ndarray         # day index since start of recording (for learning curves)
    test: np.ndarray        # bool: in held-out period
    carbs: np.ndarray       # (B, T)
    kmult: np.ndarray
    mets: np.ndarray
    obs: np.ndarray
    G0: np.ndarray

    def subset(self, mask):
        return Windows(self.pid, *(getattr(self, k)[mask] for k in
                                   ["t0", "day", "test", "carbs", "kmult", "mets", "obs", "G0"]))

    def __len__(self):
        return len(self.t0)


def build_windows(pid, grid, meal_table, beta) -> Windows:
    inp = prepare_inputs(grid, beta)
    cutoff = split_cutoff(grid)
    start = grid.index[0].normalize()
    m = meal_table[(meal_table["pid"] == pid) & meal_table["clean"]]
    rows = {k: [] for k in ["t0", "day", "test", "carbs", "kmult", "mets", "obs", "G0"]}
    for t0 in m["ts"]:
        t0 = t0.floor(config.GRID)
        idx = pd.date_range(t0 - pd.Timedelta(minutes=5 * PRE_STEPS), periods=T_WIN, freq=config.GRID)
        w = inp.reindex(idx)
        obs = w["glucose"].to_numpy()
        if np.isnan(obs[PRE_STEPS + 1:]).mean() > 0.2 or np.isnan(obs[:PRE_STEPS + 1]).all():
            continue
        first = obs[~np.isnan(obs)][0]
        rows["t0"].append(t0)
        rows["day"].append((t0.normalize() - start).days)
        rows["test"].append(t0 >= cutoff)
        rows["carbs"].append(w["carbs_avail"].fillna(0).to_numpy())
        rows["kmult"].append(w["kmult"].fillna(1).to_numpy())
        rows["mets"].append(w["mets"].fillna(1).to_numpy())
        rows["obs"].append(obs)
        rows["G0"].append(first)
    return Windows(pid, *(np.array(rows[k]) for k in rows))


# ----------------------------------------------------------------- fitting

def _residuals(lt, W: Windows, Gb, BW, prior_mu, glob, prior_weight=1.0):
    sim = M.simulate(lt, Gb, BW, W.G0, W.carbs, W.kmult, W.mets, obs=W.obs,
                     n_assim=PRE_STEPS + 1, glob=glob)
    post = slice(PRE_STEPS + 1, None)
    r = (sim[:, post] - W.obs[:, post]) / SIGMA
    r = r[~np.isnan(r)]
    prior = prior_weight * (lt - prior_mu) / TAU
    return np.concatenate([r, prior])


def fit_twin(W: Windows, Gb, BW, prior_mu, glob=M.Globals(), prior_weight=1.0):
    """MAP fit of the 4 personal parameters. With zero windows, returns the prior."""
    prior_mu = np.clip(prior_mu, M.LOWER + 1e-6, M.UPPER - 1e-6)
    if len(W) == 0:
        return prior_mu.copy(), np.nan
    res = least_squares(_residuals, prior_mu, args=(W, Gb, BW, prior_mu, glob, prior_weight),
                        bounds=(M.LOWER, M.UPPER), method="trf", x_scale=1.0, max_nfev=200)
    n = len(res.fun) - 4
    return res.x, float(np.sqrt(np.mean((res.fun[:n] * SIGMA) ** 2)))


def predict_windows(lt, W: Windows, Gb, BW, glob=M.Globals()):
    return M.simulate(lt, Gb, BW, W.G0, W.carbs, W.kmult, W.mets, obs=W.obs,
                      n_assim=PRE_STEPS + 1, glob=glob)


def score(sim, W: Windows, horizon_steps=24):
    """RMSE over the first `horizon_steps` after the meal (default 2 h) + peak error."""
    post = slice(PRE_STEPS + 1, PRE_STEPS + 1 + horizon_steps)
    err = sim[:, post] - W.obs[:, post]
    rmse = float(np.sqrt(np.nanmean(err ** 2)))
    peak_err = float(np.nanmean(np.abs(np.nanmax(sim[:, post], 1) - np.nanmax(W.obs[:, post], 1))))
    return rmse, peak_err


# ----------------------------------------------------------------- EHR -> parameter prior

class EHRPrior:
    """Ridge regression from fasting labs to log twin parameters."""

    def __init__(self):
        self.model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                   RidgeCV(alphas=np.logspace(-1, 3, 20)))
        self.fallback = M.DEFAULT.copy()

    def fit(self, bio: pd.DataFrame, thetas: dict):
        pids = [p for p in thetas if p in bio.index]
        X = bio.loc[pids, EHR_FEATURES].astype(float)
        Y = np.vstack([thetas[p] for p in pids])
        self.model.fit(X, Y)
        self.fallback = np.median(Y, axis=0)
        return self

    def predict(self, bio_row) -> np.ndarray:
        x = pd.DataFrame([bio_row[EHR_FEATURES].astype(float)])
        return np.clip(self.model.predict(x)[0], M.LOWER, M.UPPER)
