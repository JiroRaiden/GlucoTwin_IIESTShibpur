"""The physiological core of the digital twin.

A reduced Bergman minimal model with a two-compartment gut, driven by the
patient's meals (carbs) and wearable activity (METs).

States (all per patient, simulated minute by minute):
    Q1, Q2  grams of carbohydrate in the gut (stomach -> intestine)
    G       plasma glucose, mg/dL
    X       remote insulin action, 1/min (how hard insulin is clearing glucose)

Equations (dt = 1 min):
    dQ1/dt = -ka_m * Q1                     + meal carbs (instant bolus)
    dQ2/dt =  ka_m * Q1 - ka_m * Q2
    Ra     =  f * ka_m * Q2 * 1000 / (VG * BW)            [mg/dL/min appearing in blood]
    dG/dt  = -SG * E(t) * (G - Gb)  -  X * G  +  Ra
    dX/dt  = -P2 * X  +  P2 * S * max(G - Gb, 0)

    E(t)   = 1 + ALPHA * max(METs - 1, 0)   exercise boosts insulin-independent uptake
    ka_m   = ka / (1 + BETA * (fat + protein) / carbs)    fat & protein slow gastric emptying

Personal parameters (calibrated per patient):
    S   disposition: insulin response strength x insulin sensitivity  (low in T2D)
    SG  glucose effectiveness: clearance without insulin              (low in T2D)
    ka  carbohydrate absorption rate
    f   carbohydrate scale: bioavailability x meal-logging error

Population constants (shared by everyone):
    P2 = 0.025/min (insulin action time-constant ~40 min, Bergman 1979)
    VG = 1.6 dL/kg (glucose distribution volume, Dalla Man 2007)
    ALPHA, BETA chosen by pooled fit (see calibrate.py)

Why this reduction: CGMacros has no plasma insulin, so insulin secretion and
insulin sensitivity cannot be separated. We fit their product, S, which is
exactly the clinical "disposition index" idea - and it is the thing T2D breaks.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

P2 = 0.025
VG = 1.6
SUBSTEPS = 5          # 5 x 1-min Euler steps per 5-min grid step

PARAM_NAMES = ("S", "SG", "ka", "f")
# log-space bounds (physiologically plausible ranges)
LOWER = np.log(np.array([1e-6, 3e-4, 0.008, 0.1]))
UPPER = np.log(np.array([2e-3, 0.15, 0.12, 2.5]))
DEFAULT = np.log(np.array([4e-5, 0.012, 0.03, 1.0]))


@dataclass
class Globals:
    alpha: float = 0.25   # per MET above rest
    beta: float = 0.3     # fat+protein slowing of absorption


@dataclass
class TwinParams:
    S: float
    SG: float
    ka: float
    f: float
    Gb: float             # basal glucose, mg/dL (from overnight CGM, or lab fasting glucose)
    BW: float             # body weight, kg

    @property
    def log_theta(self) -> np.ndarray:
        return np.log([self.S, self.SG, self.ka, self.f])

    @classmethod
    def from_log(cls, log_theta, Gb, BW):
        S, SG, ka, f = np.exp(log_theta)
        return cls(S=S, SG=SG, ka=ka, f=f, Gb=Gb, BW=BW)

    def to_dict(self):
        return {k: float(v) for k, v in asdict(self).items()}


def simulate(log_theta, Gb, BW, G0, carbs, kmult, mets, obs=None, n_assim=0, gain=0.6,
             glob: Globals = Globals(), return_state=False):
    """Vectorised simulation of B independent windows on the 5-min grid.

    log_theta : (B, 4) or (4,)   personal parameters, log-space
    Gb, BW, G0: (B,) or scalars  basal glucose, body weight (kg), glucose at window start
    carbs     : (B, T) grams of available carbohydrate eaten in each 5-min step
    kmult     : (B, T) absorption-rate multiplier from the most recent meal's composition
    mets      : (B, T) activity (METs), 1.0 = rest
    obs       : (B, T) observed CGM, used only for the first n_assim steps
    n_assim   : steps of data assimilation; after that the twin runs free (= forecast)
    gain      : nudging gain in [0,1]; 1 = snap to the sensor, 0 = ignore it

    Returns G (B, T): the twin's glucose at each grid step (value at the START of the step,
    after assimilating that step's observation when t < n_assim).
    """
    carbs = np.atleast_2d(np.asarray(carbs, float))
    B, T = carbs.shape
    lt = np.broadcast_to(np.asarray(log_theta, float), (B, 4))
    S, SG, ka, f = (np.exp(lt[:, i]) for i in range(4))
    Gb = np.broadcast_to(np.asarray(Gb, float), (B,))
    BW = np.broadcast_to(np.asarray(BW, float), (B,))
    kmult = np.broadcast_to(np.asarray(kmult, float), (B, T))
    mets = np.broadcast_to(np.asarray(mets, float), (B, T))
    ra_scale = f * 1000.0 / (VG * BW)

    G = np.array(np.broadcast_to(np.asarray(G0, float), (B,)), copy=True)
    X = np.zeros(B)
    Q1 = np.zeros(B)
    Q2 = np.zeros(B)
    out = np.empty((B, T))

    for t in range(T):
        if obs is not None and t < n_assim:
            o = obs[:, t]
            m = ~np.isnan(o)
            G[m] = G[m] + gain * (o[m] - G[m])
        out[:, t] = G
        Q1 += carbs[:, t]
        kam = ka * kmult[:, t]
        E = 1.0 + glob.alpha * np.maximum(mets[:, t] - 1.0, 0.0)
        for _ in range(SUBSTEPS):
            Ra = ra_scale * kam * Q2
            dQ1 = -kam * Q1
            dQ2 = kam * Q1 - kam * Q2
            dG = -SG * E * (G - Gb) - X * G + Ra
            dX = -P2 * X + P2 * S * np.maximum(G - Gb, 0.0)
            Q1 += dQ1
            Q2 += dQ2
            G += dG
            X = np.maximum(X + dX, 0.0)
        G = np.clip(G, 40.0, 450.0)

    if return_state:
        return out, {"G": G, "X": X, "Q1": Q1, "Q2": Q2}
    return out


def composition_multiplier(carbs, fat, protein, beta):
    """Absorption slow-down from fat+protein relative to carbs (gastric emptying)."""
    carbs = np.maximum(np.nan_to_num(carbs), 5.0)
    return 1.0 / (1.0 + beta * (np.nan_to_num(fat) + np.nan_to_num(protein)) / carbs)
