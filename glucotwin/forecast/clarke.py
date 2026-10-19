"""Clarke Error Grid (Clarke et al., Diabetes Care 1987) - the clinical yardstick for glucose predictors.

Zone A: clinically accurate (within 20 %, or both hypoglycaemic)
Zone B: benign error - would not lead to inappropriate treatment
Zone C: overcorrection risk
Zone D: dangerous failure to detect
Zone E: erroneous treatment (hypo read as hyper or vice-versa)
"""
from __future__ import annotations

import numpy as np


def zone(ref: float, pred: float) -> str:
    if (ref <= 70 and pred <= 70) or (0.8 * ref <= pred <= 1.2 * ref):
        return "A"
    if (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
        return "E"
    if (70 <= ref <= 290 and pred >= ref + 110) or (130 <= ref <= 180 and pred <= (7 / 5) * ref - 182):
        return "C"
    if (ref >= 240 and 70 <= pred <= 180) or (ref <= 175 / 3 and 70 <= pred <= 180) or \
            (175 / 3 <= ref <= 70 and pred >= (6 / 5) * ref):
        return "D"
    return "B"


def zone_percentages(ref, pred) -> dict:
    ref, pred = np.asarray(ref, float), np.asarray(pred, float)
    m = ~(np.isnan(ref) | np.isnan(pred))
    z = np.array([zone(r, p) for r, p in zip(ref[m], pred[m])])
    return {k: round(100 * float(np.mean(z == k)), 2) for k in "ABCDE"}
