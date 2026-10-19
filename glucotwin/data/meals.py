"""Turn every logged meal into a measured glucose response.

For each meal at time t0 we read the CGM curve and compute:

  baseline   median glucose in [t0-15, t0]          (where the person started)
  peak       max glucose in [t0, t0+150]            (how high they went)
  rise       peak - baseline                        (the spike itself, mg/dL)
  t_peak     minutes from t0 to the peak
  iauc_120   incremental area under the curve above baseline, 0-120 min
             (the standard glycaemic-response measure in nutrition science)
  hyper      1 if glucose crossed 180 mg/dL in the 2 h after the meal
             (ADA post-prandial target -> our clinical "adverse event")
  post_mets  mean activity in the 60 min after eating (did they walk?)

A response is only trusted ("clean") when the CGM covers >= 80 % of the window
and no other meal starts within 2 h - otherwise two meals' spikes blur together
and the measured response would be attributed to the wrong food.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from glucotwin import config


def _window(g: pd.Series, t0, start_min: int, end_min: int) -> pd.Series:
    return g.loc[t0 + pd.Timedelta(minutes=start_min): t0 + pd.Timedelta(minutes=end_min)]


def meal_response(grid: pd.DataFrame, t0: pd.Timestamp) -> dict:
    g = grid["glucose"]
    t0 = t0.floor(config.GRID)

    pre = _window(g, t0, -config.BASELINE_MIN, 0)
    post = _window(g, t0, 0, config.RESPONSE_MIN)
    expected = config.RESPONSE_MIN // config.GRID_MIN + 1
    coverage = post.notna().sum() / expected

    out = {"coverage": coverage}
    if pre.notna().sum() == 0 or post.notna().sum() < 3:
        return out | {"baseline": np.nan, "peak": np.nan, "rise": np.nan,
                      "t_peak": np.nan, "iauc_120": np.nan, "hyper": np.nan}

    baseline = float(pre.median())
    peak_win = _window(g, t0, 0, 150)
    peak = float(peak_win.max())
    t_peak = (peak_win.idxmax() - t0).total_seconds() / 60

    w120 = _window(g, t0, 0, config.IAUC_MIN).interpolate(limit_area="inside")
    above = np.clip(w120.to_numpy() - baseline, 0, None)
    above = above[~np.isnan(above)]
    iauc = float(np.trapezoid(above, dx=config.GRID_MIN)) if len(above) > 1 else np.nan

    out |= {
        "baseline": baseline,
        "peak": peak,
        "rise": peak - baseline,
        "t_peak": t_peak,
        "iauc_120": iauc,
        "hyper": float(w120.max() >= config.HYPERGLYCEMIA_MGDL),
    }

    # wearable context: activity and heart rate around the meal
    for col, name, a, b in [("mets", "pre_mets", -60, 0), ("mets", "post_mets", 0, 60),
                            ("hr", "pre_hr", -30, 0), ("hr", "post_hr", 0, 60)]:
        if col in grid:
            out[name] = float(_window(grid[col], t0, a, b).mean())
    return out


def build_meal_table(bio: pd.DataFrame, grids: dict, meals: dict) -> pd.DataFrame:
    """One row per meal across all participants, with responses + who ate it."""
    rows = []
    for pid, m in meals.items():
        grid = grids[pid]
        starts = m["ts"].sort_values().to_numpy()
        for _, meal in m.iterrows():
            t0 = meal["ts"]
            r = meal_response(grid, t0)
            # minutes to next / since previous meal (overlap check)
            later = starts[starts > np.datetime64(t0)]
            earlier = starts[starts < np.datetime64(t0)]
            r["min_to_next_meal"] = (later[0] - np.datetime64(t0)) / np.timedelta64(1, "m") if len(later) else np.inf
            r["min_since_prev_meal"] = (np.datetime64(t0) - earlier[-1]) / np.timedelta64(1, "m") if len(earlier) else np.inf
            rows.append({"pid": pid, **meal.to_dict(), **r})

    df = pd.DataFrame(rows)
    df["hour"] = df["ts"].dt.hour + df["ts"].dt.minute / 60
    df["clean"] = (
        (df["coverage"] >= config.MIN_COVERAGE)
        & (df["min_to_next_meal"] >= 120)
        & df["carbs"].notna()
        & df["rise"].notna()
    )
    df = df.merge(bio[["status", "hba1c", "bmi", "homa_ir"]], left_on="pid", right_index=True, how="left")
    return df
