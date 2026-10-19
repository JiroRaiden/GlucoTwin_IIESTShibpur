"""Load CGMacros and put every stream on one shared 5-minute clock.

CGMacros ships one CSV per participant with a row per *minute*:
    Timestamp, Libre GL, Dexcom GL, HR, Calories (Activity), METs,
    Meal Type, Calories, Carbs, Protein, Fat, Fiber, Amount Consumed, Image path
plus a single bio.csv with demographics + fasting labs per participant.

Design decisions (defend these in Q&A):
* Dexcom is the primary glucose signal: it samples every 5 min (Libre: 15 min),
  so it has 3x the temporal resolution needed to see a spike forming.
* We resample to a 5-min grid. A 1-min grid would be ~80 % empty glucose cells;
  5 min is the sensor's real resolution, so nothing is invented.
* Glucose gaps up to 15 min are linearly interpolated (that is what a CGM
  receiver does). Longer gaps stay NaN and are excluded downstream - we never
  fabricate physiology.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from glucotwin import config

# ------------------------------------------------------------------ discovery

_PID_RE = re.compile(r"CGMacros-(\d+)\.csv$", re.IGNORECASE)


def find_participant_files(root: Path = config.RAW_DIR) -> dict[int, Path]:
    """Return {participant_id: csv_path} for every CGMacros-0XX.csv under root."""
    files = {}
    for p in Path(root).rglob("CGMacros-*.csv"):
        m = _PID_RE.search(p.name)
        if m:
            files[int(m.group(1))] = p
    if not files:
        raise FileNotFoundError(
            f"No CGMacros-XXX.csv files under {root}. "
            "Download + extract the dataset first (see scripts/get_data.md)."
        )
    return dict(sorted(files.items()))


def _find_one(root: Path, name: str) -> Path:
    hits = [p for p in Path(root).rglob(name) if not p.name.startswith("DataDictionary")]
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


# ------------------------------------------------------------------ bio / labs

_BIO_RENAME = {
    "age": "age",
    "gender": "sex",
    "bmi": "bmi",
    "body weight": "weight_lb",
    "height": "height_in",
    "self-identify": "self_identify",
    "a1c pdl (lab)": "hba1c",
    "fasting glu - pdl (lab)": "fasting_glucose",
    "insulin": "fasting_insulin",
    "triglycerides": "triglycerides",
    "cholesterol": "cholesterol",
    "hdl": "hdl",
    "non hdl": "non_hdl",
    "ldl (cal)": "ldl",
    "vldl (cal)": "vldl",
    "cho/hdl ratio": "chol_hdl_ratio",
}


def load_bio(root: Path = config.RAW_DIR) -> pd.DataFrame:
    """Demographics + fasting labs, one row per participant, clean column names.

    Data-quality fixes (all documented in the README):
    * The data dictionary says HbA1c is in mmol/mol, but values span 4.6-8.5,
      which is unambiguously the NGSP percentage scale. We treat it as %.
    * Dictionary flags LDL == 800 and VLDL == 400 as calculation errors -> NaN.
    * Chol/HDL ratio range tops out at 400, another sentinel -> NaN if > 20.
    """
    raw = pd.read_csv(_find_one(root, "bio.csv"))
    cols = {c: c.strip() for c in raw.columns}
    raw = raw.rename(columns=cols)

    # participant id column: the release names it 'subject' but be tolerant
    id_col = next((c for c in raw.columns if c.lower() in {"subject", "participant", "id", "subject id"}), None)
    out = pd.DataFrame()
    if id_col is not None:
        out["pid"] = raw[id_col].astype(str).str.extract(r"(\d+)")[0].astype(int)
    else:
        warnings.warn("bio.csv has no subject column; assuming row order == participant 1..N")
        out["pid"] = np.arange(1, len(raw) + 1)

    lower = {c.lower(): c for c in raw.columns}
    for key, new in _BIO_RENAME.items():
        if key in lower:
            out[new] = raw[lower[key]]

    num_cols = [c for c in out.columns if c not in {"pid", "sex", "self_identify"}]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    if "ldl" in out:
        out.loc[out["ldl"] >= 800, "ldl"] = np.nan
    if "vldl" in out:
        out.loc[out["vldl"] >= 400, "vldl"] = np.nan
    if "chol_hdl_ratio" in out:
        out.loc[out["chol_hdl_ratio"] > 20, "chol_hdl_ratio"] = np.nan

    # HOMA-IR: standard insulin-resistance index = glucose(mg/dL) * insulin(uU/mL) / 405.
    # It becomes a prior for the twin's insulin-sensitivity parameter in Phase 2.
    if {"fasting_glucose", "fasting_insulin"} <= set(out.columns):
        out["homa_ir"] = out["fasting_glucose"] * out["fasting_insulin"] / 405.0

    out["status"] = out["hba1c"].map(glycemic_status)
    return out.drop_duplicates("pid").set_index("pid").sort_index()


def glycemic_status(a1c: float) -> str:
    """ADA HbA1c categories."""
    if pd.isna(a1c):
        return "unknown"
    if a1c >= config.A1C_DIABETES:
        return "T2D"
    if a1c >= config.A1C_PREDIABETES:
        return "Prediabetes"
    return "Healthy"


# ------------------------------------------------------------------ participant time series

_TS_RENAME = {
    "timestamp": "ts",
    "libre gl": "glucose_libre",
    "dexcom gl": "glucose",
    "hr": "hr",
    "calories (activity)": "active_kcal",
    "mets": "mets",
    "meal type": "meal_type",
    "calories": "meal_kcal",
    "carbs": "carbs",
    "protein": "protein",
    "fat": "fat",
    "fiber": "fiber",
    "amount consumed": "amount_consumed",
    "image path": "image_path",
}


def load_participant_raw(path: Path) -> pd.DataFrame:
    """Read one participant CSV as-is (1-min rows), with clean column names."""
    df = pd.read_csv(path, low_memory=False)
    lower = {c.strip().lower(): c for c in df.columns}
    df = df[[lower[k] for k in _TS_RENAME if k in lower]].rename(
        columns={lower[k]: v for k, v in _TS_RENAME.items() if k in lower}
    )
    df["ts"] = pd.to_datetime(df["ts"], format="mixed", errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts")
    for c in ["glucose_libre", "glucose", "hr", "active_kcal", "mets",
              "meal_kcal", "carbs", "protein", "fat", "fiber", "amount_consumed"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "mets" in df:
        df["mets"] = df["mets"] / 10.0  # dictionary: Fitbit METs are stored x10
    return df.reset_index(drop=True)


def to_grid(raw: pd.DataFrame, max_gap_min: int = 15) -> pd.DataFrame:
    """Resample the sensor streams onto the shared 5-min clock.

    glucose      : mean of Dexcom readings in the bin, short gaps interpolated
    glucose_libre: kept for the sensor-agreement check, interpolated the same way
    hr, mets     : mean over the bin (wearable state)
    active_kcal  : sum over the bin (energy expended)
    carbs etc.   : sum over the bin (a meal that starts in this bin)
    """
    s = raw.set_index("ts")
    agg = {
        "glucose": "mean", "glucose_libre": "mean", "hr": "mean", "mets": "mean",
        "active_kcal": "sum", "carbs": "sum", "protein": "sum", "fat": "sum",
        "fiber": "sum", "meal_kcal": "sum",
    }
    agg = {k: v for k, v in agg.items() if k in s.columns}
    g = s.resample(config.GRID).agg(agg)

    limit = max_gap_min // config.GRID_MIN
    for c in ["glucose", "glucose_libre"]:
        if c in g:
            g[c] = g[c].interpolate(limit=limit, limit_area="inside")
    g["meal_start"] = 0
    meals = extract_meal_events(raw)
    if len(meals):
        bins = meals["ts"].dt.floor(config.GRID)
        g.loc[g.index.isin(bins), "meal_start"] = 1
    return g


def extract_meal_events(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per logged meal: when, what type, and its macronutrients.

    A meal row is any row whose Meal Type is filled in. Snacks without macros
    are kept (carbs NaN) so they can still flag overlapping windows.
    """
    if "meal_type" not in raw:
        return pd.DataFrame(columns=["ts", "meal_type", "carbs", "protein", "fat", "fiber", "meal_kcal"])
    m = raw[raw["meal_type"].notna() & (raw["meal_type"].astype(str).str.strip() != "")].copy()
    m["meal_type"] = m["meal_type"].astype(str).str.strip().str.title()
    m.loc[m["meal_type"].str.startswith("Snack"), "meal_type"] = "Snack"   # "Snack", "Snacks", "Snack 1"
    keep = [c for c in ["ts", "meal_type", "carbs", "protein", "fat", "fiber",
                        "meal_kcal", "amount_consumed"] if c in m]
    m = m[keep]
    # the same meal can be logged on two consecutive minutes (start/end photo): collapse
    m = m.sort_values("ts")
    m = m[~((m["ts"].diff() < pd.Timedelta("10min")) & (m["meal_type"] == m["meal_type"].shift()))]
    return m.reset_index(drop=True)


def load_all(root: Path = config.RAW_DIR, verbose: bool = True):
    """Load every participant. Returns (bio, grids{pid: df}, meals{pid: df})."""
    bio = load_bio(root)
    grids, meals = {}, {}
    for pid, path in find_participant_files(root).items():
        raw = load_participant_raw(path)
        grids[pid] = to_grid(raw)
        meals[pid] = extract_meal_events(raw)
        if verbose:
            g = grids[pid]
            print(f"  P{pid:03d}: {len(raw):>6} rows | {g.index[0]:%Y-%m-%d} -> {g.index[-1]:%Y-%m-%d} "
                  f"| glucose coverage {g['glucose'].notna().mean():5.1%} | meals {len(meals[pid])}")
    return bio, grids, meals
