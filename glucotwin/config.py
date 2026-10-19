"""Central configuration: paths and clinical constants.

Every threshold used anywhere in the project lives here, with its source,
so a judge (or you, in the jury Q&A) can point to one place and defend it.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
# Raw CGMacros download lives outside git (licence is CC BY-NC-SA; we fetch, never redistribute).
RAW_DIR = Path(os.environ.get("CGMACROS_DIR", DATA_DIR / "raw"))
PROCESSED_DIR = DATA_DIR / "processed"
FIG_DIR = REPO_ROOT / "reports" / "figures"

# ---------------------------------------------------------------- time grid
# Dexcom G6 samples every 5 min, so that is the true resolution of the glucose signal.
GRID = "5min"
GRID_MIN = 5

# ---------------------------------------------------------------- clinical constants
# ADA Standards of Care: HbA1c (%) cut-offs for diagnosis.
A1C_PREDIABETES = 5.7
A1C_DIABETES = 6.5

# ADA: post-prandial (1-2 h after start of meal) target for people with diabetes is < 180 mg/dL.
HYPERGLYCEMIA_MGDL = 180.0
# International consensus on CGM metrics (Battelino 2019): time-in-range is 70-180 mg/dL.
TIR_LOW, TIR_HIGH = 70.0, 180.0

# Meal-response window definitions
BASELINE_MIN = 15      # glucose baseline = median of the 15 min before the meal
RESPONSE_MIN = 180     # follow each meal for 3 h
IAUC_MIN = 120         # incremental AUC over the first 2 h (standard in nutrition literature)
MIN_COVERAGE = 0.8     # need >=80 % of CGM samples present in the window to trust a response

# Reproducibility for every synthetic component
SEED = 2026
