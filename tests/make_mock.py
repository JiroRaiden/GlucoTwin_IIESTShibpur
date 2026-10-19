"""Generate a tiny fake dataset in the exact CGMacros file format (for pipeline tests only)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/mock_cgmacros")
rng = np.random.default_rng(0)
bio_rows = []
for pid in range(1, 9):
    a1c = rng.choice([5.2, 6.0, 7.2])
    sens = {5.2: 0.6, 6.0: 1.0, 7.2: 1.6}[a1c] * rng.uniform(0.6, 1.4)
    base = {5.2: 90, 6.0: 105, 7.2: 140}[a1c]
    t = pd.date_range("2027-03-01 06:00", periods=60 * 24 * 8, freq="1min")
    g = np.full(len(t), float(base))
    rows = pd.DataFrame({"Timestamp": t.strftime("%m/%d/%Y %H:%M")})
    meal_cols = {k: [np.nan] * len(t) for k in ["Calories", "Carbs", "Protein", "Fat", "Fiber", "Amount Consumed"]}
    meal_type = [None] * len(t)
    for day in range(8):
        for h, mt in [(8, "Breakfast"), (13, "Lunch"), (19, "Dinner")]:
            i = day * 1440 + (h - 6) * 60 + int(rng.integers(0, 30))
            carbs = rng.uniform(15, 110)
            meal_type[i] = mt
            meal_cols["Carbs"][i] = carbs
            meal_cols["Protein"][i] = rng.uniform(5, 40)
            meal_cols["Fat"][i] = rng.uniform(5, 40)
            meal_cols["Fiber"][i] = rng.uniform(0, 12)
            meal_cols["Calories"][i] = carbs * 4 + 300
            meal_cols["Amount Consumed"][i] = 100
            k = np.arange(0, 240)
            g[i:i + 240] += sens * carbs * (k / 45) * np.exp(1 - k / 45)
    g += rng.normal(0, 3, len(t))
    dex = np.where(np.arange(len(t)) % 5 == 0, g, np.nan)
    lib = np.where(np.arange(len(t)) % 15 == 0, g * 0.93, np.nan)
    rows["Libre GL"] = lib
    rows["Dexcom GL"] = dex
    rows["HR"] = rng.normal(75, 8, len(t)).round()
    rows["Calories (Activity)"] = rng.uniform(1, 4, len(t))
    rows["METs"] = rng.integers(10, 40, len(t))
    rows["Meal Type"] = meal_type
    for k, v in meal_cols.items():
        rows[k] = v
    rows["Image path"] = None
    d = out / f"CGMacros-{pid:03d}"
    d.mkdir(parents=True, exist_ok=True)
    rows.to_csv(d / f"CGMacros-{pid:03d}.csv", index=False)
    bio_rows.append({"subject": pid, "Age": int(rng.integers(25, 65)), "Gender": rng.choice(["M", "F"]),
                     "BMI": rng.uniform(21, 38), "Body weight": 170, "Height": 66, "Self-identify": "White",
                     "A1c PDL (Lab)": a1c, "Fasting GLU - PDL (Lab)": base, "Insulin": rng.uniform(4, 25),
                     "Triglycerides": rng.uniform(60, 260), "Cholesterol": 190, "HDL": 50, "Non HDL": 140,
                     "LDL (Cal)": rng.uniform(70, 180), "VLDL (Cal)": 25, "Cho/HDL Ratio": 3.8})
pd.DataFrame(bio_rows).to_csv(out / "bio.csv", index=False)
print("mock written to", out)
