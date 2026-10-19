"""Synthetic-but-consistent EHR history, exported as FHIR R4 (ABDM-compatible).

WHY THIS EXISTS
The brief asks for static EHR data: demographics, past diagnoses, lab results,
genetic markers. CGMacros gives us REAL demographics and REAL fasting labs, but
no diagnosis history, medications, family history or genotype. We synthesise
only those missing pieces, and every synthetic field is *conditioned on the
participant's real labs*, so the record is clinically coherent:

  real HbA1c >= 6.5      -> T2D diagnosis (E11.9), usually on metformin
  real LDL/TG high       -> hyperlipidaemia (E78.5), often on a statin
  real BMI >= 30         -> obesity (E66.9)
  age/BMI/T2D            -> probability of hypertension (I10)
  T2D status             -> higher chance of TCF7L2 rs7903146 risk alleles
                            (strongest common T2D variant, OR ~1.4 per allele)

Every synthetic value is (a) seeded -> reproducible, and (b) tagged with
meta.tag = "synthetic" in the FHIR output so nobody mistakes it for real data.

WHY FHIR R4
India's Ayushman Bharat Digital Mission (ABDM) health records are exchanged as
FHIR R4 bundles. Emitting FHIR means the twin could, in principle, ingest a real
patient's ABHA-linked record instead of our synthetic one with no code change
to the model side.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from glucotwin import config

SYNTH_TAG = {"system": "https://glucotwin.example/tags", "code": "synthetic",
             "display": "Synthetic field generated from real labs"}
REAL_TAG = {"system": "https://glucotwin.example/tags", "code": "cgmacros",
            "display": "Real measurement from CGMacros (PhysioNet)"}

# ICD-10 + display
ICD = {
    "T2D": ("E11.9", "Type 2 diabetes mellitus without complications"),
    "Prediabetes": ("R73.03", "Prediabetes"),
    "Obesity": ("E66.9", "Obesity, unspecified"),
    "Overweight": ("E66.3", "Overweight"),
    "Hyperlipidaemia": ("E78.5", "Hyperlipidaemia, unspecified"),
    "Hypertension": ("I10", "Essential (primary) hypertension"),
}

# LOINC codes for the real labs we carry over
LOINC = {
    "hba1c": ("4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood", "%"),
    "fasting_glucose": ("1558-6", "Fasting glucose [Mass/volume] in Serum or Plasma", "mg/dL"),
    "fasting_insulin": ("20448-7", "Insulin [Units/volume] in Serum or Plasma", "u[IU]/mL"),
    "triglycerides": ("2571-8", "Triglyceride [Mass/volume] in Serum or Plasma", "mg/dL"),
    "cholesterol": ("2093-3", "Cholesterol [Mass/volume] in Serum or Plasma", "mg/dL"),
    "hdl": ("2085-9", "HDL Cholesterol [Mass/volume] in Serum or Plasma", "mg/dL"),
    "ldl": ("13457-7", "LDL Cholesterol (calculated) [Mass/volume]", "mg/dL"),
    "bmi": ("39156-5", "Body mass index (BMI) [Ratio]", "kg/m2"),
}


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def synthesize_history(bio: pd.DataFrame, seed: int = config.SEED) -> pd.DataFrame:
    """Return one row per participant with the synthetic EHR fields."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid, p in bio.iterrows():
        status = p["status"]
        age = p.get("age", np.nan)
        bmi = p.get("bmi", np.nan)
        a1c = p.get("hba1c", np.nan)
        r = {"pid": pid}

        # ---- diagnoses (anchored on real labs) -------------------------
        dx = []
        if status == "T2D":
            dx.append("T2D")
            # higher A1c -> typically longer disease duration (loose prior)
            r["years_since_t2d_dx"] = float(np.clip(rng.lognormal(np.log(3 + 2 * max(a1c - 6.5, 0)), 0.6), 0.5, 25))
        elif status == "Prediabetes":
            dx.append("Prediabetes")
            r["years_since_t2d_dx"] = 0.0
        else:
            r["years_since_t2d_dx"] = 0.0
        if bmi >= 30:
            dx.append("Obesity")
        elif bmi >= 25:
            dx.append("Overweight")
        hyperlip = (p.get("ldl", 0) or 0) >= 160 or (p.get("triglycerides", 0) or 0) >= 200
        if hyperlip:
            dx.append("Hyperlipidaemia")
        p_htn = _sigmoid(-5.0 + 0.05 * (age if age == age else 40) + 0.06 * (bmi if bmi == bmi else 27)
                         + (0.8 if status == "T2D" else 0))
        htn = rng.random() < p_htn
        if htn:
            dx.append("Hypertension")
        r["diagnoses"] = dx

        # ---- medications (Indian first-line choices) --------------------
        meds = []
        if status == "T2D":
            if rng.random() < 0.85:
                meds.append("Metformin 500 mg BD")
            if a1c > 7.5 and rng.random() < 0.5:
                meds.append(rng.choice(["Glimepiride 1 mg OD", "Sitagliptin 100 mg OD", "Vildagliptin 50 mg BD"]))
        if hyperlip and rng.random() < 0.6:
            meds.append("Atorvastatin 10 mg OD")
        if htn and rng.random() < 0.7:
            meds.append(rng.choice(["Telmisartan 40 mg OD", "Amlodipine 5 mg OD"]))
        r["medications"] = meds
        r["on_metformin"] = int(any(m.startswith("Metformin") for m in meds))

        # ---- family history ---------------------------------------------
        p_fh = {"T2D": 0.60, "Prediabetes": 0.45}.get(status, 0.25)
        r["family_history_t2d"] = int(rng.random() < p_fh)

        # ---- genetics: TCF7L2 rs7903146 (T = risk allele) -----------------
        # Prior: Hardy-Weinberg with risk-allele freq q=0.30.
        # Posterior given status via per-allele odds ratio (T2D 1.4, prediabetes 1.2).
        q = 0.30
        prior = np.array([(1 - q) ** 2, 2 * q * (1 - q), q ** 2])
        orr = {"T2D": 1.4, "Prediabetes": 1.2}.get(status, 1.0)
        post = prior * orr ** np.arange(3)
        post /= post.sum()
        k = int(rng.choice(3, p=post))
        r["tcf7l2_rs7903146"] = ["CC", "CT", "TT"][k]
        r["tcf7l2_risk_alleles"] = k
        # Polygenic risk score percentile, shifted by status
        shift = {"T2D": 0.5, "Prediabetes": 0.25}.get(status, 0.0)
        z = rng.normal(shift, 1.0)
        r["prs_t2d_percentile"] = float(np.round(100 * 0.5 * (1 + np.tanh(z * 0.8)), 1))

        # ---- longitudinal HbA1c history (-24, -12, -6 months) ------------
        rate = {"T2D": 0.25, "Prediabetes": 0.12}.get(status, 0.02)  # %-points per year
        hist = {}
        for months in (24, 12, 6):
            val = a1c - rate * months / 12 + rng.normal(0, 0.15)
            if "Metformin 500 mg BD" in meds and months == 24:
                val += 0.4  # pre-treatment value was higher
            hist[f"hba1c_minus_{months}m"] = float(np.round(np.clip(val, 4.3, 12), 1))
        r |= hist
        r["hba1c_trend_per_yr"] = float(np.round((a1c - hist["hba1c_minus_24m"]) / 2, 2))

        # ---- vitals ------------------------------------------------------
        r["systolic_bp"] = float(np.round(rng.normal(142 if htn else 122, 8)))
        r["diastolic_bp"] = float(np.round(rng.normal(90 if htn else 79, 6)))
        rows.append(r)
    return pd.DataFrame(rows).set_index("pid")


# ------------------------------------------------------------------ FHIR R4 export

def _obs(pid, code, display, value, unit, tag, when):
    return {
        "resourceType": "Observation",
        "id": f"p{pid:03d}-{code}",
        "meta": {"tag": [tag]},
        "status": "final",
        "code": {"coding": [{"system": "http://loinc.org", "code": code, "display": display}]},
        "subject": {"reference": f"Patient/p{pid:03d}"},
        "effectiveDateTime": when,
        "valueQuantity": {"value": round(float(value), 2), "unit": unit},
    }


def to_fhir_bundle(pid: int, bio_row: pd.Series, ehr_row: pd.Series, ref_date: date = date(2026, 1, 1)) -> dict:
    """One FHIR R4 'collection' Bundle per participant."""
    entries = []
    sex = str(bio_row.get("sex", "")).upper()[:1]
    entries.append({
        "resourceType": "Patient",
        "id": f"p{pid:03d}",
        "meta": {"tag": [REAL_TAG]},
        "identifier": [{"system": "https://physionet.org/content/cgmacros", "value": f"CGMacros-{pid:03d}"}],
        "gender": {"M": "male", "F": "female"}.get(sex, "unknown"),
        "birthDate": str(ref_date.year - int(bio_row["age"])) if pd.notna(bio_row.get("age")) else None,
    })
    today = ref_date.isoformat()
    for col, (code, disp, unit) in LOINC.items():
        if pd.notna(bio_row.get(col)):
            entries.append(_obs(pid, code, disp, bio_row[col], unit, REAL_TAG, today))
    for months in (24, 12, 6):
        when = (ref_date - timedelta(days=30 * months)).isoformat()
        entries.append(_obs(pid, "4548-4", LOINC["hba1c"][1], ehr_row[f"hba1c_minus_{months}m"], "%",
                            SYNTH_TAG, when) | {"id": f"p{pid:03d}-4548-4-m{months}"})
    for d in ehr_row["diagnoses"]:
        code, disp = ICD[d]
        onset = None
        if d == "T2D":
            onset = (ref_date - timedelta(days=int(365 * ehr_row["years_since_t2d_dx"]))).isoformat()
        entries.append({
            "resourceType": "Condition", "id": f"p{pid:03d}-{code}",
            "meta": {"tag": [SYNTH_TAG]}, "clinicalStatus": {"text": "active"},
            "code": {"coding": [{"system": "http://hl7.org/fhir/sid/icd-10", "code": code, "display": disp}]},
            "subject": {"reference": f"Patient/p{pid:03d}"}, "onsetDateTime": onset,
        })
    for i, m in enumerate(ehr_row["medications"]):
        entries.append({
            "resourceType": "MedicationStatement", "id": f"p{pid:03d}-med{i}",
            "meta": {"tag": [SYNTH_TAG]}, "status": "active",
            "medicationCodeableConcept": {"text": m}, "subject": {"reference": f"Patient/p{pid:03d}"},
        })
    if ehr_row["family_history_t2d"]:
        entries.append({
            "resourceType": "FamilyMemberHistory", "id": f"p{pid:03d}-fh",
            "meta": {"tag": [SYNTH_TAG]}, "status": "completed",
            "patient": {"reference": f"Patient/p{pid:03d}"},
            "relationship": {"text": "first-degree relative"},
            "condition": [{"code": {"coding": [{"system": "http://hl7.org/fhir/sid/icd-10",
                                                "code": "E11", "display": "Type 2 diabetes mellitus"}]}}],
        })
    entries.append({
        "resourceType": "Observation", "id": f"p{pid:03d}-tcf7l2", "meta": {"tag": [SYNTH_TAG]},
        "status": "final",
        "code": {"coding": [{"system": "http://loinc.org", "code": "69548-6", "display": "Genetic variant assessment"}]},
        "subject": {"reference": f"Patient/p{pid:03d}"},
        "component": [
            {"code": {"text": "dbSNP rs7903146 (TCF7L2)"}, "valueString": ehr_row["tcf7l2_rs7903146"]},
            {"code": {"text": "T2D polygenic risk score percentile"}, "valueQuantity": {"value": ehr_row["prs_t2d_percentile"], "unit": "%"}},
        ],
    })
    entries = [{k: v for k, v in e.items() if v is not None} for e in entries]
    return {"resourceType": "Bundle", "id": f"glucotwin-p{pid:03d}", "type": "collection",
            "entry": [{"resource": e} for e in entries]}


def write_fhir(bio: pd.DataFrame, ehr: pd.DataFrame, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for pid in bio.index:
        bundle = to_fhir_bundle(pid, bio.loc[pid], ehr.loc[pid])
        (out_dir / f"patient_{pid:03d}.json").write_text(json.dumps(bundle, indent=1, default=str))
    return len(bio)
