#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Engine v2 — interpolation-first hybrid lactate imputer (final).

Architecture (prespecified after the AmsterdamUMCdb portability analysis):
  Arm A (bracketing anchors available): piecewise-linear interpolation
       between the nearest measured lactate anchors around the missing
       timepoint.  This is the information ceiling of the task
       (repeated-CV R2 ~0.72 on AmsterdamUMCdb complete cases).
  Arm B (no bracketing anchor): XGBoost fallback trained on MIMIC-IV
       sepsis-3 complete trajectories using anchor-free features
       (lac0, HR SD, RR SD, age, male, vasopressor escalation),
       hyperparameters from the original manuscript grid
       (lr 0.01, depth 3, subsample 0.6; 5-fold CV within MIMIC).

v1 formula (lac0*0.85 + sofa_delta*0.20 + (10-hr_sd)*0.05 + rr_sd*0.03)
is kept only as an evaluation reference.

Outputs:
  engine_v2_lac6_anchorfree.json / engine_v2_lac12_anchorfree.json
      (trained fallback XGBoost models, ready for deployment)
  engine_v2_results.json  (all benchmark numbers)
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, train_test_split
from xgboost import XGBRegressor

DATA = "/Users/zhangrui/WorkBuddy/2026-09-18-15-20-42/amsterdam_sepsis/data"
AGE_MID = {"18-39": 30, "40-49": 45, "50-59": 55, "60-69": 65,
           "70-79": 75, "80+": 85}
BASE_WEIGHT, SOFA_IMPACT, HRV_FACTOR, RRV_FACTOR = 0.85, 0.20, 0.05, 0.03

LAC = "乳酸_Lactate(mmol/L)"
HR, RR = "心率(次/min)", "呼吸频率(次/min)"
NOREPI = "持续注射泵入(去甲肾上腺素)(ug/kg/min)"
GRID = ["Day1_00:00", "Day1_06:00", "Day1_12:00", "Day1_18:00",
        "Day2_00:00", "Day2_06:00", "Day2_12:00", "Day2_18:00"]
FALLBACK_FEATS = ["lac0", "hr_sd", "rr_sd", "age", "male", "vaso_esc"]


def build_mimic():
    df = pd.read_pickle(f"{DATA}/mimic_full.pkl")
    sep = df[df["sepsis3"] == True].copy()  # noqa: E712
    m = pd.DataFrame(index=sep.index)
    for w, g in [("lac0", "Day1_00:00"), ("lac6", "Day1_06:00"),
                 ("lac12", "Day1_12:00"), ("lac24", "Day2_00:00"),
                 ("lac48", "Day2_12:00")]:
        m[w] = sep[f"{LAC}_{g}"]
    m["hr_sd"] = sep[[f"{HR}_{g}" for g in GRID]].std(axis=1, ddof=1)
    m["rr_sd"] = sep[[f"{RR}_{g}" for g in GRID]].std(axis=1, ddof=1)
    base_dose = sep[f"{NOREPI}_Day1_00:00"].fillna(0)
    max_dose = sep[[f"{NOREPI}_{g}" for g in GRID[1:7]]].max(axis=1).fillna(0)
    m["vaso_esc"] = ((max_dose >= 0.1)
                     & ((max_dose - base_dose) >= 0.05)).astype(float)
    m["age"] = pd.to_numeric(sep["age"], errors="coerce")
    m["male"] = (sep["gender"] == "M").astype(float)
    return m.dropna().query("lac0 >= 0.3")


def build_amsterdam():
    a = pd.read_csv(f"{DATA}/validation_patient_level.csv")
    cc = a.dropna(subset=["lac0", "lac6", "lac12", "lac24", "lac48"]).copy()
    cc = cc[cc.lac0 >= 0.3].reset_index(drop=True)
    cc["age"] = cc.agegroup.map(AGE_MID).fillna(65.0)
    cc["male"] = (cc.gender == "Man").astype(float)
    cc["vaso_esc"] = cc.vaso_escalation.astype(float)
    return cc


def interp(a, b, ta, tb, t):
    return a + (b - a) * (t - ta) / (tb - ta)


def metrics(yt, yp):
    return {"MSE": float(mean_squared_error(yt, yp)),
            "MAE": float(mean_absolute_error(yt, yp)),
            "R2": float(r2_score(yt, yp))}


def main():
    mim, ams = build_mimic(), build_amsterdam()
    print(f"MIMIC training rows: {len(mim)}   Amsterdam test: {len(ams)}")

    # ---- train + save fallback models (anchor-free features) ----
    fb = {}
    for tgt in ("lac6", "lac12"):
        mdl = XGBRegressor(n_estimators=400, learning_rate=0.01,
                           max_depth=3, subsample=0.6,
                           objective="reg:squarederror", random_state=2026,
                           n_jobs=4).fit(mim[FALLBACK_FEATS], mim[tgt])
        path = f"{DATA}/engine_v2_{tgt}_anchorfree.json"
        mdl.save_model(path)
        fb[tgt] = mdl
        print(f"saved {path}")

    # ---- Arm A benchmark: bracketed values, complete cases ----
    # repeated-CV pooled estimate for interpolation (deterministic ->
    # identical across folds; computed once on all 866)
    lin6 = interp(ams.lac0, ams.lac24, 0, 24, 6)
    lin12 = interp(ams.lac0, ams.lac24, 0, 24, 12)
    yt_all = np.concatenate([ams.lac6.values, ams.lac12.values])
    resA = {
        "n_patients": int(len(ams)), "n_values": int(len(yt_all)),
        "v2_armA_interpolation": metrics(yt_all, np.concatenate([lin6, lin12])),
        "v1_frozen_formula": metrics(
            yt_all, np.concatenate([
                (ams.lac0 * BASE_WEIGHT + ams.sofa_delta * SOFA_IMPACT
                 + (10.0 - ams.hr_sd) * HRV_FACTOR
                 + ams.rr_sd * RRV_FACTOR).values] * 2)),
    }
    # S12 260-subset head-to-head (same split as imputation_portability.py)
    _, te = train_test_split(np.arange(len(ams)), test_size=0.3,
                             random_state=2026)
    s = ams.iloc[te]
    yt_s = np.concatenate([s.lac6.values, s.lac12.values])
    resA["s12_260_subset"] = {
        "v2_armA_interpolation": metrics(
            yt_s, np.concatenate([
                interp(s.lac0, s.lac24, 0, 24, 6),
                interp(s.lac0, s.lac24, 0, 24, 12)])),
        "v1_frozen_formula": metrics(
            yt_s, np.concatenate([
                (s.lac0 * BASE_WEIGHT + s.sofa_delta * SOFA_IMPACT
                 + (10.0 - s.hr_sd) * HRV_FACTOR
                 + s.rr_sd * RRV_FACTOR).values] * 2)),
    }

    # ---- Arm B benchmark: anchor-free fallback (no lac24/lac48) ----
    # frozen MIMIC-trained fallback vs v1 formula, all 866 complete cases
    p6 = fb["lac6"].predict(ams[FALLBACK_FEATS])
    p12 = fb["lac12"].predict(ams[FALLBACK_FEATS])
    resB = {
        "design": "predict lac6/lac12 without lac24/lac48 anchors "
                  "(frozen MIMIC-trained fallback vs v1 formula)",
        "v2_armB_xgb_fallback": metrics(yt_all, np.concatenate([p6, p12])),
        "v1_frozen_formula": resA["v1_frozen_formula"],
    }

    # ---- full-cohort coverage (application view) ----
    a = pd.read_csv(f"{DATA}/validation_patient_level.csv")
    ind = a[a.lac6.isna() & a.lac12.isna()]
    has_later_anchor = ind.lac24.notna() | ind.lac48.notna()
    res_cov = {
        "cohort_n": int(len(a)),
        "indeterminate_n": int(len(ind)),
        "indeterminate_with_later_anchor_n": int(has_later_anchor.sum()),
        "indeterminate_with_later_anchor_pct":
            float(has_later_anchor.mean()),
        "indeterminate_needing_fallback_n":
            int((~has_later_anchor).sum()),
        "indeterminate_needing_fallback_pct":
            float((~has_later_anchor).mean()),
        "lac0_available_in_indeterminate_pct":
            float(ind.lac0.notna().mean()),
    }

    out = {"engine": "v2 interpolation-first hybrid",
           "fallback_features": FALLBACK_FEATS,
           "mimic_training_rows": int(len(mim)),
           "armA_bracketed_benchmark": resA,
           "armB_anchor_free_benchmark": resB,
           "full_cohort_coverage": res_cov}
    with open(f"{DATA}/engine_v2_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== Arm A (bracketed, n=866; 1732 values) ===")
    for k in ("v2_armA_interpolation", "v1_frozen_formula"):
        v = resA[k]
        print(f"  {k:24s} MSE={v['MSE']:.3f} MAE={v['MAE']:.3f} "
              f"R2={v['R2']:.3f}")
    print("  [S12 260-subset head-to-head]")
    for k, v in resA["s12_260_subset"].items():
        print(f"    {k:24s} MSE={v['MSE']:.3f} MAE={v['MAE']:.3f} "
              f"R2={v['R2']:.3f}")
    print("\n=== Arm B (anchor-free fallback, n=866) ===")
    for k in ("v2_armB_xgb_fallback", "v1_frozen_formula"):
        v = resB[k]
        print(f"  {k:24s} MSE={v['MSE']:.3f} MAE={v['MAE']:.3f} "
              f"R2={v['R2']:.3f}")
    print("\n=== Full-cohort coverage ===")
    for k, v in res_cov.items():
        print(f"  {k}: {v}")
    print("\nsaved engine_v2_results.json")


if __name__ == "__main__":
    main()
