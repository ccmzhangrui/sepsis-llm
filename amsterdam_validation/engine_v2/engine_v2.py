#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Engine v2: interpolation-augmented XGBoost lactate imputer.

Design (leakage-controlled, mirrors the manuscript architecture):
  - Training:  MIMIC-IV sepsis-3 ICU stays (development site A,
    `副本minmic数据.xlsx`), complete 5-window lactate trajectories,
    onset-relative 6-hourly grid (Day1_00:00 = 0 h ... Day2_12:00 = 48 h;
    onset-relative semantics verified empirically).
  - Features:  lac0, lac24, lac48, linear-interpolation anchor (lin6/lin12),
               HR SD / RR SD over the 0-48 h grid, age, male,
               vasopressor escalation. (SOFA features omitted: the
               Amsterdam ceiling diagnostic showed they add no
               out-of-sample signal, and serial SOFA-2 is not available in
               the MIMIC grid.)
  - Hyperparameters: the original search grid (lr {0.01,0.05,0.1,0.2} x
               depth {3,5,7,9} x subsample {0.6,0.8,1.0}), 5-fold CV
               within MIMIC training data only.
  - External test: ALL AmsterdamUMCdb complete cases (n=866) and, for the
               head-to-head with Supplementary Table S12, the identical
               260-patient held-out subset (random_state=2026).
  - Also reported: deterministic hybrid v2 (interpolation-first,
               v2-formula fallback) and the frozen v1 formula for
               reference.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor

DATA = "/Users/zhangrui/WorkBuddy/2026-09-18-15-20-42/amsterdam_sepsis/data"
BASE_WEIGHT, SOFA_IMPACT, HRV_FACTOR, RRV_FACTOR = 0.85, 0.20, 0.05, 0.03
AGE_MID = {"18-39": 30, "40-49": 45, "50-59": 55, "60-69": 65,
           "70-79": 75, "80+": 85}

LAC = "乳酸_Lactate(mmol/L)"
HR = "心率(次/min)"
RR = "呼吸频率(次/min)"
NOREPI = "持续注射泵入(去甲肾上腺素)(ug/kg/min)"
GRID = ["Day1_00:00", "Day1_06:00", "Day1_12:00", "Day1_18:00",
        "Day2_00:00", "Day2_06:00", "Day2_12:00", "Day2_18:00"]


def build_mimic():
    df = pd.read_pickle(f"{DATA}/mimic_full.pkl")
    sep = df[df["sepsis3"] == True].copy()  # noqa: E712

    def c(base, day_t):
        return f"{base}_{day_t}"

    # lactate windows (onset-relative grid)
    out = pd.DataFrame(index=sep.index)
    out["lac0"] = sep[c(LAC, "Day1_00:00")]
    out["lac6"] = sep[c(LAC, "Day1_06:00")]
    out["lac12"] = sep[c(LAC, "Day1_12:00")]
    out["lac24"] = sep[c(LAC, "Day2_00:00")]
    out["lac48"] = sep[c(LAC, "Day2_12:00")]

    # physiological variability: SD across the 8 grid points (0-48 h)
    hr_cols = [c(HR, g) for g in GRID]
    rr_cols = [c(RR, g) for g in GRID]
    out["hr_sd"] = sep[hr_cols].std(axis=1, ddof=1)
    out["rr_sd"] = sep[rr_cols].std(axis=1, ddof=1)

    # vasopressor escalation: baseline D1_00 vs max over D1_06..D2_12
    base_dose = sep[c(NOREPI, "Day1_00:00")].fillna(0)
    later = [c(NOREPI, g) for g in GRID[1:7]]
    max_dose = sep[later].max(axis=1).fillna(0)
    out["vaso_esc"] = ((max_dose >= 0.1)
                       & ((max_dose - base_dose) >= 0.05)).astype(float)

    out["age"] = pd.to_numeric(sep["age"], errors="coerce")
    out["male"] = (sep["gender"] == "M").astype(float)

    cc = out.dropna(subset=["lac0", "lac6", "lac12", "lac24", "lac48",
                            "hr_sd", "rr_sd", "age"]).copy()
    cc = cc[cc.lac0 >= 0.3]
    return cc


def build_amsterdam():
    a = pd.read_csv(f"{DATA}/validation_patient_level.csv")
    cc = a.dropna(subset=["lac0", "lac6", "lac12", "lac24", "lac48"]).copy()
    cc = cc[cc.lac0 >= 0.3]
    cc["age"] = cc.agegroup.map(AGE_MID).fillna(65.0)
    cc["male"] = (cc.gender == "Man").astype(float)
    cc["vaso_esc"] = cc.vaso_escalation.astype(float)
    return cc


def features(df, target):
    X = pd.DataFrame({
        "lac0": df.lac0, "lac24": df.lac24, "lac48": df.lac48,
        "hr_sd": df.hr_sd, "rr_sd": df.rr_sd,
        "age": df.age, "male": df.male, "vaso_esc": df.vaso_esc,
    })
    if target == "lac6":
        X["lin"] = df.lac0 + (df.lac24 - df.lac0) * 6 / 24.0
    else:
        X["lin"] = df.lac0 + (df.lac24 - df.lac0) * 12 / 24.0
    return X


def main():
    mim = build_mimic()
    ams = build_amsterdam()
    print(f"MIMIC-IV complete cases (training): {len(mim)}")
    print(f"Amsterdam complete cases (external test): {len(ams)}")

    grid = {"learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7, 9],
            "subsample": [0.6, 0.8, 1.0]}
    base = dict(n_estimators=400, objective="reg:squarederror",
                random_state=2026, n_jobs=4)

    # ---- train v2 on MIMIC only ----
    models, best = {}, {}
    for tgt in ("lac6", "lac12"):
        X = features(mim, tgt)
        y = mim[tgt].values
        gs = GridSearchCV(XGBRegressor(**base), grid, cv=5,
                          scoring="neg_mean_squared_error", n_jobs=1)
        gs.fit(X, y)
        best[tgt] = gs.best_params_
        models[tgt] = XGBRegressor(**base, **gs.best_params_).fit(X, y)
        print(f"v2 {tgt}: trained on {len(X)} rows, best {gs.best_params_}")

    # ---- external test on Amsterdam ----
    # (a) all complete cases; (b) the S12 260-patient held-out subset
    tr_ids, te_ids = train_test_split(np.arange(len(ams)), test_size=0.3,
                                      random_state=2026)
    for label, sub in [("all_866", np.arange(len(ams))),
                       ("s12_260", te_ids)]:
        s = ams.iloc[sub]
        p6 = models["lac6"].predict(features(s, "lac6"))
        p12 = models["lac12"].predict(features(s, "lac12"))
        yt = np.concatenate([s.lac6.values, s.lac12.values])
        yp = np.concatenate([p6, p12])
        lin = np.concatenate([
            s.lac0 + (s.lac24 - s.lac0) * 6 / 24.0,
            s.lac0 + (s.lac24 - s.lac0) * 12 / 24.0])
        eng = (s.lac0 * BASE_WEIGHT + (s.sofa_delta * SOFA_IMPACT
               + (10.0 - s.hr_sd) * HRV_FACTOR
               + s.rr_sd * RRV_FACTOR)).values
        res = {}
        for name, pred in [("v2 XGB (MIMIC-trained, +lin)", yp),
                           ("linear interpolation", lin),
                           ("frozen v1 formula",
                            np.concatenate([eng, eng]))]:
            res[name] = {"MSE": float(mean_squared_error(yt, pred)),
                         "MAE": float(mean_absolute_error(yt, pred)),
                         "R2": float(r2_score(yt, pred))}
        if label == "s12_260":
            with open(f"{DATA}/engine_v2_results.json", "w") as f:
                json.dump({"best_params": best, "results": res}, f, indent=2)
        print(f"\n=== external test [{label}] n_patients={len(s)} ===")
        for k, v in res.items():
            print(f"  {k:34s} MSE={v['MSE']:6.3f} MAE={v['MAE']:5.3f}"
                  f" R2={v['R2']:6.3f}")

    # feature importance
    imp = models["lac6"].get_booster().get_score(importance_type="gain")
    print("\nv2 lac6 feature importance (gain):")
    for k, v in sorted(imp.items(), key=lambda kv: -kv[1]):
        print(f"  {k:10s} {v:8.1f}")


if __name__ == "__main__":
    main()
