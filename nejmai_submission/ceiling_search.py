#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ceiling-model search on AmsterdamUMCdb (honest out-of-sample estimates).

Repeated 5-fold patient-level CV on ALL complete cases (n=866) comparing:
  1. linear interpolation (anchor-only baseline)
  2. interpolation + OLS residual correction (physiology features)
  3. blend 0.8*interp + 0.2*MIMIC-trained v2 XGB
  4. MIMIC-trained v2 XGB (frozen, cross-site)
  5. in-site XGB + lin features (reference upper bound of ML)

The winner defines the final Engine v2 architecture.
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold
from xgboost import XGBRegressor

DATA = "/Users/zhangrui/WorkBuddy/2026-09-18-15-20-42/amsterdam_sepsis/data"
AGE_MID = {"18-39": 30, "40-49": 45, "50-59": 55, "60-69": 65,
           "70-79": 75, "80+": 85}


def load_ams():
    a = pd.read_csv(f"{DATA}/validation_patient_level.csv")
    cc = a.dropna(subset=["lac0", "lac6", "lac12", "lac24", "lac48"]).copy()
    cc = cc[cc.lac0 >= 0.3].reset_index(drop=True)
    cc["lin6"] = cc.lac0 + (cc.lac24 - cc.lac0) * 6 / 24.0
    cc["lin12"] = cc.lac0 + (cc.lac24 - cc.lac0) * 12 / 24.0
    cc["slope"] = (cc.lac24 - cc.lac0) / 24.0
    cc["age"] = cc.agegroup.map(AGE_MID).fillna(65.0)
    cc["male"] = (cc.gender == "Man").astype(float)
    cc["vaso_esc"] = cc.vaso_escalation.astype(float)
    return cc


def train_mimic_xgb(cc):
    """Frozen v2 XGB trained on MIMIC complete cases (as in engine_v2.py)."""
    LAC = "乳酸_Lactate(mmol/L)"
    HR, RR = "心率(次/min)", "呼吸频率(次/min)"
    NOREPI = "持续注射泵入(去甲肾上腺素)(ug/kg/min)"
    GRID = ["Day1_00:00", "Day1_06:00", "Day1_12:00", "Day1_18:00",
            "Day2_00:00", "Day2_06:00", "Day2_12:00", "Day2_18:00"]
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
    m = m.dropna().query("lac0 >= 0.3")

    def feats(d, tgt):
        X = d[["lac0", "lac24", "lac48", "hr_sd", "rr_sd", "age", "male",
               "vaso_esc"]].copy()
        X["lin"] = (d.lac0 + (d.lac24 - d.lac0) * (6 if tgt == "lac6" else 12)
                    / 24.0)
        return X

    models = {}
    for tgt in ("lac6", "lac12"):
        mdl = XGBRegressor(n_estimators=400, learning_rate=0.01,
                           max_depth=3, subsample=0.6,
                           objective="reg:squarederror", random_state=2026,
                           n_jobs=4).fit(feats(m, tgt), m[tgt])
        models[tgt] = (mdl, feats)
    return models, m


def main():
    cc = load_ams()
    print(f"complete cases: {len(cc)}")
    mim_models, mim_cc = train_mimic_xgb(cc)
    print(f"MIMIC training rows: {len(mim_cc)}")

    PHYS = ["sofa_delta", "hr_sd", "rr_sd", "age", "male", "vaso_esc",
            "slope", "lac0", "lac24", "lac48"]

    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=2026)
    y_all = {}
    preds = {k: [] for k in ["interp", "interp+OLS", "blend.8/.2",
                             "mimicXGB", "insiteXGB"]}
    truths = []
    fold_idx = []
    for tr, te in rkf.split(cc):
        ctr, cte = cc.iloc[tr], cc.iloc[te]
        for tgt, col in [("lac6", "lac6"), ("lac12", "lac12")]:
            lin_col = "lin6" if tgt == "lac6" else "lin12"
            yt = cte[col].values
            lin = cte[lin_col].values
            truths.append(yt)
            preds["interp"].append(lin)

            # interp + OLS residual correction (fit on train folds)
            resid = ctr[col].values - ctr[lin_col].values
            ols = LinearRegression().fit(ctr[PHYS], resid)
            preds["interp+OLS"].append(
                lin + ols.predict(cte[PHYS]))

            # MIMIC-trained frozen XGB
            mdl, feats = mim_models[tgt]
            px = mdl.predict(feats(cte, tgt))
            preds["mimicXGB"].append(px)
            preds["blend.8/.2"].append(0.8 * lin + 0.2 * px)

            # in-site XGB + lin (upper reference)
            xgb = XGBRegressor(n_estimators=400, learning_rate=0.01,
                               max_depth=3, subsample=0.6,
                               objective="reg:squarederror",
                               random_state=2026, n_jobs=4)
            xgb.fit(feats(ctr, tgt), ctr[col].values)
            preds["insiteXGB"].append(xgb.predict(feats(cte, tgt)))

    yt = np.concatenate(truths)
    out = {"n": int(len(cc)), "design": "repeated 5-fold CV x5, 866 patients",
           "methods": {}}
    print(f"\n=== repeated-CV honest estimates (n={len(cc)} patients, "
          f"{len(yt)} masked values) ===")
    for k, ps in preds.items():
        yp = np.concatenate(ps)
        r = {"MSE": float(mean_squared_error(yt, yp)),
             "MAE": float(mean_absolute_error(yt, yp)),
             "R2": float(r2_score(yt, yp))}
        out["methods"][k] = r
        print(f"  {k:14s} MSE={r['MSE']:6.3f} MAE={r['MAE']:5.3f}"
              f" R2={r['R2']:6.3f}")
    with open(f"{DATA}/ceiling_search.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved ceiling_search.json")


if __name__ == "__main__":
    main()
