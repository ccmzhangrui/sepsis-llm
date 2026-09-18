#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ceiling diagnostic for lactate imputation in AmsterdamUMCdb.

Question: with within-patient anchor lactate values (0/24/48 h) PLUS
physiology, what is the best achievable R2 on masked 6 h/12 h lactate
in AmsterdamUMCdb?  In-site XGBoost (upper bound for any cross-site
model) with and without interpolation features, plus the frozen formula
with a linear-interpolation anchor term added.

Same masked complete-case design as imputation_portability.py
(n=866, 70/30 split, fixed seed) so results are directly comparable to
Supplementary Table S12.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor

DATA = "/Users/zhangrui/WorkBuddy/2026-09-18-15-20-42/amsterdam_sepsis/data"
BASE_WEIGHT, SOFA_IMPACT, HRV_FACTOR, RRV_FACTOR = 0.85, 0.20, 0.05, 0.03
AGE_MID = {"18-39": 30, "40-49": 45, "50-59": 55, "60-69": 65,
           "70-79": 75, "80+": 85}


def main():
    df = pd.read_csv(f"{DATA}/validation_patient_level.csv")
    cc = df.dropna(subset=["lac0", "lac6", "lac12", "lac24", "lac48"]).copy()
    cc = cc[cc.lac0 >= 0.3]
    print(f"complete cases: {len(cc)}  (identical to imputation_portability)")

    # interpolation anchors
    cc["lin6"] = cc.lac0 + (cc.lac24 - cc.lac0) * 6 / 24.0
    cc["lin12"] = cc.lac0 + (cc.lac24 - cc.lac0) * 12 / 24.0
    # anchor spread / trajectory steepness features
    cc["slope_0_24"] = (cc.lac24 - cc.lac0) / 24.0
    cc["slope_24_48"] = (cc.lac48 - cc.lac24) / 24.0
    cc["lac_mean_anchor"] = cc[["lac0", "lac24", "lac48"]].mean(axis=1)
    cc["age"] = cc.agegroup.map(AGE_MID).fillna(65.0)
    cc["male"] = (cc.gender == "Man").astype(float)
    cc["vaso_esc"] = cc.vaso_escalation.astype(float)

    base_cols = ["lac0", "lac24", "lac48", "sofa0", "sofa_delta",
                 "hr_sd", "rr_sd", "age", "male", "vaso_esc"]
    lin_cols = ["lin6", "lin12", "slope_0_24", "slope_24_48",
                "lac_mean_anchor"]
    # per-target feature sets: lin12-target uses lin12 not lin6 etc.
    Xb = cc[base_cols].copy()
    Xb6 = pd.concat([Xb, cc[["lin6", "slope_0_24", "lac_mean_anchor"]]], axis=1)
    Xb12 = pd.concat([Xb, cc[["lin12", "slope_0_24", "lac_mean_anchor"]]], axis=1)

    y6, y12 = cc.lac6.values, cc.lac12.values
    idx = np.arange(len(cc))
    tr, te = train_test_split(idx, test_size=0.3, random_state=2026)
    print(f"train {len(tr)} / test {len(te)}")

    grid = {"learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7, 9],
            "subsample": [0.6, 0.8, 1.0]}
    base = dict(n_estimators=400, objective="reg:squarederror",
                random_state=2026, n_jobs=4)

    def fit_eval(X, label):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        preds = {}
        for tgt, y in [("lac6", y6), ("lac12", y12)]:
            gs = GridSearchCV(XGBRegressor(**base), grid, cv=5,
                              scoring="neg_mean_squared_error", n_jobs=1)
            gs.fit(Xtr, y[tr])
            m = XGBRegressor(**base, **gs.best_params_).fit(Xtr, y[tr])
            preds[tgt] = m.predict(Xte)
            print(f"  [{label}] {tgt} best: {gs.best_params_}")
        yp = np.concatenate([preds["lac6"], preds["lac12"]])
        yt = np.concatenate([y6[te], y12[te]])
        return (mean_squared_error(yt, yp), mean_absolute_error(yt, yp),
                r2_score(yt, yp))

    results = {}
    mse, mae, r2 = fit_eval(Xb, "in-site XGB, NO lin features")
    results["in-site XGB (no lin)"] = (mse, mae, r2)
    mse, mae, r2 = fit_eval(Xb6, "in-site XGB + lin features")
    results["in-site XGB (+ lin)"] = (mse, mae, r2)

    # oracle: in-sample fit on ALL data (train==test) -> absolute ceiling
    def fit_insample(X, label):
        preds = {}
        for tgt, y in [("lac6", y6), ("lac12", y12)]:
            gs = GridSearchCV(XGBRegressor(**base), grid, cv=5,
                              scoring="neg_mean_squared_error", n_jobs=1)
            gs.fit(X, y)
            m = XGBRegressor(**base, **gs.best_params_).fit(X, y)
            preds[tgt] = m.predict(X)
        yp = np.concatenate([preds["lac6"], preds["lac12"]])
        yt = np.concatenate([y6, y12])
        results[label] = (mean_squared_error(yt, yp),
                          mean_absolute_error(yt, yp), r2_score(yt, yp))
    fit_insample(Xb6, "ORACLE in-sample (+ lin)")

    te_cc = cc.iloc[te]
    eng = (te_cc.lac0 * BASE_WEIGHT + te_cc.sofa_delta * SOFA_IMPACT
           + (10.0 - te_cc.hr_sd) * HRV_FACTOR + te_cc.rr_sd * RRV_FACTOR).values
    lin = np.concatenate([te_cc.lin6.values, te_cc.lin12.values])
    yt = np.concatenate([y6[te], y12[te]])
    results["frozen formula (ref S12)"] = (
        mean_squared_error(yt, np.concatenate([eng, eng])),
        mean_absolute_error(yt, np.concatenate([eng, eng])),
        r2_score(yt, np.concatenate([eng, eng])))
    results["linear interpolation (ref S12)"] = (
        mean_squared_error(yt, lin), mean_absolute_error(yt, lin), r2_score(yt, lin))
    # hybrid: interpolation + frozen-formula residual correction (no training)
    hybrid = 0.5 * (lin + np.concatenate([eng, eng]))
    results["naive 50/50 hybrid"] = (
        mean_squared_error(yt, hybrid), mean_absolute_error(yt, hybrid),
        r2_score(yt, hybrid))

    print("\n=== Amsterdam ceiling diagnostic (same test set as S12) ===")
    print(f"test-set lactate variance: {yt.var():.2f} (SD {yt.std():.2f})")
    for k, (mse, mae, r2) in results.items():
        print(f"{k:34s} MSE={mse:6.3f}  MAE={mae:5.3f}  R2={r2:6.3f}")
    print("\nR2 0.9 would require MSE <= "
          f"{yt.var() * 0.1:.3f} on this test set.")


if __name__ == "__main__":
    main()
