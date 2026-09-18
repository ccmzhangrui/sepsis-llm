#!/usr/bin/env python3
"""Site-adapted portability evaluation of the XGBoost trajectory-imputation
framework in AmsterdamUMCdb.

Mirrors the development-cohort design (Supplementary Table S2/S3 of the
manuscript):
  - feature family: trajectory/laboratory variables + physiological
    variability (HR SD, MAP SD, RR SD) + vasopressor intensity
  - hyperparameter grid: learning rate {0.01,0.05,0.1,0.2},
    max depth {3,5,7,9}, subsample {0.6,0.8,1.0} (grid search, 5-fold CV
    within the training partition)
  - comparators: linear interpolation, mean imputation, random forest,
    MICE (IterativeImputer), and the frozen released-engine formula
    (reported transparently)
  - complete-case masking: patients with measured lactate at all of
    0/6/12/24/48 h; 6 h and 12 h values masked and reconstructed;
    70/30 patient-level split (fixed seed), metrics on the held-out 30%
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
    r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor

DATA = "/Users/zhangrui/WorkBuddy/2026-09-18-15-20-42/amsterdam_sepsis/data"
BASE_WEIGHT, SOFA_IMPACT = 0.85, 0.20
HRV_FACTOR, RRV_FACTOR = 0.05, 0.03
AGE_MID = {"18-39": 30, "40-49": 45, "50-59": 55, "60-69": 65,
           "70-79": 75, "80+": 85}


def main():
    df = pd.read_csv(f"{DATA}/validation_patient_level.csv")
    cc = df.dropna(subset=["lac0", "lac6", "lac12", "lac24", "lac48"]) \
        .copy()
    cc = cc[cc.lac0 >= 0.3]
    print(f"complete cases: {len(cc)}", flush=True)

    feat_cols = ["lac0", "lac24", "lac48", "sofa0", "sofa_delta",
                 "hr_sd", "rr_sd", "mort28", "vaso_escalation"]
    X = pd.DataFrame({
        "lac0": cc.lac0, "lac24": cc.lac24, "lac48": cc.lac48,
        "sofa0": cc.sofa0, "sofa_delta": cc.sofa_delta,
        "hr_sd": cc.hr_sd, "rr_sd": cc.rr_sd,
        "age": cc.agegroup.map(AGE_MID).fillna(65.0),
        "male": (cc.gender == "Man").astype(float),
        "vaso_esc": cc.vaso_escalation.astype(float),
    })
    y6, y12 = cc.lac6.values, cc.lac12.values

    Xtr, Xte, y6tr, y6te, y12tr, y12te = train_test_split(
        X, y6, y12, test_size=0.3, random_state=2026)

    grid = {"learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7, 9],
            "subsample": [0.6, 0.8, 1.0]}
    base = dict(n_estimators=400, objective="reg:squarederror",
                random_state=2026, n_jobs=4)
    best = {}
    for target, ytr in [("lac6", y6tr), ("lac12", y12tr)]:
        gs = GridSearchCV(XGBRegressor(**base), grid, cv=5,
                          scoring="neg_mean_squared_error", n_jobs=1)
        gs.fit(Xtr, ytr)
        best[target] = gs.best_params_
        print(f"best {target}: {gs.best_params_}", flush=True)

    m6 = XGBRegressor(**base, **best["lac6"]).fit(Xtr, y6tr)
    m12 = XGBRegressor(**base, **best["lac12"]).fit(Xtr, y12tr)
    p6, p12 = m6.predict(Xte), m12.predict(Xte)

    rf6 = RandomForestRegressor(n_estimators=300, random_state=2026,
                                n_jobs=4).fit(Xtr, y6tr)
    rf12 = RandomForestRegressor(n_estimators=300, random_state=2026,
                                 n_jobs=4).fit(Xtr, y12tr)
    r6, r12 = rf6.predict(Xte), rf12.predict(Xte)

    # MICE on trajectory columns
    mice_tr = Xtr[["lac0", "lac24", "lac48", "sofa0", "sofa_delta",
                   "hr_sd", "rr_sd"]]
    mice_te = Xte[["lac0", "lac24", "lac48", "sofa0", "sofa_delta",
                   "hr_sd", "rr_sd"]]
    imp6 = IterativeImputer(random_state=2026, max_iter=20)
    tr6 = mice_tr.copy(); tr6["y"] = y6tr
    imp6.fit(tr6)
    te6 = mice_te.copy(); te6["y"] = np.nan
    mi6 = imp6.transform(te6)[:, -1]
    imp12 = IterativeImputer(random_state=2026, max_iter=20)
    tr12 = mice_tr.copy(); tr12["y"] = y12tr
    imp12.fit(tr12)
    te12 = mice_te.copy(); te12["y"] = np.nan
    mi12 = imp12.transform(te12)[:, -1]

    te = cc.loc[Xte.index]
    eng = (te.lac0 * BASE_WEIGHT + te.sofa_delta * SOFA_IMPACT
           + (10.0 - te.hr_sd) * HRV_FACTOR + te.rr_sd * RRV_FACTOR) \
        .values
    lin6 = (te.lac0 + (te.lac24 - te.lac0) * 6 / 24.0).values
    lin12 = (te.lac0 + (te.lac24 - te.lac0) * 12 / 24.0).values
    meanimp = te[["lac0", "lac24", "lac48"]].mean(axis=1).values

    y_true = np.concatenate([y6te, y12te])
    methods = {
        "XGBoost (site-adapted)": np.concatenate([p6, p12]),
        "Random Forest": np.concatenate([r6, r12]),
        "MICE": np.concatenate([mi6, mi12]),
        "Linear interpolation": np.concatenate([lin6, lin12]),
        "Mean imputation": np.concatenate([meanimp, meanimp]),
        "Frozen released-engine formula": np.concatenate([eng, eng]),
    }
    out = {"n_complete_cases": int(len(cc)),
           "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
           "best_params": best, "methods": {}}
    for name, yp in methods.items():
        out["methods"][name] = {
            "MSE": float(mean_squared_error(y_true, yp)),
            "MAE": float(mean_absolute_error(y_true, yp)),
            "R2": float(r2_score(y_true, yp)),
        }
        print(f"{name:32s} MSE={out['methods'][name]['MSE']:.3f} "
              f"MAE={out['methods'][name]['MAE']:.3f} "
              f"R2={out['methods'][name]['R2']:.3f}", flush=True)

    # per-window metrics for the site-adapted XGBoost
    out["per_window"] = {
        "lac6": {"MSE": float(mean_squared_error(y6te, p6)),
                 "MAE": float(mean_absolute_error(y6te, p6)),
                 "R2": float(r2_score(y6te, p6))},
        "lac12": {"MSE": float(mean_squared_error(y12te, p12)),
                  "MAE": float(mean_absolute_error(y12te, p12)),
                  "R2": float(r2_score(y12te, p12))},
    }
    # feature importance (gain) for the 6h model
    imp_gain = m6.get_booster().get_score(importance_type="gain")
    out["feature_importance_lac6"] = imp_gain

    with open(f"{DATA}/imputation_portability.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved imputation_portability.json", flush=True)


if __name__ == "__main__":
    main()
