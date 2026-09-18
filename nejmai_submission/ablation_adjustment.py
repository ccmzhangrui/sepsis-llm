#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ablation / robustness package for the AmsterdamUMCdb frozen validation.

A. Severity-adjusted alert-outcome associations (logistic regression:
   outcome ~ alert_red + baseline SOFA-2 + age group) -> adjusted OR.
B. Rule-source decomposition of red alerts (which deterministic audit
   rule fired; outcome rate per rule) -> every alert traceable.
C. Measurement-density -> anchor-coverage gradient: among patients with
   missing 6 h / 12 h lactate, fraction with a later anchor (24/48 h)
   by number of measured lactate windows (the Engine v2 ceiling chain).

Outputs data/ablation_adjustment.json.
"""
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA = "/Users/zhangrui/WorkBuddy/2026-09-18-15-20-42/amsterdam_sepsis/data"
df = pd.read_csv(f"{DATA}/validation_patient_level.csv")
OUT = {}

LACW = ["lac0", "lac6", "lac12", "lac24", "lac48"]


# ---------------------------------------------------------------- A
def adjusted_or(outcome):
    """Logistic regression outcome ~ alert_red + sofa0 + agegroup.
    statsmodels preferred; bootstrap CIs if unavailable."""
    d = df[["alert_red", "sofa0", "agegroup", outcome]].copy()
    d["y"] = d[outcome].astype(int)
    d["a"] = d["alert_red"].astype(int)
    try:
        import statsmodels.api as sm
        X = pd.get_dummies(d[["a", "sofa0", "agegroup"]],
                           columns=["agegroup"], drop_first=True).astype(float)
        X = sm.add_constant(X)
        res = sm.Logit(d["y"], X).fit(disp=0)
        or_ = float(np.exp(res.params["a"]))
        lo, hi = np.exp(res.conf_int().loc["a"]).tolist()
        method = "statsmodels logistic regression"
    except Exception:
        from sklearn.linear_model import LogisticRegression
        X = pd.get_dummies(d[["a", "sofa0", "agegroup"]],
                           columns=["agegroup"], drop_first=True).astype(float)
        y = d["y"].values
        rng = np.random.default_rng(42)
        ors = []
        for _ in range(2000):
            idx = rng.integers(0, len(d), len(d))
            m = LogisticRegression(penalty=None, max_iter=500).fit(
                X.values[idx], y[idx])
            ors.append(np.exp(m.coef_[0][0]))
        m = LogisticRegression(penalty=None, max_iter=500).fit(X.values, y)
        or_ = float(np.exp(m.coef_[0][0]))
        lo, hi = np.percentile(ors, [2.5, 97.5]).tolist()
        method = "sklearn logistic regression, 2000x bootstrap CI"
    red, grn = df.alert_red == True, df.alert_red == False  # noqa: E712
    r1 = d["y"][red].mean(); r0 = d["y"][grn].mean()
    n1, n0 = red.sum(), grn.sum()
    # crude RR with Katz log CI
    rr = r1 / r0
    se = np.sqrt((1 - r1) / (n1 * r1) + (1 - r0) / (n0 * r0))
    rr_lo, rr_hi = float(rr * np.exp(-1.96 * se)), float(rr * np.exp(1.96 * se))
    return {"outcome": outcome, "red_rate": round(100 * r1, 1),
            "green_rate": round(100 * r0, 1),
            "red_n": f"{int(d['y'][red].sum())}/{n1}",
            "green_n": f"{int(d['y'][grn].sum())}/{n0}",
            "crude_rr": round(float(rr), 2),
            "crude_rr_ci": [round(rr_lo, 2), round(rr_hi, 2)],
            "adjusted_or": round(or_, 2),
            "adjusted_or_ci": [round(float(lo), 2), round(float(hi), 2)],
            "method": method}


OUT["adjusted"] = {
    "mort28": adjusted_or("mort28"),
    "deterioration_ref": adjusted_or("deterioration_ref"),
    "vaso_escalation": adjusted_or("vaso_escalation"),
    "covariates": "baseline SOFA-2 score (continuous) + age group "
                  "(18-39, 40-49, 50-59, 60-69, 70-79, 80+ years)"}

# ---------------------------------------------------------------- B
r1 = df.sofa_delta >= 2
r2 = (df.lac48 > 2.0) & (df.lac_clearance < 10)
r3 = (df.hr_sd < 10) & (df.sofa_delta > 0)
recon = r1 | r2 | r3
match = (recon == df.alert_red).mean()

pat = []
lab = {0: "No rule (green alert)", 1: "Delta SOFA-2 >= 2 only",
       2: "48 h lactate > 2.0 with clearance < 10% only",
       3: "HR SD < 10 bpm with Delta SOFA-2 > 0 only",
       4: "Multiple rules"}
code = np.zeros(len(df), dtype=int)
only = r1.astype(int) + r2.astype(int) + r3.astype(int)
code[(~recon)] = 0
code[recon & (only == 1) & r1] = 1
code[recon & (only == 1) & r2] = 2
code[recon & (only == 1) & r3] = 3
code[recon & (only > 1)] = 4
for k in (1, 2, 3, 4, 0):
    m = code == k
    pat.append({"rule": lab[k], "n": int(m.sum()),
                "pct": round(100 * m.mean(), 1),
                "mort28_pct": round(100 * df.mort28[m].mean(), 1),
                "deterioration_pct": round(100 * df.deterioration_ref[m].mean(), 1)})

OUT["rule_decomposition"] = {
    "patterns": pat,
    "reconstruction_match": round(100 * match, 1),
    "note": ("Reapplication of the frozen audit rules to the exported "
             "patient-level matrix reproduced "
             f"{int(recon.sum())} red alerts vs {int(df.alert_red.sum())} "
             f"issued by the engine (match {(100*match):.1f}%). "
             "48 h lactate and clearance are sparsely measured, so rule 2 "
             "fires rarely."),
}

# ---------------------------------------------------------------- C
nmeas = df[LACW].notna().sum(axis=1)
tiers = [(0, 1, "0-1"), (2, 2, "2"), (3, 3, "3"), (4, 4, "4"), (5, 5, "5")]
anchor6 = df.lac24.notna() | df.lac48.notna()   # later anchor for lac6/lac12
grad = []
for lo, hi, lab_ in tiers:
    m = (nmeas >= lo) & (nmeas <= hi)
    sub = df[m]
    miss6 = sub.lac6.isna()
    miss12 = sub.lac12.isna()
    a6 = (miss6 & anchor6[m])
    a12 = (miss12 & anchor6[m])
    grad.append({
        "tier": lab_, "n": int(m.sum()),
        "miss6_n": int(miss6.sum()),
        "miss6_anchor_n": int(a6.sum()),
        "miss6_anchor_pct": (round(100 * a6.sum() / miss6.sum(), 1)
                             if miss6.sum() else None),
        "miss12_n": int(miss12.sum()),
        "miss12_anchor_n": int(a12.sum()),
        "miss12_anchor_pct": (round(100 * a12.sum() / miss12.sum(), 1)
                              if miss12.sum() else None)})
tot6, tot12 = df.lac6.isna().sum(), df.lac12.isna().sum()
OUT["density_gradient"] = {
    "tiers": grad,
    "total_miss6": int(tot6), "total_miss12": int(tot12),
    "total_miss_either": int((df.lac6.isna() | df.lac12.isna()).sum()),
    "note": ("Later anchor = measured 24 h or 48 h lactate. Coverage = "
             "share of missing middle-window values that can enter the "
             "interpolation arm (R2 0.68-0.72) rather than the "
             "anchor-free fallback (R2 0.43)."),
}

with open(f"{DATA}/ablation_adjustment.json", "w") as f:
    json.dump(OUT, f, indent=2)
print(json.dumps(OUT, indent=2)[:4000])
print("\nSAVED data/ablation_adjustment.json")
