#!/usr/bin/env python3
"""AmsterdamUMCdb external validation of the frozen Sepsis Agentic-LLM CDSS
execution engine (identical coefficients to the released repository code).

Reads: hourly_features.csv, hourly_gcs.csv, vaso_intervals.csv,
       rrt_intervals.csv, vent_intervals.csv, cohort_suspected_infection.csv,
       admissions_first.csv
Writes: validation_patient_level.csv, validation_metrics.json
"""
import json

import numpy as np
import pandas as pd

DATA = "/Users/zhangrui/WorkBuddy/2026-09-18-15-20-42/amsterdam_sepsis/data"
BASE = "/Users/zhangrui/Documents/sofa2.0/AmsterdamUBD"
HOUR = 3600_000.0

BASE_WEIGHT, SOFA_IMPACT = 0.85, 0.20
HRV_FACTOR, RRV_FACTOR = 0.05, 0.03


def sd(vals):
    v = [x for x in vals if np.isfinite(x)]
    return float(np.std(v)) if len(v) >= 2 else 0.0


def brain_score(g):
    if not np.isfinite(g):
        return np.nan
    if g < 6:
        return 4
    if g <= 8:
        return 3
    if g <= 12:
        return 2
    if g <= 14:
        return 1
    return 0


def resp_score(pf, adv):
    if not np.isfinite(pf):
        return np.nan
    if pf <= 75 and adv:
        return 4
    if pf <= 150 and adv:
        return 3
    if pf <= 225:
        return 2
    if pf <= 300:
        return 1
    return 0


def cvs_score(mapmin, ne, other, dopa):
    if ne > 0 or other or dopa > 0:
        if dopa > 0 and ne == 0 and not other:
            return 4 if dopa > 40 else (3 if dopa > 20 else 2)
        if ne > 0.4:
            return 4
        if ne > 0.2 and other:
            return 4
        if ne > 0.2:
            return 3
        if ne > 0 and other:
            return 3
        return 2
    if not np.isfinite(mapmin):
        return np.nan
    if mapmin < 40:
        return 4
    if mapmin < 50:
        return 3
    if mapmin < 60:
        return 2
    if mapmin < 70:
        return 1
    return 0


def liver_score(bili_umol):
    if not np.isfinite(bili_umol):
        return np.nan
    b = bili_umol / 17.1
    if b > 12:
        return 4
    if b > 6:
        return 3
    if b > 3:
        return 2
    if b > 1.2:
        return 1
    return 0


def kidney_score(cr_umol, rrt):
    if rrt:
        return 4
    if not np.isfinite(cr_umol):
        return np.nan
    c = cr_umol / 88.4
    if c > 3.5:
        return 3
    if c > 2.0:
        return 2
    if c > 1.2:
        return 1
    return 0


def hemo_score(plt):
    if not np.isfinite(plt):
        return np.nan
    if plt <= 50:
        return 4
    if plt <= 80:
        return 3
    if plt <= 100:
        return 2
    if plt <= 150:
        return 1
    return 0


def window_value(series_dict, h_center, half=3):
    vals = [v for h, v in series_dict.items()
            if h_center - half <= h <= h_center + half and np.isfinite(v)]
    return float(np.median(vals)) if vals else np.nan


def main():
    print("loading inputs ...", flush=True)
    hourly = pd.read_csv(f"{DATA}/hourly_features.csv")
    gcs_h = pd.read_csv(f"{DATA}/hourly_gcs.csv")
    vaso = pd.read_csv(f"{DATA}/vaso_intervals.csv")
    rrt = pd.read_csv(f"{DATA}/rrt_intervals.csv")
    vent = pd.read_csv(f"{DATA}/vent_intervals.csv")
    si = pd.read_csv(f"{DATA}/cohort_suspected_infection.csv")
    onset_map = dict(zip(si.admissionid.astype(int),
                         si.suspected_infection_ms.astype(float)))
    adm = pd.read_csv(f"{BASE}/admissions.csv").set_index("admissionid")

    # FiO2 fractions -> %
    if "fio2" in hourly:
        hourly.loc[hourly.fio2 <= 1.0, "fio2"] *= 100.0

    by_adm = {k: g.set_index("hr") for k, g in
              hourly.groupby("admissionid")}
    gcs_by_adm = {k: g.set_index("hr").gcs for k, g in
                  gcs_h.groupby("admissionid")}
    vaso_by_adm = {k: g for k, g in vaso.groupby("admissionid")}
    rrt_by_adm = {k: g for k, g in rrt.groupby("admissionid")}
    vent_by_adm = {k: g for k, g in vent.groupby("admissionid")}

    hours = np.arange(0, 49)
    rows = []
    ids = sorted(by_adm.keys())
    for i, aid in enumerate(ids):
        onset = onset_map.get(aid)
        if onset is None:
            continue
        g = by_adm[aid]

        def col(var):
            if var not in g.columns:
                return np.full(49, np.nan)
            s = g[var].reindex(hours)
            return s.values.astype(float)

        map_s = col("map")
        map_ni = col("map_nibp")
        with np.errstate(invalid="ignore"):
            mapmin = np.fmin(np.where(np.isfinite(map_s), map_s, np.inf),
                             np.where(np.isfinite(map_ni), map_ni, np.inf))
        mapmin = np.where(np.isfinite(mapmin), mapmin, np.nan)

        po2, fio2 = col("po2"), col("fio2")
        with np.errstate(divide="ignore", invalid="ignore"):
            pf = np.where(np.isfinite(po2) & np.isfinite(fio2) & (fio2 > 0),
                          po2 / (fio2 / 100.0), np.nan)

        adv = np.zeros(49, dtype=bool)
        v = vent_by_adm.get(aid)
        if v is not None:
            for _, r in v.iterrows():
                a = int(np.floor((r["start"] - onset) / HOUR))
                b = int(np.ceil((r["stop"] - onset) / HOUR))
                adv[max(0, a):min(49, b + 1)] = True

        ne = np.zeros(49)
        dopa = np.zeros(49)
        other = np.zeros(49, dtype=bool)
        vv = vaso_by_adm.get(aid)
        if vv is not None:
            for _, r in vv.iterrows():
                h = int(round((r["mid_ms"] - onset) / HOUR))
                if 0 <= h < 49:
                    if r["agent"] in ("norepi", "epi"):
                        ne[h] = max(ne[h], r["ugkgmin"])
                    elif r["agent"] == "dopamine":
                        dopa[h] = max(dopa[h], r["ugkgmin"])
                    else:
                        other[h] = True

        rrt_a = np.zeros(49, dtype=bool)
        rr = rrt_by_adm.get(aid)
        if rr is not None:
            for _, r in rr.iterrows():
                a = int(np.floor((r["start"] - onset) / HOUR))
                b = int(np.ceil((r["stop"] - onset) / HOUR))
                rrt_a[max(0, a):min(49, b + 1)] = True

        gcs = np.full(49, np.nan)
        gs = gcs_by_adm.get(aid)
        if gs is not None:
            vals = gs.reindex(hours).values.astype(float)
            last = np.nan
            for h in range(49):
                if np.isfinite(vals[h]):
                    last = vals[h]
                    gcs[h] = vals[h]
                elif np.isfinite(last):
                    gcs[h] = last

        cr, bili, plt_ = col("creatinine"), col("bilirubin"), \
            col("platelets")

        subs = np.full((49, 6), np.nan)
        for h in range(49):
            subs[h] = [brain_score(gcs[h]), resp_score(pf[h], adv[h]),
                       cvs_score(mapmin[h], ne[h], other[h], dopa[h]),
                       liver_score(bili[h]), kidney_score(cr[h], rrt_a[h]),
                       hemo_score(plt_[h])]
        sdf = pd.DataFrame(subs, columns=["brain", "respiration",
                                          "cardiovascular", "liver",
                                          "kidney", "hemostasis"])
        roll = sdf.rolling(24, min_periods=1).max().fillna(0)
        sofa24 = roll.sum(axis=1).values

        onset_idx = np.where(sofa24[:25] >= 2)[0]
        if len(onset_idx) == 0:
            continue
        shift = int(onset_idx[0])

        def wval(var, hc, half=3):
            return window_value(col_dict(var), hc + shift, half)

        def col_dict(var):
            if var not in g.columns:
                return {}
            return g[var].dropna().to_dict()

        ws = {h: {
            "sofa2": float(sofa24[min(h + shift, 48)]),
            "lactate": wval("lactate", h),
            "hr": wval("hr_bpm", h),
            "map": wval("map", h),
            "rr": wval("rr_br", h)} for h in [0, 6, 12, 24, 48]}

        H = [0, 6, 12, 24, 48]
        sofa_s = [ws[h]["sofa2"] for h in H]
        lac_s = [ws[h]["lactate"] for h in H]
        hr_s = [ws[h]["hr"] for h in H]
        map_s5 = [ws[h]["map"] for h in H]
        rr_s = [ws[h]["rr"] for h in H]

        hr_sd, rr_sd = sd(hr_s), sd(rr_s)
        sofa_delta = sofa_s[-1] - sofa_s[0]
        lac_clear = ((lac_s[0] - lac_s[-1]) / (lac_s[0] + 1e-5)
                     if np.isfinite(lac_s[0]) and lac_s[0] > 0
                     and np.isfinite(lac_s[-1]) else np.nan)

        imp = list(lac_s)
        hrv_impact = (10.0 - hr_sd) * HRV_FACTOR
        rrv_impact = rr_sd * RRV_FACTOR
        for j in (1, 2):
            if not np.isfinite(imp[j]) and np.isfinite(lac_s[0]):
                imp[j] = (lac_s[0] * BASE_WEIGHT
                          + sofa_delta * SOFA_IMPACT
                          + hrv_impact + rrv_impact)

        is_det = ((sofa_delta >= 2.0) or
                  (np.isfinite(lac_s[-1]) and lac_s[-1] > 2.0
                   and np.isfinite(lac_clear) and lac_clear < 0.10) or
                  (hr_sd < 10.0 and sofa_delta > 0))

        a = adm.loc[aid] if aid in adm.index else None
        death_ms = a["dateofdeath"] if a is not None else np.nan
        mort28 = bool(np.isfinite(death_ms) and death_ms >= 0
                      and death_ms <= 28 * 24 * HOUR)

        ne0 = ne48 = 0.0
        if vv is not None:
            onset2 = onset + shift * HOUR
            for _, r in vv.iterrows():
                if r["agent"] in ("norepi", "epi"):
                    dh = (r["mid_ms"] - onset2) / HOUR
                    if -3 <= dh <= 3:
                        ne0 = max(ne0, r["ugkgmin"])
                    elif 0 < dh <= 48:
                        ne48 = max(ne48, r["ugkgmin"])
        escalation = bool((ne48 >= 0.1) and (ne48 >= ne0 + 0.05))
        ref = mort28 or escalation

        rows.append({
            "admissionid": aid, "onset_shift_h": shift,
            "sofa0": sofa_s[0], "sofa48": sofa_s[-1],
            "sofa_delta": sofa_delta,
            "lac0": lac_s[0], "lac6": lac_s[1], "lac12": lac_s[2],
            "lac24": lac_s[3], "lac48": lac_s[4],
            "lac6_imp": imp[1], "lac12_imp": imp[2],
            "lac_clearance": lac_clear,
            "hr_sd": hr_sd, "rr_sd": rr_sd,
            "alert_red": bool(is_det),
            "mort28": mort28, "vaso_escalation": escalation,
            "deterioration_ref": ref,
            "agegroup": a["agegroup"] if a is not None else "",
            "gender": a["gender"] if a is not None else "",
            "weightgroup": a["weightgroup"] if a is not None else "",
            "los_days": a["lengthofstay"] if a is not None else np.nan,
        })
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(ids)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{DATA}/validation_patient_level.csv", index=False)
    print(f"analytic cohort: {len(df)}", flush=True)

    from sklearn.metrics import (cohen_kappa_score, confusion_matrix,
                                 roc_auc_score, average_precision_score,
                                 mean_squared_error, mean_absolute_error,
                                 r2_score)

    out = {}
    # ---- A. imputation fidelity (complete-case masking)
    cc = df.dropna(subset=["lac0", "lac6", "lac12", "lac24", "lac48"])
    imp_metrics = {"n_complete_cases": int(len(cc))}
    if len(cc) >= 30:
        y_true = np.concatenate([cc.lac6.values, cc.lac12.values])
        preds = {}
        hrv = (10.0 - cc.hr_sd) * HRV_FACTOR
        rrv = cc.rr_sd * RRV_FACTOR
        eng = (cc.lac0 * BASE_WEIGHT + cc.sofa_delta * SOFA_IMPACT
               + hrv + rrv).values
        lin6 = (cc.lac0 + (cc.lac24 - cc.lac0) * 6 / 24.0).values
        lin12 = (cc.lac0 + (cc.lac24 - cc.lac0) * 12 / 24.0).values
        meanimp = cc[["lac0", "lac24", "lac48"]].mean(axis=1).values
        preds["engine"] = np.concatenate([eng, eng])
        preds["linear"] = np.concatenate([lin6, lin12])
        preds["mean"] = np.concatenate([meanimp, meanimp])
        for mode, y_pred in preds.items():
            imp_metrics[mode] = {
                "MSE": float(mean_squared_error(y_true, y_pred)),
                "MAE": float(mean_absolute_error(y_true, y_pred)),
                "R2": float(r2_score(y_true, y_pred)),
            }
    out["imputation_complete_cases"] = imp_metrics
    print("imputation:", imp_metrics, flush=True)

    # ---- B. alert performance vs deterioration reference
    y = df.deterioration_ref.astype(int)
    pred = df.alert_red.astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    n = len(df)
    score = (df.sofa_delta + (df.lac48.fillna(0) > 2).astype(int)
             - df.lac_clearance.fillna(0))
    out["alert_performance"] = {
        "n": int(n), "tp": int(tp), "fp": int(fp), "tn": int(tn),
        "fn": int(fn),
        "accuracy": float((tp + tn) / n),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "cohens_kappa": float(cohen_kappa_score(y, pred)),
        "AUROC": float(roc_auc_score(y, score)),
        "AUPRC": float(average_precision_score(y, pred)),
        "prevalence": float(y.mean()),
        "mort28_rate": float(df.mort28.mean()),
        "escalation_rate": float(df.vaso_escalation.mean()),
    }
    print("alert:", out["alert_performance"], flush=True)

    # ---- C. reclassification of indeterminate cases
    ind = df[~np.isfinite(df.lac6) | ~np.isfinite(df.lac12)]
    out["reclassification"] = {
        "indeterminate_n": int(len(ind)),
        "indeterminate_share_of_cohort": float(len(ind) / len(df)),
        "reclassified_red_n": int(ind.alert_red.sum()),
        "reclassified_red_share": float(ind.alert_red.mean())
        if len(ind) else None,
    }
    print("reclassification:", out["reclassification"], flush=True)

    out["baseline"] = {
        "n": int(len(df)),
        "male_share": float((df.gender == "Man").mean()),
        "mort28": float(df.mort28.mean()),
        "sofa0_median": float(df.sofa0.median()),
        "sofa0_iqr": [float(df.sofa0.quantile(.25)),
                      float(df.sofa0.quantile(.75))],
        "lac0_median": float(df.lac0.median()),
        "los_median": float(df.los_days.median()),
        "agegroup_mode": (df.agegroup.mode().iloc[0] if len(df) else ""),
    }

    with open(f"{DATA}/validation_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved validation_metrics.json + validation_patient_level.csv",
          flush=True)


if __name__ == "__main__":
    main()
