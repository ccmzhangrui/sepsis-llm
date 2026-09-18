#!/usr/bin/env python3
"""Build hourly feature matrices + SOFA-2 subscores inputs for the
AmsterdamUMCdb sepsis-3 validation cohort.

Memory-safe: chunked read of numeric_cohort.csv, per-variable numpy
accumulation, hourly aggregation.

Simplifications vs the full SOFA-2 footnote implementation (documented in
the manuscript supplement):
  - footnote a (motor-only GCS when intubated): verbal=8 -> missing, LOCF
  - footnote c (pre-sedation GCS): LOCF of last clean GCS, no explicit
    sedation-window masking
  - footnote e (delirium drug/CAM-ICU): not implemented
  - footnote f (SpO2/FiO2 fallback): not implemented (P/F only)
  - footnotes i/n (ECMO/MCS): not implemented (ECMO n=2 rows in database)
  - urine-output pathway of the kidney domain: not implemented
    (creatinine + RRT only)
"""
import pickle

import numpy as np
import pandas as pd

DATA = "/Users/zhangrui/WorkBuddy/2026-09-18-15-20-42/amsterdam_sepsis/data"
BASE = "/Users/zhangrui/Documents/sofa2.0/AmsterdamUBD"
HOUR = 3600_000.0

ITEM_MAP = {6640: "hr", 6642: "map", 6679: "map_nibp",
            8874: "rr", 12266: "rr", 8873: "rr", 7726: "rr",
            6709: "spo2",
            9996: "po2", 7433: "po2", 9990: "pco2", 12310: "ph",
            6848: "ph", 10053: "lactate", 12311: "o2sat",
            6699: "fio2", 13076: "fio2", 16629: "fio2",
            9941: "creatinine", 6836: "creatinine", 9945: "bilirubin",
            9964: "platelets", 6797: "platelets", 9927: "potassium",
            6835: "potassium", 9943: "urea", 9947: "glucose",
            10079: "crp", 9960: "hb"}

AGG = {"hr": "median", "map": "min", "map_nibp": "min", "rr": "median",
       "spo2": "min", "po2": "min", "pco2": "median", "ph": "min",
       "lactate": "median", "fio2": "max", "creatinine": "max",
       "bilirubin": "max", "platelets": "min", "potassium": "max",
       "urea": "max", "glucose": "median", "crp": "max", "hb": "min",
       "o2sat": "min"}

GCS_EYE, GCS_MOTOR, GCS_VERBAL = 6732, 6734, 6735
EYE_MAP = {1: 4, 2: 3, 3: 2, 4: 1}
MOTOR_MAP = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}
VERBAL_MAP = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1, 8: np.nan}

VASO_ITEMS = {"Noradrenaline (Norepinefrine)": "norepi",
              "Adrenaline (Epinefrine)": "epi",
              "Dopamine (Inotropin)": "dopamine",
              "Dobutamine (Dobutrex)": "dobutamine",
              "Fenylefrine (phenylephrine)": "phenylephrine"}

RRT_ITEMIDS = {12465, 16363}
ADVRESP_ITEMIDS = {9328, 10740, 9671}

WEIGHT_MID = {"18-39": 30, "40-49": 45, "50-59": 55, "60-69": 65,
              "70-79": 75, "80-89": 85, "90+": 95, "80+": 85}

HR_MIN, HR_MAX = -2, 50

print("loading cohort + admissions ...", flush=True)
si = pd.read_csv(f"{DATA}/cohort_suspected_infection.csv")
onset_map = dict(zip(si.admissionid.astype(int),
                     si.suspected_infection_ms.astype(float)))
adm = pd.read_csv(f"{DATA}/admissions_first.csv")
cohort_ids = set(onset_map)

print("chunked read of numeric_cohort ...", flush=True)
acc = {v: [] for v in AGG}
n_rows = 0
usecols = ["admissionid", "itemid", "item", "value", "measuredat"]
for chunk in pd.read_csv(f"{DATA}/numeric_cohort.csv", usecols=usecols,
                         chunksize=5_000_000):
    n_rows += len(chunk)
    chunk["var"] = chunk.itemid.map(ITEM_MAP)
    chunk = chunk.dropna(subset=["var"])
    chunk["value_num"] = pd.to_numeric(chunk.value, errors="coerce")
    chunk = chunk.dropna(subset=["value_num"])
    chunk["onset"] = chunk.admissionid.map(onset_map)
    chunk = chunk.dropna(subset=["onset"])
    chunk["hr"] = np.floor((chunk.measuredat - chunk.onset) / HOUR) \
        .astype(int)
    chunk = chunk[(chunk.hr >= HR_MIN) & (chunk.hr <= HR_MAX)]
    for var, sub in chunk.groupby("var"):
        acc[var].append(sub[["admissionid", "hr", "value_num"]]
                        .to_numpy())
    if n_rows % 20_000_000 < 5_000_000:
        print(f"  {n_rows/1e6:.0f}M rows ...", flush=True)
print(f"  total {n_rows} rows read", flush=True)

RENAME = {"hr": "hr_bpm", "rr": "rr_br"}
frames = {}
for var, parts in acc.items():
    if not parts:
        continue
    arr = np.concatenate(parts)
    df = pd.DataFrame(arr, columns=["admissionid", "hr", "value"])
    agg = AGG[var]
    g = df.groupby(["admissionid", "hr"]).value
    s = getattr(g, agg)()
    frames[RENAME.get(var, var)] = s
    print(f"  {var}: {len(df)} rows -> {len(s)} hourly cells ({agg})",
          flush=True)

hourly = pd.DataFrame(frames)
hourly.index.names = ["admissionid", "hr"]
hourly = hourly.reset_index()
hourly.to_csv(f"{DATA}/hourly_features.csv", index=False)
print(f"hourly matrix: {hourly.shape}", flush=True)

# FiO2 fractions -> %
print("loading GCS ...", flush=True)
gcs = pd.read_csv(f"{DATA}/listitems_gcs_exact.csv",
                  usecols=["admissionid", "itemid", "valueid", "measuredat"])
gcs = gcs[gcs.admissionid.isin(cohort_ids)]
gcs["score"] = np.where(gcs.itemid == GCS_EYE, gcs.valueid.map(EYE_MAP),
              np.where(gcs.itemid == GCS_MOTOR,
                       gcs.valueid.map(MOTOR_MAP),
                       gcs.valueid.map(VERBAL_MAP)))
gcs_w = gcs.pivot_table(index=["admissionid", "measuredat"],
                        columns="itemid", values="score",
                        aggfunc="min")
gcs_w.columns = ["eye", "motor", "verbal"]
gcs_w = gcs_w.reset_index()
gcs_w["gcs"] = gcs_w[["eye", "motor", "verbal"]].sum(axis=1, min_count=3)
gcs_w = gcs_w.dropna(subset=["gcs"])
gcs_w["onset"] = gcs_w.admissionid.map(onset_map)
gcs_w["hr"] = np.floor((gcs_w.measuredat - gcs_w.onset) / HOUR).astype(int)
gcs_w = gcs_w[(gcs_w.hr >= HR_MIN) & (gcs_w.hr <= HR_MAX)]
gcs_h = gcs_w.groupby(["admissionid", "hr"]).gcs.min().reset_index()
gcs_h.to_csv(f"{DATA}/hourly_gcs.csv", index=False)
print(f"hourly gcs: {gcs_h.shape}", flush=True)

print("loading vasopressor infusions ...", flush=True)
drg = pd.read_csv(f"{DATA}/drugitems_vaso_abx.csv",
                  usecols=["admissionid", "item", "start", "stop", "dose",
                           "doseunit", "doserateunit"])
drg = drg[drg.admissionid.isin(cohort_ids)]
drg = drg[drg.item.isin(VASO_ITEMS)]
adm_w = adm.set_index("admissionid")["weightgroup"].map(WEIGHT_MID) \
    .fillna(75.0)
drg["agent"] = drg.item.map(VASO_ITEMS)
drg["w"] = drg.admissionid.map(adm_w).fillna(75.0)
drg["per_hour"] = np.where(drg.doserateunit.str.contains("uur", na=False),
                           drg.dose, drg.dose * 60.0)
drg["ugkgmin"] = drg.per_hour * 1000.0 / (drg.w * 60.0)
drg["mid_ms"] = (drg.start + drg.stop) / 2.0
vaso = drg[["admissionid", "agent", "ugkgmin", "start", "stop", "mid_ms"]]
vaso = vaso[np.isfinite(vaso.ugkgmin) & (vaso.ugkgmin > 0)]
vaso.to_csv(f"{DATA}/vaso_intervals.csv", index=False)
print(f"vaso intervals: {len(vaso)}", flush=True)

print("loading process items ...", flush=True)
proc = pd.read_csv(f"{BASE}/processitems.csv", encoding="latin-1")
proc = proc[proc.admissionid.isin(cohort_ids)]
rrt = proc[proc.itemid.isin(RRT_ITEMIDS)][["admissionid", "start", "stop"]]
rrt.to_csv(f"{DATA}/rrt_intervals.csv", index=False)
vent = proc[proc.itemid.isin(ADVRESP_ITEMIDS)][["admissionid", "start",
                                                "stop"]]
vent.to_csv(f"{DATA}/vent_intervals.csv", index=False)
print(f"rrt: {len(rrt)}, vent: {len(vent)}", flush=True)
print("ALL DONE", flush=True)
