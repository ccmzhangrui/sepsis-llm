#!/usr/bin/env python3
"""Build the suspected-infection backbone for the AmsterdamUMCdb sepsis-3
validation cohort (mirrors the MIMIC-IV/eICU development pipeline).

Definitions (Seymour/Sepsis-3 convention):
  - antibiotic event: drugitems row whose ordercategory contains
    'Antimicrob' (Injecties Antimicrobiele middelen); time = `start` (ms
    offset from ICU admission)
  - culture event: procedureorderitems row for a culture order
    (bloed/sputum/urine/liquor/wond/faeces/drain/ascites/cathetertip/keel/
    neus/rectum/perineum kweek); time = `registeredat`
  - suspected infection time: if antibiotics first, a culture must follow
    within 24 h; if culture first, antibiotics must follow within 72 h.
Output: cohort_suspected_infection.csv
"""
import pandas as pd
import numpy as np

BASE = "/Users/zhangrui/Documents/sofa2.0/AmsterdamUBD"
DATA = "/Users/zhangrui/WorkBuddy/2026-09-18-15-20-42/amsterdam_sepsis/data"

HOUR = 3600_000  # ms

CULTURE_ITEMIDS = {8097, 9189, 9194, 9192, 9193, 9190, 8418, 9200, 9191,
                   9203, 9195, 9202, 9198, 9197, 19663, 19664}

print("loading admissions ...", flush=True)
adm = pd.read_csv(f"{BASE}/admissions.csv")
print(f"  admissions: {len(adm)}")

# keep ICU admissions, first admission per patient
adm = adm.sort_values(["patientid", "admittedat"])
first_adm = adm.groupby("patientid", as_index=False).first()
print(f"  unique patients (first ICU admission): {len(first_adm)}")

print("loading antibiotic administrations ...", flush=True)
drg = pd.read_csv(f"{DATA}/drugitems_vaso_abx.csv",
                  usecols=["admissionid", "itemid", "item", "ordercategory",
                           "start", "stop", "dose", "doseunit",
                           "doserateunit", "rate", "rateunit", "action"])
abx = drg[drg["ordercategory"].str.contains("Antimicrob", case=False,
                                            na=False)]
print(f"  antimicrobial rows: {len(abx)}")
first_abx = abx.groupby("admissionid")["start"].min().rename("abx_start")

print("loading culture orders ...", flush=True)
proc = pd.read_csv(f"{BASE}/procedureorderitems.csv", encoding="latin-1")
cult = proc[proc["itemid"].isin(CULTURE_ITEMIDS)]
print(f"  culture orders: {len(cult)}")

# per-admission earliest culture within a window helper
cult_first = cult.groupby("admissionid")["registeredat"].min() \
    .rename("culture_first")
cult_all = cult.groupby("admissionid")["registeredat"].apply(list)

si = pd.concat([first_abx, cult_first], axis=1).dropna(
    subset=["abx_start"]).reset_index()

# Seymour rule with per-admission earliest culture: if the earliest culture
# precedes abx by >72h or follows abx by >24h, try the earliest culture
# AFTER abx start (within 24h); else culture before abx (within 72h).
def suspected_time(row):
    a = row["abx_start"]
    cs = cult_all.get(row["admissionid"], [])
    before = [c for c in cs if c <= a and a - c <= 72 * HOUR]
    after = [c for c in cs if c > a and c - a <= 24 * HOUR]
    if before and after:
        return min(a, min(before))
    if before:
        return min(before)  # culture first -> onset = culture time
    if after:
        return a            # abx first -> onset = abx time
    return np.nan

print("deriving suspected infection times ...", flush=True)
si["suspected_infection_ms"] = si.apply(suspected_time, axis=1)
si = si.dropna(subset=["suspected_infection_ms"])
si = si[si["suspected_infection_ms"] >= -24 * HOUR]  # allow 24h pre-ICU
print(f"  admissions with suspected infection: {len(si)}")

si.to_csv(f"{DATA}/cohort_suspected_infection.csv", index=False)
first_adm.to_csv(f"{DATA}/admissions_first.csv", index=False)
print("saved cohort_suspected_infection.csv / admissions_first.csv",
      flush=True)
