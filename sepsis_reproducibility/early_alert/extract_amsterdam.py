"""Prepare the 24-hour landmark study inputs from AmsterdamUMCdb source extracts."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "landmark"))
from prepare_amsterdam import (
    CULTURES,
    HOUR,
    brain_score,
    confirmed_hour,
    cvs_score,
    groups,
    hemo_score,
    kidney_score,
    liver_score,
    resp_score,
    support,
)

LANDMARK = 24
HORIZON = 7 * 24


def recent_median(series, low, high):
    observed = series.tail(6).where(series.between(low, high)).dropna()
    return float(observed.median()) if len(observed) >= 2 else np.nan


def extract_features(hourly, gcs, tables, stay, origin):
    hours = range(LANDMARK)
    data = hourly.set_index("hr").reindex(hours)
    fio2 = data.fio2.where(data.fio2 > 1, data.fio2 * 100).where(lambda x: x > 0)
    pf = (data.po2 / (fio2 / 100)).to_numpy()
    map_values = pd.Series(np.fmin(data["map"], data.map_nibp), index=data.index)
    resp, renal, other, ne, dopamine = support(tables, stay, origin)
    neuro = (
        gcs.set_index("hr").gcs.reindex(hours).to_numpy()
        if gcs is not None
        else np.full(LANDMARK, np.nan)
    )
    subscores = np.array(
        [
            [
                brain_score(neuro[h]),
                resp_score(pf[h], resp[h]),
                cvs_score(map_values.iat[h], ne[h], other[h], dopamine[h]),
                liver_score(data.bilirubin.iat[h]),
                kidney_score(data.creatinine.iat[h], renal[h]),
                hemo_score(data.platelets.iat[h]),
            ]
            for h in hours
        ]
    )
    highest = pd.DataFrame(subscores).max(axis=0, skipna=True)
    lac = data.lactate.tail(12).where(data.lactate.between(0.1, 40)).dropna()
    return {
        "sofa_observed": highest.sum(min_count=1),
        "sofa_components": int(highest.notna().sum()),
        "map_median": recent_median(map_values, 20, 180),
        "hr_median": recent_median(data.hr_bpm, 20, 300),
        "rr_median": recent_median(data.rr_br, 3, 100),
        "lactate_latest": float(lac.iloc[-1]) if len(lac) else np.nan,
    }


def make_cohort(directory, admissions_path, culture_orders_path):
    hourly, gcs, vent, rrt, drugs = [
        groups(directory / f"{table}.csv")
        for table in (
            "hourly_features",
            "hourly_gcs",
            "vent_intervals",
            "rrt_intervals",
            "vaso_intervals",
        )
    ]
    culture = pd.read_csv(
        culture_orders_path,
        usecols=["admissionid", "itemid", "registeredat"],
        encoding="latin-1",
    )
    culture = (
        culture.loc[culture.itemid.isin(CULTURES)]
        .groupby("admissionid")
        .registeredat.agg(list)
    )
    admissions = pd.read_csv(
        admissions_path,
        usecols=[
            "admissionid",
            "patientid",
            "admissionyeargroup",
            "admittedat",
            "dischargedat",
            "dateofdeath",
        ],
    )
    infection = pd.read_csv(directory / "cohort_suspected_infection.csv").merge(
        admissions, on="admissionid", validate="one_to_one"
    )
    for row in infection.itertuples(index=False):
        stay, origin = row.admissionid, row.suspected_infection_ms
        confirmation = confirmed_hour(culture.get(stay, []), row.abx_start, origin)
        admit = (row.admittedat - origin) / HOUR
        discharge = (row.dischargedat - origin) / HOUR
        death = (row.dateofdeath - origin) / HOUR
        if (
            not np.isfinite(confirmation)
            or confirmation > LANDMARK
            or admit > LANDMARK
            or discharge <= LANDMARK
            or death <= LANDMARK
        ):
            continue
        if stay in hourly:
            features = extract_features(
                hourly[stay], gcs.get(stay), (vent, rrt, drugs), stay, origin
            )
        else:
            features = {
                "sofa_observed": np.nan,
                "sofa_components": 0,
                "map_median": np.nan,
                "hr_median": np.nan,
                "rr_median": np.nan,
                "lactate_latest": np.nan,
            }
        death_7d = LANDMARK < death <= LANDMARK + HORIZON
        yield {
            "stay_id": stay,
            "patient_id": row.patientid,
            "era": row.admissionyeargroup,
            "snapshot_hour": LANDMARK,
            "death_7d": int(death_7d),
            "death_72h": int(LANDMARK < death <= LANDMARK + 72),
            "hours_to_recorded_7d_death": death - LANDMARK if death_7d else np.nan,
            **features,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hourly-dir", type=Path, required=True)
    parser.add_argument("--admissions", type=Path, required=True)
    parser.add_argument("--culture-orders", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = make_cohort(args.hourly_dir, args.admissions, args.culture_orders)
    pd.DataFrame.from_records(records).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
