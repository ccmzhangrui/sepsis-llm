"""Create a local landmark input from AmsterdamUMCdb hourly and source tables."""

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "amsterdam_validation"))
from run_cdss_validation import (
    brain_score,
    cvs_score,
    hemo_score,
    kidney_score,
    liver_score,
    resp_score,
)
from validate import LANDMARK

HOUR = 3_600_000
CULTURES = {
    8097,
    9189,
    9194,
    9192,
    9193,
    9190,
    8418,
    9200,
    9191,
    9203,
    9195,
    9198,
    9197,
    19663,
    19664,
}


def groups(path):
    return dict(tuple(pd.read_csv(path).groupby("admissionid", sort=False)))


def hourly_series(table, key, name):
    if key not in table:
        return np.full(LANDMARK, np.nan)
    return (
        table[key].set_index("hr")[name].reindex(range(LANDMARK)).to_numpy(dtype=float)
    )


def confirmed_hour(cultures, abx, origin):
    """Find the first matching culture that confirms this specific infection anchor."""
    if abx == origin:
        after = [c for c in cultures if origin <= c <= origin + 24 * HOUR]
        return (min(after) - origin) / HOUR if after else np.nan
    if origin in cultures and origin <= abx <= origin + 72 * HOUR:
        return (abx - origin) / HOUR
    return np.nan


def overlap_bins(start, stop, origin):
    if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
        return slice(0, 0)
    first = np.clip(int(np.floor((start - origin) / HOUR)), 0, LANDMARK)
    after = np.clip(int(np.ceil((stop - origin) / HOUR)), 0, LANDMARK)
    return slice(first, after)


def support(tables, stay, origin):
    vent, rrt, drugs = tables
    respiratory, renal, other = np.zeros((3, LANDMARK), dtype=bool)
    ne, dopamine = np.zeros((2, LANDMARK))
    for table, target in ((vent, respiratory), (rrt, renal)):
        for row in ([] if stay not in table else table[stay].itertuples(index=False)):
            target[overlap_bins(row.start, row.stop, origin)] = True
    for row in ([] if stay not in drugs else drugs[stay].itertuples(index=False)):
        bins = overlap_bins(row.start, row.stop, origin)
        if row.agent in ("norepi", "epi") and np.isfinite(row.ugkgmin):
            ne[bins] = np.maximum(ne[bins], row.ugkgmin)
        elif row.agent == "dopamine" and np.isfinite(row.ugkgmin):
            dopamine[bins] = np.maximum(dopamine[bins], row.ugkgmin)
        elif row.agent not in ("norepi", "epi", "dopamine"):
            other[bins] = True
    return respiratory, renal, other, ne, dopamine


def hr_summary(heart, baseline):
    targets = sorted(
        {
            hour
            for hour in (
                baseline,
                baseline + 6,
                baseline + 12,
                baseline + 24,
                LANDMARK - 1,
            )
            if hour < LANDMARK
        }
    )
    values = defaultdict(list)
    for hour, value in enumerate(heart):
        if hour < baseline or not 20 <= value <= 300:
            continue
        target = min(targets, key=lambda center: abs(center - hour))
        if abs(target - hour) <= 3:
            values[target].append(value)
    observed = [np.median(items) for items in values.values()]
    return (float(np.std(observed)) if len(observed) >= 2 else np.nan, len(observed))


def lactate_in(series, first, last):
    values = series[max(0, first) : last + 1]
    values = values[np.isfinite(values) & (values > 0)]
    return np.median(values) if values.size else np.nan


def records(directory, admissions, orders):
    hourly, gcs, vent, rrt, drugs = [
        groups(directory / f"{name}.csv")
        for name in (
            "hourly_features",
            "hourly_gcs",
            "vent_intervals",
            "rrt_intervals",
            "vaso_intervals",
        )
    ]
    culture = pd.read_csv(
        orders, usecols=["admissionid", "itemid", "registeredat"], encoding="latin-1"
    )
    cultures = (
        culture.loc[culture.itemid.isin(CULTURES)]
        .groupby("admissionid")
        .registeredat.agg(list)
        .to_dict()
    )
    adm = pd.read_csv(
        admissions,
        usecols=[
            "admissionid",
            "patientid",
            "admittedat",
            "dischargedat",
            "dateofdeath",
        ],
    )
    infection = pd.read_csv(directory / "cohort_suspected_infection.csv").merge(
        adm, on="admissionid", validate="one_to_one"
    )
    for row in infection.itertuples(index=False):
        stay, origin = row.admissionid, row.suspected_infection_ms
        latest_evidence = confirmed_hour(cultures.get(stay, ()), row.abx_start, origin)
        if not np.isfinite(latest_evidence):
            continue
        respiratory, renal, other, ne, dopamine = support(
            (vent, rrt, drugs), stay, origin
        )
        g = (
            hourly[stay].set_index("hr").reindex(range(LANDMARK))
            if stay in hourly
            else None
        )
        if g is None:
            continue
        fio2 = g.fio2.where(g.fio2 > 1, g.fio2 * 100).where(lambda x: x > 0)
        pf = (g.po2 / (fio2 / 100)).to_numpy()
        pressure = np.fmin(g["map"].to_numpy(), g.map_nibp.to_numpy())
        neuro = hourly_series(gcs, stay, "gcs")
        lvr, cr, platelets = (
            g.bilirubin.to_numpy(),
            g.creatinine.to_numpy(),
            g.platelets.to_numpy(),
        )
        subs = (
            pd.DataFrame(
                [
                    [
                        brain_score(neuro[h]),
                        resp_score(pf[h], respiratory[h]),
                        cvs_score(pressure[h], ne[h], other[h], dopamine[h]),
                        liver_score(lvr[h]),
                        kidney_score(cr[h], renal[h]),
                        hemo_score(platelets[h]),
                    ]
                    for h in range(LANDMARK)
                ]
            )
            .rolling(24, min_periods=1)
            .max()
        )
        low = subs.fillna(0).sum(axis=1)
        high = low + 4 * subs.isna().sum(axis=1)
        candidates = np.flatnonzero(low.iloc[:25].ge(2).to_numpy())
        if not candidates.size:
            continue
        baseline = int(candidates[0])
        matched = subs.iloc[baseline].notna() & subs.iloc[-1].notna()
        admit = (row.admittedat - origin) / HOUR
        lactate = g.lactate.to_numpy(dtype=float)
        hr_sd, hr_windows = hr_summary(g.hr_bpm.to_numpy(dtype=float), baseline)
        yield {
            "stay_id": stay,
            "patient_id": row.patientid,
            "icu_admit_hour": admit,
            "icu_discharge_hour": (row.dischargedat - origin) / HOUR,
            "infection_confirmed_hour": latest_evidence,
            "baseline_end_hour": baseline + 1,
            "features_through_hour": LANDMARK,
            "death_hour": (row.dateofdeath - origin) / HOUR,
            "vital_followup_hour": admit + 28 * 24,
            "sofa_base_low": low.iloc[baseline],
            "sofa_base_high": high.iloc[baseline],
            "sofa_late_low": low.iloc[-1],
            "sofa_late_high": high.iloc[-1],
            "matched_base": subs.iloc[baseline][matched].sum(),
            "matched_late": subs.iloc[-1][matched].sum(),
            "matched_organs": int(matched.sum()),
            "lac_base": lactate_in(lactate, baseline - 3, baseline),
            "lac_late": lactate_in(lactate, LANDMARK - 3, LANDMARK - 1),
            "hr_sd": hr_sd,
            "hr_windows": hr_windows,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hourly-dir", type=Path, required=True)
    parser.add_argument("--admissions", type=Path, required=True)
    parser.add_argument("--culture-orders", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Local-only output with restricted stay and patient keys",
    )
    args = parser.parse_args()
    pd.DataFrame.from_records(
        records(args.hourly_dir, args.admissions, args.culture_orders)
    ).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
