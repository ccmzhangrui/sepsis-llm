"""Exploratory three-cohort application of the frozen alert and lactate reconstruction code."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--work", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--frozen-source", type=Path, required=True)
parser.add_argument("--mimic-training-parent", type=Path, required=True)
parser.add_argument("--models-dir", type=Path, required=True)
parser.add_argument("--eicu-dir", type=Path, required=True)
parser.add_argument("--mimic-sofa", type=Path, required=True)
parser.add_argument("--ruijin-daily", type=Path, required=True)
parser.add_argument(
    "--ruijin-selection", type=Path, help="Optional local-only encounter manifest"
)
parser.add_argument(
    "--cohort", choices=["all", "eICU", "MIMIC-IV", "Ruijin"], default="all"
)
args = parser.parse_args()
W = args.work
spec = importlib.util.spec_from_file_location("frozen_replay", args.frozen_source)
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)
HOURS = [0, 6, 12, 24, 48]
LAC = ["lac0", "lac6", "lac12", "lac24", "lac48"]


def numeric(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def canon(series):
    return series.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)


def counts(series):
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}


def summarize_alert(df, patient, outcome):
    """Keep the exact alert calculation while marking unavailable rule inputs."""
    flags = frozen.rules(df)
    d = df.sofa48 - df.sofa0
    avail = pd.DataFrame(
        {
            "sofa_increase": d.notna(),
            "lactate_clearance": df.lac0.gt(0) & df.lac48.notna(),
            "heart_rate_sd_and_sofa": d.notna() & df.hr_sd.notna(),
        },
        index=df.index,
    )
    for col in flags:
        if (flags[col] & ~avail[col]).any():
            raise AssertionError("A rule fired without its required inputs")
    red = flags.any(axis=1)
    complete = avail.all(axis=1)
    known_negative = complete & ~red
    status = pd.Series("indeterminate", index=df.index)
    status.loc[red] = "red"
    status.loc[known_negative] = "no_red_complete"
    result = {
        "n": len(df),
        "unique_people": df[patient].nunique(),
        "all_three_evaluable": int(complete.sum()),
        "rules": {
            c: {"evaluable": int(avail[c].sum()), "triggered": int(flags[c].sum())}
            for c in flags
        },
        "status": counts(status),
    }
    for key, mask in [
        ("full_input_only", complete),
        ("evaluable_status", status.ne("indeterminate")),
        ("implementation_available_inputs", pd.Series(True, index=df.index)),
    ]:
        used = mask & df[outcome].notna()
        if not used.any():
            continue
        reference = df.loc[used, outcome].astype(bool)
        selected = red[used]
        result[key] = frozen.ratio(reference, selected, df.loc[used, patient])
        result[key]["n_with_outcome"] = int(used.sum())
    assert sum(result["status"].values()) == len(df)
    return result


def measure_windows(values, name, offset, value, ids):
    d = values.dropna(subset=[value]).copy()
    d["hour"] = np.floor(d[offset] / 60).astype(int)
    hourly = (
        d.groupby(["patientunitstayid", "hour"], sort=False)[value]
        .median()
        .rename("value")
        .reset_index()
    )
    return pd.DataFrame(
        {
            f"{name}{h}": hourly.loc[hourly.hour.between(h - 3, h + 3)]
            .groupby("patientunitstayid")
            .value.median()
            .reindex(ids)
            for h in HOURS
        },
        index=ids,
    )


def eval_vaso(p):
    cols = [
        "patientunitstayid",
        "infusionoffset",
        "drugname",
        "drugrate",
        "patientweight",
    ]
    raw = pd.read_csv(
        args.eicu_dir / "infusionDrug.csv", usecols=cols, low_memory=False
    )
    raw = raw.loc[
        raw.patientunitstayid.isin(p.index) & raw.infusionoffset.between(-180, 2880)
    ]
    name = raw.drugname.astype("string").str.lower()
    target = name.str.contains(
        r"\b(?:norepinephrine|levophed|epinephrine)\b", regex=True, na=False
    )
    raw = raw.loc[target].copy()
    name = name[target]
    rate = numeric(raw.drugrate)
    wt = numeric(raw.patientweight)
    raw["dose"] = np.select(
        [
            name.str.endswith("(mcg/kg/min)", na=False),
            name.str.endswith("(mcg/min)", na=False) & wt.between(25, 350),
        ],
        [rate, rate / wt],
        default=np.nan,
    )
    raw.loc[~raw.dose.between(0, 5), "dose"] = np.nan
    b = (
        raw[raw.infusionoffset.between(-180, 180)]
        .groupby("patientunitstayid")
        .dose.max()
    )
    later = raw[raw.infusionoffset.gt(180)].groupby("patientunitstayid").dose.max()
    b = b.reindex(p.index)
    later = later.reindex(p.index)
    involved = raw.groupby("patientunitstayid").dose.agg(["size", "count"])
    all_known = (
        involved["size"].eq(involved["count"]).reindex(p.index, fill_value=False)
    )
    evaluable = b.notna() & later.notna() & all_known
    return ((later >= 0.10) & ((later - b) >= 0.05)).astype(float).where(evaluable), {
        "n_with_explicit_unambiguous_baseline_and_followup_dose": int(evaluable.sum()),
        "n_any_target_drug": int(p.index.isin(raw.patientunitstayid).sum()),
        "n_target_drug_rows": len(raw),
        "n_standardized_drug_rows": int(raw.dose.notna().sum()),
    }


def single_benchmark(d, cols, kind):
    block = d.dropna(subset=cols)
    if len(block) < 2:
        return {"n": len(block)}
    actual = np.r_[block[cols[1]], block[cols[2]]]
    base = block[cols[0]]
    later = block[cols[3]]
    early = np.r_[base + 0.25 * (later - base), base + 0.5 * (later - base)]
    out = {
        "n": len(block),
        "n_values": len(actual),
        "linear_0_24h": frozen.metrics(actual, early),
    }
    if kind == "eicu":
        v1 = (
            0.85 * base
            + 0.20 * (block.sofa48 - block.sofa0)
            + 0.05 * (10 - block.hr_sd)
            + 0.03 * block.rr_sd
        )
        out["frozen_formula"] = frozen.metrics(actual, np.r_[v1, v1])
    return out


def booster_benchmark(block, label):
    if len(block) < 2:
        return {"n": len(block)}
    import xgboost as xgb

    predicted = []
    for t in [6, 12]:
        reg = xgb.Booster(params={"nthread": 4})
        reg.load_model(args.models_dir / f"engine_v2_lac{t}_anchorfree.json")
        cols = ["lac0", "hr_sd", "rr_sd", "age", "male", "vaso_esc"]
        predicted.append(reg.predict(xgb.DMatrix(block[cols], feature_names=cols)))
    obs = np.r_[block.lac6, block.lac12]
    return {
        "n": len(block),
        "n_values": len(obs),
        label: frozen.metrics(obs, np.r_[*predicted]),
    }


def eicu():
    p = pd.read_csv(W / "eicu_base.csv").set_index("patientunitstayid")
    for c in ["sofa0", "sofa48"]:
        p[c] = numeric(p[c]).where(numeric(p[c]).between(0, 24))
    p.loc[p.unitdischargeoffset < 0, "sofa0"] = np.nan
    p["hosp_dead"] = p.hospitaldischargestatus.map({"Expired": True, "Alive": False})
    p.loc[~p.hospitaldischargeoffset.ge(2880), "hosp_dead"] = np.nan
    p["in_hosp_death_by_day28"] = p.hosp_dead
    p.loc[
        p.hosp_dead.eq(True) & p.hospitaldischargeoffset.gt(28 * 1440),
        "in_hosp_death_by_day28",
    ] = False
    p["male"] = p.gender.map({"Male": 1, "Female": 0})
    p["cluster"] = p.uniquepid.astype("string")
    vit = pd.read_csv(W / "eicu_vitals.csv")
    la = pd.read_csv(W / "eicu_lactate.csv")
    la.labresult = numeric(la.labresult).where(numeric(la.labresult).gt(0))
    la = la[la.labmeasurenamesystem.astype("string").str.lower().eq("mmol/l")]
    p = p.join(measure_windows(la, "lac", "labresultoffset", "labresult", p.index))
    for field, label in [("hr", "Heart Rate"), ("rr", "Respiratory Rate")]:
        v = measure_windows(
            vit[vit.nursingchartcelltypevalname.eq(label)],
            field,
            "nursingchartoffset",
            "nursingchartvalue",
            p.index,
        )
        p[f"{field}_n"] = v.notna().sum(axis=1)
        p[f"{field}_sd"] = v.std(axis=1, ddof=0).where(v.notna().sum(axis=1) >= 2)
        p[f"{field}_sample_sd"] = v.std(axis=1, ddof=1)
    # For the 48-hour analysis retain only people observed at the ICU landmark.
    landmark = p.unitdischargeoffset.ge(2880) & p.cluster.notna()
    d = p.loc[landmark].copy()
    res = {
        "input_flow": json.loads((W / "eicu_flow.json").read_text()),
        "eligible_still_in_icu_at_48h": len(d),
        "eligible_hemodynamic_by_48h_availability": {
            c: int(d[c].notna().sum())
            for c in LAC + ["sofa0", "sofa48", "hr_sd", "rr_sd", "hosp_dead"]
        },
        "hospital_mortality_after_48h": summarize_alert(d, "cluster", "hosp_dead"),
        "in_hospital_death_48h_to_28days": summarize_alert(
            d, "cluster", "in_hosp_death_by_day28"
        ),
        "sofa0_at_least_two_sensitivity": summarize_alert(
            d[d.sofa0.ge(2)], "cluster", "hosp_dead"
        ),
        "masked_same_cases": single_benchmark(
            d.query("lac0>=0.3"), LAC + ["sofa0", "sofa48", "hr_sd", "rr_sd"], "eicu"
        ),
    }
    d["vaso_esc"], res["explicit_dose_coverage"] = eval_vaso(d)
    x = d.query("lac0>=0.3").copy()
    x["hr_sd"] = x.hr_sample_sd
    x["rr_sd"] = x.rr_sample_sd
    x = x.dropna(
        subset=["lac0", "lac6", "lac12", "hr_sd", "rr_sd", "age", "male", "vaso_esc"]
    )
    res["fallback_frozen_explicit_dose_subset"] = booster_benchmark(
        x, "frozen_fallback"
    )
    return res


LACS = "乳酸_Lactate(mmol/L)_"
HRS = "心率(次/min)_"
RRS = "呼吸频率(次/min)_"
NOREPI = "持续注射泵入(去甲肾上腺素)(ug/kg/min)_"
GRID = [f"Day{day}_{hour:02d}:00" for day in [1, 2] for hour in [0, 6, 12, 18]]


def mimic_features(d, historical=False):
    out = d[["stay_id", "subject_id", "age", "gender", "sepsis3"]].copy()
    for col, label in zip(
        ["lac0", "lac6", "lac12", "lac24", "lac_day2_12"],
        [GRID[x] for x in [0, 1, 2, 4, 6]],
    ):
        values = numeric(d[LACS + label])
        out[col] = values if historical else values.where(values.gt(0))
    for dest, pre in [("hr_sd", HRS), ("rr_sd", RRS)]:
        out[dest] = d[[pre + g for g in GRID]].apply(numeric).std(axis=1, ddof=1)
    baseline = numeric(d[NOREPI + GRID[0]]).fillna(0)
    maximum = d[[NOREPI + g for g in GRID[1:7]]].apply(numeric).max(axis=1).fillna(0)
    out["vaso_esc"] = ((maximum >= 0.1) & (maximum - baseline >= 0.05)).astype(float)
    out["male"] = (
        d.gender.eq("M").astype(float) if historical else d.gender.map({"M": 1, "F": 0})
    )
    out["age"] = numeric(out.age)
    return out


def mimic():
    workbook = pd.read_pickle(W / "mimic_source.pkl")
    allf = mimic_features(workbook)
    m = allf[allf.sepsis3.eq(True) & allf.age.ge(18)].copy()
    scores = pd.read_csv(
        args.mimic_sofa,
        usecols=["stay_id", "hourly_sofa2_total_hr_0", "hourly_sofa2_total_hr_48"],
    ).set_index("stay_id")
    s = scores.reindex(m.stay_id)
    measures = ["lac0", "lac6", "lac12", "lac24", "lac_day2_12"]
    features = ["hr_sd", "rr_sd", "age", "male", "vaso_esc"]
    m0 = m.query("lac0>=0.3")
    full = m0.dropna(subset=measures)
    source = pd.read_pickle(args.mimic_training_parent)
    tp = mimic_features(source, historical=True)
    train = (
        tp[tp.sepsis3.eq(True)].query("lac0>=0.3").dropna(subset=measures + features)
    )
    compatible = m0.dropna(subset=measures + features).copy()
    novel = compatible.loc[~compatible.subject_id.isin(train.subject_id)]
    obs = np.r_[full.lac6, full.lac12]
    mapped = np.r_[
        full.lac0 + 0.25 * (full.lac24 - full.lac0),
        full.lac0 + 0.5 * (full.lac24 - full.lac0),
    ]
    out = {
        "workbook_stays": len(workbook),
        "recorded_sepsis_stays": int(allf.sepsis3.eq(True).sum()),
        "adult_recorded_sepsis_stays": len(m),
        "stay_id_matches_in_existing_sofa_file": int(
            m.stay_id.isin(scores.index).sum()
        ),
        "available_0_48_sofa_value_distributions": {c: counts(s[c]) for c in s},
        "no_48h_column_in_workbook": True,
        "masked_benchmark_workbook_0h_24h_anchors": {
            "n": len(full),
            "n_values": len(obs),
            "linear_0_24h": frozen.metrics(obs, mapped),
        },
        "recorded_training_parent_stay_matches": int(
            allf.stay_id.isin(tp.stay_id).sum()
        ),
        "reconstructed_training_stays": len(train),
        "compatible_workbook_stays": len(compatible),
        "compatible_workbook_stays_in_training": int(
            compatible.stay_id.isin(train.stay_id).sum()
        ),
        "compatible_workbook_stays_without_any_training_patient": len(novel),
        "fallback_workbook_replay": booster_benchmark(
            compatible, "descriptive_replay_training_overlap"
        ),
        "fallback_patient_disjoint": booster_benchmark(novel, "frozen_fallback"),
    }
    return out


def ruijin():
    d = pd.read_pickle(W / "ruijin_registry.pkl")
    d["id"] = canon(d["住院号"])
    d["admit"] = pd.to_datetime(d.icu_admit_date, errors="coerce").dt.normalize()
    d["discharge"] = pd.to_datetime(
        d.icu_discharge_date, errors="coerce"
    ).dt.normalize()
    valid = d[d.id.notna() & d.admit.notna()].copy()
    grp = ["id", "admit"]
    check = [
        "sepsis_related_flag",
        "age_clean",
        "sex_binary",
        "SOFA-2 (Baseline)",
        "SOFA-2 (72h)",
        "discharge",
        "mortality_28d_flag",
        "date_inconsistency_flag",
    ]
    stat = valid.groupby(grp, sort=False)[check].nunique(dropna=False)
    keys = valid.groupby(grp, sort=False).size().rename("n_source_rows")
    known = (
        valid.drop_duplicates(grp, keep="first")
        .set_index(grp)
        .join(keys)
        .join(stat.gt(1).add_prefix("conflict_"))
    )
    eligible = known[
        known.sepsis_related_flag.eq(1)
        & numeric(known.age_clean).ge(18)
        & ~known.conflict_sepsis_related_flag
        & ~known.conflict_age_clean
    ].copy()
    e = eligible[
        ~eligible[[f"conflict_{k}" for k in check]].any(axis=1)
        & eligible.discharge.ge(eligible.index.get_level_values("admit"))
        & eligible.date_inconsistency_flag.eq(0)
    ].copy()
    e["proxy72"] = numeric(e["SOFA-2 (72h)"]) - numeric(e["SOFA-2 (Baseline)"]) >= 2
    e["out"] = numeric(e.mortality_28d_flag).where(
        numeric(e.mortality_28d_flag).isin([0, 1])
    )
    e = e[
        e.out.notna()
        & numeric(e["SOFA-2 (72h)"]).notna()
        & numeric(e["SOFA-2 (Baseline)"]).notna()
    ].copy()
    e48 = e[(e.discharge - e.index.get_level_values("admit")).dt.days.ge(4)].copy()
    if args.ruijin_selection:
        manifest = eligible.reset_index()[
            ["id", "admit", "n_source_rows", "conflict_mortality_28d_flag"]
        ]
        included = pd.MultiIndex.from_frame(manifest[["id", "admit"]]).isin(e.index)
        manifest["included_in_72h_exploration"] = included
        manifest.to_csv(args.ruijin_selection, index=False)
    daily = pd.read_excel(
        args.ruijin_daily, usecols=["Patient_ID", "Day_1", "Day_3", "Outcome_28d"]
    )
    daily["id"] = canon(daily.Patient_ID)
    unique = daily.loc[daily.id.notna() & ~daily.id.duplicated(keep=False)].set_index(
        "id"
    )
    sub = eligible.reset_index()[["id", "SOFA-2 (Baseline)"]].merge(
        unique[["Day_1", "Day_3"]], on="id", how="inner", validate="many_to_one"
    )
    return {
        "registry_rows": len(d),
        "sepsis_flag_rows": int(d.sepsis_related_flag.eq(1).sum()),
        "unique_encounter_keys_all": len(known),
        "repeated_encounter_keys_all": int(keys.gt(1).sum()),
        "conflicting_mortality_keys_all": int(stat.mortality_28d_flag.gt(1).sum()),
        "adult_sepsis_unique_encounter_keys": len(eligible),
        "adult_sepsis_keys_with_repeated_rows": int(eligible.n_source_rows.gt(1).sum()),
        "adult_sepsis_mortality_conflict_keys": int(
            eligible.conflict_mortality_28d_flag.sum()
        ),
        "adult_sepsis_consistent_keys_for_exploratory_72h": len(e),
        "matched_unique_daily_trajectory_keys": len(sub),
        "matched_daily_day1_agrees_with_registry_baseline": int(
            numeric(sub.Day_1).eq(numeric(sub["SOFA-2 (Baseline)"])).sum()
        ),
        "exact_48h_has_verified_data": False,
        "serial_lactate_and_hr_inputs_present": False,
        "exploratory_72h_delta_ge2_only": frozen.ratio(
            e.out.astype(bool),
            e.proxy72,
            pd.Series(e.index.get_level_values("id"), index=e.index),
        ),
        "exploratory_72h_delta_ge2_stay_date_gap_at_least_4d": frozen.ratio(
            e48.out.astype(bool),
            e48.proxy72,
            pd.Series(e48.index.get_level_values("id"), index=e48.index),
        ),
    }


results = {}
for name, fn in [("eICU", eicu), ("MIMIC-IV", mimic), ("Ruijin", ruijin)]:
    if args.cohort not in ("all", name):
        continue
    results[name] = fn()
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False)
    )
    print(name, "finished", flush=True)
