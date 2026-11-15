"""Evaluate fixed sepsis rules at a 48-hour landmark on a local cohort."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduce import ratio

LANDMARK = 48
HORIZON = 28 * 24
FIELDS = (
    "icu_admit_hour",
    "icu_discharge_hour",
    "infection_confirmed_hour",
    "baseline_end_hour",
    "features_through_hour",
    "death_hour",
    "vital_followup_hour",
    "sofa_base_low",
    "sofa_base_high",
    "sofa_late_low",
    "sofa_late_high",
    "matched_base",
    "matched_late",
    "matched_organs",
    "lac_base",
    "lac_late",
    "hr_sd",
    "hr_windows",
)
REQUIRED = ("stay_id", "patient_id", *FIELDS)


def load(path):
    data = pd.read_csv(
        path,
        usecols=list(REQUIRED),
        dtype={"patient_id": "string", "stay_id": "string"},
    )
    if (
        data.empty
        or data.stay_id.duplicated().any()
        or data[["patient_id", "stay_id"]].isna().any().any()
    ):
        raise ValueError(
            "Provide a nonempty cohort with one row per stay and a patient key."
        )
    for name in FIELDS:
        data[name] = pd.to_numeric(data[name], errors="raise")
    if np.isinf(data[list(FIELDS)].to_numpy(dtype=float)).any():
        raise ValueError("Use blank fields for missing values.")
    required_timing = [
        "icu_admit_hour",
        "icu_discharge_hour",
        "infection_confirmed_hour",
        "baseline_end_hour",
        "features_through_hour",
    ]
    if (
        data[required_timing].isna().any().any()
        or data.icu_discharge_hour.lt(data.icu_admit_hour).any()
    ):
        raise ValueError("Missing or reversed encounter or feature timing.")
    if (
        data.features_through_hour.ne(LANDMARK) | ~data.baseline_end_hour.between(1, 25)
    ).any():
        raise ValueError(
            "Provide closed feature windows through hour 48 and a baseline end in hours 1–25."
        )
    for label in ("base", "late"):
        low, high = data[f"sofa_{label}_low"], data[f"sofa_{label}_high"]
        if (
            low.isna().ne(high.isna())
            | low.notna() & (~low.between(0, 24) | ~high.between(0, 24) | high.lt(low))
        ).any():
            raise ValueError(f"Invalid adapted SOFA-2 bounds: {label}.")
    if (
        data.hr_windows.notna().any()
        and not data.hr_windows.dropna().isin(range(6)).all()
    ):
        raise ValueError("Heart-rate window count must be between 0 and 5.")
    n = data.matched_organs
    base, late = data.matched_base, data.matched_late
    if (~n.isin(range(7)) | ~base.between(0, 4 * n) | ~late.between(0, 4 * n)).any():
        raise ValueError("Invalid matched organ counts or sums.")
    return data


def bounded_increase(lower, upper, threshold, *, strict=False):
    if strict:
        positive, negative = lower.gt(threshold), upper.le(threshold)
    else:
        positive, negative = lower.ge(threshold), upper.lt(threshold)
    return (
        pd.Series(pd.NA, index=lower.index, dtype="boolean")
        .mask(negative, False)
        .mask(positive, True)
    )


def rule_status(data, minimum_matched=None):
    if minimum_matched is None:
        lower = data.sofa_late_low - data.sofa_base_high
        upper = data.sofa_late_high - data.sofa_base_low
    else:
        lower = (data.matched_late - data.matched_base).where(
            data.matched_organs.ge(minimum_matched)
        )
        upper = lower
    sofa = bounded_increase(lower, upper, 2)
    hr = (
        data.hr_sd.lt(10)
        .astype("boolean")
        .where(data.hr_sd.ge(0) & data.hr_windows.ge(2))
    )
    delta_positive = bounded_increase(lower, upper, 0, strict=True)
    observed_lactate = data.lac_base.gt(0) & data.lac_late.gt(0)
    clearance = (data.lac_base - data.lac_late) / (
        data.lac_base.where(observed_lactate) + 1e-5
    )
    lactate = (
        (data.lac_late.gt(2) & clearance.lt(0.10))
        .astype("boolean")
        .where(observed_lactate)
    )
    return pd.DataFrame(
        {"sofa": sofa, "lactate": lactate, "heart_rate": hr & delta_positive}
    )


def disjunction(rules):
    positive = rules.fillna(False).any(axis=1)
    negative = rules.notna().all(axis=1) & ~positive
    return (
        pd.Series(pd.NA, index=rules.index, dtype="boolean")
        .mask(positive, True)
        .mask(negative, False)
    )


def counts(series, outcome):
    return {
        state: {"n": int(mask.sum()), "death": int((mask & outcome).sum())}
        for state, mask in {
            "red": series.eq(True).fillna(False),
            "no_red": series.eq(False).fillna(False),
            "indeterminate": series.isna(),
        }.items()
    }


def risk_ratio(outcome, flags, patient):
    estimate = ratio(outcome, flags, patient)
    a, b = estimate["red_events"], estimate["no_red_events"]
    if min(a, b, estimate["red_total"] - a, estimate["no_red_total"] - b) < 5:
        estimate["95_ci"] = None
    return estimate


def report(data):
    horizon = data.icu_admit_hour + HORIZON
    had_died = data.death_hour.le(LANDMARK)
    eligible = (
        data.icu_admit_hour.le(LANDMARK)
        & data.icu_discharge_hour.gt(LANDMARK)
        & data.infection_confirmed_hour.le(LANDMARK)
        & ~had_died
        & horizon.gt(LANDMARK)
    )
    event = data.death_hour.gt(LANDMARK) & data.death_hour.le(horizon)
    known = event | data.death_hour.gt(horizon) | data.vital_followup_hour.ge(horizon)
    cohort = data.loc[eligible & known]
    outcome = event.loc[cohort.index]
    return {
        "analysis": "retrospective landmark reanalysis",
        "landmark_hours_from_suspected_infection": LANDMARK,
        "outcome": "recorded death after landmark and within 28 days of ICU admission",
        "flow": {
            "source_entries": len(data),
            "death_at_or_before_landmark": int(had_died.sum()),
            "eligible_at_landmark": int(eligible.sum()),
            "unknown_outcome_among_eligible": int((eligible & ~known).sum()),
            "analyzed_stays": len(cohort),
            "patients": int(cohort.patient_id.nunique()),
            "deaths": int(outcome.sum()),
        },
        "unresolved_organs_bounded_0_to_4": compare(
            rule_status(cohort), outcome, cohort.patient_id
        ),
        "matched_organs_at_least_four": compare(
            rule_status(cohort, 4), outcome, cohort.patient_id
        ),
        "matched_organs_at_least_five_sensitivity": compare(
            rule_status(cohort, 5), outcome, cohort.patient_id
        ),
    }


def compare(rules, outcome, patient):
    comparators = pd.DataFrame(
        {
            "sofa_only": rules.sofa,
            "sofa_or_lactate": disjunction(rules[["sofa", "lactate"]]),
            "any_of_three": disjunction(rules),
        }
    )
    common = rules.notna().all(axis=1)
    comparables = {}
    per_rule = {}
    for name, flags in rules.items():
        ok = flags.notna()
        summary = counts(flags, outcome)
        summary["available_input_rr"] = risk_ratio(
            outcome[ok], flags[ok].astype(bool), patient[ok]
        )
        per_rule[name] = summary
    for name, flags in comparators.items():
        summary = counts(flags, outcome)
        ok = flags.notna()
        summary["determinate_status_rr"] = risk_ratio(
            outcome[ok], flags[ok].astype(bool), patient[ok]
        )
        summary["common_cohort_rr"] = risk_ratio(
            outcome[common], flags[common].astype(bool), patient[common]
        )
        comparables[name] = summary
    change = {}
    for name in ("sofa_or_lactate", "any_of_three"):
        added = comparators[name].eq(True).fillna(False) & comparators.sofa_only.eq(
            False
        ).fillna(False)
        baseline_unknown = (
            comparators[name].eq(True).fillna(False) & comparators.sofa_only.isna()
        )
        both = common & comparators.sofa_only.eq(False).fillna(False)
        change[name] = {
            "additional_red_when_sofa_negative": int(added.sum()),
            "deaths_in_additional_red": int((added & outcome).sum()),
            "red_with_indeterminate_sofa": int(baseline_unknown.sum()),
            "deaths_in_red_with_indeterminate_sofa": int(
                (baseline_unknown & outcome).sum()
            ),
            "incremental_rr_vs_no_red_on_common_cohort": risk_ratio(
                outcome[both], added[both].astype(bool), patient[both]
            ),
        }
    return {
        "all_three_determinate": int(common.sum()),
        "deaths_in_common_cohort": int((common & outcome).sum()),
        "rules": per_rule,
        "strategies": comparables,
        "additional_alerts": change,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = report(load(args.input))
    with args.input.open("rb") as source:
        output["input_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
