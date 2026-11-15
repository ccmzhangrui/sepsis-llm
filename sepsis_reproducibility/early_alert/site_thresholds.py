"""Compare frozen and local alert thresholds without refitting risk scores."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.model_selection import GroupShuffleSplit

from evaluate import BUDGET, DEATH_TIME, SEED, fraction, load, predictions

CONFIDENCE = 0.95
POLICIES = ("frozen", "local_empirical", "local_buffered")


def patient_max(stays, scores):
    """One maximum per patient, including patients whose stays are all unscored."""
    return scores.groupby(stays.patient_id, sort=False).max()


def estimate_threshold(scores, target, *, confidence=None):
    """Exclude ties at the boundary: issue alerts only strictly above it."""
    if not 0 < target < 1 or scores.empty:
        raise ValueError("A nonempty reference and a target between 0 and 1 are required.")
    if not scores.notna().any():
        return {"status": "no_scored_reference_patients"}
    n = len(scores)
    if confidence is None:
        permitted = int(np.floor(target * n))
    else:
        alpha = 1 - confidence
        permitted = min(n - 1, int(np.floor(target * n)))
        while permitted >= 0 and binom.cdf(permitted, n, target) > alpha:
            permitted -= 1
        if permitted < 0:
            return {"status": "too_few_reference_patients_for_buffer"}
    cutoff = np.sort(scores.fillna(-np.inf).to_numpy())[-permitted - 1]
    threshold = float(np.nextafter(cutoff, np.inf)) if np.isfinite(cutoff) else 0.0
    return {
        "status": "estimated",
        "threshold": threshold,
        "reference_patients_flagged": int(scores.ge(threshold).sum()),
        "reference_patients": n,
        "allowed_reference_ranks": permitted,
    }


def workload(stays, risk, threshold):
    alerts = risk.ge(threshold)
    patient_alerts = patient_max(stays, alerts)
    y = stays.death_7d
    positives = y.eq(1)
    event_count = int(positives.sum())
    true_alerts = int((alerts & positives).sum())
    known_alerts = int((alerts & y.notna()).sum())
    times = stays.loc[alerts, DEATH_TIME].dropna()
    volume = int(alerts.sum())
    return {
        "eligible_stays": len(stays),
        "eligible_patients": len(patient_alerts),
        "scored_stays": int(risk.notna().sum()),
        "alerts": volume,
        "alerts_per_100_stays": 100 * volume / len(stays),
        "patients_with_alert_per_100": 100 * float(patient_alerts.mean()),
        "known_7d_outcomes": int(y.notna().sum()),
        "death_7d_events": event_count,
        "death_7d_alerts": true_alerts,
        "death_7d_missed": event_count - true_alerts,
        "death_7d_capture": fraction(true_alerts, event_count),
        "death_7d_ppv_known": fraction(true_alerts, known_alerts),
        "alerts_with_unknown_7d_outcome": volume - known_alerts,
        "death_72h_alerts": int((alerts & stays.death_72h.eq(1)).sum()),
        "median_hours_to_recorded_death_among_alerts": (
            float(times.median()) if not times.empty else None
        ),
    }


def compare(reference, eval_stays, model, target=BUDGET, *, strategies=None):
    if reference.patient_id.isin(eval_stays.patient_id).any():
        raise ValueError("The local reference and evaluation must have disjoint patients.")
    reference_scores = {name: r for name, r, _, _ in predictions(reference, model)}
    components = {}
    for name, rs, _, _ in predictions(eval_stays, model):
        if strategies is not None and name not in strategies:
            continue
        pts = patient_max(reference, reference_scores[name])
        thresholds = {
            "frozen": {"status": "original", "threshold": model["policies"][name]["threshold"]},
            "local_empirical": estimate_threshold(pts, target),
            "local_buffered": estimate_threshold(pts, target, confidence=CONFIDENCE),
        }
        components[name] = {
            policy: {
                "operating_rule": rule,
                "evaluation": (
                    workload(eval_stays, rs, rule["threshold"])
                    if "threshold" in rule
                    else None
                ),
            }
            for policy, rule in thresholds.items()
        }
    return {
        "design": "Independent local reference, evaluated without refitting risk coefficients",
        "target_fraction_of_patients_with_any_alert": target,
        "buffer_confidence_if_patient_scores_are_iid": CONFIDENCE,
        "reference_patients": int(reference.patient_id.nunique()),
        "reference_stays": len(reference),
        "strategies": components,
    }


def stress_test(stays, model, target, repeats):
    """Test threshold sensitivity across repeated within-period patient splits."""
    rows = {name: [] for name in POLICIES}
    split = GroupShuffleSplit(n_splits=repeats, train_size=0.35, random_state=SEED)
    for a, b in split.split(stays, groups=stays.patient_id):
        result = compare(stays.iloc[a], stays.iloc[b], model, target, strategies=("lactate",))
        for policy, details in result["strategies"]["lactate"].items():
            if details["evaluation"] is not None:
                rows[policy].append(details["evaluation"])
    summaries = {}
    for name, series in rows.items():
        if not series:
            continue
        rates = pd.DataFrame(series)
        summaries[name] = {
            "evaluated_splits": len(rates),
            "splits_above_target_patient_rate": int(
                rates.patients_with_alert_per_100.gt(100 * target).sum()
            ),
            "median_and_5th_95th_percentiles_across_splits": {
                col: rates[col].quantile([0.50, 0.05, 0.95]).tolist()
                for col in (
                    "eligible_stays",
                    "alerts_per_100_stays",
                    "patients_with_alert_per_100",
                    "death_7d_capture",
                    "death_7d_ppv_known",
                )
                if rates[col].notna().any()
            },
        }
    return {
        "design": "Repeated patient-disjoint splits within one cohort; descriptive, not temporal",
        "target_fraction_of_patients_with_any_alert": target,
        "planned_splits": repeats,
        "reference_patient_fraction": 0.35,
        "policies_for_frozen_lactate_strategy": summaries,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for command in ("compare", "stress"):
        cmd = subs.add_parser(command)
        cmd.add_argument("--input", type=Path, required=True)
        cmd.add_argument("--model", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--target", type=float, default=BUDGET)
        if command == "compare":
            cmd.add_argument("--reference", type=Path, required=True)
        else:
            cmd.add_argument("--repeats", type=int, default=40)
    args = parser.parse_args()
    if not 0 < args.target < 1:
        parser.error("--target must be strictly between 0 and 1")
    if args.command == "stress" and args.repeats < 1:
        parser.error("--repeats must be positive")
    df = load(args.input)
    model = json.loads(args.model.read_text())
    if args.command == "compare":
        result = compare(load(args.reference), df, model, args.target)
    else:
        result = stress_test(df, model, args.target, args.repeats)
    args.output.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
