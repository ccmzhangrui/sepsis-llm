"""Evaluate high-sensitivity review alongside the original urgent alert."""

import argparse
import json
from decimal import Decimal, ROUND_CEILING
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from evaluate import fit, fraction, load, predictions, split

AGE_MIDPOINT = {
    "18-39": 28.5, "40-49": 44.5, "50-59": 54.5,
    "60-69": 64.5, "70-79": 74.5, "80+": 85.0,
}
TARGETS = (0.90, 0.95, 1.00)


def load_review(path, *, development=False):
    stays = load(path, development=development)
    stays["age_midpoint"] = (
        stays.age_group.map(AGE_MIDPOINT) if "age_group" in stays else np.nan
    )
    return stays


def lactate_predictions(stays, model):
    return next((risk, alert) for name, risk, alert, _ in predictions(stays, model)
                if name == "lactate")


def review_threshold(outcome, risk, urgent, target):
    """Highest cutoff covering the target fraction of known reference events."""
    events = outcome.eq(1)
    event_count = int(events.sum())
    if event_count == 0:
        raise ValueError("Reference must contain known 7-day deaths to set a review threshold.")
    required = int((Decimal(str(target)) * event_count).to_integral_value(rounding=ROUND_CEILING))
    remaining = events & ~urgent & risk.notna()
    required_from_risk = required - int((events & ~remaining).sum())
    if required_from_risk <= 0:
        return None
    return float(risk[remaining].nlargest(required_from_risk).iloc[-1])


def breakdown(stays, risk, urgent, threshold, *, review=True, review_all=False):
    unknown = risk.isna() & ~urgent if review else pd.Series(False, index=stays.index)
    secondary = (
        risk.ge(threshold) & ~urgent if review and threshold is not None
        else pd.Series(False, index=stays.index)
    )
    universal_review = ~urgent & ~unknown & ~secondary if review_all else pd.Series(
        False, index=stays.index
    )
    routed = urgent | unknown | secondary | universal_review
    deaths = stays.death_7d.eq(1)
    early = stays.death_72h.eq(1)
    known = stays.death_7d.notna()
    return {
        "urgent_alerts": int(urgent.sum()),
        "secondary_review_risk": int(secondary.sum()),
        "secondary_review_unscored": int(unknown.sum()),
        "additional_review_for_full_coverage": int(universal_review.sum()),
        "any_alert_or_review": int(routed.sum()),
        "per_100_stays_alert_or_review": 100 * float(routed.mean()),
        "known_7d_deaths": int(deaths.sum()),
        "7d_deaths_urgent": int((urgent & deaths).sum()),
        "7d_deaths_secondary_risk": int((secondary & deaths).sum()),
        "7d_deaths_secondary_unscored": int((unknown & deaths).sum()),
        "7d_deaths_additional_review_for_full_coverage": int((universal_review & deaths).sum()),
        "7d_deaths_missed": int((~routed & deaths).sum()),
        "7d_deaths_all_tiers_capture": fraction(int((routed & deaths).sum()), int(deaths.sum())),
        "72h_deaths_missed": int((~routed & early).sum()),
        "known_label_7d_ppv_all_tiers": fraction(int((routed & deaths).sum()), int((known & routed).sum())),
        "unlabelled_stays": int((~known).sum()),
    }


def compare(reference, evaluation, baseline, age_model=None):
    if reference.patient_id.isin(evaluation.patient_id).any():
        raise ValueError("Reference and evaluation patients must be disjoint.")
    ref_base, ref_urgent = lactate_predictions(reference, baseline)
    eval_base, eval_urgent = lactate_predictions(evaluation, baseline)
    scores = {"frozen_risk": (ref_base, eval_base)}
    if age_model is not None:
        ref_age, _ = lactate_predictions(reference, age_model)
        eval_age, _ = lactate_predictions(evaluation, age_model)
        scores["age_plus_frozen_inputs"] = (
            ref_age.fillna(ref_base), eval_age.fillna(eval_base)
        )
    comparisons = {}
    for name, (ref_risk, eval_risk) in scores.items():
        # All review policies leave the original urgent alert set intact.
        result = {}
        for target in TARGETS:
            cutoff = review_threshold(reference.death_7d, ref_risk, ref_urgent, target)
            result[str(target)] = {
                "cutoff": cutoff,
                "reference": breakdown(reference, ref_risk, ref_urgent, cutoff),
                "later_evaluation": breakdown(evaluation, eval_risk, eval_urgent, cutoff),
            }
        known_scored = evaluation.death_7d.notna() & eval_risk.notna()
        observed = evaluation.loc[known_scored, "death_7d"]
        result["scored_7d_auroc"] = (
            float(roc_auc_score(observed, eval_risk[known_scored]))
            if observed.nunique() == 2 else None
        )
        if evaluation.death_7d.eq(1).any():
            cutoff = review_threshold(evaluation.death_7d, eval_risk, eval_urgent, 1.0)
            result["retrospective_zero_miss_diagnostic"] = breakdown(
                evaluation, eval_risk, eval_urgent, cutoff
            )
        comparisons[name] = result
    return {
        "reference_stays": len(reference),
        "evaluation_stays": len(evaluation),
        "frozen_urgent_only": breakdown(evaluation, eval_base, eval_urgent, None, review=False),
        "universal_review_benchmark": breakdown(
            evaluation, eval_base, eval_urgent, None, review_all=True
        ),
        "review_policies": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("develop", "evaluate"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--input", type=Path, required=True)
        cmd.add_argument("--baseline", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--age-model", type=Path, required=name == "develop")
        if name == "evaluate":
            cmd.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    if args.command == "develop":
        all_stays = load_review(args.input, development=True)
        _, reference, evaluation, _ = split(all_stays)
        age_model, _, _ = fit(all_stays, additional_features=("age_midpoint",))
    else:
        reference = load_review(args.reference)
        evaluation = load_review(args.input)
        age_model = json.loads(args.age_model.read_text()) if args.age_model else None
    result = compare(reference, evaluation, baseline, age_model)
    if args.command == "develop":
        args.age_model.write_text(json.dumps(age_model, allow_nan=False, indent=2) + "\n")
    args.output.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
