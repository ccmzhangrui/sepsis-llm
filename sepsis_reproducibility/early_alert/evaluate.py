"""Fit or evaluate a fixed 24-hour sepsis prognostic research protocol."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

SCHEMA = "suspected_infection_24h_7d_mortality_v1"
SEED = 2026
BUDGET = 0.20
N_BOOTSTRAP = 1000
ORGAN = ("sofa_observed", "sofa_components")
VITAL = (*ORGAN, "map_median", "hr_median", "rr_median")
FEATURES = {"organ": ORGAN, "vitals": VITAL, "lactate": (*VITAL, "lactate_latest")}
METADATA = ("stay_id", "patient_id", "snapshot_hour")
OUTCOMES = ("death_7d", "death_72h")
DEATH_TIME = "hours_to_recorded_7d_death"
PAIRS = (("organ", "vitals"), ("vitals", "lactate"), ("organ", "lactate"))


def load(path, *, development=False):
    df = pd.read_csv(path, dtype={"stay_id": "string", "patient_id": "string"})
    required = {*METADATA, *FEATURES["lactate"]}
    if development:
        required.update(("era", *OUTCOMES))
    if missing := required.difference(df):
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if (
        df.empty
        or df.stay_id.duplicated().any()
        or df[["stay_id", "patient_id"]].isna().any().any()
    ):
        raise ValueError(
            "Provide one row per eligible stay with a stable patient identifier."
        )
    for col in ("snapshot_hour", *FEATURES["lactate"]):
        df[col] = pd.to_numeric(df[col], errors="raise")
    if df.snapshot_hour.ne(24).any():
        raise ValueError("This protocol scores completed snapshots at hour 24 only.")
    if not df.sofa_components.isin(range(7)).all():
        raise ValueError("sofa_components must be an integer from 0 to 6.")
    score = df.sofa_observed
    if (score.notna() & ~score.between(0, df.sofa_components * 4)).any():
        raise ValueError(
            "The observed organ score lies outside the measured component range."
        )
    for col, lo, hi in [
        ("map_median", 20, 180),
        ("hr_median", 20, 300),
        ("rr_median", 3, 100),
        ("lactate_latest", 0.1, 40),
    ]:
        df[col] = df[col].where(df[col].between(lo, hi))
    for name in OUTCOMES:
        df[name] = pd.to_numeric(df[name], errors="raise") if name in df else np.nan
        if not df[name].dropna().isin([0, 1]).all():
            raise ValueError(f"{name} must contain only 0, 1 or blank.")
    if (df.death_72h.eq(1) & df.death_7d.eq(0)).any():
        raise ValueError(
            "Recorded death by 72 hours must also count by day 7 when both are known."
        )
    df[DEATH_TIME] = (
        pd.to_numeric(df[DEATH_TIME], errors="raise") if DEATH_TIME in df else np.nan
    )
    t = df[DEATH_TIME]
    if (t.notna() & (~t.gt(0) | ~t.le(168) | ~df.death_7d.eq(1))).any():
        raise ValueError(f"{DEATH_TIME} is optional and valid only for 7-day deaths.")
    if (t.notna() & df.death_72h.notna() & df.death_72h.ne(t.le(72))).any():
        raise ValueError(f"{DEATH_TIME} must agree with known 72-hour outcomes.")
    return df


def available(df, route, extra=()):
    columns = (*FEATURES[route], *extra)
    return df.sofa_components.ge(2) & df.loc[:, columns].notna().all(axis=1)


def train_component(train, route, extra=()):
    columns = [*FEATURES[route], *extra]
    data = train[available(train, route, extra) & train.death_7d.notna()]
    if min(data.death_7d.eq(1).sum(), data.death_7d.eq(0).sum()) < 10:
        raise ValueError(
            f"Too few outcome events or non-events to fit the {route} model."
        )
    x = data[columns].to_numpy(dtype=float)
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    estimator = LogisticRegression(C=0.5, max_iter=1000).fit(
        (x - center) / scale, data.death_7d
    )
    weights = estimator.coef_[0] / scale
    return {
        "fit_stays": len(data),
        "fit_events": int(data.death_7d.sum()),
        "coefficients": dict(zip(columns, weights.tolist())),
        "intercept": float(estimator.intercept_[0] - weights @ center),
    }


def linear_scores(df, model, policy):
    allowed = {
        "organ": ("organ",),
        "vitals": ("organ", "vitals"),
        "lactate": ("organ", "vitals", "lactate"),
    }[policy]
    scores = pd.Series(np.nan, index=df.index)
    used = pd.Series("indeterminate", index=df.index)
    for route in allowed:
        component = model["components"][route]
        ok = available(df, route, model.get("additional_features", ()))
        weights = pd.Series(component["coefficients"])
        scores.loc[ok] = df.loc[ok, weights.index].dot(weights) + component["intercept"]
        used.loc[ok] = route
    return scores, used


def thresholds_at_budget(probabilities):
    """Include at most floor(budget*n) calibration rows without breaking ties."""
    scores = sorted(probabilities.dropna(), reverse=True)
    if not scores:
        raise ValueError("No observed inputs in the operating-point cohort.")
    maximum = int(BUDGET * len(scores))
    return float(np.nextafter(scores[maximum], np.inf))


def split(df):
    if set(df.era) != {"2003-2009", "2010-2016"}:
        raise ValueError("Expected both original Amsterdam admission-year bands.")
    newer = df.loc[df.era.eq("2010-2016")]
    older = df.loc[df.era.eq("2003-2009") & ~df.patient_id.isin(newer.patient_id)]
    a, b = next(
        GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED).split(
            older, groups=older.patient_id
        )
    )
    fit, calibration = older.iloc[a], older.iloc[b]
    assert not set(fit.patient_id) & set(calibration.patient_id)
    assert not set(older.patient_id) & set(newer.patient_id)
    return fit, calibration, newer, int(df.era.eq("2003-2009").sum() - len(older))


def fit(df, *, additional_features=()):
    train, calibration, test, removed = split(df)
    model = {
        "schema": SCHEMA,
        "alert_budget_in_development": BUDGET,
        "fit_era": "2003-2009",
        "primary_comparator": "organ",
        "components": {k: train_component(train, k, additional_features) for k in FEATURES},
        "policies": {},
    }
    if additional_features:
        model["additional_features"] = list(additional_features)
    for name in FEATURES:
        scores, _ = linear_scores(calibration, model, name)
        ok = scores.notna() & calibration.death_7d.notna()
        y = calibration.loc[ok, "death_7d"]
        if min(y.eq(1).sum(), y.eq(0).sum()) < 5:
            raise ValueError("Too few outcomes in the probability calibration cohort.")
        calibrator = LogisticRegression(C=1.0, max_iter=1000).fit(
            scores[ok].to_numpy()[:, None], y
        )
        slope, intercept = float(calibrator.coef_[0][0]), float(
            calibrator.intercept_[0]
        )
        if slope <= 0:
            slope = 0.0
            intercept = float(logit(y.mean()))
        p = pd.Series(expit(intercept + slope * scores), index=scores.index)
        cutoff = thresholds_at_budget(p)
        model["policies"][name] = {
            "intercept": intercept,
            "slope": slope,
            "threshold": cutoff,
            "calibration_scored": int(p.notna().sum()),
            "calibration_alerts": int(p.ge(cutoff).sum()),
        }
    partitions = {
        key: {
            "stays": len(x),
            "patients": int(x.patient_id.nunique()),
            "deaths_7d": int(x.death_7d.sum()),
        }
        for key, x in (
            ("fit", train),
            ("calibration", calibration),
            ("later_test", test),
        )
    }
    partitions["older_stays_removed_due_to_patient_overlap"] = removed
    return model, partitions, test


def predictions(df, model):
    if model.get("schema") != SCHEMA:
        raise ValueError("Unexpected frozen alert model.")
    for name in FEATURES:
        scores, route = linear_scores(df, model, name)
        policy = model["policies"][name]
        risk = pd.Series(
            expit(policy["intercept"] + policy["slope"] * scores), index=scores.index
        )
        alert = risk.ge(policy["threshold"])
        yield name, risk, alert, route


def transfer_thresholds(reference, evaluation, model):
    """Use only site-reference scores to set thresholds on unchanged risks."""
    if reference.patient_id.isin(evaluation.patient_id).any():
        raise ValueError("Reference and evaluation patients must be disjoint.")
    local = deepcopy(model)
    details = {}
    for name, risk, _, _ in predictions(reference, model):
        threshold = thresholds_at_budget(risk)
        local["policies"][name]["threshold"] = threshold
        details[name] = {
            "threshold": threshold,
            "scored": int(risk.notna().sum()),
            "alerts": int(risk.ge(threshold).sum()),
        }
    return local, {
        "reference_stays": len(reference),
        "reference_patients": int(reference.patient_id.nunique()),
        "target_fraction_of_scored_stays": BUDGET,
        "policies": details,
    }


def fraction(num, den):
    return float(num / den) if den else None


def metrics(df, risk, alert, endpoint):
    y = df[endpoint]
    known = y.notna()
    scored = risk.notna()
    positive = y.eq(1)
    evaluable = known & scored
    tp = int((alert & positive).sum())
    events = int(positive.sum())
    label = {
        "known_endpoints": int(known.sum()),
        "events": events,
        "scored_known": int(evaluable.sum()),
        "events_in_scored": int((positive & scored).sum()),
        "events_in_indeterminate": int((positive & ~scored).sum()),
        "recorded_deaths_with_alert": tp,
        "alerts_with_known_endpoint": int((known & alert).sum()),
        "ppv": fraction(tp, int((known & alert).sum())),
        "event_detection_all_eligible_known": fraction(tp, events),
        "sensitivity_scored": fraction(tp, int((scored & positive).sum())),
        "specificity_scored": fraction(
            int((scored & y.eq(0) & ~alert).sum()), int((scored & y.eq(0)).sum())
        ),
    }
    if evaluable.any() and endpoint == "death_7d":
        r, outcome = risk[evaluable], y[evaluable]
        label.update(
            {
                "brier": float(((r - outcome) ** 2).mean()),
                "mean_predicted_risk": float(r.mean()),
                "observed_risk": float(outcome.mean()),
            }
        )
        if outcome.nunique() == 2:
            label.update(
                {
                    "auroc": float(roc_auc_score(outcome, r)),
                    "average_precision": float(average_precision_score(outcome, r)),
                }
            )
    return label


def paired_intervals(df, preds, a, b):
    # All strategies share the same measured organ-score gate.
    risk_a, flag_a = preds[a]
    risk_b, flag_b = preds[b]
    if not risk_a.notna().equals(risk_b.notna()):
        raise RuntimeError("Paired comparison requires the same scoring coverage.")
    valid = df.death_7d.notna()
    y = df.loc[valid, "death_7d"].to_numpy()
    if len(y) < 2:
        return {}
    clusters = list(df.loc[valid].groupby("patient_id", sort=False).indices.values())
    if len(clusters) < 2:
        return {}
    delta_alert = flag_b[valid].to_numpy(dtype=float) - flag_a[valid].to_numpy(
        dtype=float
    )
    p, q = risk_a[valid].to_numpy(), risk_b[valid].to_numpy()
    error_difference = (q - y) ** 2 - (p - y) ** 2
    rng = np.random.default_rng(SEED)
    rows = []
    for sample in rng.integers(0, len(clusters), size=(N_BOOTSTRAP, len(clusters))):
        indices = np.concatenate([clusters[i] for i in sample])
        subset = y[indices]
        losses = error_difference[indices]
        losses = losses[np.isfinite(losses)]
        diff = delta_alert[indices]
        rows.append(
            (
                100 * diff.mean(),
                100 * (diff * subset).sum() / subset.sum() if subset.sum() else np.nan,
                losses.mean() if len(losses) else np.nan,
            )
        )
    cases = np.asarray(rows)
    return {
        name: np.quantile(col[np.isfinite(col)], [0.025, 0.975]).tolist()
        for i, name in enumerate(("alerts_per_100", "capture_pct_points", "brier"))
        if (col := cases[:, i])[np.isfinite(col)].size
    }


def capacity_weights(scores, slots):
    """Allocate ties fractionally, independent of outcomes and row order."""
    if slots == 0:
        return np.zeros(len(scores))
    boundary = np.partition(scores, len(scores) - slots)[len(scores) - slots]
    above = scores > boundary
    tied = scores == boundary
    return above + tied * ((slots - above.sum()) / tied.sum())


def matched_capacity(df, collected):
    common = df.death_7d.notna()
    for risk, _ in collected.values():
        common &= risk.notna()
    y = df.loc[common, "death_7d"].to_numpy(dtype=float)
    scores = {k: r[common].to_numpy() for k, (r, _) in collected.items()}
    slots = int(BUDGET * len(y))
    events = int(y.sum())
    result = {
        "design": "Retrospective ranking on the same scored stays with known outcomes",
        "fraction_of_common_scored_stays": BUDGET,
        "common_scored_known": len(y),
        "events": events,
        "slots_per_policy": slots,
        "policies": {},
        "comparisons": {},
    }
    captured = {k: float(capacity_weights(r, slots) @ y) for k, r in scores.items()}
    for k, caught in captured.items():
        result["policies"][k] = {
            "expected_deaths_with_alert": caught,
            "sensitivity_common_scored": fraction(caught, events),
            "ppv": fraction(caught, slots),
        }
    clusters = list(df.loc[common].groupby("patient_id", sort=False).indices.values())
    draws = {pair: [] for pair in PAIRS}
    rng = np.random.default_rng(SEED)
    if len(clusters) > 1 and events:
        for _ in range(N_BOOTSTRAP):
            sample = rng.integers(len(clusters), size=len(clusters))
            indices = np.concatenate([clusters[i] for i in sample])
            outcomes = y[indices]
            n_events = outcomes.sum()
            if not n_events:
                continue
            capacity = int(BUDGET * len(indices))
            catch = {
                k: capacity_weights(r[indices], capacity) @ outcomes
                for k, r in scores.items()
            }
            for a, b in PAIRS:
                draws[a, b].append(100 * (catch[b] - catch[a]) / n_events)
    for a, b in PAIRS:
        draws_ab = draws[a, b]
        result["comparisons"][f"{b}_vs_{a}"] = {
            "difference_in_expected_captured_deaths": captured[b] - captured[a],
            "capture_difference_pct_points": fraction(
                100 * (captured[b] - captured[a]), events
            ),
            "paired_cluster_bootstrap_95_ci_pct_points": (
                np.quantile(draws_ab, [0.025, 0.975]).tolist() if draws_ab else None
            ),
        }
    return result


def summarize(df, model, *, include_capacity_ranking=True):
    collected, results = {}, {}
    for name, risk, alert, route in predictions(df, model):
        collected[name] = (risk, alert)
        results[name] = {
            "scored": int(risk.notna().sum()),
            "indeterminate": int(risk.isna().sum()),
            "routes": {str(k): int(v) for k, v in route.value_counts().items()},
            "alerts": int(alert.sum()),
            "alerts_per_100_eligible": 100 * float(alert.mean()),
            **{key: metrics(df, risk, alert, key) for key in OUTCOMES},
        }
        hours = df.loc[alert, DEATH_TIME].dropna()
        if not hours.empty:
            q1, median, q3 = hours.quantile([0.25, 0.5, 0.75])
            results[name]["hours_from_assessment_to_recorded_death_among_alerts"] = {
                "n": len(hours),
                "median": float(median),
                "q1": float(q1),
                "q3": float(q3),
            }
    comparisons = {}
    for a, b in PAIRS:
        prev, nxt = collected[a][1], collected[b][1]
        events = df.death_7d.eq(1)
        comparisons[f"{b}_vs_{a}"] = {
            "added_alerts": int((nxt & ~prev).sum()),
            "removed_alerts": int((prev & ~nxt).sum()),
            "added_alerts_with_7d_death": int((nxt & ~prev & events).sum()),
            "removed_alerts_with_7d_death": int((prev & ~nxt & events).sum()),
            "paired_cluster_bootstrap_95_ci": paired_intervals(df, collected, a, b),
        }
    result = {
        "eligible": len(df),
        "patients": int(df.patient_id.nunique()),
        "policies": results,
        "comparisons": comparisons,
    }
    if include_capacity_ranking:
        result["matched_alert_capacity"] = matched_capacity(df, collected)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for command in ("develop", "evaluate"):
        run = subs.add_parser(command)
        run.add_argument("--input", type=Path, required=True)
        run.add_argument("--model", type=Path, required=True)
        run.add_argument("--output", type=Path, required=True)
        if command == "evaluate":
            run.add_argument(
                "--capacity-reference",
                type=Path,
                help="Earlier local cohort, with no patients shared with the evaluation input.",
            )
    args = parser.parse_args()
    development = args.command == "develop"
    df = load(args.input, development=development)
    if development:
        model, partitions, test = fit(df)
        args.model.write_text(json.dumps(model, allow_nan=False, indent=2) + "\n")
        result = {"partitions": partitions, **summarize(test, model)}
    else:
        model = json.loads(args.model.read_text())
        local = None
        if args.capacity_reference is not None:
            local, reference = transfer_thresholds(load(args.capacity_reference), df, model)
        result = summarize(df, model)
        if local is not None:
            result["local_capacity_thresholds"] = {
                "reference": reference,
                "evaluation": summarize(df, local, include_capacity_ranking=False),
            }
    args.output.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
