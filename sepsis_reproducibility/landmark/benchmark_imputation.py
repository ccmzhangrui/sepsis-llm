"""Compare retrospective 6- and 12-hour lactate reconstruction by held-out patient."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduce import metrics

SEED = 2026
BOOTSTRAPS = 2000
ANCHORS = ["lac0", "lac24"]
TARGETS = ["lac6", "lac12"]


def design(matrix, admissions):
    groups = pd.read_csv(admissions, usecols=["admissionid", "patientid"])
    data = pd.read_csv(matrix, usecols=["admissionid", *ANCHORS, *TARGETS]).merge(
        groups, on="admissionid", validate="one_to_one", how="left"
    )
    if data.patientid.isna().any():
        raise ValueError("Every analytical stay needs a patient identifier.")
    return data.loc[
        data.lac0.ge(0.30) & data[["lac6", "lac12", "lac24"]].gt(0).all(axis=1)
    ].reset_index(drop=True)


def predictions(train, test):
    from xgboost import XGBRegressor

    x_train, x_test = train[ANCHORS], test[ANCHORS]
    a = test.lac0.to_numpy()
    b = test.lac24.to_numpy()
    results = {
        "linear_interpolation": np.column_stack(
            [a + 0.25 * (b - a), a + 0.50 * (b - a)]
        ),
        "average_of_two_anchors": np.repeat(((a + b) / 2)[:, None], 2, axis=1),
    }
    models = {name: [] for name in ("chained_equations", "random_forest", "xgboost")}
    fitted = {}
    for target in TARGETS:
        blocks = [*ANCHORS, target]
        imputer = IterativeImputer(random_state=SEED, max_iter=20)
        imputer.fit(train[blocks])
        values = x_test.assign(**{target: np.nan})[blocks]
        learned = imputer.transform(values)[:, -1]
        models["chained_equations"].append(learned)
        target_fit = next(
            step
            for step in reversed(imputer.imputation_sequence_)
            if step.feat_idx == 2
        )
        fitted[target] = {
            "intercept": float(target_fit.estimator.intercept_),
            "coefficients": dict(
                zip(
                    [blocks[i] for i in target_fit.neighbor_feat_idx],
                    map(float, target_fit.estimator.coef_),
                )
            ),
        }
        frozen = fitted[target]["intercept"] + sum(
            coef * x_test[feature].to_numpy()
            for feature, coef in fitted[target]["coefficients"].items()
        )
        if not np.allclose(frozen, learned, atol=1e-10):
            raise ValueError(
                "Frozen two-anchor predictor differs from the holdout benchmark."
            )

        forest = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=4
        )
        forest.fit(x_train, train[target])
        models["random_forest"].append(forest.predict(x_test))

        xgb = GridSearchCV(
            XGBRegressor(n_estimators=300, random_state=SEED, n_jobs=2),
            {"max_depth": [2, 3, 5], "learning_rate": [0.03, 0.10]},
            cv=GroupKFold(n_splits=5),
            scoring="neg_mean_squared_error",
            n_jobs=2,
        )
        xgb.fit(x_train, train[target], groups=train.patientid)
        models["xgboost"].append(xgb.predict(x_test))
    return {
        **results,
        **{name: np.column_stack(values) for name, values in models.items()},
    }, fitted


def intervals(test, actual, estimates):
    patients = list(test.groupby("patientid", sort=False).indices.values())
    base = estimates["linear_interpolation"]
    other = [name for name in estimates if name != "linear_interpolation"]
    rng = np.random.default_rng(SEED)
    values = {k: [] for k in other}
    for draw in rng.integers(0, len(patients), size=(BOOTSTRAPS, len(patients))):
        rows = np.concatenate([patients[i] for i in draw])
        y = actual[rows]
        err = y - base[rows]
        denom = ((y - y.mean()) ** 2).mean()
        if denom <= 0:
            continue
        for name in other:
            err2 = y - estimates[name][rows]
            mse_delta = (err2**2 - err**2).mean()
            mae_delta = (np.abs(err2) - np.abs(err)).mean()
            values[name].append([-mse_delta / denom, mse_delta, mae_delta])
    return {
        name: {
            metric: np.quantile(draws, [0.025, 0.975], axis=0)[:, i].tolist()
            for i, metric in enumerate(("R2", "MSE", "MAE"))
        }
        for name, draws in values.items()
    }


def benchmark(data):
    fit, held = next(
        GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED).split(
            data, groups=data.patientid
        )
    )
    train, test = data.iloc[fit].reset_index(drop=True), data.iloc[held].reset_index(
        drop=True
    )
    if set(train.patientid) & set(test.patientid):
        raise ValueError("Patient overlap in the held-out split.")
    actual = test[TARGETS].to_numpy()
    estimates, fitted = predictions(train, test)
    return {
        "scope": "Retrospective reconstruction with both 0-hour and 24-hour anchors.",
        "anchor_definition": "Same two inputs for all methods. Targets must be measured. No 48-hour requirement.",
        "seed": SEED,
        "cluster_bootstraps": BOOTSTRAPS,
        "n_stays": len(data),
        "n_patients": int(data.patientid.nunique()),
        "train_stays": len(train),
        "train_patients": int(train.patientid.nunique()),
        "test_stays": len(test),
        "test_patients": int(test.patientid.nunique()),
        "test_values": actual.size,
        "train_test_shared_patients": 0,
        "metrics": {
            name: {
                "joint": metrics(actual.ravel(), estimate.ravel()),
                **{
                    target: metrics(actual[:, i], estimate[:, i])
                    for i, target in enumerate(TARGETS)
                },
            }
            for name, estimate in estimates.items()
        },
        "paired_delta_95_ci_method_minus_interpolation": intervals(
            test, actual, estimates
        ),
    }, {
        "schema": "two_observed_lactate_anchors_v1",
        "fit_scope": "AmsterdamUMCdb training partition, seed 2026",
        "training_patients": int(train.patientid.nunique()),
        "training_stays": len(train),
        "targets": fitted,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--admissions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-model", type=Path)
    args = parser.parse_args()
    result, fitted = benchmark(design(args.matrix, args.admissions))
    if args.frozen_model:
        args.frozen_model.write_text(
            json.dumps(fitted, indent=2, allow_nan=False) + "\n"
        )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
