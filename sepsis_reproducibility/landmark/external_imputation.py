"""Compare the frozen Amsterdam imputer against linear interpolation without refitting."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduce import metrics


def compare(data, model):
    if model.get("schema") != "two_observed_lactate_anchors_v1":
        raise ValueError("Unexpected frozen imputer schema.")
    usable_anchors = data.lac0.ge(0.30) & data.lac24.gt(0)
    results = {
        "source_entries": len(data),
        "two_anchors_available": int(usable_anchors.sum()),
    }
    for hour in (6, 12):
        target = f"lac{hour}"
        use = usable_anchors & data[target].gt(0)
        part = data.loc[use]
        coef = model["targets"][target]
        pred = coef["intercept"] + sum(
            coef["coefficients"][var] * part[var] for var in ("lac0", "lac24")
        )
        linear = part.lac0 + (hour / 24) * (part.lac24 - part.lac0)
        valid = np.isfinite(pred) & pred.gt(0)
        result = {
            "measured_with_anchors": len(part),
            "positive_learned_estimates": int(valid.sum()),
        }
        common = part[valid]
        if len(common) >= 2:
            result["linear_on_same_stays"] = metrics(common[target], linear[valid])
            result["frozen_on_same_stays"] = metrics(common[target], pred[valid])
        if len(part) >= 2:
            result["linear_on_all_stays_with_anchors"] = metrics(part[target], linear)
        results[target] = result
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Local CSV with stay_id and lactate columns lac0, lac6, lac12, lac24 (mmol/L)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).with_name("amsterdam_two_anchor_model.json"),
    )
    args = parser.parse_args()
    columns = ["stay_id", "lac0", "lac6", "lac12", "lac24"]
    data = pd.read_csv(args.input, usecols=columns, dtype={"stay_id": "string"})
    if data.stay_id.isna().any() or data.stay_id.duplicated().any():
        raise ValueError("Provide one row per stay and a nonmissing stay identifier.")
    data[columns[1:]] = (
        data[columns[1:]]
        .apply(pd.to_numeric, errors="raise")
        .replace([np.inf, -np.inf], np.nan)
    )
    args.output.write_text(
        json.dumps(
            compare(data, json.loads(args.model.read_text())), indent=2, allow_nan=False
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
