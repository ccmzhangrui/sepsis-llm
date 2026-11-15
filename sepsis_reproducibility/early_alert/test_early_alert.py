import unittest

import numpy as np
import pandas as pd

from evaluate import (
    FEATURES,
    SCHEMA,
    capacity_weights,
    linear_scores,
    split,
    thresholds_at_budget,
    transfer_thresholds,
)
from extract_amsterdam import extract_features


class EarlyAlertTests(unittest.TestCase):
    def test_equal_capacity_gives_identical_scores_equal_weight(self):
        scores = np.array([0.2, 0.8, 0.5, 0.5, 0.5])
        weights = capacity_weights(scores, 2)
        np.testing.assert_allclose(weights, [0, 1, 1 / 3, 1 / 3, 1 / 3])
        self.assertAlmostEqual(weights.sum(), 2)

    def test_frozen_threshold_does_not_break_boundary_ties(self):
        probabilities = pd.Series([0.8, 0.8, 0.8, 0.7, 0.65] + [0.2] * 5)
        self.assertEqual(probabilities.ge(thresholds_at_budget(probabilities)).sum(), 0)
        probabilities.iat[0] = 0.9
        self.assertEqual(probabilities.ge(thresholds_at_budget(probabilities)).sum(), 1)

    def test_policy_falls_back_without_treating_missing_organs_as_normal(self):
        data = pd.DataFrame(
            {
                "sofa_observed": [2, 2, 2, 2],
                "sofa_components": [3, 3, 3, 1],
                "map_median": [70, 70, np.nan, 70],
                "hr_median": [80, 80, 80, 80],
                "rr_median": [15, 15, 15, 15],
                "lactate_latest": [2, np.nan, 2, 2],
            }
        )
        model = {
            "components": {
                name: dict(coefficients=dict.fromkeys(features, 0), intercept=i)
                for i, (name, features) in enumerate(FEATURES.items())
            }
        }
        scores, routes = linear_scores(data, model, "lactate")
        self.assertEqual(routes.tolist(), ["lactate", "vitals", "organ", "indeterminate"])
        np.testing.assert_allclose(scores.to_numpy(), [2, 1, 0, np.nan])

    def test_no_hours_at_or_after_scoring_landmark_are_used(self):
        data = pd.DataFrame(
            {
                "hr": [20, 23, 24],
                "map": [65, 75, 30],
                "map_nibp": [np.nan] * 3,
                "fio2": [np.nan] * 3,
                "po2": [np.nan] * 3,
                "bilirubin": [np.nan] * 3,
                "creatinine": [np.nan, np.nan, 500],
                "platelets": [160, 160, 10],
                "lactate": [2, 3, 20],
                "hr_bpm": [75, 85, 150],
                "rr_br": [18, 20, 40],
            }
        )
        features = extract_features(data, None, ({}, {}, {}), 1, 0)
        self.assertEqual(features["sofa_components"], 2)
        self.assertEqual(features["sofa_observed"], 1)
        self.assertEqual(features["map_median"], 70)
        self.assertEqual(features["hr_median"], 80)
        self.assertEqual(features["lactate_latest"], 3)

    def test_patient_cannot_appear_in_development_and_later_test(self):
        data = pd.DataFrame(
            {
                "patient_id": ["a", "a", "b", "c", "d", "d", "e"],
                "era": ["2003-2009"] * 5 + ["2010-2016"] * 2,
            }
        )
        train, calibration, later, removed = split(data)
        self.assertEqual(removed, 1)
        self.assertEqual(set(later.patient_id), {"d", "e"})
        self.assertEqual(set(train.patient_id) | set(calibration.patient_id), {"a", "b", "c"})
        self.assertFalse(set(train.patient_id) & set(calibration.patient_id))

    def test_local_thresholds_move_without_refitting_or_reading_outcomes(self):
        nonoverlap = pd.DataFrame({"patient_id": ["other"]})
        reference = pd.DataFrame({"patient_id": list("abcde")})
        reference["sofa_components"] = 3
        reference["sofa_observed"] = [0, 1, 2, 3, 4]
        for feature in FEATURES["lactate"][2:]:
            reference[feature] = np.nan
        model = {
            "schema": SCHEMA,
            "components": {
                name: dict(coefficients=dict.fromkeys(features, 1), intercept=0)
                for name, features in FEATURES.items()
            },
            "policies": {
                name: dict(slope=1, intercept=-4, threshold=0.5) for name in FEATURES
            },
        }
        local, report = transfer_thresholds(reference, nonoverlap, model)
        self.assertEqual(local["components"], model["components"])
        self.assertEqual(model["policies"]["lactate"]["threshold"], 0.5)
        self.assertGreater(local["policies"]["lactate"]["threshold"], 0.5)
        self.assertEqual(report["policies"]["lactate"]["alerts"], 1)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            transfer_thresholds(reference, reference.iloc[:1], model)


if __name__ == "__main__":
    unittest.main()
