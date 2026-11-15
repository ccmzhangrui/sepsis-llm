import unittest

import numpy as np
import pandas as pd
from scipy.stats import binom

from evaluate import DEATH_TIME
from site_thresholds import estimate_threshold, patient_max, workload


class SiteThresholdTests(unittest.TestCase):
    def test_empirical_cutoff_does_not_alert_ties_or_unknown_scores(self):
        risks = pd.Series([0.7, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, np.nan, np.nan])
        result = estimate_threshold(risks, 0.20)
        self.assertEqual(result["reference_patients"], 10)
        self.assertEqual(result["reference_patients_flagged"], 2)
        with_boundary_tie = estimate_threshold(risks, 0.10)
        self.assertEqual(with_boundary_tie["reference_patients_flagged"], 0)

    def test_buffer_respects_one_sided_coverage_target(self):
        scores = pd.Series(np.arange(100) / 100)
        result = estimate_threshold(scores, 0.20, confidence=0.95)
        empirical = estimate_threshold(scores, 0.20)
        allowed = result["allowed_reference_ranks"]
        self.assertLessEqual(binom.cdf(allowed, 100, 0.20), 0.05)
        self.assertGreater(binom.cdf(allowed + 1, 100, 0.20), 0.05)
        self.assertGreater(result["threshold"], empirical["threshold"])

    def test_small_or_unscored_reference_reports_inadequacy(self):
        self.assertEqual(
            estimate_threshold(pd.Series([0.7] * 10), 0.20, confidence=0.95)["status"],
            "too_few_reference_patients_for_buffer",
        )
        self.assertEqual(
            estimate_threshold(pd.Series([np.nan] * 20), 0.20)["status"],
            "no_scored_reference_patients",
        )

    def test_workload_uses_distinct_patient_and_stay_denominators(self):
        data = pd.DataFrame(
            {
                "patient_id": ["a", "a", "b", "c"],
                "death_7d": [1, np.nan, 1, 0],
                "death_72h": [1, np.nan, 0, 0],
                DEATH_TIME: [24, np.nan, 96, np.nan],
            }
        )
        scores = pd.Series([0.8, 0.9, np.nan, 0.1])
        self.assertEqual(len(patient_max(data, scores)), 3)
        self.assertTrue(np.isnan(patient_max(data, scores)["b"]))
        result = workload(data, scores, 0.60)
        self.assertAlmostEqual(result["patients_with_alert_per_100"], 100 / 3)
        self.assertEqual(result["alerts_per_100_stays"], 50)
        self.assertEqual(result["death_7d_missed"], 1)
        self.assertEqual(result["alerts_with_unknown_7d_outcome"], 1)
        self.assertEqual(result["death_7d_ppv_known"], 1.0)


if __name__ == "__main__":
    unittest.main()
