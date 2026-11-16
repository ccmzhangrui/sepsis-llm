import unittest

import numpy as np
import pandas as pd

from review_policy import breakdown, review_threshold


class ReviewPolicyTests(unittest.TestCase):
    def test_cutoff_counts_preexisting_alerts_and_unscored_events_and_includes_ties(self):
        deaths = pd.Series([1, 1, 1, 1, 0, 0])
        risk = pd.Series([0.9, np.nan, 0.8, 0.2, 0.2, 0.1])
        alerts = pd.Series([True, False, False, False, False, False])
        cutoff = review_threshold(deaths, risk, alerts, 1.0)
        report = breakdown(pd.DataFrame({"death_7d": deaths, "death_72h": deaths}), risk, alerts, cutoff)
        self.assertEqual(cutoff, 0.2)
        self.assertEqual(report["urgent_alerts"], 1)
        self.assertEqual(report["secondary_review_unscored"], 1)
        self.assertEqual(report["any_alert_or_review"], 5)
        self.assertEqual(report["7d_deaths_missed"], 0)

    def test_reference_full_capture_is_not_reported_as_evaluation_full_capture(self):
        reference = pd.Series([1, 1, 0])
        risks = pd.Series([0.9, 0.4, 0.2])
        alerts = risks.ge(0.7)
        cutoff = review_threshold(reference, risks, alerts, 1.0)
        evaluation = pd.DataFrame({"death_7d": [1, 0, 1], "death_72h": [1, 0, 0]})
        self.assertEqual(breakdown(evaluation, risks, alerts, cutoff)["7d_deaths_missed"], 1)
        all_covered = breakdown(evaluation, risks, alerts, cutoff, review_all=True)
        self.assertEqual(all_covered["urgent_alerts"], 1)
        self.assertEqual(all_covered["7d_deaths_missed"], 0)
        self.assertEqual(all_covered["additional_review_for_full_coverage"], 1)


if __name__ == "__main__":
    unittest.main()
