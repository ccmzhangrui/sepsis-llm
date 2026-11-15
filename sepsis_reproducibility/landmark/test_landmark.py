import unittest

import numpy as np
import pandas as pd

from prepare_amsterdam import HOUR, confirmed_hour, hr_summary, overlap_bins
from reconstruct import reconstruct
from validate import disjunction, report, rule_status


class LandmarkTests(unittest.TestCase):
    def row(self, **fields):
        values = dict(
            patient_id="synthetic",
            icu_admit_hour=0.0,
            icu_discharge_hour=90.0,
            infection_confirmed_hour=6.0,
            death_hour=np.nan,
            vital_followup_hour=672.0,
            sofa_base_low=3,
            sofa_base_high=3,
            sofa_late_low=4,
            sofa_late_high=4,
            matched_base=3,
            matched_late=4,
            matched_organs=5,
            lac_base=np.nan,
            lac_late=np.nan,
            hr_sd=3,
            hr_windows=1,
        )
        return {**values, **fields}

    def test_unknown_lactate_and_insufficient_heart_rate_are_not_negative(self):
        rules = rule_status(pd.DataFrame([self.row()]))
        self.assertFalse(rules.sofa.iloc[0])
        self.assertTrue(pd.isna(rules.heart_rate.iloc[0]))
        self.assertTrue(pd.isna(disjunction(rules).iloc[0]))

    def test_positive_observed_rule_survives_other_missing_inputs(self):
        rules = rule_status(pd.DataFrame([self.row(lac_base=3, lac_late=3)]))
        self.assertTrue(disjunction(rules).iloc[0])

    def test_unresolved_bounds_and_matched_score_are_distinct(self):
        data = pd.DataFrame(
            [
                self.row(
                    sofa_late_low=5, sofa_late_high=9, sofa_base_high=7, matched_late=6
                )
            ]
        )
        self.assertTrue(pd.isna(rule_status(data).sofa.iloc[0]))
        self.assertTrue(rule_status(data, 4).sofa.iloc[0])
        self.assertTrue(pd.isna(rule_status(data, 6).sofa.iloc[0]))

    def test_landmark_excludes_early_events_late_eligibility_and_incomplete_followup(
        self,
    ):
        rows = [self.row(death_hour=x) for x in [48, 49, 672, 673]]
        rows.append(self.row(infection_confirmed_hour=49))
        rows.append(self.row(vital_followup_hour=100))
        flow = report(pd.DataFrame(rows))["flow"]
        self.assertEqual(flow["analyzed_stays"], 3)
        self.assertEqual(flow["deaths"], 2)
        self.assertEqual(flow["unknown_outcome_among_eligible"], 1)

    def test_hourly_input_and_support_respect_cutoff(self):
        self.assertEqual(overlap_bins(48 * HOUR, 49 * HOUR, 0), slice(48, 48))
        self.assertEqual(overlap_bins(-HOUR, 2 * HOUR, 0), slice(0, 2))
        self.assertEqual(overlap_bins(-3 * HOUR, -2 * HOUR, 0), slice(0, 0))
        self.assertEqual(confirmed_hour([4 * HOUR, 9 * HOUR], 7 * HOUR, 7 * HOUR), 2)
        hr = np.full(48, np.nan)
        hr[3] = 85
        self.assertEqual(hr_summary(hr, 0)[1], 1)
        self.assertTrue(np.isnan(hr_summary(hr, 0)[0]))

    def test_reconstruction_preserves_measurement_and_uses_original_anchors(self):
        self.assertEqual(
            reconstruct({0: 2.0, 24: 6.0}),
            {6: (3.0, "interpolated"), 12: (4.0, "interpolated")},
        )
        self.assertEqual(
            reconstruct({0: 2, 6: 3}), {6: (3, "observed"), 12: (None, "indeterminate")}
        )


if __name__ == "__main__":
    unittest.main()
