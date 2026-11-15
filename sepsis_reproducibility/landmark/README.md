# Frozen research protocol for independent validation

The 48-hour analysis compares three fixed strategies: adapted SOFA-2 increase alone, increase or lactate clearance, and the union of all three original rules. Report red, no red, and indeterminate statuses on the entire eligible cohort. Use the common determinate cohort when comparing the same stays, and report its coverage alongside the outcome counts. A wide interval or small common cohort limits what can be concluded about incremental value.

The imputation test is a separate retrospective experiment. It compares two-anchor linear interpolation with a supervised two-anchor predictor extracted from an iterative-imputer fit in AmsterdamUMCdb. The frozen coefficients come from the training partition only. Both methods require observed 0-hour and 24-hour lactate. Keep measured values unchanged. When reconstructing a naturally missing value, mark it indeterminate if it has no two observed brackets. Neither imputed value feeds an alert. The included Amsterdam benchmark compares models on the same anchors with a patient-disjoint holdout, clustered bootstrap intervals and group-wise cross-validation for tuning.

## Run on a local independent cohort

Use Python 3.11 or later. From the repository root:

```sh
python -m pip install -r sepsis_reproducibility/requirements.txt
python sepsis_reproducibility/landmark/validate.py --input /path/to/local_landmark.csv --output /path/to/local_alert_summary.json
python sepsis_reproducibility/landmark/external_imputation.py --input /path/to/local_lactate.csv --output /path/to/local_lactate_summary.json
```

The external lactate CSV has one row per `stay_id` and measured `lac0,lac6,lac12,lac24` (mmol/L; blanks for missing). Use actual equivalent collection windows, and record the source time and its mapping. The command scores the frozen regression without fitting on the external cohort.

### Alert input format

One row per adult ICU stay meeting the source eligibility definition. Keep patient and encounter keys local. Use hours since the first event of the qualifying antibiotic–culture pair, which is time zero. Include timing at the same precision used to determine ICU presence and outcome. Use no observations acquired at or after hour 48. Where result-availability times exist, require features to have been available by the landmark. A flag from this research evaluation has no clinical-action threshold.

| Columns | Construction |
|---|---|
| `stay_id,patient_id` | Unique local encounter key and stable parent patient key. |
| `icu_admit_hour,icu_discharge_hour` | ICU entry and exit. An eligible patient is alive and in the ICU at hour 48. |
| `infection_confirmed_hour` | Time of the later event in the qualifying pair. Both events must be known by hour 48. |
| `baseline_end_hour,features_through_hour` | The end of the first qualifying completed hourly bin in the first 25 bins, and 48. Derive qualifying organ dysfunction from the observed lower score bound of at least 2. |
| `death_hour,vital_followup_hour` | Time of recorded death, if available, and last hour of vital status ascertainment. The outcome includes deaths strictly after hour 48 and through ICU admission plus 28 days. Alive-at-horizon requires ascertainment through that horizon. Keep unknown outcomes blank. |
| `sofa_base_low,sofa_base_high,sofa_late_low,sofa_late_high` | Six adapted organ subscores (0–4 each). Use the worst observed subscore over the previous 24 completed hourly bins (a shorter window at baseline). An unavailable component contributes the interval 0–4, yielding lower and upper total bounds. A rule can be positive or negative when its threshold is unambiguous over the full range. |
| `matched_base,matched_late,matched_organs` | Sum formed at both endpoints using exactly the same subset of observed organ systems, and the number of systems in that subset (0–6). Use 0 for the two sums when the count is zero. The candidate analysis requires at least four shared systems; the five-system and full-range analyses assess sensitivity to coverage. These sums are partial scores. |
| `lac_base,lac_late` | Median of available positive lactate measurements (mmol/L) in completed baseline bins 0–3 hours back and bins 45–47 at follow-up. If unavailable, leave blank. |
| `hr_sd,hr_windows` | Population SD across measured heart-rate window medians in beats/min and number of medians (0–5). Assign each hourly median once to the nearest of baseline, baseline + 6, +12, +24 and bin 47, within three bins after the baseline. Use source heart rates 20–300 beats/min and omit targets after bin 47. Ties go to the earlier target. Leave SD missing with fewer than two window medians. |

Fixed thresholds are a score rise ≥2, measured late lactate >2 mmol/L with fractional clearance <0.10, and low heart-rate SD <10 beats/min with a score rise >0. A known positive rule makes the corresponding strategy positive. A negative strategy requires all its included rules to be known negative. A mixed set with an unresolved rule remains indeterminate. The report includes denominators and recorded deaths for each status, per-rule summaries, and crude patient-clustered risk-ratio intervals where cell counts support them. There is no refitting of alert thresholds to the external outcomes.

## Reproduce the Amsterdam exploratory analysis

The input adapter consumes the already-derived AmsterdamUMCdb hourly tables and the original admissions and procedure-order CSVs. The repository's existing Amsterdam pipeline describes the source tables and units. The adapter uses measured or event times as recorded. Delayed chart availability was not reconstructed in this evaluation. The registry endpoint is recorded death by day 28. The Amsterdam adapter operationally treats no recorded eligible death as a non-event, so differences in ascertainment must be reported before comparing cohorts.

```sh
python sepsis_reproducibility/landmark/prepare_amsterdam.py --hourly-dir "$AMSTERDAM_DERIVED" --admissions "$AMSTERDAM_ADMISSIONS" --culture-orders "$AMSTERDAM_ORDERS" --output "$PRIVATE_WORK/landmark_input.csv"
python sepsis_reproducibility/landmark/validate.py --input "$PRIVATE_WORK/landmark_input.csv" --output "$PRIVATE_WORK/landmark_results.json"
python -m pip install -r sepsis_reproducibility/multicohort/requirements.txt
python sepsis_reproducibility/landmark/benchmark_imputation.py --matrix "$AMSTERDAM_MATRIX" --admissions "$AMSTERDAM_ADMISSIONS" --output "$PRIVATE_WORK/imputation_results.json" --frozen-model "$PRIVATE_WORK/two_anchor_model.json"
python -m unittest discover -v -s sepsis_reproducibility/landmark
```

AmsterdamUMCdb was used to revise this protocol. Its results are exploratory. The eICU replay in `multicohort/` uses ICU admission as its index and its own derived scores. The fixed infection-indexed landmark protocol requires independently mapped inputs and follow-up. Share code, definitions and aggregate results only.
