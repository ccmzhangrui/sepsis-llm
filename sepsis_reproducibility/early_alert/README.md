# 24-hour suspected-infection alert: external evaluation

This research workflow estimates the chance of recorded death in the seven days
after a single assessment, 24 hours after suspected infection. It was developed
in AmsterdamUMCdb admissions from 2003–2009 and tested in held-out admissions
from 2010–2016, with no patients shared between periods. It is a prognostic
study, separate from the [48-hour rule replay and lactate reconstruction](../landmark/README.md).
It is intended for retrospective or silent evaluation. Transfer to clinical
alerting has not been established.

## Run the frozen model locally

Start in this directory and use Python 3.11 or later. Prepare one row per eligible
stay using the definitions below. Keep clinical files and any patient-level
derivatives outside the repository.

```sh
python -m pip install -r requirements.txt
python evaluate.py evaluate --input /path/to/site_evaluation.csv \
  --model amsterdam_24h_v1.json --output /path/to/site_summary.json
```

The output gives counts, coverage, the share and number of stays flagged, the
share of recorded 7-day deaths flagged, positive predictive value,
discrimination, Brier score, mean risk versus observed risk, and paired
patient-cluster bootstrap intervals. The secondary endpoint is recorded death
within 72 hours after assessment. The equal-volume analysis ranks the *same*
scored stays with known outcomes and assigns the top 20% (rounded down) to each
strategy, allocating tied slots fractionally. It compares rankings after the
fact; it cannot set a threshold for incoming patients.

The frozen threshold was chosen separately for each strategy using an older
development calibration subset, targeting at most 20% of scored stays.
In the later Amsterdam test set of 2,642 eligible stays, the lactate-inclusive
strategy scored 2,608 and flagged 44.7% of eligible stays, demonstrating that a
nominal alert capacity does not transfer automatically. At equal volume
(521 slots), it identified 149 recorded deaths, compared with 130 for organ
score plus vitals and an expected 116.8 for organ score alone (tied scores).

### Optional local threshold comparison

Use an **earlier**, separate local reference cohort. Freeze local threshold
estimates from those scores before evaluating on the later site cohort. The
program rejects patient overlap; the site must enforce chronological separation.
Outcome labels are unnecessary in the reference CSV. The original Amsterdam
threshold results remain in the same report for comparison. This step only
adjusts the operating threshold; it does not recalibrate risk probabilities or
guarantee the same alert rate in the later sample.

```sh
python evaluate.py evaluate --input /path/to/site_evaluation.csv \
  --model amsterdam_24h_v1.json --capacity-reference /path/to/earlier_site_reference.csv \
  --output /path/to/site_summary_with_local_thresholds.json
```

### Compare three operating points using patient counts

`site_thresholds.py` keeps the same risk model and compares the original
cutoff with two cutoffs estimated from an unlabeled local reference: an
empirical quantile and a more conservative order statistic. The target is
the fraction of eligible **patients** with any alert (default 20%), using
each patient's maximum score across their eligible stays. The report also
shows workload per stay, missed 7-day deaths, and positive predictive value.
Unscored stays and patients stay in the workload denominators. The patient
horizon and collection of eligible stays must match across cohorts.

```sh
python site_thresholds.py compare --reference /path/to/recent_local_reference.csv \
  --input /path/to/later_local_evaluation.csv --model amsterdam_24h_v1.json \
  --target 0.20 --output /path/to/site_workload.json
```

Use separate periods close to the intended evaluation period, with no repeated
patients across them. The buffering rule chooses the largest number of
reference scores allowed above the cutoff whose lower-tail binomial CDF under
the target is at most 0.05. The one-sided 95% interpretation concerns the
population rate of patients with any alert under independent, identically
distributed patient scores. It is not a cap on the observed rate or a remedy
for drift. The command reports reference insufficiency rather than substituting
another threshold silently. `stress` provides descriptive repeated patient
splits within one cohort; these are not independent validation studies.

In the Amsterdam test period, a repeated within-period check found median
workloads of 44.7, 18.9, and 16.4 alerts per 100 stays respectively, with
median 7-day death capture of 68.5%, 44.9%, and 42.8%. However, the buffered
cutoff estimated from 2003–2009 still flagged 43.8 per 100 stays in 2010–2016.
Treat old local reference data as a possible source of threshold failure.

## Cohort and inputs

Suspected infection is anchored to the earlier timestamp of a paired IV
antibiotic administration and body-fluid culture order. A culture must follow
the antibiotic within 24 hours; after a culture the antibiotic must follow within
72 hours; either way, both must be documented by 24 hours after the earlier
timestamp for cohort eligibility. Amsterdam selected the first qualifying pair
using its published local extraction code (linked below). Include stays alive
and still in critical care at hour 24; the Amsterdam critical-care database
includes ICU and medium-care settings. Count elapsed hours from the infection
anchor; predictors use only the completed bins in `[0, 24)`. For site evaluation,
use only results available by assessment time where availability timestamps
exist. Amsterdam retrospectively used chart measurement, order, and therapy
interval times. It cannot establish when every result was available to clinicians.

| Field | Definition |
| --- | --- |
| `stay_id` | Unique stay identifier; stays occupy one row each. |
| `patient_id` | Stable patient identifier shared across repeat stays for grouping. |
| `snapshot_hour` | Must equal 24. |
| `sofa_observed` | Sum of the worst observed score for each of six organs during `[0, 24)`; Amsterdam uses the [local SOFA-2 adaptation](../../amsterdam_validation/run_cdss_validation.py). Use the same scoring and units, including recorded support; leave absent components unscored. This partial sum is not a complete SOFA-2 total. |
| `sofa_components` | Number of organ components observed (0–6); at least two and a nonmissing sum are required to score. |
| `map_median` | Median of hourly MAP in mm Hg over hours 18–23; use the lower of available invasive and noninvasive hourly values, then retain 20–180. Require two or more observed hours. |
| `hr_median`, `rr_median` | Medians of hourly heart rate (beats/min, 20–300) and respiratory rate (breaths/min, 3–100) over hours 18–23. Require two or more observed hours for each. |
| `lactate_latest` | Most recent available hourly median lactate in hours 12–23 (mmol/L, 0.1–40). Use a measured value or leave blank. |
| `death_7d`, `death_72h` | Code 1 for recorded death in `(24, 192]` or `(24, 96]` hours from anchor, respectively, and 0 when the respective event was ascertainable but not observed. Leave blank if unknown. The Amsterdam analysis used its available death record, treating absence of a recorded death as no recorded event. Document follow-up and death ascertainment at the external site. |
| `hours_to_recorded_7d_death` | Optional elapsed hours from the 24-hour assessment to recorded death, for known 7-day deaths only. Reports median and quartiles for alerts with a timestamp. This is elapsed time to recorded death, not measured gain over standard care. |

Leave missing values blank rather than using a sentinel. The three strategies
use ridge logistic models: observed organ score and component count; the same
organ features plus all three vital signs; and the same vital-sign model inputs
plus observed lactate. The richer strategies fall back to the next supported
model when inputs are missing. Stays with insufficient organ data remain
indeterminate. Heart-rate variability, lactate clearance, and imputed values do
not enter this model. The lactate comparison concerns the change in the whole
scoring strategy and can reflect differences in who had lactate measured.

## Reproduce the Amsterdam analysis

Use authorized AmsterdamUMCdb data and the
[source extraction scripts](../../amsterdam_validation/README.md) to prepare
`cohort_suspected_infection.csv`, `hourly_features.csv`, `hourly_gcs.csv`,
`vent_intervals.csv`, `rrt_intervals.csv`, and `vaso_intervals.csv`, all
aligned relative to suspected infection. Raw admissions and culture-order tables
are also required. The adapter adds the original Amsterdam admission-year band
as `era`. Development splits the earlier group into patient-disjoint fit
and calibration sets; the later group is not used to tune or calibrate.

```sh
python extract_amsterdam.py --hourly-dir /path/to/prepared_tables \
  --admissions /path/to/admissions.csv --culture-orders /path/to/procedureorderitems.csv \
  --output /path/to/private_snapshot.csv
python evaluate.py develop --input /path/to/private_snapshot.csv \
  --model /path/to/rebuilt_model.json --output /path/to/aggregate_summary.json
python -m unittest -v
```
