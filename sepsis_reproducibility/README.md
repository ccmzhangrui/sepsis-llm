# Structured sepsis analysis: reproducibility companion

This directory reproduces the retrospective, rules-only analysis on the harmonized AmsterdamUMCdb matrix described in the repository. It has no model API, clinical recommendation output, or claim of validation of a generative LLM. The 6- and 12-hour imputed lactate values do not enter the alert rules. No refitting or threshold selection is performed by this command.

## Run locally with credentialed data

Use Python 3.11 or later and the access-controlled AmsterdamUMCdb data that you obtained through its approved process. The legacy cohort scripts are in `amsterdam_validation/`. Prepare the `validation_patient_level.csv` matrix using that pipeline and keep it outside the repository.

```sh
cd sepsis_reproducibility
python -m pip install -r requirements.txt
python -m unittest -v
python reproduce.py --matrix /path/to/validation_patient_level.csv --admissions /path/to/admissions.csv --output /path/to/a_new_aggregate_report.json
```

The required fields are unique `admissionid`; `sofa0` and `sofa48`; heart-rate and respiratory-rate standard deviations `hr_sd` and `rr_sd`; measured lactate `lac0`, `lac6`, `lac12`, `lac24`, `lac48` (mmol/L; missing values blank); and binary `alert_red`, `mort28`, `vaso_escalation`. If supplied, `deterioration_ref` must equal mortality or vasopressor escalation. The admissions file supplies `admittedat`, `dateofdeath` (same millisecond origin), and `patientid`. The corrected 28-day endpoint uses `dateofdeath - admittedat` and excludes negative offsets. The composite is recomputed; risk-ratio confidence intervals group repeat admissions by patient. This endpoint differs from the original legacy script, which used the unadjusted death timestamp. Output contains aggregate estimates and the input file fingerprint, with no row-level records.

The rule replay uses fractional lactate clearance and a threshold of 0.10 (10%). Reconstruction uses two pooled masked measurements per eligible complete encounter and is retrospective; this calculation is separate from the alert replay. Later lactate availability (24/48 hours) and bracketed availability are separate measures. The legacy repository also includes the held-out portability comparison and the exploratory Engine v2 fallback; these are separate from the pooled benchmark in this directory.

Do not commit credentialed data, clinical records, local data paths, or patient-level outputs. This is research code for reproducing a retrospective analysis.

## Additional cohort analyses

See [the eICU, MIMIC-IV, and Ruijin replay instructions](multicohort/README.md) for the separate exploratory analyses and local input requirements.

## Independent-validation candidate

The [landmark workflow](landmark/README.md) provides the updated 48-hour alert comparison, missing-input handling, the patient-grouped lactate benchmark, and a frozen two-anchor imputer with a separate external evaluation command.
