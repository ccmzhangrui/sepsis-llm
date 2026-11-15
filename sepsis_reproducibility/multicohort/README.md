# Cohort replay for eICU, MIMIC-IV, and Ruijin

The parent directory contains the AmsterdamUMCdb replay, frozen rules and shared statistics. The frozen XGBoost boosters are tracked in `amsterdam_validation/engine_v2`. Keep patient-level files and intermediate datasets in access-controlled local storage.

| Source | Local inputs |
|---|---|
| eICU | `patient.csv`, `admissionDx.csv`, `diagnosis.csv`, `lab.csv`, `nurseCharting.csv`, `infusionDrug.csv`, and the supplied derived `eicu_sofa2_hourly_wide.csv` with matching stay identifiers. |
| MIMIC-IV | Serial numeric workbook, matching derived hourly SOFA-2 export, and the historical `mimic_full.pkl` training parent. Use a trusted local pickle because Python deserializes its contents. |
| Ruijin | The supplied completed validation registry and the separate daily 14-day SOFA-2 workbook. |

Set each input variable to its local absolute path. Run from the repository or from the release's `GitHub_Code` directory.

```bash
python -m pip install -r sepsis_reproducibility/multicohort/requirements.txt

python sepsis_reproducibility/multicohort/extract_eicu.py "$EICU_DIR" "$PRIVATE_WORK_DIR"

python sepsis_reproducibility/multicohort/prepare_workbooks.py \
  --mimic "$MIMIC_WORKBOOK" --ruijin "$RUIJIN_WORKBOOK" --work "$PRIVATE_WORK_DIR"

python sepsis_reproducibility/multicohort/run_followup.py \
  --work "$PRIVATE_WORK_DIR" --output "$PRIVATE_WORK_DIR/aggregate_results.json" \
  --frozen-source sepsis_reproducibility/reproduce.py \
  --models-dir amsterdam_validation/engine_v2 --eicu-dir "$EICU_DIR" \
  --mimic-training-parent "$MIMIC_TRAINING_PARENT" --mimic-sofa "$MIMIC_SOFA_EXPORT" \
  --ruijin-daily "$RUIJIN_DAILY_WORKBOOK"
```

The optional `--ruijin-selection "$PRIVATE_WORK_DIR/ruijin_selection.csv"` saves the local encounter index for the 72-hour exploration. This file contains restricted identifiers.

The exploratory eICU replay uses ICU admission as time zero and restricts mortality comparisons to stays observed at 48 hours. A negative status requires all three rules to be evaluable. The aggregate JSON also reports the effect of pooling indeterminate and negative stays, as in the legacy available-input implementation. Deaths following hospital discharge are outside the eICU endpoint.

The MIMIC-IV benchmark uses the workbook slots and audits overlap with training. The all-zero endpoint SOFA-2 scores are reported as a data-quality finding. The Ruijin 72-hour analysis uses score change in registry keys with internally consistent fields. These analyses use different time origins, input definitions, and endpoints.
