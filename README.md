# sepsis-llm

A Clinically Auditable Agentic Large Language Model System for Sepsis
Decision Support under Data Missingness.

## Repository contents

- `sepsis_agentic_cdss.py` — the frozen execution engine of the
  Agentic-LLM CDSS: Agent 2 trajectory-imputation formula
  (`lactate[0] * 0.85 + sofa_delta * 0.20 + (10 - hr_sd) * 0.05 +
  rr_sd * 0.03`) and Agent 3 deterministic audit rules
  (Delta SOFA-2 >= 2, OR 48 h lactate > 2.0 mmol/L with clearance < 10%,
  OR HR SD < 10 bpm with Delta SOFA-2 > 0).
- `requirements.txt` — Python dependencies for the frozen engine and the
  validation pipeline.
- `amsterdam_validation/` — fully frozen external validation of the engine
  on AmsterdamUMCdb (2003-2016; n = 3,949 Sepsis-3 encounters), executed
  after model freeze with no refitting, recalibration, or threshold
  adjustment. See `amsterdam_validation/README.md` for the cohort
  definition, pipeline scripts, machine-readable results, and key numbers.
- `engine_v2/` (under `amsterdam_validation/`) — Engine v2, the post-hoc
  architectural revision: interpolation-first lactate imputation with a
  MIMIC-IV-trained XGBoost fallback for anchor-free trajectories
  (trained models shipped as JSON).
- `nejmai_submission/` — reproducibility scripts for the NEJM AI
  submission package:
  - `ceiling_diagnostic.py` — information-theoretic ceiling analysis
    (in-site XGBoost vs oracle vs interpolation vs frozen formula).
  - `ceiling_search.py` — repeated 5-fold cross-validation comparing
    interpolation, interpolation+OLS, blend, MIMIC-XGB, and in-site XGB.
  - `engine_v2.py` — cross-site variant: MIMIC-trained XGBoost tested on
    AmsterdamUMCdb (all complete cases and the masked-subset view).
  - `engine_v2_final.py` — final Engine v2 training (saves deployable
    JSON models) plus Arm A/B benchmark and coverage analysis.
  - `build_nejmai_figures.py` — journal-standard vector figures
    (Arial, fonttype 42, no chartjunk) for the main manuscript and
    supplementary appendix.
  - `build_nejmai_package.py` — assembly of the main manuscript,
    supplementary appendix, and cover letter (structured abstract,
    word-count compliance checks, Table 1-2 transfer).
  Note: scripts contain absolute paths to the local analysis workspace;
  edit the `DATA` / `OUT` constants at the top to reproduce.

## Development and validation cohorts

| Cohort | Role | n |
|---|---|---|
| MIMIC-IV v2.2 | Development (feature selection, tuning, training) | 17,292 |
| eICU v2.0 | External validation (pre-freeze) | 11,029 |
| Ruijin Hospital 2022-2025 | Frozen retrospective clinical validation | 200 |
| AmsterdamUMCdb 2003-2016 | Fully frozen additional external validation | 3,949 |

## Citation / reference

The system and its multicenter evaluation are described in:
"A Clinically Auditable Agentic Large Language Model System for Sepsis
Decision Support under Data Missingness: A Multicenter Development and
Retrospective Validation Study" (manuscript under revision).

AmsterdamUMCdb is available to credentialed researchers via
https://amsterdammedicaldatascience.nl/amsterdamumcdb/ after completion of
the required data-use procedures. No patient-level data are distributed in
this repository.
