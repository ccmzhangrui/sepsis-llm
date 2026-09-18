#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the NEJM AI submission package:
  1. sepsis_decision_model_nejmai.docx  (main manuscript, <=3,000 words
     Introduction->Discussion, structured abstract <=300 words,
     1-2 sentence description, Tables 1-2 embedded)
  2. supplementary_appendix_nejmai.docx (TOC + Notes + Tables S1-S14 +
     S-figure legends; tables deep-copied from the Amsterdam manuscript)
  3. cover_letter_nejmai.docx

Numbers are transcribed from verified pipeline JSONs (2026-09-18).
"""
import copy
import os
from docx import Document
from docx.shared import Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

SRC = "/Users/zhangrui/Desktop/脓毒症/脓毒症大模型文章/sepsis_decision_model_revised_major_revision_amsterdam.docx"
OUTDIR = "/Users/zhangrui/Desktop/脓毒症/脓毒症大模型文章/NEJMAI_submission"
os.makedirs(OUTDIR, exist_ok=True)

src = Document(SRC)   # source of table DATA only (rebuilt, not copied)


def table_rows(tbl):
    """Extract non-empty rows of cell texts from a source table."""
    out = []
    for r in tbl.rows:
        vals = [c.text.strip() for c in r.cells]
        if any(v for v in vals):
            out.append(vals)
    return out

TITLE = ("A Clinically Auditable Agentic Large Language Model System for "
         "Sepsis Decision Support under Data Missingness: A Multicenter "
         "Development and Retrospective Validation Study")
# (name, superscript) — NEJM AI author-order compliance:
#   co-first authors first, then contributing authors, then co-senior
#   authors; ONE corresponding author (NEJM AI permits only one).
AUTHORS = [("Rui Zhang MD", "1†"),
           ("Xu Wang PhD", "2†"),
           ("Shengjun Liu MD", "3†"),
           ("Jingyi Wu PhD", "1"),
           ("Jie Huang PhD", "1"),
           ("Wencui Wang", "1"),
           ("Jiahui Wang MD", "1"),
           ("Lei Pei MD", "1"),
           ("Ruoming Tan", "1"),
           ("Lei Li", "1"),
           ("Yun Long MD", "3‡"),
           ("Longxiang Su MD", "3‡"),
           ("Hongping Qu MD", "1‡*")]
AFFILIATIONS = [
    ("1", "Department of Critical Care Medicine, Ruijin Hospital, "
          "Shanghai Jiao Tong University School of Medicine, Shanghai, "
          "China"),
    ("2", "Zhongshan Hospital, Fudan University, Shanghai, China"),
    ("3", "Department of Critical Care Medicine, State Key Laboratory of "
          "Complex Severe and Rare Diseases, Peking Union Medical College "
          "Hospital, Beijing, China"),
]
EQUAL = "† Drs. Zhang, Wang, and Liu contributed equally to this manuscript."
SENIOR = "‡ Drs. Long, Su, and Qu contributed equally as co-senior authors."
CORR = ("* Corresponding author: Hongping Qu (qhp10516@rjh.com.cn), "
        "Department of Critical Care Medicine, Ruijin Hospital, Shanghai "
        "Jiao Tong University School of Medicine, 197 Ruijin Er Road, "
        "Shanghai 200025, China")

DESCRIPTION = ("A clinically auditable agentic large-language-model system "
               "for sepsis decision support under data missingness, "
               "evaluated across four cohorts including a fully frozen "
               "external validation in AmsterdamUMCdb.")

ABSTRACT = {
 "Background":
 "Large language models (LLMs) may support clinical summarization and "
 "decision support, but their use in early sepsis is limited by "
 "hallucination risk, weak native handling of irregular time series, and "
 "missing values such as serial lactate. We developed and retrospectively "
 "validated a clinically auditable agentic-LLM system integrating temporal "
 "extraction, frozen trajectory imputation, and deterministic audit gating.",
 "Methods":
 "In a multicenter retrospective study using MIMIC-IV (development, "
 "n=17,292), eICU (external validation, n=11,029), a Ruijin Hospital "
 "clinical validation cohort (2022-2025, n=200), and AmsterdamUMCdb "
 "(fully frozen additional external validation, n=3,949), Agent 1 "
 "extracted structured and unstructured data into a six-domain temporal "
 "matrix; Agent 2 imputed missing trajectory features with a frozen "
 "XGBoost regressor; Agent 3 generated risk-stratification alerts only "
 "after deterministic rule-based audit checks.",
 "Results":
 "Extraction precision for serial SOFA-2 scores was 99.2% (95% CI "
 "98.4-99.7) in MIMIC-IV, 98.8% (97.9-99.4) in eICU, and 99.4% "
 "(98.6-99.9) in Ruijin. Agent 2 reconstructed lactate trajectories with "
 "R2 0.92 in development. Agent 3 alerts showed 96.5% concordance with "
 "blinded expert consensus; no hallucinations were identified in the 200 "
 "audited cases. In fully frozen external validation in AmsterdamUMCdb, "
 "red alerts were associated with 28-day mortality (26.2% vs 16.2%; "
 "RR 1.61, 95% CI 1.42-1.83), composite deterioration (RR 1.80), and "
 "vasopressor escalation (RR 2.16). Frozen-formula imputation degraded "
 "(R2 0.48) below within-cohort interpolation (0.68); a redesigned "
 "interpolation-first engine (v2) restored reconstruction where "
 "bracketing anchors existed (R2 0.68-0.72) but remained "
 "information-limited without later measurements (R2 0.43).",
 "Conclusions":
 "A constrained, trajectory-informed agentic-LLM workflow with "
 "deterministic auditing is technically feasible and retrospectively "
 "reproducible across four cohorts. Serial-measurement density, rather "
 "than model architecture, limits trajectory-based "
 "decision support in sparse-sampling settings. Prospective "
 "clinical-effectiveness studies remain necessary.",
}

INTRODUCTION = [
 "Sepsis, defined as life-threatening organ dysfunction caused by a "
 "dysregulated host response to infection, remains a primary driver of "
 "mortality and intensive care unit (ICU) admissions worldwide [1]. Early "
 "sepsis care is highly dynamic and time-sensitive [2]: successful "
 "resuscitation during the initial golden hours relies on the rapid, "
 "sequential synthesis of heterogeneous and evolving physiological "
 "signals, including progressive organ dysfunction (Sequential Organ "
 "Failure Assessment-2 [SOFA-2] scores [3,4]), tissue hypoperfusion "
 "(lactate kinetics [5]), and hemodynamic support intensity [2].",

 "Clinical decision support systems (CDSS) face two translational "
 "barriers. Predictive machine-learning models are black boxes that "
 "require rigid, structured inputs and are highly sensitive to data "
 "missingness \u2014 serial lactate or vital-sign trends are frequently "
 "absent in real-world electronic health records (EHRs) [7-10]. "
 "Conversely, clinical applications of generative LLMs are limited "
 "by hallucination risk, weak native handling of temporal trajectories, "
 "and the absence of verifiable audit trails [11,12].",

 "We developed a clinically auditable Agentic-LLM Clinical Support "
 "System for early sepsis, drawing on clinical agentic architectures "
 "[13,14] and multimodal clinical state representations [15]. Here, "
 "\u2018agentic\u2019 denotes a constrained workflow in which components "
 "maintain a shared patient-state representation, call external tools, "
 "perform sequential state transitions, and pass outputs through "
 "verifiable safety gates, differing from a passive chatbot that answers "
 "a single prompt. We frame this work as a retrospective clinical "
 "informatics system study with four contributions: (i) sepsis decision "
 "support formulated as a trajectory-centered rather than "
 "snapshot-centered task; (ii) integration of machine-learning "
 "imputation with auditable LLM orchestration under prespecified "
 "constraints; (iii) evaluation across development, external validation, "
 "and retrospective clinical validation cohorts with strict "
 "development\u2013validation separation; and (iv) a fully frozen, "
 "zero-adaptation external validation in AmsterdamUMCdb, including a "
 "transparent portability analysis that motivated a redesigned "
 "interpolation-first imputation engine (v2).",
]

METHODS = [
 ("Data sources and study populations", [
  "We used four cohorts with strict development\u2013validation "
  "separation (Figure 1; Supplementary Figure S1). MIMIC-IV v2.2 "
  "(n=17,292 Sepsis-3 ICU encounters) [16] was used for temporal-schema "
  "construction, Boruta feature selection, XGBoost hyperparameter "
  "tuning, model training, and prompt optimization. The eICU "
  "Collaborative Research Database v2.0 (n=11,029) [17] was used only "
  "for pre-freeze external validation. Ruijin Hospital historical "
  "records (January 1, 2022 to December 31, 2025; n=200) were used for "
  "frozen retrospective clinical validation and simulated real-time "
  "auditing. AmsterdamUMCdb v1.0 (2003\u20132016; n=3,949) [23] served "
  "as a fully frozen additional external validation cohort: suspected "
  "infection was identified by the Seymour antibiotic\u2013culture "
  "pairing criteria [24], SOFA-2 was computed hourly with a 24-hour "
  "rolling maximum, and no feature selection, refitting, threshold "
  "adjustment, prompt modification, or site-specific adaptation of any "
  "kind was performed. Because blinded expert adjudication was not "
  "available for this public-data cohort, evaluation was limited to "
  "alert\u2013outcome association, imputation portability on masked "
  "complete-case values, and outcome characterization of indeterminate "
  "cases. Reporting was aligned with STROBE/RECORD for routinely "
  "collected data and organized by TRIPOD+AI-oriented transparency "
  "domains.",
 ]),
 ("The Agentic-LLM system", [
  "The system comprises three constrained agents built on a quantized "
  "LLaMA-3.1-70B-Instruct model (4-bit AWQ) running locally on NVIDIA "
  "H800 GPUs (Supplementary Note 1). Agent 1 extracts vital signs, "
  "laboratory values, vasopressor doses, organ-support data, and "
  "note-derived context from flowsheet tables and free-text notes into a "
  "six-domain temporal matrix across five windows (0, 6, 12, 24, 48 h), "
  "using schema-constrained JSON prompts, few-shot examples, retrieved "
  "source snippets, and explicit refusal rules. When critical trajectory "
  "variables (6 h or 12 h lactate) are missing, Agent 2 invokes a frozen "
  "XGBoost trajectory regressor developed in MIMIC-IV and frozen after "
  "eICU validation. Agent 3 synthesizes the 48 h patient-state matrix "
  "into risk-stratification alerts and concise summaries, governed by a "
  "deterministic mathematical auditing layer: every generated statement "
  "must map to a verifiable data element, temporal delta, or "
  "prespecified rule (e.g., delta SOFA-2 \u22652, lactate clearance "
  "<10%, worsening mean arterial pressure despite vasopressor "
  "escalation); otherwise the recommendation is suppressed with a "
  "manual-review flag. After eICU validation, the feature list, "
  "preprocessing pipeline, model weights, thresholds, prompt templates, "
  "and audit rules were frozen; neither the Ruijin nor the "
  "AmsterdamUMCdb cohort was used for any form of adaptation "
  "(Supplementary Note 2).",
 ]),
 ("Reference standard, hallucination audit, and usability", [
  "In the Ruijin cohort, three senior ICU consultants established a "
  "blinded reference standard, adjudicating deterioration phenotypes "
  "from the original record without access to imputed values, model "
  "outputs, or risk labels (Supplementary Note 3). A hallucination was "
  "defined as any generated statement, numeric value, temporal trend, "
  "causal assertion, or recommendation unsupported by the structured "
  "matrix, a prespecified rule, or a retrievable source passage; audit "
  "combined deterministic machine checking with sentence-level manual "
  "review (Supplementary Note 4). Usability was assessed with the System "
  "Usability Scale (SUS) among 86 clinicians and analyzed as a usability "
  "endpoint only.",
 ]),
 ("Engine v2 (post-hoc architectural revision)", [
  "The AmsterdamUMCdb portability findings motivated a redesigned "
  "imputation engine, evaluated under the same leakage-controlled design "
  "and reported separately from the frozen validation. Engine v2 is "
  "interpolation-first: a missing 6 h or 12 h lactate value bracketed by "
  "measured anchors is reconstructed by piecewise-linear interpolation; "
  "when no later anchor exists, an anchor-free XGBoost fallback "
  "(initial lactate, heart-rate and respiratory-rate variability, age, "
  "sex, vasopressor escalation; original hyperparameter grid) trained "
  "only on MIMIC-IV complete trajectories (n=1,061) is applied. The "
  "fallback was benchmarked on AmsterdamUMCdb complete cases with 24 h "
  "and 48 h anchors additionally masked. Because Engine v2 was designed "
  "after observing the AmsterdamUMCdb results, it is reported as a "
  "post-hoc design iteration throughout.",
 ]),
 ("Statistical analysis and ethics", [
  "Extraction performance was summarized as precision with 95% "
  "confidence intervals; imputation performance as mean squared error "
  "(MSE), mean absolute error, and R2; decision-support performance as "
  "accuracy, sensitivity, specificity, Cohen\u2019s kappa, AUROC, and "
  "AUPRC, with the rule-of-three upper bound for the zero-event audit "
  "(Supplementary Note 5). Risk ratios (95% CIs) compare red- versus "
  "green-alert outcome rates. The Ruijin analysis was approved by the "
  "Clinical Research Ethics Committee of Ruijin Hospital (Approval No. "
  "2025\u3014515\u3015) with a waiver of informed consent; public "
  "databases were used under their respective data-use agreements.",
 ]),
]

RESULTS = [
 ("Cohort characteristics and extraction performance", [
  "The study database comprised 32,470 Sepsis-3 encounters across four "
  "independent cohorts (Supplementary Figure S1). The Ruijin cohort "
  "(n=200; median age 62.9 years, IQR 55.0\u201371.0; 47.5% male; "
  "28-day mortality 13.0%) is summarized in Table 2. Against manual "
  "double-blind chart review (n=1,200 sampled cases), Agent 1 achieved "
  "extraction precision for serial SOFA-2 scores of 99.2% (95% CI "
  "98.4\u201399.7) in MIMIC-IV, 98.8% (97.9\u201399.4) in eICU, and "
  "99.4% (98.6\u201399.9) in Ruijin, with similar precision for "
  "for serial lactate and vasopressor dosing (Table 1; Supplementary "
  "Table S1).",
 ]),
 ("Imputation performance and Ruijin clinical validation", [
  "In the development cohorts, Agent 2 reconstructed missing lactate "
  "trajectories with MSE 0.12 mmol/L and R2 0.92, outperforming linear "
  "interpolation, random forest, MICE, and mean imputation "
  "(Supplementary Table S3). In the Ruijin cohort, 64 patients had "
  "incomplete perfusion trajectories; frozen reconstruction reclassified "
  "49 of 64 (24.5% of the cohort) from indeterminate to high-risk "
  "deteriorating phenotypes (Supplementary Table S4). Against the "
  "blinded expert reference standard, Agent 3 alerts achieved 96.5% "
  "concordance (accuracy 96.5% [95% CI 93.1\u201398.5]; sensitivity "
  "95.3%; specificity 97.4%; Cohen\u2019s kappa 0.93). Under the "
  "prespecified sentence-level definition, no hallucinations were "
  "identified across the 200 audited cases (rule-of-three upper bound "
  "\u22481.5% per case); this is a retrospective audit result, not a "
  "guarantee against future errors. Performance remained consistent "
  "across infection sources, calendar years, and sensitivity "
  "exclusions, and the median SUS score was 82.5 (IQR 72.5\u201390.0) "
  "(Supplementary Tables S5\u2013S10).",
 ]),
 ("Fully frozen external validation in AmsterdamUMCdb", [
  "AmsterdamUMCdb (2003\u20132016) contributed 3,949 Sepsis-3 encounters "
  "(23,106 admissions screened; 4,291 with suspected infection; 342 "
  "excluded for SOFA-2 <2). Median age was 65 years (IQR 55\u201375), "
  "28-day mortality 21.5% (851/3,949), and vasopressor escalation "
  "39.0%. Serial-lactate sampling density was markedly lower than in "
  "the development cohorts: lactate was measured in 71.1% of patients "
  "at 0 h, 58.3% at 6 h, 53.1% at 12 h, 44.7% at 24 h, and 26.6% at "
  "48 h; 2,084 patients (52.8%) had missing 6 h or 12 h lactate, of whom "
  "1,414 (35.8%) were missing both (Supplementary Table S11).",

  "Agent 3 issued red alerts for 2,113 patients (53.5%). Red alerts "
  "were significantly associated with 28-day mortality (26.2% "
  "[553/2,113] vs 16.2% [298/1,836]; RR 1.61, 95% CI 1.42\u20131.83), "
  "composite deterioration (61.2% vs 34.0%; RR 1.80, 1.68\u20131.94), "
  "and vasopressor escalation (52.0% vs 24.1%; RR 2.16, 1.97\u20132.36; "
  "all p<0.001) (Figure 2B; Supplementary Table S13). Against the "
  "composite endpoint as reference standard, alert-level performance "
  "was accuracy 63.5%, sensitivity 67.5%, specificity 59.7%, kappa "
  "0.27, and AUROC 0.69 (AUPRC 0.57); these observational associations "
  "remain vulnerable to confounding by illness severity.",

  "On masked complete-case lactate values (866 complete cases; 260 "
  "held-out patients; 520 masked values), the frozen formula degraded "
  "relative to development (MSE 2.28 mmol/L, R2 0.48 vs 0.92) and was "
  "outperformed by linear interpolation (MSE 1.38, R2 0.68), MICE "
  "(R2 0.67), and site-adapted XGBoost (R2 0.63) (Figure 2A; "
  "Supplementary Table S12). Engine v2 restored the ceiling where "
  "bracketing anchors existed (R2 0.68; pooled complete cases R2 0.72), "
  "and a physiological residual correction added no material gain "
  "(repeated cross-validated R2 0.728 vs 0.722). On the anchor-free "
  "benchmark, the MIMIC-trained XGBoost fallback achieved R2 0.43, "
  "essentially matching the v1 formula (R2 0.44). Only 325 of 2,084 "
  "patients (15.6%) with missing 6 h or 12 h lactate had any later "
  "anchor measurement; among the indeterminate group, 900 (43.2%) met "
  "red-alert criteria on the basis of the remaining trajectory features, "
  "with 28-day mortality 20.3% versus 15.0% (Figure 2C).",
 ]),
]

DISCUSSION = [
 ("Principal findings", [
  "This multicenter retrospective study supports the technical "
  "feasibility of a clinically auditable, trajectory-informed "
  "agentic-LLM workflow for sepsis decision support under data "
  "missingness. Extraction fidelity exceeded 98% across development, "
  "external, and clinical validation cohorts; audit-gated alerts were "
  "reproducibly associated with hard outcomes in a fully frozen, "
  "historically distant cohort without any site adaptation; and the "
  "deterministic audit layer yielded no unsupported statements in the "
  "audited retrospective sample. These findings move generative-AI "
  "decision support from unconstrained prompt\u2013response use toward "
  "a constrained workflow in which every clinical statement is "
  "traceable to data or prespecified rules.",
 ]),
 ("Missing data and the measurement-density constraint", [
  "The fully frozen AmsterdamUMCdb validation adds a caveat of broad "
  "relevance. In a cohort with markedly lower serial-lactate sampling "
  "density (52.8% of patients missing 6 h or 12 h lactate) and a "
  "different care epoch, the frozen imputation formula degraded below "
  "within-cohort linear interpolation (R2 0.48 vs 0.68), and even "
  "site-adapted retraining did not beat interpolation (R2 0.63). The "
  "interpolation-first Engine v2 restores reconstruction where anchors "
  "exist (R2 0.68\u20130.72) but remains information-limited where they "
  "do not (R2 0.43); 84.4% of patients with missing 6 h or 12 h lactate "
  "had no later anchor measurement. These results indicate that "
  "serial-measurement density, rather than model architecture, limits "
  "trajectory-based decision support in sparse-sampling settings. "
  "This finding favours protocolised serial lactate measurement over "
  "model refinement and argues that imputation portability be "
  "evaluated directly in CDSS transportability studies.",
 ]),
 ("Limitations", [
  "First, the Ruijin clinical validation cohort was retrospective, "
  "single-centre, and small (n=200), which may limit generalizability "
  "to community or resource-limited hospitals. Second, imputation "
  "introduces uncertainty; although the system performed well in the "
  "evaluated cohorts, errors may occur where missingness mechanisms or "
  "practice patterns differ, potentially leading to over- or "
  "under-triage. Third, local deployment of a quantized 70B model "
  "requires H800-class GPU infrastructure. Fourth, this was a "
  "retrospective simulated real-time audit; live EHR integration, "
  "alert fatigue, workflow adherence, and patient outcomes were not "
  "tested. Fifth, the AmsterdamUMCdb cohort (2003\u20132016) predates "
  "the development cohorts, has lower sampling density, and was scored "
  "with simplified SOFA-2 items; the observed imputation degradation "
  "may reflect site and era heterogeneity as well as true portability "
  "limits. Sixth, alert\u2013outcome associations in AmsterdamUMCdb "
  "were assessed against composite endpoints without blinded expert "
  "adjudication, and residual confounding cannot be excluded. "
  "Prospective workflow studies and randomized trials are required to "
  "determine whether this system improves outcomes.",
 ]),
 ("Conclusions", [
  "A constrained, trajectory-informed, audit-gated agentic-LLM system "
  "for sepsis decision support under data missingness is technically "
  "feasible, clinician-facing, and retrospectively reproducible across "
  "four independent cohorts. Transparent portability analysis showed "
  "that serial-measurement density, not model architecture, limits "
  "trajectory reconstruction in sparse-sampling ICUs. Prospective "
  "clinical-effectiveness, implementation, and patient-outcome studies "
  "remain necessary before bedside deployment.",
 ]),
]

DECLARATIONS = [
 ("Contributors",
  "RZ, XW, SL, and JW (Jingyi Wu) conceived and designed the study, "
  "developed the methodology, drafted the manuscript, and critically "
  "revised it. HQ, LS, and YL supervised the study and interpreted the "
  "clinical findings. RZ, XW, JH, WW, and JW (Jingyi Wu) performed the "
  "algorithm design and validation. RZ, JW (Jiahui Wang), LP, RT, and "
  "LL contributed to case screening, data collection, and data "
  "verification. All authors had full access to all data, critically "
  "revised the manuscript, approved the final version, and had final "
  "responsibility for the decision to submit for publication."),
 ("Declaration of interests", "We declare no competing interests."),
 ("Data sharing",
  "Public MIMIC-IV, eICU, and AmsterdamUMCdb data are accessible "
  "through their standard credentialing and data-use procedures [23]. "
  "Ruijin-derived de-identified analytic summaries and data "
  "dictionaries will be made available after publication subject to "
  "institutional agreements. The complete source code of the CDSS, "
  "including the frozen engine, the AmsterdamUMCdb validation pipeline, "
  "the Engine v2 fallback models, prompt templates, audit rules, and "
  "evaluation scripts, is publicly available at "
  "https://github.com/ccmzhangrui/sepsis-llm to support independent "
  "inspection and reproducibility."),
 ("Acknowledgments",
  "We thank all patients whose historical records were analyzed, and "
  "the medical and nursing staff at Ruijin Hospital and the maintainers "
  "of the MIMIC-IV, eICU, and AmsterdamUMCdb databases. During the "
  "preparation of this manuscript, Kimi-K3 (Moonshot AI) was used to "
  "polish the text and create some elements of figures. All "
  "AI-assisted output was reviewed and edited by the authors, who take "
  "full responsibility for the accuracy and integrity of the work. No "
  "AI-assisted technology is listed as an author."),
]

REFERENCES = [
 "Singer M, Deutschman CS, Seymour CW, et al. The Third International "
 "Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA "
 "2016; 315(8): 801-810.",
 "Evans L, Alhazzani W, Alshamsi F, et al. Surviving Sepsis Campaign: "
 "international guidelines for management of sepsis and septic shock "
 "2021. Intensive Care Med 2021; 47(11): 1181-1427.",
 "Vincent JL, Moreno R, Takala J, et al. The SOFA (Sepsis-related "
 "Organ Failure Assessment) score to describe organ dysfunction/"
 "failure. Intensive Care Med 1996; 22(7): 707-710.",
 "Ferreira FL, Bota DP, Bross A, et al. Serial evaluation of the SOFA "
 "score to predict outcome in critically ill patients. JAMA 2001; "
 "286(14): 1754-1758.",
 "Hern\u00e1ndez G, Ospina-Tasc\u00f3n GA, Damiani LP, et al. Effect of "
 "a resuscitation strategy targeting peripheral perfusion status versus "
 "serum lactate levels on 28-day mortality among patients with septic "
 "shock. JAMA 2019; 321(7): 654-664.",
 "Zhang R, Long F, Wu J, et al. Dynamical trajectories of organ "
 "dysfunction in sepsis using the SOFA-2 score and early prediction "
 "from multicenter cohorts. Intensive Crit Care Nurs 2026; 95: 104413.",
 "Lim Y, Jeon B, Park SA, et al. Large language model-augmented "
 "offline reinforcement learning framework for sepsis management in "
 "critical care. npj Digit Med 2026; 9: 42.",
 "Swinckels L, Bennis FC, Ziesemer KA, et al. The Use of Deep Learning "
 "and Machine Learning on Longitudinal Electronic Health Records for "
 "the Early Detection and Prevention of Diseases: Scoping Review. "
 "J Med Internet Res 2024; 26: e48320.",
 "Shankar SV, Dhingra LS, Aminorroaya A, et al. Automated "
 "transformation of unstructured cardiovascular diagnostic reports "
 "into structured datasets using sequentially deployed large language "
 "models. Eur Heart J Digit Health 2025; 6: 783-796.",
 "Swinckels L, Bennis FC, Ziesemer KA, et al. The Use of Deep Learning "
 "and Machine Learning on Longitudinal Electronic Health Records for "
 "the Early Detection and Prevention of Diseases: Scoping Review. "
 "J Med Internet Res 2024; 26: e48320.",
 "Lee P, Bubeck S, Petro J. Benefits, limits, and risks of GPT-4 as an "
 "AI chatbot for medicine. N Engl J Med 2023; 388(13): 1233-1239.",
 "Moor M, Banerjee O, Abad ZSH, et al. Foundation models for "
 "generalist medical artificial intelligence. Nature 2023; 616(7956): "
 "259-265.",
 "Shams SM, Maldarelli ME, Yin Y, et al. A clinical support tool using "
 "artificial intelligence to diagnose pulmonary hypertension. Eur "
 "Respir J 2026; 67: in press. doi:10.1183/13993003.00125-2026.",
 "Plaat A, van Duijn M, van Stein N, et al. Agentic Large Language "
 "Models, a Survey. Journal of Artificial Intelligence Research 2025; "
 "84: 112-135.",
 "Lim Y, Jeon B, Park SA, et al. Large language model-augmented "
 "offline reinforcement learning framework for sepsis management in "
 "critical care. npj Digit Med 2026; 9: 42.",
 "Johnson AEW, Bulgarelli L, Shen L, et al. MIMIC-IV, a freely "
 "accessible electronic health record dataset. Sci Data 2023; 10(1): 1.",
 "Pollard TJ, Johnson AEW, Raffa JD, et al. The eICU Collaborative "
 "Research Database, a freely available multi-center database for "
 "critical care research. Sci Data 2018; 5: 180178.",
 "Riley RD, Ensor J, Snell KIE, et al. Calculating the sample size "
 "required for developing a clinical prediction model. BMJ 2020; 368: "
 "m441.",
 "Kursa MB, Rudnicki WR. Feature selection with the Boruta package. "
 "J Stat Softw 2010; 36: 1-13.",
 "Amann J, Blasimme A, Vayena E, et al. Explainability for artificial "
 "intelligence in healthcare: a multidisciplinary perspective. BMC Med "
 "Inform Decis Mak 2020; 20: 310.",
 "Zhang R, Long F, Zhao Z, et al. Machine learning predicts sepsis "
 "deterioration trajectories. npj Digit Med 2026; 9: 21. "
 "doi:10.1038/s41746-026-02565-x.",
 "Bangor A, Kortum PT, Miller JT. An empirical evaluation of the "
 "system usability scale. Int Journal of Hum-Computer Interaction "
 "2008; 24: 574-594.",
 "Thoral PJ, Peppink JM, Driessen RH, et al. Sharing ICU patient data "
 "responsibly under the Society of Critical Care Medicine/European "
 "Society of Intensive Care Medicine Joint Data Science Collaboration: "
 "the Amsterdam University Medical Centers Database (AmsterdamUMCdb) "
 "example. Crit Care Med 2021; 49(6): e563-e577.",
 "Seymour CW, Liu VX, Iwashyna TJ, et al. Assessment of clinical "
 "criteria for sepsis: for the Third International Consensus "
 "Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA 2016; "
 "315(8): 762-774.",
]

FIG1_LEGEND = ("Figure 1. End-to-end multicenter development and "
 "agentic-LLM clinical support pipeline. Four cohorts with strict "
 "development\u2013validation separation: MIMIC-IV (development), eICU "
 "(pre-freeze external validation), Ruijin Hospital (frozen "
 "retrospective clinical validation), and AmsterdamUMCdb (fully frozen "
 "additional external validation, no site adaptation). The pipeline "
 "runs through Agent 1 (temporal extraction), Agent 2 (trajectory "
 "imputation; Engine v2 is interpolation-first with an anchor-free "
 "XGBoost fallback), and Agent 3 (audit-gated decision), with a "
 "deterministic audit gate before clinician-facing output.")
FIG2_LEGEND = ("Figure 2. Imputation and decision performance in the "
 "fully frozen AmsterdamUMCdb validation cohort (n=3,949). A, "
 "Imputation accuracy (R\u00b2) on masked complete-case lactate values "
 "(866 complete cases; 260 held-out patients), comparing the frozen v1 "
 "formula, classical comparators, site-adapted XGBoost, and the two "
 "Engine v2 arms. B, Risk ratios (95% CIs) for 28-day mortality, "
 "composite deterioration, and vasopressor escalation among patients "
 "with red versus green alerts. C, Coverage among the 2,084 patients "
 "with missing 6 h or 12 h lactate (interpolable vs anchor-free "
 "fallback arm) and alert status within the indeterminate group.")

# ---------------------------------------------------------------- helpers
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING

def new_doc():
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(12)
    # 1-inch margins, Letter size (NEJM AI submission standard)
    for sec in doc.sections:
        sec.top_margin = sec.bottom_margin = Pt(72)
        sec.left_margin = sec.right_margin = Pt(72)
    return doc

def add_run(p, text, bold=False, size=12, italic=False, superscript=False):
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(size)
    r.bold = bold
    r.italic = italic
    if superscript:
        r.font.superscript = True
    rPr = r._element.get_or_add_rPr()
    rf = rPr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts"); rPr.insert(0, rf)
    rf.set(qn("w:eastAsia"), "Times New Roman")
    return r

def para(doc, text, bold=False, size=12, italic=False, align=None,
         space_after=6, line_spacing=None, first_indent=None,
         space_before=0):
    p = doc.add_paragraph()
    add_run(p, text, bold, size, italic)
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(space_before)
    if line_spacing:
        pf.line_spacing = line_spacing
    if first_indent is not None:
        pf.first_line_indent = Pt(first_indent)
    if align:
        p.alignment = align
    return p

def body_para(doc, text):
    """Journal-standard body paragraph: TNR 12, double-spaced, justified,
    first-line indent 0.25 in."""
    return para(doc, text, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
                line_spacing=2.0, first_indent=18, space_after=0)

def heading(doc, text, size=12, space_before=12, italic=False):
    """Level-1 heading: bold, left-aligned."""
    return para(doc, text, bold=True, size=size, italic=italic,
                space_after=6, space_before=space_before)

def subheading(doc, text):
    """Level-2 heading: bold italic."""
    return para(doc, text, bold=True, italic=True, size=12,
                space_after=4, space_before=8)

def page_break(doc):
    from docx.enum.text import WD_BREAK
    p = doc.add_paragraph()
    r = p.add_run()
    r.add_break(WD_BREAK.PAGE)
    return p

def setup_lines_and_footer(doc):
    """Continuous line numbering + centered page number in footer
    (journal submission requirements)."""
    sect = doc.sections[0]
    sectPr = sect._sectPr
    ln = OxmlElement("w:lnNumType")
    ln.set(qn("w:countBy"), "1")
    ln.set(qn("w:start"), "1")
    ln.set(qn("w:distance"), "240")
    ln.set(qn("w:restart"), "continuous")
    sectPr.append(ln)
    fp = sect.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    r = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    rf = OxmlElement("w:rFonts"); rf.set(qn("w:ascii"), "Times New Roman")
    rf.set(qn("w:hAnsi"), "Times New Roman")
    sz = OxmlElement("w:sz"); sz.set(qn("w:val"), "20")
    rPr.append(rf); rPr.append(sz)
    t = OxmlElement("w:t"); t.text = "1"
    r.append(rPr); r.append(t)
    fld.append(r)
    fp._p.append(fld)

def wordcount(text):
    import re
    return len(re.findall(r"[A-Za-z0-9\u00c0-\u00ff]+(?:['\u2019\-]"
                          r"[A-Za-z0-9\u00c0-\u00ff]+)*", text))


# ---------------------------------------------------- journal tables
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH as _AL
from docx.shared import Twips

USABLE = 9360  # 6.5 in in twips (1-inch margins on US Letter)


def _cell_borders(cell, bottom=False):
    tcPr = cell._tc.get_or_add_tcPr()
    old = tcPr.find(qn("w:tcBorders"))
    if old is not None:
        tcPr.remove(old)
    b = OxmlElement("w:tcBorders")
    if bottom:
        el = OxmlElement("w:bottom")
        el.set(qn("w:val"), "single"); el.set(qn("w:sz"), "6")
        el.set(qn("w:space"), "0"); el.set(qn("w:color"), "000000")
        b.append(el)
    tcPr.append(b)


def _set_tblpr_child(tblPr, tag):
    """Remove any existing child of this tag, return a fresh element
    appended at the end (avoids duplicate w:tblW etc.)."""
    for old in tblPr.findall(qn(f"w:{tag}")):
        tblPr.remove(old)
    el = OxmlElement(f"w:{tag}")
    tblPr.append(el)
    return el


def styled_table(doc, rows, size=9):
    """Rebuild a table in clean journal (booktabs) style:
    top/bottom rules only, header underline, no vertical lines,
    TNR 9 pt, proportional fixed column widths, header repeats."""
    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]
    t = doc.add_table(rows=len(rows), cols=ncols)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    tblPr = t._tbl.tblPr

    layout = _set_tblpr_child(tblPr, "tblLayout")
    layout.set(qn("w:type"), "fixed")
    tw = _set_tblpr_child(tblPr, "tblW")
    tw.set(qn("w:w"), str(USABLE)); tw.set(qn("w:type"), "dxa")
    mar = _set_tblpr_child(tblPr, "tblCellMar")
    for side, v in (("top", 30), ("left", 80), ("bottom", 30),
                    ("right", 80)):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"), str(v)); el.set(qn("w:type"), "dxa")
        mar.append(el)
    borders = _set_tblpr_child(tblPr, "tblBorders")
    for side, val, sz in (("top", "single", "12"), ("bottom", "single", "12")):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:val"), val); el.set(qn("w:sz"), sz)
        el.set(qn("w:space"), "0"); el.set(qn("w:color"), "000000")
        borders.append(el)
    for side in ("left", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:val"), "none"); el.set(qn("w:sz"), "0")
        el.set(qn("w:space"), "0"); el.set(qn("w:color"), "auto")
        borders.append(el)

    # proportional column widths from longest line per column
    maxlen = []
    for j in range(ncols):
        m = max((max((len(ln) for ln in r[j].split("\n")), default=1)
                 for r in rows), default=1)
        maxlen.append(max(5, min(m, 60)))
    tot = sum(maxlen)
    widths = [max(680, int(USABLE * m / tot)) for m in maxlen]
    widths[-1] += USABLE - sum(widths)  # exact fit
    grid = t._tbl.find(qn("w:tblGrid"))
    for gc in list(grid):
        grid.remove(gc)
    for w in widths:
        gc = OxmlElement("w:gridCol"); gc.set(qn("w:w"), str(w))
        grid.append(gc)

    # header repeat
    trPr = t.rows[0]._tr.get_or_add_trPr()
    th = OxmlElement("w:tblHeader"); th.set(qn("w:val"), "true")
    trPr.append(th)

    for i, r in enumerate(rows):
        for j, val in enumerate(r):
            cell = t.rows[i].cells[j]
            cell.width = Twips(widths[j])
            left = maxlen[j] > 22
            first = True
            for ln in val.split("\n"):
                p = cell.paragraphs[0] if first else cell.add_paragraph()
                first = False
                add_run(p, ln, bold=(i == 0), size=size)
                p.alignment = _AL.LEFT if (left or i == 0 and j == 0) \
                    else _AL.CENTER
                if i == 0 and not left:
                    p.alignment = _AL.CENTER
                pf = p.paragraph_format
                pf.space_before = Pt(0); pf.space_after = Pt(0)
                pf.line_spacing = 1.0
            if i == 0:
                _cell_borders(cell, bottom=True)
    return t

# ================================================== MAIN MANUSCRIPT
main = new_doc()
C = WD_ALIGN_PARAGRAPH.CENTER

# ---- title page ----
para(main, TITLE, bold=True, size=16, align=C, space_after=16,
     space_before=24, line_spacing=1.5)
p = para(main, "", align=C, space_after=10, line_spacing=1.5)
for i, (name, sup) in enumerate(AUTHORS):
    if i:
        add_run(p, ", ", size=12)
    add_run(p, name, size=12)
    add_run(p, sup, size=12, superscript=True)
for num, txt in AFFILIATIONS:
    p = para(main, "", align=C, size=10, space_after=4, line_spacing=1.25)
    add_run(p, num, size=10, superscript=True)
    add_run(p, " " + txt, size=10, italic=True)
para(main, EQUAL, size=10, align=C, space_after=2, space_before=8)
para(main, SENIOR, size=10, align=C, space_after=2)
para(main, CORR, size=10, align=C, space_after=0, line_spacing=1.25)
page_break(main)

# ---- short description & abstract ----
para(main, "Short description", bold=True, space_after=4)
body_para(main, DESCRIPTION)
para(main, "", space_after=6)
heading(main, "Abstract", size=13)
for sec in ("Background", "Methods", "Results", "Conclusions"):
    p = main.add_paragraph()
    add_run(p, sec + ". ", bold=True)
    add_run(p, ABSTRACT[sec])
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 2.0
    p.paragraph_format.space_after = Pt(0)
abs_words = sum(wordcount(v) for v in ABSTRACT.values())

# ---- body ----
heading(main, "Introduction", size=13)
for t in INTRODUCTION:
    body_para(main, t)
heading(main, "Methods", size=13)
for h, ps in METHODS:
    subheading(main, h)
    for t in ps:
        body_para(main, t)
heading(main, "Results", size=13)
for h, ps in RESULTS:
    subheading(main, h)
    for t in ps:
        body_para(main, t)
heading(main, "Discussion", size=13)
for h, ps in DISCUSSION:
    subheading(main, h)
    for t in ps:
        body_para(main, t)

main_words = (sum(wordcount(t) for t in INTRODUCTION)
              + sum(wordcount(t) for _, ps in METHODS for t in ps)
              + sum(wordcount(t) for _, ps in RESULTS for t in ps)
              + sum(wordcount(t) for _, ps in DISCUSSION for t in ps))

for h, t in DECLARATIONS:
    heading(main, h, size=13)
    body_para(main, t)

page_break(main)
heading(main, "References", size=13)
for i, ref in enumerate(REFERENCES, 1):
    p = para(main, f"{i}. {ref}", size=11, space_after=2,
             line_spacing=1.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    p.paragraph_format.left_indent = Pt(18)
    p.paragraph_format.first_line_indent = Pt(-18)

page_break(main)
heading(main, "Tables", size=13)
para(main, "Table 1. Performance of Agent 1 on dynamic feature "
     "extraction across cohorts.", bold=True, space_after=6,
     line_spacing=1.5)
main.element.body  # (tables are rebuilt below, not copied)
styled_table(main, table_rows(src.tables[0]))
para(main, "", space_after=8)
para(main, "Table 2. Baseline characteristics and early serial clinical "
     "measures in the Ruijin clinical validation cohort (2022\u20132025).",
     bold=True, space_after=6, line_spacing=1.5)
styled_table(main, table_rows(src.tables[1]))

page_break(main)
heading(main, "Figure legends", size=13)
para(main, FIG1_LEGEND, space_after=8, line_spacing=2.0,
     align=WD_ALIGN_PARAGRAPH.JUSTIFY)
para(main, FIG2_LEGEND, space_after=8, line_spacing=2.0,
     align=WD_ALIGN_PARAGRAPH.JUSTIFY)

setup_lines_and_footer(main)
main.save(f"{OUTDIR}/sepsis_decision_model_nejmai.docx")
print(f"main manuscript saved. abstract={abs_words} words, "
      f"intro->discussion={main_words} words")

# ================================================== SUPPLEMENT
sup = new_doc()
para(sup, "Supplementary Appendix", bold=True, size=14, space_after=4)
para(sup, TITLE, italic=True, size=11, space_after=2)
para(sup, "Rui Zhang, Xu Wang, Shengjun Liu, et al.", size=11,
     space_after=10)

heading(sup, "Table of contents")
toc = ["Supplementary Note 1: Extended multi-agent system configuration",
       "Supplementary Note 2: Leakage-controlled feature selection and "
       "frozen XGBoost training",
       "Supplementary Note 3: Blinded expert reference standard",
       "Supplementary Note 4: Hallucination audit SOP",
       "Supplementary Note 5: Additional robustness and outcome analyses",
       "Supplementary Tables S1\u2013S14",
       "Supplementary Figures S1\u2013S3 (legends)"]
for t in toc:
    para(sup, t, space_after=2)

NOTES = {
 "Supplementary Note 1: Extended Multi-Agent System Configuration, State "
 "Transitions, and Prompt/Tool Protocol":
 "Our local deployment utilized the quantized LLaMA-3.1-70B-Instruct "
 "model running on 4-bit precision (AWQ quantization) to match simulated "
 "real-time inference requirements (<5 seconds per patient pipeline in "
 "the local test environment). The local server was equipped with 8 x "
 "NVIDIA H800 GPUs (80GB VRAM each). The system was containerized using "
 "Docker (v20.10.8) and orchestrated via Python 3.8. The agentic "
 "workflow used a shared patient-state store, schema-constrained "
 "extraction prompts, tool calls to SOFA-2/lactate calculators and the "
 "frozen XGBoost imputation module, and deterministic audit gates before "
 "output release.",
 "Supplementary Note 2: Leakage-Controlled Feature Selection, Boruta "
 "Implementation, and Frozen XGBoost Training":
 "Candidate variables were selected based on prior trajectory-modeling "
 "work [6,21]. For lactate and SOFA-2 imputation, the Boruta algorithm "
 "in R (v4.3.1) was run over 500 iterations within the MIMIC-IV "
 "development cohort only. Highly correlated parameters (r>0.60) were "
 "consolidated to reduce overfitting. The selected feature set, XGBoost "
 "hyperparameters, preprocessing workflow, thresholds, and audit rules "
 "were frozen after eICU external validation and before Ruijin "
 "validation; neither Ruijin nor AmsterdamUMCdb data were used for "
 "feature selection, model fitting, threshold setting, prompt "
 "optimization, or calibration.",
 "Supplementary Note 3: Blinded Expert Reference Standard":
 "Three senior ICU consultants independently reviewed original EHR "
 "records while blinded to imputed values, model outputs, and audit "
 "results. For patients with missing lactate values, adjudicators "
 "reviewed only observed data and subsequent clinical course. "
 "Disagreements were resolved by consensus, and pre-consensus agreement "
 "was recorded.",
 "Supplementary Note 4: Hallucination Audit SOP":
 "Each generated sentence was mapped to structured data, source text, "
 "or a prespecified audit rule. Unsupported numeric values, temporal "
 "trends, causal assertions, or recommendations were counted as "
 "hallucinations. Deterministic audit was followed by manual ICU "
 "consultant review. Zero observed events in 200 cases should be "
 "reported with uncertainty and not interpreted as evidence that future "
 "events cannot occur.",
 "Supplementary Note 5: Additional Robustness, Baseline, Calibration, "
 "and Outcome Analyses":
 "This note documents the additional robustness, comparator, "
 "calibration, and outcome analyses specified for the revised study: "
 "simulated missingness robustness, GRU-D/BRITS/SAITS/Transformer "
 "imputation baselines where feasible, LLM ablations, decision-curve "
 "analysis, calibration slope/intercept, and outcome characterization of "
 "the 49 reclassified patients. The AmsterdamUMCdb fully frozen "
 "validation analyses (suspected-infection screening, hourly SOFA-2 "
 "computation with 24-hour rolling maxima, frozen-engine application, "
 "alert\u2013outcome association testing, masked complete-case "
 "imputation-portability comparison, Engine v2 evaluation, and "
 "reclassification outcome characterization) are included in the "
 "repository under amsterdam_validation/ (including engine_v2/).",
}
for h, t in NOTES.items():
    heading(sup, h)
    para(sup, t)

CAPTIONS = [
 ("Supplementary Table S1: Detailed Variables Mapping and Schema "
  "Definitions", 2),
 ("Supplementary Table S2: Hyperparameter Search Space and Tuning "
  "Results", 3),
 ("Supplementary Table S3: Imputation Performance Across Classical and "
  "Contemporary Algorithms (development cohorts)", 4),
 ("Supplementary Table S4: Reclassification Matrix and Outcome "
  "Characterization of Indeterminate Cases (Ruijin Cohort, n=64)", 5),
 ("Supplementary Table S5: Diagnostic Performance by Sepsis Infection "
  "Source", 6),
 ("Supplementary Table S6: Diagnostic Performance Stratified by Calendar "
  "Year (Ruijin Cohort)", 7),
 ("Supplementary Table S7: System Usability Scale (SUS) Survey "
  "Demographics (N=86)", 8),
 ("Supplementary Table S8: Detailed System Usability Scale (SUS) Item "
  "Scores (0\u20134 scale)", 9),
 ("Supplementary Table S9: Sensitivity Analysis Excluding Specific "
  "Patient Subgroups", 10),
 ("Supplementary Table S10: Validation Performance on MIMIC-IV Cohort "
  "(n=17,292)", 11),
 ("Supplementary Table S11: Baseline Characteristics and Lactate "
  "Sampling Availability in the Fully Frozen AmsterdamUMCdb Validation "
  "Cohort (n=3,949)", 12),
 ("Supplementary Table S12: Lactate Imputation Portability in "
  "AmsterdamUMCdb (Masked Complete Cases: n=866; Held-out Test 260 "
  "Patients, 520 Masked Values)", 13),
 ("Supplementary Table S13: Agent 3 Alert\u2013Outcome Associations in "
  "the Fully Frozen AmsterdamUMCdb Validation Cohort (n=3,949)", 14),
]
heading(sup, "Supplementary Tables")
for cap, idx in CAPTIONS:
    para(sup, cap + ".", bold=True, space_after=3)
    styled_table(sup, table_rows(src.tables[idx]))
    if idx == 13:
        para(sup, "For reference, the frozen XGBoost trajectory regressor "
             "achieved MSE 0.12 mmol/L and R2 0.92 in the development "
             "cohorts (Table S3). The site-adapted XGBoost model was "
             "refitted within AmsterdamUMCdb using the original "
             "hyperparameter search grid (optimal: learning rate 0.01, "
             "max depth 3, subsample 0.6) under the same masked "
             "complete-case design. Engine v2 (interpolation-first "
             "hybrid) reduces to linear interpolation on this "
             "complete-case benchmark by construction; its anchor-free "
             "fallback arm is tabulated above. Among the 2,084 patients "
             "with missing 6 h or 12 h lactate in the full cohort, "
             "1,759 (84.4%) had no later anchor measurement.",
             size=10, space_after=8)
    if idx == 14:
        para(sup, "Against the composite deterioration endpoint as "
             "reference standard: accuracy 63.5%, sensitivity 67.5%, "
             "specificity 59.7%, Cohen\u2019s kappa 0.27, AUROC 0.69 "
             "(AUPRC 0.57). Red alerts were issued for 2,113/3,949 "
             "patients (53.5%). RR = risk ratio (95% CI).", size=10,
             space_after=8)

# ---- S14: Diversity / representativeness (NEJM AI requirement) ----
para(sup, "Supplementary Table S14: Disease Background and "
     "Representativeness of the Study Group (NEJM AI diversity-in-"
     "research requirement).", bold=True, space_after=3)
S14 = [
 ["Dimension", "Background (sepsis)", "This study group"],
 ["Condition", "Sepsis is a leading cause of ICU admission and death "
  "worldwide; 28-day mortality is typically 15\u201330% in ICU "
  "cohorts, with wide variation by era, region, and case mix [1,2].",
  "Four independent ICU cohorts spanning the USA (MIMIC-IV, eICU), "
  "China (Ruijin), and the Netherlands (AmsterdamUMCdb), and care "
  "epochs from 2003\u20132016 to 2022\u20132025 (n=32,470 Sepsis-3 "
  "encounters)."],
 ["Geographic representation",
  "Sepsis phenotypes, coding practices, and measurement density differ "
  "across health systems.",
  "Single-centre US academic (MIMIC-IV), multicentre US (eICU), "
  "single-centre Chinese academic (Ruijin), and Dutch academic "
  "(AmsterdamUMCdb) sites; low- and middle-income and community "
  "settings are not represented."],
 ["Era representation",
  "Sepsis care (Surviving Sepsis Campaign bundles, lactate monitoring) "
  "evolved substantially over two decades.",
  "AmsterdamUMCdb (2003\u20132016) predates eICU (2014\u20132015) and "
  "Ruijin (2022\u20132025); imputation-portability degradation is "
  "reported transparently and partly reflects era heterogeneity."],
 ["Age and sex",
  "Sepsis incidence and mortality rise with age; sex differences in "
  "presentation and outcomes are documented.",
  "Age distributions were similar across cohorts (Ruijin median 62.9 "
  "years; AmsterdamUMCdb 65 years); male share 47.5% (Ruijin) and "
  "63.5% (AmsterdamUMCdb, 60 patients with unknown gender)."],
 ["Race and ethnicity",
  "Disparities in sepsis recognition and outcomes exist across racial "
  "and ethnic groups.",
  "Race/ethnicity was available in MIMIC-IV only; the Ruijin cohort "
  "was Han Chinese-predominant and AmsterdamUMCdb Dutch-predominant. "
  "Generalizability to other racial and ethnic groups cannot be "
  "established and is a limitation."],
 ["Generalizability comment",
  "Findings may not extend to community or resource-limited hospitals, "
  "pediatric populations, or settings without high-frequency "
  "physiological monitoring.",
  "The fully frozen AmsterdamUMCdb validation was designed to test "
  "portability across era and sampling density; remaining gaps "
  "are acknowledged in the Limitations."],
]
tbl = styled_table(sup, S14)

heading(sup, "Supplementary Figure legends")
para(sup, "Supplementary Figure S1. STROBE/RECORD-style cohort "
     "construction across the four databases. AmsterdamUMCdb screening "
     "flow (23,106 admissions; 4,291 suspected infections by Seymour "
     "antibiotic\u2013culture pairing; 342 excluded for SOFA-2 <2; "
     "3,949 Sepsis-3 analytic encounters) is shown in full; roles of "
     "each cohort in the leakage-controlled design are annotated.",
     space_after=4)
para(sup, "Supplementary Figure S2. Feature importance (mean |SHAP "
     "value|) of the Engine v2 anchor-free fallback XGBoost model for "
     "6 h lactate, computed on the MIMIC-IV training data (n=1,061 "
     "complete trajectories). Initial lactate dominates, consistent "
     "with the information-limited anchor-free regime.", space_after=4)
para(sup, "Supplementary Figure S3. Bland\u2013Altman analysis of "
     "imputed versus measured lactate in AmsterdamUMCdb complete cases "
     "(n=866; 6 h and 12 h values pooled). A, Frozen v1 formula; "
     "B, Engine v2 interpolation-first arm. Bias and 95% limits of "
     "agreement are annotated and should be interpreted relative to "
     "clinically meaningful lactate thresholds.", space_after=4)
sup.save(f"{OUTDIR}/supplementary_appendix_nejmai.docx")
print("supplementary appendix saved")

# ================================================== COVER LETTER
cl = new_doc()
para(cl, "Cover Letter", bold=True, size=14, space_after=12)
para(cl, "Isaac S. Kohane, MD, PhD", space_after=2)
para(cl, "Editor-in-Chief, NEJM AI", space_after=2)
para(cl, "Department of Biomedical Informatics, Harvard Medical School",
     space_after=12)
para(cl, "Dear Dr. Kohane,", space_after=8)
cl_paras = [
 "We are pleased to submit our manuscript entitled \u201cA Clinically "
 "Auditable Agentic Large Language Model System for Sepsis Decision "
 "Support under Data Missingness: A Multicenter Development and "
 "Retrospective Validation Study\u201d for consideration as an Original "
 "Research article in NEJM AI.",

 "This study evaluates a constrained, clinically auditable agentic-LLM "
 "architecture for sepsis decision support across four independent "
 "cohorts with strict development\u2013validation separation: MIMIC-IV "
 "(development, n=17,292), eICU (external validation, n=11,029), a "
 "Ruijin Hospital clinical validation cohort (2022\u20132025, n=200), "
 "and a fully frozen additional external validation in AmsterdamUMCdb "
 "(2003\u20132016, n=3,949) processed with no site-specific adaptation "
 "of any kind. Extraction precision exceeded 98% in all cohorts; "
 "audit-gated alerts were associated with 28-day mortality (RR 1.61, "
 "95% CI 1.42\u20131.83), composite deterioration (RR 1.80), and "
 "vasopressor escalation (RR 2.16) in the fully frozen cohort; and no "
 "hallucinations were identified in the 200 audited retrospective cases.",

 "We believe three findings are of particular interest to NEJM AI "
 "readers. First, we report transparently that a frozen imputation "
 "formula did not transfer losslessly (R2 0.48 vs 0.92 in development), "
 "and that even site-adapted retraining did not beat simple "
 "interpolation \u2014 a portability failure that many publications "
 "omit. Second, our redesigned interpolation-first engine (v2) restored "
 "reconstruction to the information ceiling where anchor measurements "
 "existed (R2 0.68\u20130.72) but remained information-limited without "
 "later measurements (R2 0.43). Third, a coverage analysis showed that "
 "84.4% of patients with missing 6 h or 12 h lactate had no later "
 "anchor measurement, identifying serial-measurement density \u2014 "
 "not model architecture \u2014 as the factor that most limits "
 "trajectory-based decision support in sparse-sampling ICUs, a "
 "conclusion with direct implications for monitoring policy.",

 "Reporting was aligned with STROBE/RECORD and organized by "
 "TRIPOD+AI-oriented transparency domains; a disease-background and "
 "representativeness table is included in the Supplementary Appendix "
 "per NEJM AI policy. The complete source code, frozen engine, "
 "AmsterdamUMCdb validation pipeline, and Engine v2 fallback models "
 "are publicly available at https://github.com/ccmzhangrui/sepsis-llm. "
 "In accordance with NEJM AI policy on AI-assisted technologies, we "
 "disclose that Kimi-K3 (Moonshot AI) was used during manuscript "
 "preparation to polish the text and create some elements of figures; "
 "all AI-assisted output was reviewed and edited by the authors, who "
 "take full responsibility for the accuracy and integrity of the work. "
 "No AI-assisted technology is listed as an author.",

 "This manuscript is not under consideration elsewhere. All authors "
 "have read and approved the final version and declare no competing "
 "interests. The Ruijin cohort was approved by the Clinical Research "
 "Ethics Committee of Ruijin Hospital (Approval No. 2025\u3014515\u3015) "
 "with a waiver of informed consent; public databases were used under "
 "their respective data-use agreements.",

 "Thank you for your consideration. We look forward to your response.",
]
for t in cl_paras:
    para(cl, t, space_after=8)
para(cl, "Sincerely,", space_after=12)
para(cl, "On behalf of all authors,", space_after=2)
para(cl, "Hongping Qu, MD (Corresponding author)", bold=True,
     space_after=2)
para(cl, "Department of Critical Care Medicine, Ruijin Hospital, "
     "Shanghai Jiao Tong University School of Medicine, Shanghai, China",
     space_after=2)
para(cl, "E-mail: qhp10516@rjh.com.cn", space_after=2)
cl.save(f"{OUTDIR}/cover_letter_nejmai.docx")
print("cover letter saved")
print("ALL DONE:", OUTDIR)
