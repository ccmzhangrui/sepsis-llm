#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NEJM AI submission figures — all rebuilt to journal standard.

Design system (NEJM/NEJM AI style):
  - Arial (Helvetica fallback); panel letters bold 12 pt; axis text 8-9 pt
  - Muted NEJM palette; no chartjunk; no titles on figures
  - Vector PDF (fonttype 42, editable text) + 300 dpi PNG
  - Explicit coordinates with verified gaps (no overlapping borders)

Data sources (verified): engine_v2_results.json, imputation_portability.json,
validation_metrics.json, manuscript_numbers.json, validation_patient_level.csv
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

DATA = "/Users/zhangrui/WorkBuddy/2026-09-18-15-20-42/amsterdam_sepsis/data"
OUT = "/Users/zhangrui/Desktop/脓毒症/脓毒症大模型文章/NEJMAI_submission"
os.makedirs(OUT, exist_ok=True)

FONT = "Arial"
try:
    font_manager.findfont("Arial", fallback_to_default=False)
except Exception:
    FONT = "Helvetica"
plt.rcParams.update({
    "font.family": FONT, "font.size": 9,
    "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.labelsize": 9, "axes.titlesize": 9,
    "figure.dpi": 150, "savefig.dpi": 300,
    "pdf.fonttype": 42,
})
print("font:", FONT)
NAVY, BLUE, LBLUE = "#1F4E79", "#2E75B6", "#B8CCE4"
RED, DRED = "#C0392B", "#8E2A20"
GRAY, LGRAY = "#595959", "#F2F2F2"
TEAL = "#2A7F62"
INK = "#262626"

A = pd.read_csv(f"{DATA}/validation_patient_level.csv")
cc = A.dropna(subset=["lac0", "lac6", "lac12", "lac24", "lac48"]).copy()
cc = cc[cc.lac0 >= 0.3]


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", pad_inches=0.05,
                dpi=300)
    plt.close(fig)
    print("saved", name)


def panel_letter(ax, s, x=-0.08, y=1.05):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="top", ha="left", color=INK)


# ================= Figure 1: pipeline (4 cohorts) =================
def fig1():
    fig = plt.figure(figsize=(6.9, 5.1))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(x, y, w, h, fc, ec, lw=1.0):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.6",
            fc=fc, ec=ec, lw=lw, zorder=2))

    def arrow(p0, p1, lw=1.0, ms=9):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>",
                                     mutation_scale=ms, color="#555555",
                                     lw=lw, zorder=1))

    # --- top: four cohorts ---
    cohorts = [
        ("MIMIC-IV v2.2", "Development", "n=17,292", NAVY),
        ("eICU v2.0", "External validation", "n=11,029", BLUE),
        ("Ruijin Hospital", "Clinical validation", "n=200", TEAL),
        ("AmsterdamUMCdb", "Frozen external validation", "n=3,949", GRAY),
    ]
    W, H, GAP, X0, Y0 = 22.4, 15.0, 1.9, 1.0, 80.0
    centers = []
    for i, (name, role, n, col) in enumerate(cohorts):
        x = X0 + i * (W + GAP)
        centers.append(x + W / 2)
        box(x, Y0, W, H, "white", col, lw=1.1)
        ax.text(x + W/2, Y0 + H - 3.2, name, ha="center", va="center",
                fontsize=8.6, fontweight="bold", color=col, zorder=3)
        ax.text(x + W/2, Y0 + H/2 - 1.4, role, ha="center", va="center",
                fontsize=7.2, color=INK, zorder=3)
        ax.text(x + W/2, Y0 + 2.2, n, ha="center", va="center",
                fontsize=8.2, fontweight="bold", color=INK, zorder=3)

    # freeze divider between eICU (col 2) and Ruijin (col 3)
    xdiv = X0 + 2 * W + GAP + GAP / 2  # 48.65
    ax.plot([xdiv, xdiv], [97.6, 79.0], color="#999999", lw=0.9,
            ls=(0, (4, 3)), zorder=1)
    ax.text(xdiv, 98.6, "model freeze", fontsize=7.0, ha="center",
            color="#666666", style="italic")

    # --- EHR ingestion bus (clean bus: 4 stubs + junction dots) ---
    ybus = 76.0
    ax.plot([centers[0], centers[-1]], [ybus, ybus], color="#888888",
            lw=0.9, zorder=1)
    for c in centers:
        ax.plot([c, c], [Y0, ybus], color="#888888", lw=0.9, zorder=1)
        ax.plot([c], [ybus], marker="o", ms=2.4, color="#888888",
                zorder=1)
    ax.text(centers[0] - 0.6, 77.5, "EHR ingestion", fontsize=7.0,
            ha="left", color="#666666", style="italic")

    # --- middle: three agents ---
    agents = [
        ("Agent 1", "Temporal extraction",
         "Six-domain temporal matrix\n(0, 6, 12, 24, 48 h windows)", NAVY),
        ("Agent 2", "Trajectory imputation",
         "Frozen XGBoost regressor;\nEngine v2: interpolation-first\n+ anchor-free fallback", RED),
        ("Agent 3", "Audit-gated decisions",
         "Deterministic rule checks:\nSOFA-2 delta, lactate kinetics,\nheart-rate variability", NAVY),
    ]
    AW, AH, AX0, AGAP, AY = 27.5, 23.0, 6.0, 4.75, 45.0
    agent_x = []
    for i, (tag, name, desc, col) in enumerate(agents):
        x = AX0 + i * (AW + AGAP)
        agent_x.append(x)
        fc = "#FDF6F5" if col == RED else "#F7F9FC"
        box(x, AY, AW, AH, fc, col, lw=1.1)
        ax.text(x + AW/2, AY + AH - 3.2, tag, ha="center", va="center",
                fontsize=8.8, fontweight="bold", color=col, zorder=3)
        ax.text(x + AW/2, AY + AH - 7.0, name, ha="center", va="center",
                fontsize=8.0, color=INK, zorder=3)
        ax.text(x + AW/2, AY + 7.0, desc, ha="center", va="center",
                fontsize=7.0, color="#444444", linespacing=1.55, zorder=3)
    a_centers = [x + AW/2 for x in agent_x]

    # bus -> Agent 1 (single vertical arrow at Agent 1 center)
    arrow((a_centers[0], ybus), (a_centers[0], AY + AH + 0.4))

    # horizontal chain Agent 1 -> Agent 2 -> Agent 3
    for i in range(2):
        arrow((agent_x[i] + AW + 0.3, AY + AH/2),
              (agent_x[i] + AW + AGAP - 0.3, AY + AH/2))

    # --- fan-in collector: all agents -> audit gate ---
    ycol = 40.0
    for c in a_centers:
        ax.plot([c, c], [AY - 0.3, ycol], color="#888888", lw=0.9,
                zorder=1)
    ax.plot([a_centers[0], a_centers[-1]], [ycol, ycol], color="#888888",
            lw=0.9, zorder=1)
    arrow((a_centers[0], ycol), (a_centers[0], 35.6))

    # --- bottom: audit gate -> clinician-facing report (left to right) ---
    gw = 27.5
    box(6.0, 12, gw, 23, "#FDF6F5", RED)
    ax.text(6.0 + gw/2, 31.5, "Audit gate", ha="center", va="center",
            fontsize=8.8, fontweight="bold", color=DRED, zorder=3)
    ax.text(6.0 + gw/2, 21.5,
            "Rule checks against the\nstructured matrix;\nunsupported statements\n"
            "suppressed with a\nmanual-review flag",
            ha="center", va="center", fontsize=7.2, color="#444444",
            linespacing=1.5, zorder=3)

    rx0 = 6.0 + gw + AGAP          # 38.25
    rw = 98.0 - rx0                 # to x = 98
    box(rx0, 12, rw, 23, LGRAY, "#444444")
    ax.text(rx0 + rw/2, 31.5, "Clinician-facing CDSS report",
            ha="center", va="center", fontsize=8.8, fontweight="bold",
            color=INK, zorder=3)
    ax.text(rx0 + rw/2, 21.0,
            "Risk-stratified alert with auditable data lineage:\n"
            "every statement linked to structured data,\n"
            "temporal deltas, or prespecified rules",
            ha="center", va="center", fontsize=7.4, color="#444444",
            linespacing=1.6, zorder=3)
    # audit gate -> report (horizontal, mid-height)
    arrow((6.0 + gw + 0.3, 23.5), (rx0 - 0.3, 23.5))
    save(fig, "Figure_1")


# ================= Figure 2: performance (3 panels) =================
def fig2():
    fig = plt.figure(figsize=(6.9, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0])

    # --- Panel A: imputation R2 (horizontal dot chart) ---
    axA = fig.add_subplot(gs[0, 0])
    methods = [
        ("Engine v2, interpolation-first", 0.685, TEAL),
        ("Linear interpolation", 0.685, GRAY),
        ("MICE", 0.673, GRAY),
        ("Site-adapted XGBoost", 0.635, GRAY),
        ("Random Forest", 0.602, GRAY),
        ("Mean imputation", 0.571, GRAY),
        ("Frozen v1 formula", 0.480, DRED),
        ("Engine v2, anchor-free fallback", 0.428, BLUE),
    ]
    methods = sorted(methods, key=lambda m: m[1])
    y = np.arange(len(methods))
    vals = [m[1] for m in methods]
    cols = [m[2] for m in methods]
    axA.hlines(y, 0, vals, color="#BBBBBB", lw=1.0, zorder=1)
    axA.scatter(vals, y, s=38, color=cols, zorder=3, edgecolor="white",
                linewidth=0.6)
    for yi, v in zip(y, vals):
        axA.text(v + 0.015, yi, f"{v:.2f}", va="center", fontsize=7.4,
                 color=INK)
    axA.set_yticks(y, [m[0] for m in methods])
    axA.set_xlim(0, 0.84)
    axA.set_xlabel("R\u00b2, masked complete cases (AmsterdamUMCdb)")
    axA.spines[["top", "right"]].set_visible(False)
    panel_letter(axA, "A")

    # --- Panel B: alert-outcome forest plot ---
    axB = fig.add_subplot(gs[0, 1])
    rows = [
        ("28-day mortality", 1.61, 1.42, 1.83),
        ("Composite deterioration", 1.80, 1.68, 1.94),
        ("Vasopressor escalation", 2.16, 1.97, 2.36),
    ]
    yb = np.arange(len(rows))[::-1]
    for yi, (_, rr, lo, hi) in zip(yb, rows):
        axB.plot([lo, hi], [yi, yi], color=DRED, lw=1.6, zorder=2)
        axB.scatter([rr], [yi], s=46, color=DRED, marker="s", zorder=3)
        axB.text(3.05, yi, f"{rr:.2f} ({lo:.2f}\u2013{hi:.2f})",
                 va="center", fontsize=7.2, color=INK)
    axB.axvline(1.0, color="#888888", lw=0.9, ls=(0, (4, 3)))
    axB.set_yticks(yb, [r[0] for r in rows])
    axB.set_xscale("log")
    axB.set_xlim(0.8, 5.2)
    axB.set_xticks([1.0, 2.0, 3.0], ["1.0", "2.0", "3.0"])
    axB.set_xlabel("Risk ratio, red vs green alert (log scale)")
    axB.spines[["top", "right"]].set_visible(False)
    axB.tick_params(axis="y", length=0)
    panel_letter(axB, "B")

    # --- Panel C: coverage among indeterminate patients ---
    axC = fig.add_subplot(gs[1, :])
    axC.barh([1], [325], height=0.42, color=TEAL, edgecolor="white",
             lw=0.6)
    axC.barh([1], [1759], left=[325], height=0.42, color="#C9CDD4",
             edgecolor="white", lw=0.6)
    axC.text(325/2, 1, "Interpolable\n325 (15.6%)", ha="center",
             va="center", fontsize=7.8, color="white", fontweight="bold",
             linespacing=1.4)
    axC.text(325 + 1759/2, 1, "No later anchor — fallback arm  1,759 "
             "(84.4%)", ha="center", va="center", fontsize=7.8,
             color="#3A3F47")
    axC.barh([0.30], [900], height=0.42, color=DRED, edgecolor="white",
             lw=0.6)
    axC.barh([0.30], [1184], left=[900], height=0.42, color="#E4B7B2",
             edgecolor="white", lw=0.6)
    axC.text(450, 0.30, "Red alert 900 (43.2%)", ha="center", va="center",
             fontsize=7.8, color="white", fontweight="bold")
    axC.text(900 + 1184/2, 0.30, "No red alert 1,184 (56.8%)",
             ha="center", va="center", fontsize=7.8, color=DRED)
    axC.set_yticks([1, 0.30],
                   ["Missing 6 h or 12 h lactate (n=2,084)",
                    "Alert status among indeterminate patients"])
    axC.set_xlim(0, 2084)
    axC.set_ylim(-0.1, 1.25)
    axC.set_xlabel("Patients (n)")
    axC.spines[["top", "right", "left"]].set_visible(False)
    axC.tick_params(left=False)
    panel_letter(axC, "C")
    save(fig, "Figure_2")


# ================= Figure S1: cohort flow =================
def figS1():
    """Each label line is <= 26 chars (verified: column inner width
    ~111 pt, ~28 chars at 7.0 pt Arial) so text never exceeds borders."""
    fig = plt.figure(figsize=(6.9, 5.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    LINE_H, PAD_H, VGAP = 3.4, 4.2, 3.6   # per text line / padding / gap

    def box(x, y, w, h, fc="white", ec="#333333", lw=0.9):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.6",
            fc=fc, ec=ec, lw=lw, zorder=2))

    def text_block(x, y, w, h, lines, fs=7.0, tc=INK, bold=False,
                   n_line=None):
        """Draw lines centered; n_line (if given) extra bold below."""
        all_lines = list(lines)
        if n_line:
            all_lines = all_lines + [n_line]
        total = len(all_lines) * LINE_H
        ytop = y + h / 2 + total / 2
        for k, ln in enumerate(all_lines):
            is_n = n_line and k == len(all_lines) - 1
            ax.text(x + w/2, ytop - (k + 0.5) * LINE_H, ln, ha="center",
                    va="center", fontsize=8.0 if is_n else fs,
                    fontweight="bold" if (bold or is_n) else "normal",
                    color=tc, zorder=3)

    cols = [
        ("MIMIC-IV v2.2", NAVY, [
            (["ICU encounters meeting", "Sepsis-3 criteria"], "n = 17,292"),
            (["Development: feature", "selection, tuning,", "training"], None),
        ]),
        ("eICU v2.0", BLUE, [
            (["Sepsis-3 encounters", "across multiple centres"], "n = 11,029"),
            (["Pre-freeze external", "validation (portability)"], None),
        ]),
        ("Ruijin Hospital\n(2022\u20132025)", TEAL, [
            (["Retrospective clinical", "validation cohort"], "n = 200"),
            (["Frozen clinical validation", "and simulated", "real-time auditing"], None),
        ]),
        ("AmsterdamUMCdb\n(2003\u20132016)", GRAY, [
            (["ICU admissions", "screened"], "n = 23,106"),
            (["Suspected infection", "(Seymour antibiotic\u2013", "culture pairing)"], "n = 4,291"),
            (["Sepsis-3 analytic cohort", "(342 excluded,", "SOFA-2 < 2)"], "n = 3,949"),
            (["Fully frozen external", "validation (no", "adaptation)"], None),
        ]),
    ]
    W, GAP, X0 = 22.4, 1.9, 1.0
    last_bottom = {}
    for i, (name, col, steps) in enumerate(cols):
        x = X0 + i * (W + GAP)
        # header (2 lines max)
        box(x, 89, W, 10, fc=col, ec=col)
        ax.text(x + W/2, 94, name, ha="center", va="center", fontsize=8.0,
                fontweight="bold", color="white", linespacing=1.4,
                zorder=3)
        prev_y = 89
        for j, (lines, n) in enumerate(steps):
            nl = len(lines) + (1 if n else 0)
            h = nl * LINE_H + PAD_H
            y = prev_y - VGAP - h
            box(x, y, W, h, fc="white", ec=col)
            text_block(x, y, W, h, lines, n_line=n)
            ax.add_patch(FancyArrowPatch(
                (x + W/2, prev_y), (x + W/2, y + h + 0.15),
                arrowstyle="-|>", mutation_scale=7, color="#777777",
                lw=0.8))
            prev_y = y
        last_bottom[i] = prev_y

    # summary strip connected to every column (dashed drops)
    strip_y, strip_h = 3.5, 7.5
    box(1.0, strip_y, 98.0, strip_h, fc=LGRAY, ec="#444444")
    ax.text(50, strip_y + strip_h/2,
            "Total Sepsis-3 encounters across four cohorts: n = 32,470",
            ha="center", va="center", fontsize=8.4, fontweight="bold",
            color=INK, zorder=3)
    for i in range(4):
        x = X0 + i * (W + GAP) + W/2
        ax.plot([x, x], [last_bottom[i], strip_y + strip_h + 0.2],
                color="#AAAAAA", lw=0.8, ls=(0, (3, 3)), zorder=1)
    save(fig, "Supplementary_Figure_S1")


# ================= Figure S2: SHAP of Engine v2 fallback =================
def figS2():
    import shap
    from xgboost import XGBRegressor
    LAC = "乳酸_Lactate(mmol/L)"
    HR, RR = "心率(次/min)", "呼吸频率(次/min)"
    NOREPI = "持续注射泵入(去甲肾上腺素)(ug/kg/min)"
    GRID = ["Day1_00:00", "Day1_06:00", "Day1_12:00", "Day1_18:00",
            "Day2_00:00", "Day2_06:00", "Day2_12:00", "Day2_18:00"]
    FEATS = ["lac0", "hr_sd", "rr_sd", "age", "male", "vaso_esc"]
    NAMES = {"lac0": "Initial lactate (0 h)", "hr_sd": "Heart-rate SD",
             "rr_sd": "Respiratory-rate SD", "age": "Age", "male": "Male sex",
             "vaso_esc": "Vasopressor escalation"}

    df = pd.read_pickle(f"{DATA}/mimic_full.pkl")
    sep = df[df["sepsis3"] == True].copy()  # noqa: E712
    m = pd.DataFrame(index=sep.index)
    for w, g in [("lac0", "Day1_00:00"), ("lac6", "Day1_06:00")]:
        m[w] = sep[f"{LAC}_{g}"]
    m["hr_sd"] = sep[[f"{HR}_{g}" for g in GRID]].std(axis=1, ddof=1)
    m["rr_sd"] = sep[[f"{RR}_{g}" for g in GRID]].std(axis=1, ddof=1)
    base_dose = sep[f"{NOREPI}_Day1_00:00"].fillna(0)
    max_dose = sep[[f"{NOREPI}_{g}" for g in GRID[1:7]]].max(axis=1).fillna(0)
    m["vaso_esc"] = ((max_dose >= 0.1)
                     & ((max_dose - base_dose) >= 0.05)).astype(float)
    m["age"] = pd.to_numeric(sep["age"], errors="coerce")
    m["male"] = (sep["gender"] == "M").astype(float)
    m = m.dropna().query("lac0 >= 0.3")

    mdl = XGBRegressor()
    mdl.load_model(f"{DATA}/engine_v2_lac6_anchorfree.json")
    explainer = shap.TreeExplainer(mdl)
    sv = explainer.shap_values(m[FEATS])
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.barh(np.arange(len(FEATS)), mean_abs[order], color=NAVY, height=0.62,
            edgecolor="white")
    ax.set_yticks(np.arange(len(FEATS)), [NAMES[FEATS[i]] for i in order])
    for yi, v in enumerate(mean_abs[order]):
        ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=7.6,
                color=INK)
    ax.set_xlim(0, mean_abs.max() * 1.18)
    ax.set_xlabel("Mean |SHAP value| (mmol/L), Engine v2 fallback model")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "Supplementary_Figure_S2")


# ================= Figure S3: Bland-Altman =================
def figS3():
    BASE_WEIGHT, SOFA_IMPACT, HRV_FACTOR, RRV_FACTOR = 0.85, 0.20, 0.05, 0.03
    yt = np.concatenate([cc.lac6.values, cc.lac12.values])
    eng = (cc.lac0 * BASE_WEIGHT + cc.sofa_delta * SOFA_IMPACT
           + (10.0 - cc.hr_sd) * HRV_FACTOR + cc.rr_sd * RRV_FACTOR).values
    y_eng = np.concatenate([eng, eng])
    lin6 = cc.lac0 + (cc.lac24 - cc.lac0) * 6 / 24.0
    lin12 = cc.lac0 + (cc.lac24 - cc.lac0) * 12 / 24.0
    y_v2 = np.concatenate([lin6.values, lin12.values])

    lo = min((y_eng - yt).min(), (y_v2 - yt).min()) - 0.8
    hi = max((y_eng - yt).max(), (y_v2 - yt).max()) + 0.8

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.2))
    fig.subplots_adjust(wspace=0.32, left=0.09, right=0.985,
                        bottom=0.16, top=0.88)
    for ax, yp, lab, col in [
            (axes[0], y_eng, "Frozen v1 formula", DRED),
            (axes[1], y_v2, "Engine v2 (interpolation)", TEAL)]:
        mean = (yt + yp) / 2
        diff = yp - yt
        bias, sd = diff.mean(), diff.std(ddof=1)
        ax.scatter(mean, diff, s=4, alpha=0.25, color=col, edgecolor="none")
        ax.axhline(bias, color=INK, lw=1.0)
        for k in (-1.96, 1.96):
            ax.axhline(bias + k * sd, color="#777777", lw=0.9,
                       ls=(0, (4, 3)))
        ax.text(0.97, 0.06, f"bias {bias:+.2f} mmol/L",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.8, color=INK)
        ax.text(0.97, 0.18, f"95% LoA {bias - 1.96*sd:+.2f} to "
                f"{bias + 1.96*sd:+.2f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.8, color="#555555")
        ax.text(0.03, 0.97, lab, transform=ax.transAxes, ha="left",
                va="top", fontsize=8.4, fontweight="bold", color=col)
        ax.set_xlabel("Mean of measured and imputed lactate (mmol/L)")
        ax.set_ylabel("Imputed \u2212 measured (mmol/L)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(lo, hi)
    panel_letter(axes[0], "A", x=-0.16)
    panel_letter(axes[1], "B", x=-0.16)
    save(fig, "Supplementary_Figure_S3")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["fig1", "fig2", "figS1", "figS2", "figS3"]
    if "fig1" in which:
        fig1()
    if "fig2" in which:
        fig2()
    if "figS1" in which:
        figS1()
    if "figS2" in which:
        figS2()
    if "figS3" in which:
        figS3()
    print("FIGURES DONE:", ", ".join(which))
