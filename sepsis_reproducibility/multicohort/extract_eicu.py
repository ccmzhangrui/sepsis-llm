from pathlib import Path
import pandas as pd, numpy as np
import re, json, sys

src = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(exist_ok=True, parents=True)


def read(name, cols):
    return pd.read_csv(src / f"{name}.csv", usecols=cols, low_memory=False)


def iter_csv(name, cols):
    return pd.read_csv(
        src / f"{name}.csv", usecols=cols, chunksize=1_000_000, low_memory=False
    )


id = "patientunitstayid"
p = read(
    "patient",
    [
        id,
        "patienthealthsystemstayid",
        "uniquepid",
        "hospitalid",
        "age",
        "gender",
        "apacheadmissiondx",
        "hospitaladmitoffset",
        "hospitaldischargeoffset",
        "hospitaldischargestatus",
        "unitdischargeoffset",
        "unitdischargestatus",
    ],
)
if p[id].duplicated().any():
    raise ValueError("Duplicate ICU stay ID")
p.age = pd.to_numeric(
    p.age.astype(str).str.replace(">", "", regex=False), errors="coerce"
)
sepsis_text = r"\b(?:sepsis|septic (?:shock|emia)|septicemia)\b"


def diag(name, cols, time, labels):
    x = read(name, cols)
    flag = pd.Series(False, index=x.index)
    for l in labels:
        flag |= (
            x[l]
            .astype("string")
            .str.contains(sepsis_text, case=False, regex=True, na=False)
        )
    if "icd9code" in x:
        flag |= x.icd9code.astype("string").str.contains(
            r"(?<![\w.])(?:038(?:\.\d+)?|995\.9[12]|785\.52|A4[01](?:\.[A-Z0-9]+)?|R65\.2[01])(?![\w.])",
            case=False,
            regex=True,
            na=False,
        )
    return set(x.loc[flag & x[time].between(-1440, 1440), id])


a = diag(
    "admissionDx",
    [id, "admitdxenteredoffset", "admitdxpath", "admitdxname", "admitdxtext"],
    "admitdxenteredoffset",
    ["admitdxpath", "admitdxname", "admitdxtext"],
)
d = diag(
    "diagnosis",
    [id, "diagnosisoffset", "diagnosisstring", "icd9code"],
    "diagnosisoffset",
    ["diagnosisstring"],
)
b = set(
    p.loc[
        p.apacheadmissiondx.astype("string").str.contains(
            sepsis_text, case=False, regex=True, na=False
        ),
        id,
    ]
)
p["sepsis_early"] = p[id].isin(a | d | b)
adult = p.age.ge(18)
sel = p.loc[adult & p.sepsis_early].copy()
flow = {
    "patient_rows": len(p),
    "adult_admissions": int(adult.sum()),
    "diagnosis_or_apache_early_all": len(a | d | b),
    "adult_documented_sepsis": len(sel),
    "early_admissionDx_sepsis_all": len(a),
    "early_active_sepsis_all": len(d),
    "apache_admissiondx_sepsis_all": len(b),
}
print(flow, flush=True)
sofa = read(
    "eicu_sofa2_hourly_wide",
    [id, "hourly_sofa2_total_hr_0", "hourly_sofa2_total_hr_48"],
)
if sofa[id].duplicated().any():
    raise ValueError("Duplicate SOFA stay ID")
sel = sel.merge(sofa, on=id, how="left", validate="one_to_one", indicator=True).rename(
    columns={"hourly_sofa2_total_hr_0": "sofa0", "hourly_sofa2_total_hr_48": "sofa48"}
)
flow["eligible_linked_sofa_count"] = int(sel._merge.eq("both").sum())
# The notebook already computes rolling component scores; no second rolling maximum.
sel.loc[sel.unitdischargeoffset.lt(2880) | sel.unitdischargeoffset.isna(), "sofa48"] = (
    np.nan
)
sel.drop(columns="_merge").to_csv(out / "eicu_base.csv", index=False)
(out / "eicu_flow.json").write_text(json.dumps(flow, indent=2))
ids = set(sel[id])
print("Streaming lactate", flush=True)
cols = [
    id,
    "labresultoffset",
    "labname",
    "labresult",
    "labmeasurenamesystem",
    "labmeasurenameinterface",
]
labs = []
for i, q in enumerate(iter_csv("lab", cols)):
    mask = (
        q[id].isin(ids)
        & q.labname.astype("string").str.fullmatch(r"(?i)lactate|lactic acid", na=False)
        & q.labresultoffset.between(-180, 2880)
    )
    labs.append(q.loc[mask].copy())
    if i % 10 == 0:
        print("lab", i, flush=True)
pd.concat(labs, ignore_index=True).to_csv(out / "eicu_lactate.csv", index=False)
print("Streaming heart and respiratory rates", flush=True)
cols = [id, "nursingchartoffset", "nursingchartcelltypevalname", "nursingchartvalue"]
for i, q in enumerate(iter_csv("nurseCharting", cols)):
    mask = (
        q[id].isin(ids)
        & q.nursingchartoffset.between(0, 2880)
        & q.nursingchartcelltypevalname.isin(["Heart Rate", "Respiratory Rate"])
    )
    z = q.loc[mask].copy()
    z.nursingchartvalue = pd.to_numeric(z.nursingchartvalue, errors="coerce")
    z = z.loc[
        (
            z.nursingchartcelltypevalname.eq("Heart Rate")
            & z.nursingchartvalue.between(20, 300)
        )
        | (
            z.nursingchartcelltypevalname.eq("Respiratory Rate")
            & z.nursingchartvalue.between(3, 100)
        )
    ]
    z.to_csv(
        out / "eicu_vitals.csv", mode="w" if not i else "a", header=not i, index=False
    )
    if i % 20 == 0:
        print("vitals", i, flush=True)
print("Extract complete", flush=True)
