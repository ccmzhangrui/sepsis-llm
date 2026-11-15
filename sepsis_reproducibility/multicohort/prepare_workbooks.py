"""Extract the columns required for replay from local workbooks."""

import argparse
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

MIMIC_FIELDS = {"stay_id", "subject_id", "age", "gender", "sepsis3"}
MIMIC_PREFIXES = (
    "乳酸_Lactate(mmol/L)_",
    "心率(次/min)_",
    "呼吸频率(次/min)_",
    "持续注射泵入(去甲肾上腺素)(ug/kg/min)_",
)
RUIJIN_FIELDS = {
    "住院号",
    "sex_binary",
    "age_clean",
    "icu_admit_date",
    "icu_discharge_date",
    "sepsis_related_flag",
    "date_inconsistency_flag",
    "mortality_28d_flag",
    "SOFA-2 (Baseline)",
    "SOFA-2 (72h)",
}


def extract(path, fields, prefixes=()):
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = workbook.worksheets[0].values
        headers = next(rows)
        if missing := fields.difference(headers):
            raise ValueError(
                f"Required workbook columns are missing: {sorted(missing)}"
            )
        cols = [
            (i, value)
            for i, value in enumerate(headers)
            if isinstance(value, str)
            and (value in fields or value.startswith(prefixes))
        ]
        return pd.DataFrame(
            ([row[i] for i, _ in cols] for row in rows),
            columns=[value for _, value in cols],
        )
    finally:
        workbook.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic", type=Path, required=True)
    parser.add_argument("--ruijin", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    mimic = extract(args.mimic, MIMIC_FIELDS, MIMIC_PREFIXES).dropna(subset=["stay_id"])
    if mimic.stay_id.duplicated().any():
        raise ValueError("MIMIC-IV workbook contains repeated stay_id values.")
    mimic.to_pickle(args.work / "mimic_source.pkl")
    ruijin = extract(args.ruijin, RUIJIN_FIELDS)
    ruijin.to_pickle(args.work / "ruijin_registry.pkl")


if __name__ == "__main__":
    main()
