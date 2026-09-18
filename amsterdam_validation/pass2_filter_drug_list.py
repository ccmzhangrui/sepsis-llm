#!/usr/bin/env python3
"""Pass 2: filter drugitems (vasopressors/inotropes/antibiotics) and
listitems (GCS components, ventilation) from AmsterdamUMCdb CSV exports."""
import csv
import re
import sys
import time

BASE = "/Users/zhangrui/Documents/sofa2.0/AmsterdamUBD"
OUT_DIR = "/Users/zhangrui/WorkBuddy/2026-09-18-15-20-42/amsterdam_sepsis/data"

VASO_RX = re.compile(
    r"noradrenaline|norfine|adrenaline|epinefrine|dopamine|dobutamine|"
    r"vasopressine|phenylephrine|efedrine|milrinone|levosimendan|"
    r"metaraminol|noradrenalin", re.IGNORECASE)

ABX_RX = re.compile(r"antimicrob", re.IGNORECASE)

GCS_RX = re.compile(
    r"^(ogen|motoriek|verbaal|actief|robot|gcs)|glasgow|"
    r"beadem|ventilat|intubat", re.IGNORECASE)


def scan(path, out_name, filters, name_cols):
    t0 = time.time()
    n = kept = 0
    with open(path, newline="", encoding="utf-8", errors="replace") as fi, \
            open(f"{OUT_DIR}/{out_name}", "w", newline="") as fo:
        reader = csv.reader(fi)
        header = next(reader)
        w = csv.writer(fo)
        w.writerow(header)
        idx = {c: i for i, c in enumerate(header)}
        col_ids = [idx[c] for c in name_cols if c in idx]
        for row in reader:
            n += 1
            text = " | ".join(row[i] for i in col_ids if i < len(row))
            if any(rx.search(text) for rx in filters):
                w.writerow(row)
                kept += 1
            if n % 5_000_000 == 0:
                print(f"  {out_name}: {n/1e6:.0f}M rows, kept {kept}, "
                      f"{time.time()-t0:.0f}s", flush=True)
    print(f"DONE {out_name}: {n} rows, kept {kept}, "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "drugs":
        scan(f"{BASE}/drugitems.csv", "drugitems_vaso_abx.csv",
             [VASO_RX, ABX_RX], ["item", "ordercategory"])
    elif which == "list":
        scan(f"{BASE}/listitems.csv", "listitems_gcs_vent.csv",
             [GCS_RX], ["item"])
