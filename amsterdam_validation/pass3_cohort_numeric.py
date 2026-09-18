#!/usr/bin/env python3
"""Pass 3: targeted extraction of ALL numericitems rows for cohort
admissions (suspected-infection backbone) from the nested zip stream."""
import csv
import io
import struct
import time
import zlib

import pandas as pd
import zipfile_deflate64 as zipfile

SRC = "/Users/zhangrui/Documents/sofa2.0/AmsterdamUBD/numericitems.zip"
DATA = "/Users/zhangrui/WorkBuddy/2026-09-18-15-20-42/amsterdam_sepsis/data"


def parse_inner_local_header(stream):
    sig = stream.read(4)
    assert sig == b"PK\x03\x04"
    hdr = stream.read(26)
    (ver, flags, method, mtime, mdate, crc, csize, usize,
     fnlen, extralen) = struct.unpack("<HHHHHIIIHH", hdr)
    fname = stream.read(fnlen).decode("utf-8")
    extra = stream.read(extralen)
    real_csize, real_usize = csize, usize
    i = 0
    while i + 4 <= len(extra):
        tag, sz = struct.unpack("<HH", extra[i:i + 4])
        body = extra[i + 4:i + 4 + sz]
        if tag == 0x0001:
            vals = struct.unpack("<" + "Q" * (sz // 8), body)
            if len(vals) >= 2:
                real_usize, real_csize = vals[0], vals[1]
            elif len(vals) == 1:
                real_usize = real_csize = vals[0]
        i += 4 + sz
    return fname, real_csize, method


class LimitedReader(io.RawIOBase):
    def __init__(self, raw, n):
        self.raw = raw
        self.remaining = n

    def readable(self):
        return True

    def readinto(self, b):
        if self.remaining <= 0:
            return 0
        n = min(len(b), self.remaining)
        data = self.raw.read(n)
        if not data:
            return 0
        self.remaining -= len(data)
        b[:len(data)] = data
        return len(data)


class Inflater(io.RawIOBase):
    def __init__(self, raw_):
        self.raw_ = raw_
        self.d = zlib.decompressobj(-15)
        self.buf = b""
        self.eof = False

    def readable(self):
        return True

    def readinto(self, b):
        while not self.buf and not self.eof:
            chunk = self.raw_.read(1024 * 1024)
            if not chunk:
                self.buf = self.d.flush()
                self.eof = True
            else:
                self.buf = self.d.decompress(chunk)
        n = min(len(b), len(self.buf))
        b[:n] = self.buf[:n]
        self.buf = self.buf[n:]
        return n


NEEDED_ITEMIDS = {
    # vitals
    6640, 6642, 6679, 8874, 12266, 8873, 7726, 6709,
    # blood gas
    9996, 7433, 9990, 12310, 6848, 10053, 12311, 6699, 13076, 16629,
    # chemistry
    9941, 6836, 9945, 9964, 6797, 9927, 6835, 9943, 9947, 10079, 9960,
}


def main():
    cohort = pd.read_csv(f"{DATA}/cohort_suspected_infection.csv",
                         usecols=["admissionid"])
    keep = set(cohort["admissionid"].astype(int))
    print(f"cohort admissions: {len(keep)}", flush=True)

    t0 = time.time()
    z = zipfile.ZipFile(SRC)
    member = z.infolist()[0]
    raw = z.open(member)
    fname, data_size, method = parse_inner_local_header(raw)
    print(f"inner: {fname}, {data_size} bytes, method {method}", flush=True)
    stream = LimitedReader(raw, data_size)
    if method == 8:
        stream = Inflater(stream)
    text = io.TextIOWrapper(io.BufferedReader(stream, 1024 * 1024),
                            encoding="utf-8", errors="replace", newline="")
    reader = csv.reader(text)
    header = next(reader)
    out = f"{DATA}/numeric_cohort.csv"
    n = kept = 0
    with open(out, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(header)
        for row in reader:
            n += 1
            try:
                aid = int(row[0])
            except (ValueError, IndexError):
                continue
            if aid not in keep:
                continue
            try:
                iid = int(row[1])
                iname = row[2]
            except (ValueError, IndexError):
                continue
            if iid in NEEDED_ITEMIDS or "urine" in iname.lower():
                w.writerow(row)
                kept += 1
            if n % 20_000_000 == 0:
                print(f"  {n/1e6:.0f}M rows, kept {kept}, "
                      f"{time.time()-t0:.0f}s", flush=True)
    print(f"DONE: {n} scanned, {kept} kept, {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
