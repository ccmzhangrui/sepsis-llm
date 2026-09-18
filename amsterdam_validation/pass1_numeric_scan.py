#!/usr/bin/env python3
"""Pass 1: stream numericitems from AmsterdamUMCdb nested zip (deflate64).

Structure: numericitems.zip (deflate64) -> inner 'numericitems.zip' (zip64,
single stored member 'numericitems.csv').

Single streaming pass:
  A) enumerate distinct (itemid, item) with row counts -> items_numeric.csv
  B) filter rows whose item name matches clinical regex patterns
     -> numeric_filtered.csv
"""
import csv
import io
import re
import struct
import sys
import time
import zlib

import zipfile_deflate64 as zipfile

SRC = "/Users/zhangrui/Documents/sofa2.0/AmsterdamUBD/numericitems.zip"
OUT_DIR = "/Users/zhangrui/WorkBuddy/2026-09-18-15-20-42/amsterdam_sepsis/data"

# broad Dutch/English clinical item-name patterns (case-insensitive)
PATTERNS = [
    r"lactaat", r"kreat", r"ureum", r"bilirubin", r"thrombo", r"trombo",
    r"hartfreq", r"hartritme", r"bloeddruk", r"\babp\b", r"ademfreq",
    r"ademhaling", r"pao2", r"fio2", r"spo2", r"saturatie", r"etco2",
    r"pco2", r"po2", r"\bph\b", r"kalium", r"natrium", r"glucose",
    r"temperatuur", r"zuurstof", r"peep", r"tidal", r"plateau",
    r"leuko", r"hemoglobine", r"\bhb\b", r"hematocriet", r"\bck\b",
    r"troponine", r"crp\b", r"procalcitonine", r"albumine", r"\baf\b",
    r"\basat\b", r"\balat\b", r"inr\b", r"\bpt\b", r"aptt",
    r"urineproductie", r"diurese",
]
RX = re.compile("|".join(PATTERNS), re.IGNORECASE)


def parse_inner_local_header(stream):
    """Read the inner zip local file header from a binary stream.

    Returns (filename, data_size) where data_size is the compressed size of
    the stored member (== uncompressed size for method 0).
    """
    sig = stream.read(4)
    assert sig == b"PK\x03\x04", f"bad inner sig: {sig!r}"
    hdr = stream.read(26)
    (ver, flags, method, mtime, mdate, crc, csize, usize,
     fnlen, extralen) = struct.unpack("<HHHHHIIIHH", hdr)
    fname = stream.read(fnlen).decode("utf-8")
    extra = stream.read(extralen)
    real_csize, real_usize = csize, usize
    if csize == 0xFFFFFFFF or usize == 0xFFFFFFFF or flags & 0x08:
        # zip64 extra field 0x0001
        i = 0
        while i + 4 <= len(extra):
            tag, sz = struct.unpack("<HH", extra[i:i + 4])
            body = extra[i + 4:i + 4 + sz]
            if tag == 0x0001:
                vals = struct.unpack("<" + "Q" * (sz // 8), body)
                # order: usize, csize (only those that are 0xFFFFFFFF appear,
                # but with data-descriptor flags both are typically present)
                if len(vals) >= 2:
                    real_usize, real_csize = vals[0], vals[1]
                elif len(vals) == 1:
                    real_usize = real_csize = vals[0]
            i += 4 + sz
    assert method in (0, 8), f"unexpected inner method {method}"
    return fname, real_csize, flags, method


class LimitedReader(io.RawIOBase):
    """Expose exactly n bytes from an underlying stream."""

    def __init__(self, raw, n):
        self.raw = raw
        self.remaining = n

    def readable(self):
        return True

    def read(self, size=-1):
        if self.remaining <= 0:
            return b""
        if size is None or size < 0 or size > self.remaining:
            size = self.remaining
        data = self.raw.read(size)
        self.remaining -= len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n


def main():
    t0 = time.time()
    z = zipfile.ZipFile(SRC)
    member = z.infolist()[0]
    raw = z.open(member)  # deflate64 stream of the inner zip
    fname, data_size, flags, method = parse_inner_local_header(raw)
    print(f"inner member: {fname}, bytes: {data_size}, "
          f"flags=0x{flags:x}, method={method}", flush=True)

    limited = LimitedReader(raw, data_size)
    if method == 8:
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

        limited = Inflater(limited)
    text = io.TextIOWrapper(io.BufferedReader(limited, 1024 * 1024),
                            encoding="utf-8", errors="replace", newline="")
    reader = csv.reader(text)
    header = next(reader)
    print("csv header:", header, flush=True)
    idx = {name: i for i, name in enumerate(header)}
    i_item = idx["item"]
    i_itemid = idx["itemid"]

    item_counts = {}
    n_rows = 0
    n_kept = 0
    out_path = f"{OUT_DIR}/numeric_filtered.csv"
    with open(out_path, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(header)
        for row in reader:
            n_rows += 1
            if len(row) <= max(i_item, i_itemid):
                continue
            key = (row[i_itemid], row[i_item])
            item_counts[key] = item_counts.get(key, 0) + 1
            if RX.search(row[i_item]):
                w.writerow(row)
                n_kept += 1
            if n_rows % 20_000_000 == 0:
                print(f"  {n_rows/1e6:.0f}M rows, kept {n_kept}, "
                      f"{time.time()-t0:.0f}s", flush=True)
    with open(f"{OUT_DIR}/items_numeric.csv", "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["itemid", "item", "n_rows"])
        for (iid, name), c in sorted(item_counts.items(),
                                     key=lambda kv: -kv[1]):
            w.writerow([iid, name, c])
    print(f"DONE: {n_rows} rows scanned, {n_kept} kept, "
          f"{len(item_counts)} distinct items, {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
