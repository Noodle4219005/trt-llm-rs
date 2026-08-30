#!/usr/bin/env python3
"""Aggregate the D1 experiment's per-run result.csv files into one dataset.

Applies the pre-registered validity rules BEFORE reporting anything:
  * aiperf exit 0
  * request_count == 320
  * output sequence length within 1 token of 128
  * ARM=d1 runs must show >= 352 POST /generate in the Python worker's access
    log, i.e. the request actually traversed the Rust adapter. A d1 run whose
    counter did not fire is void, not slow.

Usage:
    python3 scripts/d1-aggregate.py OUTDIR... > runs.csv
Summary goes to stderr so stdout stays a clean CSV.
"""
from __future__ import annotations

import csv
import io
import statistics
import sys
from pathlib import Path

FIELDS = [
    "arm", "rep", "job_id", "node", "itl_mean_ms", "ttft_mean_ms",
    "request_throughput_rps", "output_token_throughput_tps",
    "output_seq_len_mean", "request_count", "benchmark_duration_s",
    "mechanism_fired", "aiperf_rc", "neighbours_start", "neighbours_end",
    "valid", "invalid_reason",
]
EXPECTED_REQUESTS = 320
MIN_MECHANISM_FIRINGS = 352      # 320 measured + 32 warmup
GATE = 1.15                      # pre-registered: mean(d1) <= 1.15 * mean(a1r)


def why_invalid(row: dict[str, str]) -> str:
    def num(key: str) -> float:
        text = row.get(key, "")
        return float(text) if text not in ("", None) else float("nan")

    if row.get("aiperf_rc") != "0":
        return f"aiperf exit {row.get('aiperf_rc')}"
    if abs(num("request_count") - EXPECTED_REQUESTS) > 0.5:
        return f"request_count {row.get('request_count')} != {EXPECTED_REQUESTS}"
    if abs(num("output_seq_len_mean") - 128) > 1.0:
        return f"output_seq_len_mean {row.get('output_seq_len_mean')} off 128"
    if row["arm"] == "d1" and num("mechanism_fired") < MIN_MECHANISM_FIRINGS:
        return (f"mechanism fired {row.get('mechanism_fired')} "
                f"< {MIN_MECHANISM_FIRINGS}: requests did not traverse the Rust adapter")
    return ""


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2

    rows: list[dict[str, str]] = []
    for outdir in argv[1:]:
        path = Path(outdir) / "result.csv"
        if not path.is_file():
            print(f"missing {path}", file=sys.stderr)
            continue
        # aiperf writes CRLF on its single-value metric rows, so a bare \r can
        # end up inside a field of result.csv. Python's csv module treats that
        # as a row terminator and misaligns every column after it -- silently,
        # which is worse than crashing. Strip CR before parsing; the raw
        # artifact itself is left untouched.
        # newline="" is load-bearing: without it Python's universal-newline
        # translation turns the embedded CR into a LF *before* we can strip it,
        # which splits the row instead of repairing it.
        with path.open("r", newline="") as raw:
            text = raw.read().replace("\r", "")
        with io.StringIO(text) as handle:
            for row in csv.DictReader(handle):
                reason = why_invalid(row)
                row["valid"] = "no" if reason else "yes"
                row["invalid_reason"] = reason
                rows.append(row)

    rows.sort(key=lambda r: (r["arm"], int(r["rep"])))
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

    def itls(arm: str) -> list[float]:
        return [float(r["itl_mean_ms"]) for r in rows
                if r["arm"] == arm and r["valid"] == "yes"]

    say = lambda *a: print(*a, file=sys.stderr)
    say("\n=== D1 summary (valid runs only) ===")
    for arm in ("a1r", "d1"):
        values = itls(arm)
        invalid = [r for r in rows if r["arm"] == arm and r["valid"] == "no"]
        if values:
            spread = statistics.stdev(values) if len(values) > 1 else 0.0
            say(f"{arm}: n={len(values)} ITL mean={statistics.mean(values):.4f} ms "
                f"stdev={spread:.4f} values={[round(v, 4) for v in values]}")
        else:
            say(f"{arm}: no valid runs")
        for row in invalid:
            say(f"  INVALID {arm} rep{row['rep']} job {row['job_id']}: {row['invalid_reason']}")

    base, treat = itls("a1r"), itls("d1")
    if len(base) >= 1 and len(treat) >= 1:
        ratio = statistics.mean(treat) / statistics.mean(base)
        say(f"\nITL ratio d1/a1r = {ratio:.4f} (pre-registered gate <= {GATE})")
        say("VERDICT:", "PASS" if ratio <= GATE else "FAIL")
        if len(base) < 3 or len(treat) < 3:
            say("NOTE: fewer than three valid runs in an arm -- the noise floor is "
                "not established, so this verdict is provisional, not confirmatory.")
    else:
        say("\nno verdict: an arm has no valid runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
