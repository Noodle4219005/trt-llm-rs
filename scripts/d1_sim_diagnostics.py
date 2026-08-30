#!/usr/bin/env python3
"""Print the decode-side diagnostics from one `trt-llm-rs sim --json` report.

Separate file rather than an inline `python3 -c` inside a nested `bash -c`:
job 314768 produced the whole sweep and then died on the quoting.
"""
import json
import sys

for path in sys.argv[1:]:
    with open(path) as handle:
        report = json.load(handle)
    g, d = report["goodput"], report["diagnostics"]
    print(
        f"{path.rsplit('/', 1)[-1]:<16} "
        f"goodput={g['goodput_req_s']:7.2f} good={g['good_frac'] * 100:5.1f}% "
        f"itl_mean={g['itl']['mean']:6.2f}ms cap={d['final_decode_cap']:6.1f} "
        f"decodeC={d['mean_decode_concurrency']:6.1f} "
        f"step={d['observed_step_ms']:6.2f}ms "
        f"refusals={d['decode_refusals']} "
        f"extrapolated={d['extrapolated_beyond_calibration']}"
    )
