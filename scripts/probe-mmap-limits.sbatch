#!/bin/bash
# Which limit refuses our mmap: commit charge, or RLIMIT_AS?
#
# Job 316229's decode worker died inside OUR np.memmap(mode="c") at
# numpy/_core/memmap.py:280 with ENOMEM, after ~16 of 24 shards. Two
# explanations fit equally well and demand opposite fixes:
#
#   (c) commit charge -- MAP_PRIVATE writable reserves every page. mode="r"
#       (MAP_SHARED, PROT_READ) reserves nothing and the fix is one character.
#   (b) RLIMIT_AS    -- address space is capped near 200 GB per rank. Then NO
#       zero-copy scheme can work, because zero-copy means mapping all 235 GB,
#       and the fix has to bound what is mapped at once instead.
#
# Guessing costs a 16-GPU run. Measuring costs about 3 SU.
#SBATCH -J probe-mmap-limits
#SBATCH -A acd110018
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:H200:1
#SBATCH -c 12
#SBATCH -t 00:10:00
#SBATCH -o /work/u5727520/trt-llm-rs-runs/logs/%x-%j.out
#SBATCH -e /work/u5727520/trt-llm-rs-runs/logs/%x-%j.err
set -uo pipefail
SIF=/work/u5727520/dynamo-qwen-smoke/images/tensorrtllm-runtime-1.4.1.sif
MODEL=/work/u4063814/hf_cache/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/e156cb4efae43fbee1a1ab073f946a1377e6b969
BIND="--bind /work/u4063814,/work/u5727520,/home/u5727520"
apptainer exec --nv $BIND "$SIF" python3 - "$MODEL" <<'PY'
import glob, os, resource, sys

print("=== limits as the rank sees them ===")
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
inf = resource.RLIM_INFINITY
fmt = lambda v: "unlimited" if v == inf else f"{v/2**30:.1f} GiB"
print(f"RLIMIT_AS   soft={fmt(soft)} hard={fmt(hard)}")
for f in ("/proc/sys/vm/overcommit_memory", "/proc/sys/vm/overcommit_ratio"):
    try: print(f"{f} = {open(f).read().strip()}")
    except OSError as e: print(f"{f}: {e}")
for k in ("MemTotal", "CommitLimit", "Committed_AS"):
    for ln in open("/proc/meminfo"):
        if ln.startswith(k): print("  " + ln.strip())
for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"):
    try: print(f"{p} = {open(p).read().strip()}")
    except OSError as e: print(f"{p}: {e}")

files = sorted(glob.glob(os.path.join(sys.argv[1], "*.safetensors")))
total = sum(os.path.getsize(f) for f in files)
print(f"\n=== {len(files)} shards, {total/2**30:.1f} GiB total ===")

import mmap as _m
for label, access in (("mode='r'  MAP_SHARED PROT_READ", _m.ACCESS_READ),
                      ("mode='c'  MAP_PRIVATE writable", _m.ACCESS_COPY)):
    held, mapped = [], 0
    try:
        for f in files:
            fh = open(f, "rb")
            held.append((fh, _m.mmap(fh.fileno(), 0, access=access)))
            mapped += os.path.getsize(f)
        print(f"{label}: ALL {len(files)} mapped, {mapped/2**30:.1f} GiB")
    except OSError as e:
        print(f"{label}: FAILED after {len(held)}/{len(files)} "
              f"({mapped/2**30:.1f} GiB) -- {e}")
    finally:
        for fh, mm in held:
            mm.close(); fh.close()
PY
