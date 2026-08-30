#!/bin/bash
# Two things the mode='c' -> mode='r' fix depends on, neither of which may be guessed:
#   1. Does MAP_SHARED|PROT_READ actually cost zero commit charge? The whole fix
#      is that 8 ranks x 220 GiB stops counting against CommitLimit=1603 GiB.
#   2. Does torch.from_numpy accept a read-only memmap -- warning, or exception?
#      mode='r' gives non-writable arrays. If torch refuses them the fix is dead.
#SBATCH -J probe-mmap-readonly
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
apptainer exec --nv $BIND --env PYTHONPATH=/home/u5727520/trt-llm-rs/scripts "$SIF" python3 - "$MODEL" <<'PY'
import glob, os, sys, warnings

def committed_gib():
    for ln in open("/proc/meminfo"):
        if ln.startswith("Committed_AS"):
            return int(ln.split()[1]) / 2**20

files = sorted(glob.glob(os.path.join(sys.argv[1], "*.safetensors")))
size = sum(os.path.getsize(f) for f in files) / 2**30
print(f"{len(files)} shards, {size:.1f} GiB")

import numpy as np
print("\n=== 1. commit charge per mapping mode ===")
for mode in ("r", "c"):
    before = committed_gib()
    held = [np.memmap(f, dtype=np.uint8, mode=mode) for f in files]
    after = committed_gib()
    print(f"  mode={mode!r}: Committed_AS {before:.1f} -> {after:.1f} GiB "
          f"(delta {after-before:+.1f} for {size:.1f} GiB mapped)")
    del held

print("\n=== 2. torch on a read-only memmap ===")
import torch
arr = np.memmap(files[0], dtype=np.uint8, mode="r")
print(f"  numpy writeable flag: {arr.flags.writeable}")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    try:
        t = torch.from_numpy(arr[:4096])
        print(f"  torch.from_numpy: OK, {t.dtype} {tuple(t.shape)}")
        print(f"  warnings: {[str(w.message)[:70] for w in caught] or 'none'}")
        v = t.view(torch.bfloat16)
        print(f"  .view(bfloat16): OK {tuple(v.shape)}; first={v[0].item():.6f}")
        g = v.cuda()
        print(f"  .cuda(): OK, device={g.device}, matches={torch.equal(g.cpu(), v)}")
    except Exception as e:
        print(f"  torch.from_numpy FAILED: {type(e).__name__}: {e}")

print("\n=== 3. does anything in the load path write to the weights? ===")
print("  (a write to a PROT_READ page is SIGSEGV, not an exception)")
PY
