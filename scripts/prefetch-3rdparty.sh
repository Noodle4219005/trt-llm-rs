#!/bin/bash
# Pre-fetch TensorRT-LLM's 15 third-party repositories on the LOGIN NODE.
#
# WHY THIS EXISTS: rc22 has NO git submodules (.gitmodules is empty). It pulls 15 repos through
# CMake FetchContent at configure time -- cutlass, DeepGEMM, FlashMLA, eigen, cppzmq, googletest
# and friends (3rdparty/fetch_content.json). Compute nodes on this cluster have no route to the
# internet; the login node does (github 200, gitlab 301, git ls-remote works). So the sources are
# fetched here, where there is network and no compute, and the build runs offline on a node.
#
# This is network I/O, not compilation -- the same reasoning that puts `cargo fetch` and this
# repo's own fetch-deps.sh on the login node. Nothing is compiled by this script.
#
# The cache goes on /work, NOT the default <project>/3rdparty/.cache_3rdparty: that default is
# under /home, which has 7 GB free, and cutlass alone would eat a large share of it.
set -euo pipefail

SRC=${SRC:-/home/u5727520/trt-llm-rs/third_party/TensorRT-LLM}
CACHE=${CACHE:-/work/u5727520/trtllm-3rdparty-cache}
mkdir -p "$CACHE"

echo "source: $SRC"
echo "cache:  $CACHE  ($(df -h "$CACHE" | tail -1 | awk '{print $4}') free)"

python3 - "$SRC/3rdparty/fetch_content.json" "$CACHE" <<'PY'
import json, subprocess, sys, os, re
spec, cache = sys.argv[1], sys.argv[2]
deps = json.load(open(spec))
deps = deps if isinstance(deps, list) else deps.get('dependencies', [])
ok = fail = skip = 0
for d in deps:
    name, url, tag = d.get('name'), d.get('git_repository'), d.get('git_tag')
    if not url or not name:
        print(f"  -- {name}: no git_repository, skipped"); skip += 1; continue
    # ${github_base_url} is substituted by cmake at configure time; assume public GitHub here.
    url = url.replace('${github_base_url}', 'https://github.com')
    bare = os.path.join(cache, f"{name}.git")
    if os.path.isdir(bare):
        print(f"  == {name}: cache present"); ok += 1; continue
    print(f"  ++ {name} <- {url} @ {tag}")
    # A bare mirror, so `git clone --reference` can satisfy objects from disk. Full history, not
    # shallow: a shallow bare cannot serve as a reference for an arbitrary tag later.
    r = subprocess.run(['git','clone','--bare','--quiet',url,bare])
    if r.returncode: print(f"  !! {name}: clone failed rc={r.returncode}"); fail += 1
    else: ok += 1
print(f"\n  ok={ok} failed={fail} skipped={skip} of {len(deps)}")
sys.exit(1 if fail else 0)
PY

echo
du -sh "$CACHE"
echo "Pass this to the build as -DTRTLLM_FETCHCONTENT_CACHE=$CACHE"
