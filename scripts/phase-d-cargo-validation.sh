#!/usr/bin/env bash
set -euo pipefail

artifact="${TRTLLM_RS_ARTIFACT_DIR:-/home/u5727520/trt-llm-rs/results/phase-d-cargo/2026-08-30}"
mkdir -p "$artifact"

probe_artifact="/home/u5727520/my-llm-wiki/knowledge/hpc/2026-08-30-25a-hgpn-hardware-topology.md"
if [[ ! -s "$probe_artifact" ]]; then
  /home/u5727520/my-llm-wiki/bin/hpc-probe > "$probe_artifact"
fi
protoc_artifact="$artifact/protoc"
TRTLLM_RS_PROTOC_ARTIFACT_DIR="$protoc_artifact" bash scripts/phase-d-protoc-bootstrap.sh
export PROTOC="$(<"$protoc_artifact/protoc-path.txt")"


set +e
CARGO_NET_OFFLINE=true cargo test \
  --manifest-path crates/dynamo/Cargo.toml \
  --features dynamo-v1 \
  --all-targets \
  --no-fail-fast 2>&1 | tee "$artifact/cargo-test.log"
cargo_status=${PIPESTATUS[0]}

./scripts/verify-source-tree.sh 2>&1 | tee "$artifact/source-tree.log"
source_status=${PIPESTATUS[0]}
set -e

if (( cargo_status != 0 )); then
  exit "$cargo_status"
fi
exit "$source_status"
