#!/usr/bin/env bash
set -euo pipefail

artifact="${TRTLLM_RS_PROTOC_FETCH_ARTIFACT_DIR:-/home/u5727520/trt-llm-rs/results/phase-d-protoc-fetch/2026-08-30}"
mkdir -p "$artifact"

set +e
CARGO_NET_OFFLINE=false cargo fetch \
  --manifest-path tools/protoc-bootstrap/Cargo.toml 2>&1 | tee "$artifact/cargo-fetch.log"
fetch_status=${PIPESTATUS[0]}
set -e

exit "$fetch_status"
