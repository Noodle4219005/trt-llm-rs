#!/usr/bin/env bash
set -euo pipefail

artifact="${TRTLLM_RS_PROTOC_ARTIFACT_DIR:-/home/u5727520/trt-llm-rs/results/phase-d-protoc/2026-08-30}"
mkdir -p "$artifact"

path_file="$artifact/protoc-path.txt"
log_file="$artifact/bootstrap.log"
set +e
CARGO_NET_OFFLINE=true cargo run \
  --manifest-path tools/protoc-bootstrap/Cargo.toml \
  --locked \
  --quiet > "$path_file" 2> "$log_file"
cargo_status=$?
set -e
if (( cargo_status != 0 )); then
  cat "$log_file" >&2
  exit "$cargo_status"
fi

protoc_path="$(<"$path_file")"
if [[ ! -x "$protoc_path" ]]; then
  echo "bootstrap did not produce an executable protoc: $protoc_path" >&2
  exit 1
fi
"$protoc_path" --version | tee "$artifact/protoc-version.txt"
printf '%s\n' "$protoc_path" > "$path_file"
