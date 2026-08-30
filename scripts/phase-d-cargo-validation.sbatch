#!/bin/bash
# Phase D Cargo validation for crates/dynamo, rerun after the protoc gate.
#
# Job 314078 reached etcd-client v0.17.0's build script and died in 57s with
# "Could not find `protoc`". Read-only diagnosis found no protoc in PATH, in
# the module catalogue, in the installed Spack inventory, or in Python's
# grpc_tools. The fix is source-visible rather than an opaque install:
# tools/protoc-bootstrap pins protoc-bin-vendored = 3.2.0, and
# phase-d-cargo-validation.sh runs it, verifies the binary is executable,
# records its version, and exports PROTOC before Cargo starts.
#
# Only PROTOC is exported, not PROTOC_INCLUDE: every .proto actually reached
# by this build imports local files only -- etcd-client's rpc.proto imports
# auth.proto/kv.proto, and dynamo-llm's kserve.proto imports model_config.proto.
# The dynamo protos that do import google/protobuf/* live under lib/sidecar/*
# and deploy/inference-gateway, which are not in dynamo-backend-common's graph.
#
# Resource shape is the one nano4 forces, not a preference: p_dev carries
# MinTRES=gres/gpu=1 so a CPU-only job never admits, and one GPU caps CPUs at
# 12 (1 * 12). Asking for more GPUs to buy build parallelism costs slightly
# MORE, because SU is GPU-hour * 60 and the scaling is sublinear -- see
# knowledge/hpc/2026-08-29-you-cannot-buy-build-parallelism-on-nano4-and-8gpus-accepts-one-gpu.md
# Billing follows elapsed time, not the requested limit, so -t is generous.
#SBATCH -J trtllm-rs-phase-d-cargo
#SBATCH -A acd110018
#SBATCH -p dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --gres=gpu:H200:1
#SBATCH -t 01:30:00
#SBATCH -o /home/u5727520/trt-llm-rs/results/phase-d-cargo/2026-08-30-protoc/slurm-%x-%j.out
#SBATCH -e /home/u5727520/trt-llm-rs/results/phase-d-cargo/2026-08-30-protoc/slurm-%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

printf 'node=%s job=%s cpus=%s\n' "$(hostname)" "$SLURM_JOB_ID" "$SLURM_CPUS_PER_TASK"

export CARGO_BUILD_JOBS="$SLURM_CPUS_PER_TASK"
export TRTLLM_RS_ARTIFACT_DIR=/home/u5727520/trt-llm-rs/results/phase-d-cargo/2026-08-30-protoc

bash scripts/phase-d-cargo-validation.sh
