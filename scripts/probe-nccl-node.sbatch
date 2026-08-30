#!/bin/bash
# Is 25a-hgpn143's NCCL broken, or were three configurations?
#
# Jobs 316457, 316497 and 316571 ran three DIFFERENT prefill configurations
# (TP2xPP2+attention-DP, 2xTP4, 4xTP2 @ kv=0.50) on 25a-hgpn[142-143] and all
# three died the same way: the decode worker printed its eight NVLS banners and
# then produced NOTHING -- it never reached "Detecting local TP group", never
# finished registration, never served. The one run that DID serve (316364, same
# code) was on 25a-hgpn[022-023] and emitted 457 more lines at that point.
#
# I attributed each failure to its configuration. The node pair was printed in
# every sacct line and I never treated it as a variable. This costs 8 GPU x ~2
# minutes to settle.
#SBATCH -J probe-nccl-node
#SBATCH -A acd110018
#SBATCH -p dev
#SBATCH -N 1
#SBATCH -w 25a-hgpn143
#SBATCH --gres=gpu:H200:8
#SBATCH -c 96
#SBATCH -t 00:04:00
#SBATCH -o /work/u5727520/trt-llm-rs-runs/logs/%x-%j.out
#SBATCH -e /work/u5727520/trt-llm-rs-runs/logs/%x-%j.err
set -uo pipefail
SIF=/work/u5727520/dynamo-qwen-smoke/images/tensorrtllm-runtime-1.4.1.sif
echo "node=$(hostname)  $(date -Is)"
timeout 150 apptainer exec --nv --bind /work/u5727520 "$SIF" \
  torchrun --nproc_per_node=8 --nnodes=1 --master_port=$((20000 + SLURM_JOB_ID % 10000)) \
  /work/u5727520/trt-llm-rs-runs/probe/nccl_probe.py
echo "exit=$?  $(date -Is)"
