#!/bin/bash
# AIConfigurator layout search, then score its candidates under OUR metric.
#
# AIConfigurator filters on TTFT/TPOT targets and models no scheduler, no KV
# behaviour and no queueing. It is a generator of candidate topologies, not a
# judge of this deployment - so every row it returns goes through `trt-llm-rs
# tune`, which scores it on per-request good_frac in simulation.
#
# CPU-only. The GPU request is forced by nano4's p_dev QOS (MinTRES=gres/gpu=1),
# which also caps -c at 12 per GPU.
#SBATCH -J aic-tune
#SBATCH -A acd110018
#SBATCH -p dev
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

AIC=/work/u5727520/aic/venv/bin/aiconfigurator
RS=/work/u5727520/.trtllm-rs-target/release/trt-llm-rs
# aiconfigurator's safe_mkdir only allows save-dirs under CWD, $HOME, /tmp,
# /workspace, /var/tmp or the tempdir. A --save-dir on /work is accepted on the
# command line and then throws when it writes, losing every artifact after a
# successful search - so this has to live under $HOME.
OUT=$HOME/aic-results
MODEL=Qwen/Qwen3-235B-A22B-FP8

# The scored workload, exactly: ISL 4000 / OSL 200, TTFT 3000 ms, ITL 20 ms,
# 16 GPUs, and --prefix 0 because the scored run busts the prefix cache.
COMMON=(--model-path "$MODEL" --system h200_sxm --total-gpus 16
        --isl 4000 --osl 200 --ttft 3000 --tpot 20
        --prefix 0 --database-mode SILICON --top-n 5)

for BACKEND in trtllm vllm sglang; do
  echo "############ AIConfigurator: backend=$BACKEND ############"
  "$AIC" cli support --model-path "$MODEL" --system h200_sxm --backend "$BACKEND" 2>&1 | tail -12
  rm -rf "$OUT/$BACKEND"
  "$AIC" cli default "${COMMON[@]}" --backend "$BACKEND" --save-dir "$OUT/$BACKEND" 2>&1 | tail -80
  echo "---- artifacts ----"
  find "$OUT/$BACKEND" -maxdepth 3 -name "*.csv" -o -maxdepth 3 -name "*_config.yaml" | sort
done

for BACKEND in trtllm vllm sglang; do
  echo "############ trt-llm-rs scoring of $BACKEND candidates ############"
  if [ -d "$OUT/$BACKEND" ]; then
    "$RS" -c configs/qwen3-4p1d.toml tune --save-dir "$OUT/$BACKEND" \
        --system h200_sxm --backend "$BACKEND" 2>&1 | tail -40
  else
    echo "(no save-dir for $BACKEND)"
  fi
done
echo "===== DONE ====="
