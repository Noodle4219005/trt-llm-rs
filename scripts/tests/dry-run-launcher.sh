#!/bin/bash
# Run the launcher's control flow with the cluster stubbed out.
#
# `bash -n` parses; it does not execute. Forty-nine lines of a dead heredoc body
# sat in this script for several commits, syntactically valid as a sequence of
# commands named `attn_backend:` and `moe_config:`, and would have printed a
# screenful of "command not found" into a 163 SU run's log. Nothing caught it:
# not the syntax check, not the tests, not the edits that walked past it.
#
# This stubs srun, apptainer, sbatch, scontrol, curl and nvidia-smi, points the
# script at a temporary directory, and runs it far enough to write both engine
# configs and lay out the workers. Anything that would have been a runtime error
# up to that point becomes one here, for free.
set -uo pipefail
SCRIPT="${1:-$(dirname "$0")/../stage-d-235b-disagg.sbatch}"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$WORK/bin"
for cmd in srun apptainer sbatch scontrol nvidia-smi curl mpirun aiperf; do
  cat > "$WORK/bin/$cmd" <<STUB
#!/bin/bash
echo "[stub $cmd] \$*" >> "$WORK/calls.log"
exit 0
STUB
  chmod +x "$WORK/bin/$cmd"
done
# scontrol must produce hostnames or the topology split is meaningless.
cat > "$WORK/bin/scontrol" <<'STUB'
#!/bin/bash
printf 'node-a\nnode-b\n'
STUB
chmod +x "$WORK/bin/scontrol"

export PATH="$WORK/bin:$PATH"
export SLURM_JOB_ID=999999
export SLURM_JOB_NODELIST="node-[a-b]"
export SLURM_SUBMIT_DIR="$PWD"

# Stop before anything that needs a GPU: the first worker launch.
STOP='########## NCCL preflight ##########'
sed "/$STOP/,\$d" "$SCRIPT" > "$WORK/head.sh"

echo "=== running the launcher head with the cluster stubbed ==="
bash "$WORK/head.sh" > "$WORK/out.log" 2> "$WORK/err.log"
rc=$?

fail=0
if grep -qiE "command not found|unbound variable|No such file or directory|syntax error" "$WORK/err.log"; then
  echo "!!! runtime errors:"; grep -iE "command not found|unbound variable|No such file|syntax error" "$WORK/err.log" | sort -u | head -10 | sed 's/^/    /'
  fail=1
fi
for role in prefill decode; do
  f=$(find "$WORK" /work/u5727520/trt-llm-rs-runs/out/stage-d-999999 -name "engine-$role.yaml" 2>/dev/null | head -1)
  if [ -z "$f" ]; then echo "!!! engine-$role.yaml was not written"; fail=1
  else
    echo "  engine-$role.yaml written, $(wc -l < "$f") lines"
    grep -qE '\$\{|\$[A-Z]' "$f" && { echo "!!!   it still contains an unexpanded variable"; sed -n '1,20p' "$f" | grep -n '\$' | sed 's/^/      /'; fail=1; }
  fi
done
[ $fail = 0 ] && echo "=== dry run clean (exit $rc) ===" || echo "=== dry run FOUND PROBLEMS ==="
exit $fail
