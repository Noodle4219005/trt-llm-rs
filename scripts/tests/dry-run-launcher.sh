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

# The half this cannot execute still gets checked, because the half it could
# execute is not where the last two defects were. Forty-nine lines of dead
# heredoc sat after the engine-config writer; a verdict block landed inside
# `cleanup ()` and produced `echo "echo; echo "########## verdict ##########"`,
# which bash accepts -- the quotes close and `##########` starts a comment --
# and which the cut above never reached. Both were syntactically valid garbage
# in a region the runner did not enter.
echo "=== static checks over the whole script ==="
static_fail=0

# An `echo` line with an odd number of double quotes. That is the exact
# signature of the cleanup-block bug -- `echo "echo; echo "###...` has three --
# and it is narrow enough not to fire on a multi-line `python3 -c "` string,
# whose opening and closing lines legitimately carry one each.
awk 'BEGIN{bad=0}
     /^[[:space:]]*#/ {next}
     /^[[:space:]]*echo([[:space:]]|;)/ {
       n=gsub(/"/,"&");
       if (n%2==1) { print "    line " NR ": echo with " n " quotes: " $0; bad=1 }
     }
     END{exit bad}' "$SCRIPT" || static_fail=1

# A bare `key: value` at the start of a line outside a heredoc is the dead-YAML
# signature. Anything indented, commented, or a shell label is fine.
grep -nE '^[a-z_]+:[[:space:]]+[^ ]' "$SCRIPT" \
  | grep -vE '^[0-9]+:[[:space:]]*#' \
  | while read -r hit; do echo "    dead YAML? $hit"; done \
  | grep -q . && static_fail=1

if [ $static_fail = 0 ]; then echo "  static checks clean"; else echo "!!! static checks found problems"; fi

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
[ "$static_fail" = 1 ] && fail=1
[ $fail = 0 ] && echo "=== dry run clean (exit $rc) ===" || echo "=== dry run FOUND PROBLEMS ==="
exit $fail
