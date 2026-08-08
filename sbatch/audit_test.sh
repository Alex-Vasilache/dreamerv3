#!/bin/bash
# Submit a pytest run to a free GPU node and block until it finishes, printing
# the summary. Used by the overnight audit so each edit/test cycle is one call.
#
#   ./sbatch/audit_test.sh embodied/tests/test_foo.py [more tests...]
#
# Falls back across partitions so the audit never stalls on a full queue.
set -uo pipefail

REPO=/apps/unit/DoyaU/vasilache/apps/code/dreamerv3_audit
LOGS=/work/DoyaU/vasilache/work/slurm_logs
TESTS="$*"
[ -z "$TESTS" ] && { echo "usage: $0 <test files>"; exit 2; }

for part in "gpu-v100 --gres=gpu:v100:1" "gpu-p100 --gres=gpu:p100:1 --nodelist=saion-gpu[11-14]" "intel"; do
  jid=$(REPO="$REPO" TESTS="$TESTS" sbatch -p $part -J audit \
        "$REPO/sbatch/run_pytest_cpu.sbatch" 2>/dev/null | awk '{print $NF}')
  [ -n "$jid" ] && break
done
[ -z "${jid:-}" ] && { echo "SUBMIT FAILED"; exit 2; }

out="$LOGS/audit-$jid.out"
for _ in $(seq 1 240); do
  if [ -f "$out" ] && grep -qaE "passed|failed|error in|no tests ran" "$out"; then break; fi
  sleep 5
done
sed 's/\x1b\[[0-9;]*m//g' "$out" 2>/dev/null | grep -aE "^(FAILED|ERROR)|passed|failed|no tests ran" | tail -20
echo "--- job $jid  log: $out"
