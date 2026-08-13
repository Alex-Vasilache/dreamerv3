#!/bin/bash
# One command for the final pass: scores -> figures -> paper.
#
# Everything the analysis needs is derived from the run directories and the
# geometry .npz files, so this is idempotent and can be run as often as data
# lands. It refuses to quietly produce a figure from half a batch: pass
# --expect N to assert the number of complete runs per cell, and it stops if
# any cell is short.
#
#   experiments/arm_comparison/finalize.sh                 # whatever is done
#   experiments/arm_comparison/finalize.sh --expect 4      # the real thing
#   experiments/arm_comparison/finalize.sh --at-step 3000000
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PAPER="${PAPER:-/apps/unit/DoyaU/vasilache/apps/code/26_04_HRL-paper}"
FIGS="$PAPER/figures/motivation"
AC="$REPO/experiments/arm_comparison"
GG="$REPO/experiments/goal_geometry"

AT_STEP="${AT_STEP:-4000000}"
EXPECT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --expect) EXPECT="$2"; shift 2 ;;
    --at-step) AT_STEP="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$AC/results" "$FIGS"
JSON="$AC/results/arms_$((AT_STEP / 1000000))M.json"

echo "=== 1/4  scores at ${AT_STEP} steps ==="
python3 "$AC/collect_scores.py" --at-step "$AT_STEP" --out "$JSON" | tee \
  "$AC/results/summary_$((AT_STEP / 1000000))M.txt"

if [ -n "$EXPECT" ]; then
  short=$(python3 - "$JSON" "$EXPECT" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
want = int(sys.argv[2])
bad = [f'{k} n={v["n"]}' for k, v in sorted(data['cells'].items())
       if v['n'] < want]
missing = [f'{t}|{a}' for t, _ in data['tasks'] for a, _ in data['arms']
           if f'{t}|{a}' not in data['cells']]
print('; '.join(bad + [m + ' n=0' for m in missing]))
PY
)
  if [ -n "$short" ]; then
    echo
    echo "STOP: cells short of $EXPECT complete seeds -> $short"
    echo "(rerun without --expect to produce provisional figures anyway)"
    exit 1
  fi
  echo "all cells have >= $EXPECT complete seeds"
fi

echo
echo "=== 2/4  endpoint figure ==="
python3 "$AC/make_figure_arms.py" --json "$JSON" \
  --out "$AC/arm_comparison" --copy-to "$FIGS"

echo
echo "=== 3/4  learning curves + goal-code landscape ==="
python3 "$AC/make_figure_curves.py" --json "$JSON" \
  --out "$AC/arm_curves" --copy-to "$FIGS"
python3 "$GG/make_figure_landscape.py" --results "$GG/results" \
  --out "$GG/goal_code_landscape" --copy-to "$FIGS"

echo
echo "=== 4/4  paper ==="
cd "$PAPER" || exit 1
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
if grep -qE '^! ' main.log; then
  echo "LaTeX errors:"; grep -E '^! ' -A 3 main.log | head -20; exit 1
fi
echo "built $(pdfinfo main.pdf | awk '/Pages/{print $2}') pages"
grep -E 'LaTeX Warning: Reference' main.log | head -5
echo
echo "figures now in $FIGS:"
ls -la "$FIGS"/arm_comparison.pdf "$FIGS"/arm_curves.pdf \
  "$FIGS"/goal_code_landscape.pdf 2>/dev/null
