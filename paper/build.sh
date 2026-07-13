#!/bin/bash
# Rebuild the paper PDF. Optionally regenerate figures first:
#   ./build.sh --figures
set -euo pipefail
cd "$(dirname "$0")"

if [ "${1:-}" = "--figures" ]; then
  (
    source /etc/profile.d/modules.sh && module load python/3.7.3
    source /apps/unit/DoyaU/vasilache/apps/rl_env/bin/activate
    python3 make_figures.py
  )
fi

pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
echo "built $(pwd)/main.pdf"
