#!/usr/bin/env bash
# Publish the robot work to github.com/oist/dreamerv3 as a clean snapshot.
#
#   tools/publish_oist.sh            # commit HEAD's shareable files onto oist/main
#   DRY_RUN=1 tools/publish_oist.sh  # build the snapshot and list it, push nothing
#
# The OIST repo does not share this branch's history: each publish is one
# commit on oist/main holding only the files below, so personal notes, the
# HRL experiment log, Saion launchers and CLAUDE.md never reach it, not even
# in old commits. Run from a clean checkout of smartphone-robot-bridge.
set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE=${REMOTE:-oist}
URL=https://github.com/oist/dreamerv3.git
KEEP=(dreamerv3 embodied tools/robot docs/robot ROBOT.md README.md LICENSE
      setup.py requirements.txt requirements-robot.txt Dockerfile entrypoint.sh
      .dockerignore plot.py scores baselines.yaml)

[ -z "$(git status --porcelain)" ] || { echo "commit or stash first" >&2; exit 1; }
git remote get-url "$REMOTE" >/dev/null 2>&1 || git remote add "$REMOTE" "$URL"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
tree=$tmp/tree
mkdir "$tree"
git archive HEAD "${KEEP[@]}" | tar -x -C "$tree"

# Personal or internal bits inside otherwise shareable files.
rm -rf "$tree/docs/robot/history"
sed -i.bak 's/^\(    wandb_entity:\).*/\1 '"''"'/' "$tree/dreamerv3/configs.yaml"
sed -i.bak 's/^They are dated records: scripts they name may since have been replaced, see$/They are dated records: scripts they name may since have been replaced./' "$tree/ROBOT.md"
sed -i.bak '/^`docs\/robot\/history\/` for the older guides\.$/d' "$tree/ROBOT.md"
find "$tree" -name '*.bak' -delete
cat > "$tree/.gitignore" <<'EOF'
__pycache__/
*.py[cod]
*.egg-info
.pytest_cache
.venv/
dist
wandb/
MUJOCO_LOG.TXT
tools/robot/robot.env
EOF

if grep -rIl -e vasilache -e /work/DoyaU -e /bucket/DoyaU "$tree"; then
  echo "personal paths left in the files above; fix them first" >&2; exit 1
fi

export GIT_INDEX_FILE=$tmp/index
git --work-tree="$tree" add -A .
snapshot=$(git write-tree)
unset GIT_INDEX_FILE

parent=()
if git fetch -q "$REMOTE" main 2>/dev/null; then
  parent=(-p "$REMOTE/main")
  if [ "$(git rev-parse "$REMOTE/main^{tree}")" = "$snapshot" ]; then
    echo "oist/main already has this snapshot"; exit 0
  fi
fi

if [ -n "${DRY_RUN:-}" ]; then
  git ls-tree -r --name-only "$snapshot" | awk -F/ '{print $1"/"$2}' | sort | uniq -c
  exit 0
fi

source=$(git rev-parse --short HEAD)
commit=$(git commit-tree "$snapshot" "${parent[@]}" -F - <<EOF
Snapshot of smartphone-robot-bridge $source

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)
git push "$REMOTE" "$commit:refs/heads/main"
echo "published $source as $(git rev-parse --short "$commit") on $REMOTE/main"
