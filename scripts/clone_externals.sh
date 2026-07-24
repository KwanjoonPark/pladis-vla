#!/bin/bash
# Clone (or verify) the external checkouts pinned in scripts/externals.lock.
#
#   bash scripts/clone_externals.sh          # clone missing, verify existing
#   bash scripts/clone_externals.sh --check  # verify only, never clone
#
# Existing checkouts are never modified: a SHA mismatch is reported (exit 1)
# and left for the operator to resolve — this script must not silently move
# a checkout another experiment may be running from.
set -u
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/experiments/load_machine_env.sh"
LOCK="$REPO/scripts/externals.lock"
CHECK_ONLY="${1:-}"
fail=0

while IFS=$'\t' read -r name dest url sha; do
  case "$name" in ''|\#*) continue ;; esac
  dest_expanded="$(eval echo "$dest")"
  if [ ! -d "$dest_expanded/.git" ]; then
    if [ "$CHECK_ONLY" = "--check" ]; then
      echo "[externals] MISSING $name at $dest_expanded"; fail=1; continue
    fi
    echo "[externals] cloning $name -> $dest_expanded"
    mkdir -p "$(dirname "$dest_expanded")"
    git clone "$url" "$dest_expanded" || { fail=1; continue; }
    git -C "$dest_expanded" checkout --detach "$sha" || fail=1
  fi
  head="$(git -C "$dest_expanded" rev-parse HEAD 2>/dev/null || echo none)"
  if [ "$head" != "$sha" ]; then
    echo "[externals] SHA MISMATCH $name: HEAD=$head expected=$sha ($dest_expanded)"
    fail=1
  else
    dirty="$(git -C "$dest_expanded" status --short 2>/dev/null | grep -cv '^??' || true)"
    if [ "$dirty" -gt 0 ]; then
      echo "[externals] WARN $name: $dirty tracked modification(s) on top of $sha"
    else
      echo "[externals] OK $name @ ${sha:0:9}"
    fi
  fi
done < "$LOCK"

exit "$fail"
