#!/usr/bin/env bash
# Wait for a sync that is already running, then bring everything else up to date.
#
#   scripts/refresh-after-sync.sh [--every SECONDS] [--timeout SECONDS]
#
# A long first sync of a shared library can take hours, and the indexing that has
# to follow it cannot start while it runs. This waits, then does the rest: a short
# catch-up sync, the CLIP and face pass over whatever is new, and the export that
# puts it in front of the GUI. Meant to be run detached:
#
#   systemd-run --user --unit=photos-refresh -p MemoryMax=2600M \
#       scripts/refresh-after-sync.sh
set -uo pipefail

EVERY=60
TIMEOUT=$((12 * 3600))
while [ $# -gt 0 ]; do
  case "$1" in
    --every) EVERY="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,14p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

PHOTOS="${PHOTOS_BIN:-$HOME/.local/bin/photos}"
[ -x "$PHOTOS" ] || PHOTOS="$HOME/Work/icloud-photo-workspace/.venv/bin/photos"
[ -x "$PHOTOS" ] || { echo "cannot find the photos command" >&2; exit 1; }

running() {
  "$PHOTOS" --json status --offline 2>/dev/null \
    | python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get("sync_running") else 1)'
}

waited=0
while running; do
  if [ "$waited" -ge "$TIMEOUT" ]; then
    echo "the sync is still going after ${TIMEOUT}s; leaving it alone" >&2
    exit 1
  fi
  sleep "$EVERY"
  waited=$((waited + EVERY))
done
echo "the sync finished after ${waited}s of waiting; refreshing"
exec "$PHOTOS" refresh
