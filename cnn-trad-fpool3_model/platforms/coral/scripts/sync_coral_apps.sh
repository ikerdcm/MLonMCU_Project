#!/usr/bin/env bash
# Sync coralmicro/apps changes → platforms/coral/apps for git tracking.
# Run this before committing whenever you change anything in coralmicro/apps/.
#
# Usage: ./scripts/sync_coral_apps.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORAL_DIR="$(dirname "$SCRIPT_DIR")"
SRC="$CORAL_DIR/coralmicro/apps"
DST="$CORAL_DIR/apps"

if [[ ! -d "$SRC" ]]; then
    echo "ERROR: $SRC not found"
    exit 1
fi

echo "=== Syncing coralmicro/apps → apps/ ==="

# Top-level CMakeLists.txt (add_subdirectory entries)
cp "$SRC/CMakeLists.txt" "$DST/CMakeLists.txt"
echo "  CMakeLists.txt"

# Sync each app directory (source files only, no build artifacts)
for APP in kws_coral kws_live kws_bench single_inference; do
    if [[ -d "$SRC/$APP" ]]; then
        mkdir -p "$DST/$APP"
        rsync -a --include="*.cc" --include="*.c" --include="*.h" \
                 --include="CMakeLists.txt" --exclude="*" \
                 "$SRC/$APP/" "$DST/$APP/"
        echo "  $APP/"
    fi
done

echo ""
echo "Done. Changed files in apps/ are now visible in the main repo."
echo "Check with: git status"
