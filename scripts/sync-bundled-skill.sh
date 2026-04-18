#!/usr/bin/env bash
# Mirror the canonical skills/vemem/ into integrations/openclaw/plugin/skills/vemem/
# so the openclaw plugin bundles its companion skill (see openclaw docs:
# "Plugins + skills" — plugins ship skills by listing `skills` dirs in
# openclaw.plugin.json, paths relative to the plugin root).
#
# Run this after any change to skills/vemem/ before committing. A future CI
# check could enforce bit-for-bit equality.

set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
src="$here/skills/vemem"
dst="$here/integrations/openclaw/plugin/skills/vemem"

if [ ! -d "$src" ]; then
  echo "error: canonical skill source not found at $src" >&2
  exit 1
fi

rm -rf "$dst"
mkdir -p "$(dirname "$dst")"
cp -r "$src" "$dst"
echo "synced $src -> $dst"
