# Bundled skill — mirror, not source

This directory is a **mirror** of the canonical skill at [`../../../../skills/vemem/`](../../../../skills/vemem/) so the openclaw plugin can ship its companion skill in-tree (per the [openclaw "Plugins + skills"](https://docs.openclaw.ai/tools/skills#plugins-skills) pattern).

**Do not edit files here directly.** Edit [`skills/vemem/`](../../../../skills/vemem/) at the repo root, then run:

```bash
./scripts/sync-bundled-skill.sh
```

That script copies the canonical skill into this directory. A future CI check will enforce bit-for-bit equality.

## Why a mirror instead of a symlink

When a user copies `integrations/openclaw/plugin/` into `~/.openclaw/extensions/vemem-bridge/`, a symlink would point outside the copied tree and break. A plain directory copy survives the install.
