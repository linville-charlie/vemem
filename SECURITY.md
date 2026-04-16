# Security policy

## Scope

vemem is a self-hosted Python library. The "security surface" is:

1. **The library code itself.** Issues that let a malicious caller escape the Store / Encoder Protocol contracts, read or corrupt data they shouldn't, or cause the MCP server to misbehave for a well-formed client.
2. **Dependencies** (`lancedb`, `insightface`, `open_clip_torch`, `pyarrow`, `mcp`, etc.) — we update pinned versions when upstream advisories are published, on a best-effort basis for a personal OSS project.
3. **Documentation that could mislead a deployer into a non-compliant configuration.** The COMPLIANCE.md guidance is part of this — errors or dangerous defaults there count.

Out of scope:
- How downstream applications collect consent, store images, or build user-facing surfaces on top of vemem.
- InsightFace / CLIP model weight provenance. The library does not ship weights; you install them.
- Local-only denial of service (fill your own disk with embeddings).

## Supported versions

vemem is pre-alpha (v0.x). Only the latest `main` receives fixes. No backport commitment for v0.x.

## Reporting a vulnerability

Prefer **GitHub's private security advisory** channel for the [linville-charlie/vemem](https://github.com/linville-charlie/vemem) repository. That keeps the thread encrypted and lets us coordinate a fix before public disclosure.

If you can't use GitHub advisories, email the author listed in `pyproject.toml` with subject `SECURITY`. Expect a response within two weeks — this is a personal project, not a 24/7 security team.

## What to include

- A minimal reproduction (a test case is ideal)
- The version / commit you're reporting against
- Expected vs. observed behavior
- Any assessment of impact you already have

## What we'll do

1. Acknowledge receipt within two weeks
2. Investigate and reproduce
3. Agree on a fix + disclosure timeline with you
4. Ship the fix + credit you in the release notes (unless you'd rather stay anonymous)

## Defensive-disclosure notes for deployers

- LanceDB 0.19 segfaults on CPython 3.14; the project pins local dev to 3.13 via `.python-version`. If you upgrade CPython without checking, you may see a hard crash at `LanceDBStore()` construction — the library's error surface cannot catch segfaults.
- The MCP server exposes `observe_image` — a client that can talk to your stdio server can insert biometric data into your store. Only expose the MCP server to clients you trust.
- `forget(entity_id)` physically removes data from version history via `optimize(cleanup_older_than=0)`. A caller that can run `forget` on an attacker-chosen `entity_id` can cause data loss. Gate this operation behind authentication at the MCP / CLI layer.
- `restrict` → `unrestrict` is reversible and trivial; do not rely on it as an erasure path.
- The event log persists across store reopens. Rotating your LanceDB store wipes the log — do not do this casually; use `prune_events(older_than=...)` instead.
