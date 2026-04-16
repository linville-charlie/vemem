# `vm` CLI walkthrough

A short end-to-end tour of the command-line interface. Every command honors
`$VEMEM_HOME` (default `~/.vemem`) for the store path and `--actor` for
attribution; reads support `--format json` for scripting.

## Install

```bash
uv sync                 # installs the package + the `vm` console script
vm --help               # should print the command list
```

## Observe an image

Run the InsightFace detector + ArcFace encoder on a local image and persist
one observation per detected face.

```bash
vm observe ~/Pictures/charlie-meetup.jpg
# obs_a1b2c3d4...
```

The output line is the observation id — copy it into the next command.

## Label that observation as a new entity

```bash
vm label obs_a1b2c3d4... --name Charlie
# ent_7f8a9b0c...
# labeled 1 obs → Charlie
```

The first `label` of an unknown name creates the entity; subsequent labels of
that name re-attach to the same id.

## Attach a fact

```bash
vm remember Charlie --fact "runs a cafe in Austin"
# remembered fact fact_... on Charlie
```

## Identify a new photo

```bash
vm identify ~/Pictures/suspect.jpg
#                   Candidates
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Entity ID   Name      Kind      Confidence  Method   ┃
# ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
# ┃ ent_7f...   Charlie   instance       0.913  user_... ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

Use `--format json` for scripting:

```bash
vm identify ~/Pictures/suspect.jpg --format json | jq '.detections[0].candidates[0]'
```

## Recall what we know

```bash
vm recall Charlie
# Charlie
#                   Facts
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Fact ID   Content                            Source ┃
# ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
# ┃ fact_...  runs a cafe in Austin              user   ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

## Inspect the full state

```bash
vm inspect Charlie
```

Shows: id, kind, modality, status, aliases, created/last-seen, current
binding count, active facts, and the most recent events from the event log.

## Forget — irreversible hard delete

```bash
vm forget Charlie
# DANGER: forget deletes all observations, bindings, facts, events, and
# relationships for Charlie. This is NOT reversible by `vm undo`.
# Are you sure? [y/N]: y
# forgotten ent_7f8a9b0c... — counts: {'observations': 1, ...}
```

Pass `--yes` / `-y` to skip the prompt in scripts. The confirmation is
deliberately unskippable by default because forget is **not reversible** by
`vm undo` — it prunes LanceDB version history (GDPR Art. 17) so the
biometric vectors are gone from disk.

## Undo

All ops except `forget` are reversible for 30 days by the same actor:

```bash
vm remember Charlie --fact "likes pour-over"
vm undo
# undone event 12 (new event 13)
```

## Corrections

```bash
vm merge ent_abc ent_def --yes       # "these are the same"
vm relabel obs_xyz --name Dana       # "that's actually Dana, not Charlie"
vm restrict Charlie                  # GDPR Art. 18 — stop using for inference
vm unrestrict Charlie
```

## Export for portability

```bash
vm export Charlie -o charlie.json               # GDPR Art. 20
vm export Charlie --include-embeddings -o ...   # full vectors (rarely wanted)
```

## Tooling helpers

```bash
vm list                             # active instance entities
vm list --kind type --status all    # include tombstones
vm export-tools -o tools.json       # OpenAI function-calling schemas
vm serve-mcp                        # launch the MCP server
```
