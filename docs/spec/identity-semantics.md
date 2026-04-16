# Identity Semantics Specification (T1)

Status: **draft v0.1** — 2026-04-16
Scope: the `label / relabel / merge / split / forget / undo` operations, plus the type-vs-instance model. Load-bearing for every other part of the library. Written before any core code.

---

## 1. Purpose

Real use produces identity corrections: *"that's not Charlie, that's Dana"*; *"unknown_7 and unknown_12 are the same person, merge them"*; *"Charlie after a haircut no longer matches — split the cluster"*; *"forget Charlie entirely."* If the library has no coherent answer for these, attached facts rot on the first correction and the memory becomes untrustworthy. This spec is the answer.

## 2. Core principle — evidence vs. interpretation

An **observation** is immutable evidence: a detected region in a specific image at a specific time, embedded by a specific encoder. Observations never change.

An **entity** is a mutable interpretation: "these observations all refer to the same real-world thing." Corrections are changes to interpretation, not to evidence.

A **binding** connects observations to entities. Bindings are append-only; corrections add new bindings that supersede old ones. The history of bindings is the audit trail.

This split makes every correction operation tractable: we never rewrite evidence, we only change which evidence is currently believed to belong to which interpretation.

## 3. Data model

### 3.1 Observation (immutable)

| field | type | notes |
|---|---|---|
| `id` | uuid | content-hash of `(source_hash, bbox, detector_id)` — deterministic |
| `source_uri` | str | image URI supplied by caller (may be a path, URL, or opaque ID — library does not fetch it at recall time) |
| `source_hash` | str | SHA-256 of the image bytes at observation time — the durable identifier. If the file at `source_uri` changes, the hash doesn't. |
| `bbox` | `[x,y,w,h]` | detection region in pixel space (whole-image observations use the full frame) |
| `detector_id` | str | e.g. `insightface/buffalo_l@0.7.3` — version is part of the ID so a detector upgrade is a new detector |
| `modality` | enum | `face \| object \| scene \| audio` (face only in v0) |
| `detected_at` | timestamp | |
| `source_ts` | timestamp? | for video/stream frames that carry their own time |
| `source_frame` | int? | frame index in a video, for deduplication hints |

**Observations are append-only and idempotent.** Re-observing the same `(image_bytes, bbox, detector)` returns the existing `id`. Embeddings are NOT stored on the Observation row — see §3.1a.

### 3.1a Embedding (append-only, one per encoder per observation)

| field | type | notes |
|---|---|---|
| `id` | uuid | |
| `observation_id` | uuid | foreign key |
| `encoder_id` | str | e.g. `insightface/arcface@0.7.3` — **encoder version is part of the ID**; an encoder upgrade produces new rows, not overwrites |
| `vector` | fixed-length float array | L2-normalized at write time for cosine-compatible search |
| `dim` | int | redundant with encoder_id but denormalized for filtered search |
| `created_at` | timestamp | |
| `key_id` | str? | if per-entity crypto-erase is enabled (§4.5) |

Splitting Observation from Embedding enforces the invariant: the observation is the thing we saw; embeddings are how various encoders described it. Adding support for a new encoder later means appending new Embedding rows against existing observations — no Observation mutation.

Multiple embeddings per `(observation, encoder)` are allowed for ensemble encoders, but the normal case is one.

### 3.1b Source image policy

The library does NOT store image bytes. It stores the `source_uri` (caller's reference) and `source_hash` (content identity). If the caller deletes the image, observations remain but point to a dead URI — `recall()` returns the observations as orphaned and flags them in a warning. Consumers that need image bytes must manage storage separately.

**Rationale:** image bytes are often large, frequently duplicative (many observations per image), and the heaviest regulatory burden when biometric. Pushing storage responsibility to the caller keeps the library small and lets users plug in Blob/S3/local-dir without the library becoming a photo manager.

**Exception for `forget()`:** the library explicitly can NOT delete the source image (it never stored it). The library documents clearly that the user is responsible for deleting source images when forgetting an entity.

### 3.2 Entity (mutable metadata, stable id)

| field | type | notes |
|---|---|---|
| `id` | uuid | stable across renames and merges |
| `kind` | enum | `instance` (default) \| `type` — see §5 |
| `name` | str | canonical display name; not identity |
| `aliases` | `list[str]` | alternate names ("Charlie", "Charles Smith", "that guy from ops"). Searched by `recall(query=name)`. |
| `modality` | enum | `face \| object \| scene \| audio`. The modality of this entity's bindings. An entity is single-modality in v0; cross-modal binding (face↔voice) is a v0.1 feature via linked entities. |
| `status` | enum | `active` \| `merged_into` \| `forgotten` |
| `merged_into_id` | uuid? | populated when `status = merged_into` |
| `created_at` / `last_seen` | timestamp | |

`status=merged_into` preserves audit trail. `status=forgotten` is an opaque tombstone; see §4.5 and §7.

**Modality is enforced at binding-time.** You cannot bind a face observation to an object entity. `merge()` across modalities is rejected (§4.3).

### 3.3 Binding (append-only, bi-temporal)

| field | type | notes |
|---|---|---|
| `id` | uuid | |
| `observation_id` | uuid | |
| `entity_id` | uuid | |
| `polarity` | enum | `positive` ("obs IS entity") \| `negative` ("obs is NOT entity") |
| `confidence` | float | 1.0 for user labels; encoder similarity for auto |
| `method` | enum | `user_label` \| `auto_suggest` \| `llm_assist` \| `migrated` \| `user_reject` |
| `valid_from` | timestamp | when this binding became the current answer |
| `valid_to` | timestamp? | `null` if still current; set when superseded |
| `recorded_at` | timestamp | when the row was written (for audit) |
| `actor` | str | who made the decision (user id, agent name) |

Bi-temporal shape borrowed from Zep / Graphiti. `valid_from` / `valid_to` is the library's belief timeline; `recorded_at` is the system timeline. A query "what did we think about this observation at time T?" is `WHERE valid_from <= T AND (valid_to IS NULL OR valid_to > T)`. This gives us correction, undo, and audit in one mechanism — no separate `valid:bool` + `superseded_by` pointer chain.

**Negative bindings** (`polarity=negative`) capture "this observation is NOT this entity" — the durable correction signal that Apple Photos and Google Photos use internally ("This is not X" in the UI). They:
- do not claim identity; they forbid it
- never expire on their own
- are what `split()` leaves behind, preventing the auto-clusterer from re-merging
- are what `relabel` emits against the old entity in addition to a new `positive` binding

The **current positive binding** for an observation is the one row with `polarity=positive` and `valid_to IS NULL`. At most one per observation.

### 3.4 Fact (bi-temporal statements about an entity)

| field | type | notes |
|---|---|---|
| `id` | uuid | |
| `entity_id` | uuid | the entity this fact describes |
| `content` | text | natural language. Schema does NOT force structure ("Charlie runs marathons" is stored as-is). Callers that want structure wrap in JSON; the library treats `content` as opaque text. |
| `source` | enum | `user \| vlm \| llm \| import` |
| `actor` | str | who recorded it |
| `valid_from` | timestamp | when this fact became believed |
| `valid_to` | timestamp? | when retracted; `null` if still believed |
| `recorded_at` | timestamp | system timeline |
| `provenance_entity_id` | uuid? | for merged/split entities — the entity this fact was originally attached to. Preserved across merges so conflict-reconciliation can identify source. |

Facts use the same bi-temporal shape as Bindings (§3.3) for consistency. Retraction = set `valid_to=now`. A fact retracted at T is still true for queries "what did we believe at T-1?"

### 3.5 Event (timestamped things that happened)

| field | type | notes |
|---|---|---|
| `id` | uuid | |
| `entity_id` | uuid | the entity this event is about (or the primary participant) |
| `content` | text | |
| `source` | enum | `user \| vlm \| llm \| import` |
| `occurred_at` | timestamp | when the event took place in the world |
| `recorded_at` | timestamp | when we wrote the row |
| `provenance_entity_id` | uuid? | as in §3.4 |

Events are immutable once written. A "correction" to an event (wrong time, wrong entity) is a `retract` + `insert new event`, not an update. Recall by entity, by time range, or both.

### 3.6 Relationship (directed edge between entities)

| field | type | notes |
|---|---|---|
| `id` | uuid | |
| `from_entity_id` | uuid | subject |
| `to_entity_id` | uuid | object |
| `relation_type` | str | free-form; `instance_of` / `part_of` / `located_at` / `cofounder_of` / etc. |
| `source` | enum | `user \| vlm \| llm \| import` |
| `valid_from` / `valid_to` | timestamps | bi-temporal, same shape as Facts |
| `recorded_at` | timestamp | |
| `provenance_from_id` / `provenance_to_id` | uuid? | each end's provenance across merges |

Relationships are directed. An inverse relationship is a separate row (the library does not auto-mirror). This matches RDF convention and avoids the "half a row vs whole row" confusion when one side merges or is forgotten.

**Self-loops** (e.g. after merge: "Charlie cofounder_of Charlie" because both ends of the edge merged into Charlie) are detected and collapsed — the row is retracted (`valid_to=now`) with a note in the EventLog.

`instance_of` is a dedicated, enforced relation — used for the type-vs-instance model (§5).

### 3.7 EventLog (append-only audit + undo stack)

| field | type | notes |
|---|---|---|
| `id` | monotonic int | |
| `op` | enum | `label \| relabel \| merge \| split \| forget \| remember \| retract_fact \| auto_suggest_commit \| undo` |
| `payload` | json | operation-specific (see §4); size-capped at 1MB per row |
| `actor` | str | structured `"kind:id"` — e.g. `"user:alice"`, `"mcp:claude"`, `"cli:local"`, `"agent:my_assistant"` |
| `affected_entity_ids` | `list[uuid]` | denormalized for the "show me everything that touched Charlie" query. Indexed. |
| `at` | timestamp | |
| `reversible_until` | timestamp? | default `at + 30 days`; `null` means non-reversible (e.g. `forget`) |
| `reversed_by` | event_id? | if this event has been undone |

**EventLog never stores embeddings or image bytes.** Only IDs and counts. This keeps the log safe to retain past observation-prune for audit purposes without re-creating biometric liability. (GDPR Art. 17 — erasure must cover biometric data itself, not metadata about the fact that a deletion happened.)

**Retention.** EventLog rows older than `reversible_until` are no longer functionally needed for undo. Operators may run `prune_events(older_than=...)` to trim the log; the default retention is 1 year. Rows for `forget` operations are kept longer (5 years default) as the durable audit of erasure — they already contain no biometric content.

## 4. Operations

### 4.0 `identify(image_or_embedding, k=5, min_confidence=0.5, prefer="instance")` → `list[Candidate]`

The read-path bridge. Does not mutate state.

- Accepts either raw `image` (library runs detector + encoder) or a precomputed `embedding` (caller already ran the encoder).
- Queries the current positive bindings' embeddings via ANN (LanceDB vector search) and returns top-k candidate entities, each with `confidence`, `matched_observation_ids`, and `binding.method`.
- Respects `negative` bindings: an entity with a negative binding against the query observation is excluded from results, regardless of embedding similarity.
- Applies per-encoder, per-modality thresholds (see `ThresholdConfig`). Defaults are conservative; `identify` returns fewer false positives than raw similarity would suggest.
- `prefer="instance"` biases the ranking toward `instance` entities when both instance and type match (e.g. "my red mug" beats "red mugs" when both are above threshold). `prefer="type"` inverts. `prefer="both"` returns both, ranked by raw score.

**Cross-encoder behavior:** if the query's encoder has no bindings in the gallery (e.g. query is CLIP, gallery is all InsightFace), `identify` returns an empty list and a `no_compatible_encoder` warning. It does NOT silently re-embed or fall back. This preserves the invariant from §3.1a that encoder version is part of identity-of-evidence.

### 4.1 `label(observation_ids, entity_name_or_id)` → `entity_id`

- If `entity_name` is new, create `Entity(kind=instance, name=…, modality=<derived from observations>)`.
- Create a positive `Binding` per observation: `method=user_label`, `confidence=1.0`, `valid_from=now`.
- Close any prior current positive binding on those observations by setting its `valid_to=now`.
- Emit `EventLog{op=label, payload={observation_ids, entity_id, prior_binding_ids}}`.

**Ambiguity rules:**
- If the observations span multiple modalities, reject with `ModalityMismatch`.
- If some observations are already bound to a different entity (user previously labeled them as something else), `label()` is equivalent to `relabel()` for those observations — the library does NOT silently merge the two entities. Emit both positive (to new target) and negative (against old entity) bindings.
- If `entity_name_or_id` refers to a `forgotten` or `merged_into` entity, reject with `EntityUnavailable`.

Reversible by recreating prior bindings from the payload.

### 4.2 `relabel(observation_id, new_entity_name_or_id)` → `binding_id`

Sugar for `label` of a single observation onto a different entity, PLUS a `negative` binding emitted against the old entity so the auto-clusterer never re-attaches the same observation.

**UX rule borrowed from Apple Photos:** identity is the entity `id`, not the entity `name`. Renaming entity A to a name previously held by entity B does NOT merge them. Library never merges by name — only by explicit `merge()`.

### 4.3 `merge(entity_ids, keep=entity_id or "oldest")` → `entity_id`

"These are the same."

- Pick a `winner` (kept) and `losers`. Default `keep="oldest"`.
- **Modality check:** reject with `ModalityMismatch` if not all entities share the same `modality`.
- **Kind check:** reject with `KindMismatch` if mixing `instance` and `type`. The library suggests using `instance_of` instead.
- For each current positive binding on a loser: close it (`valid_to=now`) and open an equivalent binding on the winner (`method=migrated`, same observation, `valid_from=now`).
- **Negative bindings on losers are dropped**, not migrated. They asserted "obs X is not loser Y" — after merge that would become "obs X is not winner W," but X has no negative relationship to W's *other* observations, so the assertion doesn't transfer cleanly. Closing them on the loser (valid_to=now) preserves history without making false claims.
- Move all `Fact / Event / Relationship` rows: update `entity_id → winner_id`, set `provenance_entity_id = loser_id` to preserve origin.
- **Self-loop collapse on relationships:** any relationship whose `from` and `to` both map to the winner after merge is retracted (valid_to=now) and logged. Example: "Charlie cofounder_of AnotherCharlie" becomes a self-loop if we merge the two Charlies — retract.
- Mark each loser `status=merged_into, merged_into_id=winner_id`. Do not delete the entity row.
- Emit `EventLog{op=merge, payload={winner_id, loser_ids, closed_binding_ids, opened_binding_ids, dropped_negative_ids, collapsed_relationship_ids, moved_fact_ids}}`.

Fact conflicts (e.g. winner says "lives in Austin," loser says "lives in Boston") are NOT auto-resolved. Both facts become active under the winner with different `provenance_entity_id`. The library exposes `recall(entity).conflicts()` so an LLM or user can reconcile.

### 4.4 `split(entity_id, groups: list[list[observation_id]])` → `list[entity_id]`

"This is actually N different entities."

- `groups[0]` stays on the original `entity_id`. Each subsequent group becomes a new `Entity` with the same `modality` and `kind` as the original.
- **Ungrouped observations** (positive bindings to the original entity that aren't listed in any `group`) stay bound to the original. Split is not required to cover every observation — a partial split is legal.
- Move bindings accordingly: close old positive bindings (`valid_to=now`) and open new ones on the correct entity. **Emit `negative` bindings cross-wise** — each group's observations get `negative` bindings against every *other* split group's entity. This is the durable signal preventing the auto-clusterer from silently re-merging what the user just split.
- **Pre-existing negative bindings on the original** stay on the original (they asserted a relationship to "the pre-split entity" which is still represented by the original's `id`).
- **Facts, events, relationships DO NOT auto-migrate** on split. They remain on the original entity by default. The caller may pass `fact_policy={"copy_to_all" | "keep_original" | "manual"}` — default `keep_original` because we cannot know which split owns the statement.
- Emit `EventLog{op=split, payload={original_id, group_entity_ids, group_observation_ids, fact_policy}}`.

### 4.5 `forget(entity_id, grace_days=0)` — privacy-critical

Default (`grace_days=0`): hard delete.

1. For each Observation currently bound **only** to this entity (no other positive bindings to different entities): delete the Observation and all its Embedding rows. For Observations with positive bindings to OTHER entities (e.g. bound to both `my_red_mug` and `red_mugs`), keep the Observation and Embeddings — only delete THIS entity's bindings to them. Multi-bound observations survive; single-bound observations are purged.
2. Delete all `Binding` rows (positive AND negative) with `entity_id == this`.
3. Delete all `Fact / Event / Relationship` rows with `entity_id == this` (or either end for Relationships). Relationships with one end on this entity are deleted entirely — the edge has no meaning without both nodes.
4. For `type` entities with `instance_of` inbound edges: forgetting a type does NOT cascade to the instances. The instances lose the `instance_of` edge but keep their own data. The library warns if orphan instances are created.
5. Mark `Entity.status = forgotten` and null out `name` and `aliases`. Keep the row as an opaque tombstone for audit consistency (opaque = no name, no embedding-derivable content, no link back to biometric artifacts beyond "an entity with this ID was forgotten at this time").
6. Call `lancedb.optimize(Prune{older_than: 0})` on affected tables — **this is the step that physically removes the biometric data from older table versions**. Without it, LanceDB's time-travel retains the vectors for the default 7 days. GDPR Art. 17 does not accept "eventually" as compliance.
7. Emit `EventLog{op=forget, payload={entity_id, counts_only}, reversible_until=null}`. Payload contains counts for audit, never the erased content.

**Alternative erasure strategy: crypto-erase.** For deployments where step 5 is infeasible (distributed caches, cold backups, remote object storage), the library supports per-entity encryption keys on embeddings. `forget()` then destroys the key — a single O(1) operation renders all ciphertext unreadable, across every location the data lives. ISO/IEC has formalized this pattern as Renewable Biometric References. Not v0 scope, but the embedding column shape must not preclude it: store `(embedding_bytes, key_id)` so we can add key-destruction later without a migration.

`grace_days>0` keeps a recoverable tombstone for the grace period (biometric data retained but soft-deleted). After the grace period, an automatic sweep performs the hard-delete + prune. Grace mode is opt-in and MUST be disabled for strict-erasure jurisdictions (GDPR Art. 9, Illinois BIPA) without explicit user consent to delayed delete. Industry consensus on retention window: 14–30 days.

**`forget` is not reversible via `undo()`.** The EventLog records that a forget happened, but has no way to reconstruct the purged biometric data. If grace mode is on, the user can `restore(entity_id)` within the grace window — a separate, explicit operation.

**Backup policy.** Regulators accept "beyond use" backups that age out on normal rotation, not a restore-and-patch path. The library does not provide a `restore_from_backup(entity_id)` operation for a forgotten entity. Operators must NOT build one on top of us. Documented loudly in the README.

### 4.6 `undo(event_id=None)` — time-limited reversal

- If `event_id` is `None`: undo the most recent reversible event by this actor.
- Reject if `event.reversible_until` is null or in the past.
- Apply the inverse using the payload. Under the bi-temporal model this is concrete:
  - `label` undo: re-open the previously-closed bindings (clear their `valid_to`) and close the new ones (`valid_to=now`).
  - `merge` undo: close the `migrated` bindings on the winner, re-open the corresponding bindings on each loser, move facts back using `provenance_entity_id`, flip loser status `merged_into → active`.
  - `split` undo: invert the split — close the new-entity bindings, re-open the original's closed bindings, clear the cross-wise negatives, delete the split-off entity rows (or tombstone if any facts were attached).
  - Mark the original event's `reversed_by` to this new event.
- Emit a new `EventLog{op=undo, payload={undone_event_id}}`.

Undo of undo is allowed (redo). Undo chains deeper than one step must be explicit — we do not silently cascade.

### 4.7 `restrict(entity_id)` / `unrestrict(entity_id)` — GDPR Art. 18

"Stop using this entity for inference, but don't delete it yet." Used during dispute resolution ("is this really me?") or while the user decides whether to `forget`. Strictly less destructive than `forget`.

- Entity is excluded from `identify()` results (treated as if all its positive bindings are inactive).
- No new bindings can be opened against it.
- Facts/events remain readable via `recall()` for the owner; identification writes are blocked.
- `unrestrict()` reverses, re-activating the entity.
- Emits `EventLog{op=restrict | unrestrict}`.

### 4.8 `export(entity_id or "all") → json` — GDPR Art. 20

Data portability. Returns a structured dump of everything the library holds for the subject: observations metadata (not raw images — caller must export those separately), binding history, facts, events, relationships, event log entries. Embedding vectors are included by flag (`include_embeddings=False` by default, since raw biometric vectors in a user-facing export are often worse than useless).

The export format is versioned JSON. Round-tripping via `import_subject(export_json)` restores the records on another instance — useful for migrations and testing, and is the only legitimate way to "restore" a forgotten entity (the user who was forgotten must re-supply their data; the operator must not re-materialize it from backup).

### 4.9 Auto-suggest (deferred to v0.1, stubbed in v0)

When unlabeled observations are seen, the library computes a suggested `entity_id` via ANN + threshold + optional voting (cf. M3-Agent's accumulated-weight pattern). **Suggestions never commit bindings automatically in v0.** They surface via `identify()` with `confidence < 1.0` and a `method=auto_suggest` hint, and must be committed by `label()`. This matches Apple Photos' "Review People" pattern and keeps the hot path deterministic (design principle #2 in the vision doc).

v0.1 may introduce `auto_commit_above: float` as an opt-in, per-entity-type threshold.

## 5. Type vs. instance

`Entity.kind ∈ {instance, type}`. Default `instance`.

- **instance**: a specific physical thing — "my red mug," "Charlie (my coworker)."
- **type**: a class — "red mugs," "Golden Retrievers."

Relationship model borrowed from Wikidata / RDF: an instance may have an `instance_of` relationship to a type (an enforced relation type, §3.6). Observations may bind to either — a coffee-shop photo can carry both `binding(obs, my_red_mug)` and `binding(obs, red_mugs)`. Bindings to types carry lower confidence and do not block instance-level bindings.

**Disjointness constraint.** An entity cannot be simultaneously `instance_of` and a supertype/subclass of the same type node. This is the "class/instance punning" failure mode documented in Wikidata quality studies — once allowed, retrieval fuses class-level and instance-level facts and is not recoverable by any later correction. Reject at the API.

**Design the instance/class split at ingest, not post-hoc.** Embedding similarity alone will collapse "my red mug" and "red mugs" into one cluster, after which no correction op can untangle which facts belong where. `observe()` takes an optional `intended_kind` hint for callers that already know.

**Identify-time policy.** When an observation could bind to both a matching instance and a matching type (e.g. the photo is of my red mug, and it's also a red mug), `identify()` returns BOTH by default, ranked by `prefer` param. Callers who want only one kind filter with `kind="instance"` or `kind="type"`.

**v0 scope:** `kind` field exists; only `instance` is exercised end-to-end. `type` entities can be created but type-level auto-clustering and type prototypes are v0.1. Do not design v0 features around types — but do not paint us into a corner either.

## 6. Concurrency

Personal project, single-writer default. Don't over-engineer:

- Each operation is one LanceDB transaction when possible; `merge_insert` with `when_matched_update_all` for binding supersede.
- Cross-table ops (`merge`, `split`, `forget`) execute in a fixed order (bindings → facts → relationships → entity → prune) and write the EventLog entry LAST, after all data changes have committed. If the process dies mid-op, the EventLog will NOT record it and the partial changes are visible to reads. A `repair()` CLI command scans for inconsistent states (e.g. an entity with no current positive bindings that isn't marked `merged_into` or `forgotten`) and either finishes or rolls back based on recorded payload hints in a sidecar `InFlightOps` table.
- **Reader isolation.** LanceDB reads see the latest committed version of each table, not a globally consistent snapshot. During a merge in flight, a reader could see the bindings moved but facts not yet moved. Acceptable for a personal-project single-writer model; callers needing a consistent snapshot use `store.snapshot()` which pins a version across all tables.
- **Multi-writer (v0.1).** Once MCP clients and the local library may write concurrently: optimistic concurrency via `Entity.version_stamp` bumped on each write. `merge`, `split`, `forget` take a version stamp and retry once on conflict. `label` / `relabel` / `remember` are last-write-wins (bindings are append-only so there's nothing to lose).

## 7. GDPR / biometric hygiene (non-negotiable rules)

| Rule | Why |
|---|---|
| `forget()` prunes LanceDB old versions | Without prune, biometric vectors live in version history for 7 days (LanceDB default). GDPR Art. 17 requires actual erasure. |
| EventLog never stores embeddings or image bytes | Audit records must not themselves be a regulated biometric artifact. |
| Library ships with `consent_required=True` as default config | For face modality specifically. Attempts to `observe(modality=face)` without an entity carrying `consent_source` metadata emit a warning and can be set to raise. |
| `forget` is irreversible by `undo` | Reversible forgets would defeat erasure. Grace mode is a separate, opt-in feature, not an undo path. |
| Restore-from-backup is the operator's responsibility | Document clearly: if the user restores a backup older than the forget, they restore the biometric data with it. The library cannot police this. |
| `restrict()` for Art. 18 | Dispute-resolution path that stops inference without deletion. |
| `export()` for Art. 20 | Data portability; includes embeddings only by flag. |
| Jurisdiction config | `compliance_profile={"gdpr", "bipa", "ccpa", "none"}` config toggle adjusts defaults: BIPA disables `grace_days>0`, GDPR requires `consent_required=True`, CCPA enables a `do_not_sell` tombstone field. Defaults to "none" for personal-project use; OSS operators pick their profile. |
| EventLog retention | Normal events: 1 year default; forget events: 5 years. Operator-configurable. Compaction via `prune_events(older_than)`. |

## 8. Open questions (park for post-spike)

1. **Voting-vs-threshold for auto-suggest** (M3-Agent uses voting weights; classical face re-ID uses threshold). Voting needs a window and an update rule — defer to v0.1 after empirical data from personal use. **Constraint:** voting may *propose* but authoritative `label()` always wins; one user vote must outweigh 50 auto-voting frames, which argues against a pure-weighted-majority scheme and for a hybrid "authoritative > consensus" rule.
2. **Cross-encoder bindings.** If `identify(image, encoder=CLIP)` matches entity X but all of X's observations are InsightFace embeddings, do we add a CLIP embedding to X's existing observations, or require explicit `reconcile_encoder(entity, encoder)`? Decision depends on the T2 spike.
3. **Co-occurrence binding groups.** M3-Agent binds face+voice co-occurrences into identity cliques (durable signal: the *pair* is what's stable, not the individual modality). v0 is face-only so this is deferred, but the Observation schema should accommodate a future `co_observation_group_id` without a migration.
4. **Fact conflict resolution UI.** `recall().conflicts()` surfaces conflicts; library does not pick. Is there a library-level heuristic we should offer (e.g. "latest wins, with flag"), or leave it entirely to the caller LLM?
5. **`merge` on instance + type**. Meaningless (they're different kinds). Library rejects with a helpful error suggesting `instance_of` instead (§4.3).
6. **Confidence aggregation across bindings.** When an entity has 50 high-confidence bindings and one low-confidence outlier, does `identify()` return max, mean, or top-k-mean? Suggest max by default, configurable.
7. **Don't bury identity semantics in query filters** (Mem0 anti-pattern: `AND`-across-entity-types silently returns empty sets). All identity operations are explicit verbs; filters are only for retrieval.
8. **Source image lifecycle.** Library stores hash, not bytes. But what about caller-provided crops stored alongside (e.g. thumbnails for UI)? Probably out of scope for the core library; a sidecar `BlobStore` adapter handles it.
9. **Entity deduplication on ingest.** If the user calls `label(obs, "Charlie")` twice with different observations and no existing "Charlie" entity, we create two Charlies. Intentional (names aren't identity, §4.2), but the library should warn when creating an entity with a name that already exists as an alias somewhere.
10. **Orphan observations.** Observations whose only bindings have been retracted, or whose entity was forgotten leaving them multi-bound (now single-bound), or whose source image was deleted. `vm orphans` CLI command surfaces them; policy (auto-delete vs keep-for-recluster) is an open question.
11. **Index freshness.** LanceDB's ANN index is periodically rebuilt. `identify()` on a freshly-labeled observation may not find it until the next rebuild. Document the staleness window; consider a `force_refresh=True` flag.
12. **Migration path when the spec itself changes.** Schema evolution (§3) will happen. Policy: semver-tagged schema version stored in `store.meta`, migration functions in `visual_entity_memory.storage.migrations`, `vm migrate` CLI.

## 9. What this spec is NOT

- Not an API spec — method signatures, error hierarchies, and type annotations live in `src/visual_entity_memory/core/` and are derived from this.
- Not a storage spec — the LanceDB table layout is a separate doc, but this spec constrains what must be storable (immutable observations + append-only embeddings/bindings/events, prune-on-forget, bi-temporal fact/relationship rows).
- Not a clustering algorithm spec — that lives in `docs/spec/auto-clustering.md` (v0.1).
- Not an MCP wire-format spec — tool schemas, payload shapes, and transport concerns live in `docs/spec/mcp-tools.md`.
- Not a performance spec — index freshness windows, query latency budgets, and ANN tuning live in `docs/spec/performance.md` if we write one.

## 10. Glossary

- **Observation** — one detection in one image, at one region, by one detector. Immutable.
- **Embedding** — one vector representing an observation, produced by one specific encoder version. Append-only.
- **Entity** — the "thing" (person, object, place, category) that observations are about. Mutable metadata, stable ID.
- **Binding** — the claim "this observation belongs to this entity." Append-only, bi-temporal. Can be positive or negative.
- **Fact / Event / Relationship** — attached knowledge. Facts are statements; Events are timestamped occurrences; Relationships are edges between entities.
- **EventLog** — audit trail and undo stack. Records what operation happened, by whom, when, and references to affected rows. Never stores biometric content.
- **Actor** — the caller: a user, an MCP-connected LLM, a CLI session, an agent. Recorded on every write for attribution.
- **Modality** — `face / object / scene / audio`. An entity has one modality in v0; cross-modal entities are v0.1.
- **Kind** — `instance` (one specific thing) vs `type` (a class). See §5.

---

**References consulted**

*Memory-system prior art (correction APIs):*
- **Zep / Graphiti** — bi-temporal edges with `valid_at`/`invalid_at`/`created_at`/`expired_at`; the non-lossy-correction pattern this spec steals. [arXiv:2501.13956](https://arxiv.org/abs/2501.13956), [graphiti](https://github.com/getzep/graphiti).
- **Mem0** — opaque scope-id model; LLM-mediated reconcile (`ADD/UPDATE/DELETE/NOOP`); no first-class merge/split. Anti-pattern to avoid: silent `AND`-across-entity-types returning empty. [Mem0 entities API](https://docs.mem0.ai/platform/features/entity-scoped-memory).
- **Letta / MemGPT** — memory blocks are shared, not merged; Context Repositories give git-backed memory branching. [Shared memory blocks](https://docs.letta.com/tutorials/shared-memory-blocks/), [Context Repositories](https://www.letta.com/blog/context-repositories).
- **Cognee** — soft/hard delete modes; "memify" post-processing prunes stale nodes. [Cognee delete](https://docs.cognee.ai/api-reference/delete/delete).
- **Supermemory** — upsert-by-customId; bulk delete by ID or container tag; no built-in undo. [Update & delete docs](https://supermemory.ai/docs/update-delete-memories/overview).

*Visual/multimodal prior art:*
- **M3-Agent** (ByteDance, 2025) — entity-centric multimodal graph, voting-weight self-correction, InsightFace `buffalo_l` + HDBSCAN clustering, cross-modal co-occurrence binding (face↔voice). [arXiv:2508.09736](https://arxiv.org/abs/2508.09736), [ByteDance-Seed/m3-agent](https://github.com/ByteDance-Seed/m3-agent).
- **MIRIX** (Jul 2025) — six memory types + multi-agent routing. No explicit correction primitives. [arXiv:2507.07957](https://arxiv.org/abs/2507.07957).
- **WorldMM** (Dec 2025) — multi-scale temporal indexing; retrieval-focused. [arXiv:2512.02425](https://arxiv.org/abs/2512.02425).
- **MemVerse** (Dec 2025) — hierarchical KG + parametric fast-path; "adaptive forgetting" = usage-driven pruning, not user-requested erasure. [arXiv:2512.03627](https://arxiv.org/abs/2512.03627).

*Photo-management UX precedents:*
- **Apple Photos** — user-driven merge by drag, emulated split via "This is not…" per-photo rejection, same-name remerge hazard (workaround: new nickname). [Reset faces thread](https://discussions.apple.com/thread/255471021), [unmerge thread](https://discussions.apple.com/thread/256144651).
- **Google Photos** — irreversible merge (documented as such, a design we explicitly reject). [Face groups](https://support.google.com/photos/answer/6128838).

*Type-vs-instance reference model:*
- **Wikidata** — `instance_of` (P31) vs `subclass_of` (P279); transitive P279 through `P31/P279*`. [P31](https://www.wikidata.org/wiki/Property:P31), [P279](https://www.wikidata.org/wiki/Property:P279).
- **Class/instance disorder** — known Wikidata quality bug, informs the disjointness constraint in §5. [arXiv:2411.15550](https://arxiv.org/abs/2411.15550).

*Privacy / erasure:*
- **GDPR Art. 17** — erasure must be actual deletion; "impossible to erase" is not a valid defense. [gdpr-info.eu/art-17](https://gdpr-info.eu/art-17-gdpr/).
- **Biometric data compliance** — Art. 9 special-category treatment; Renewable Biometric References (ISO/IEC) formalize crypto-erase. [gdprlocal.com biometric guide](https://gdprlocal.com/biometric-data-gdpr-compliance-made-simple/), [2026 erasure guide](https://univik.com/blog/gdpr-data-erasure-compliance/).

*Storage substrate:*
- **LanceDB** — table versioning, `checkout`/`restore`, `optimize(Prune{older_than})`, default 7-day retention. The mechanism that makes `forget` actually forget. [Versioning docs](https://docs.lancedb.com/tables/versioning).
