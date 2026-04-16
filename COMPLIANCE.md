# Compliance notes for vemem deployers

> **If you deploy vemem, you are the data controller under GDPR, BIPA, and CCPA — not the library author.** This document translates spec §7 into operator action items so your deployment can pass a privacy review.

vemem stores **biometric identifiers** (face embeddings) by design. That brings regulated-data obligations that the library builds primitives for but cannot enforce by itself. Use this document as a checklist when you ship something built on vemem.

---

## Regulatory surface

| Rule | Where it hits you |
|---|---|
| **GDPR Art. 9** — biometric data is special-category; processing requires explicit consent or another Art. 9(2) basis | Any face observation you keep |
| **GDPR Art. 17** — right to erasure must actually remove the data | `vemem.Vemem.forget(entity_id)` + version prune (see below) |
| **GDPR Art. 18** — right to restrict processing without deletion | `vemem.Vemem.restrict(entity_id)` |
| **GDPR Art. 20** — data portability | `vemem.Vemem.export(entity_id, include_embeddings=True)` |
| **BIPA (Illinois)** — per-violation statutory damages, no de minimis threshold | Deploying to any Illinois user requires their written consent + retention policy |
| **CCPA / CPRA** — right to know, delete, correct, opt out of sale | Same forget/export primitives apply; "sale" has a specific legal meaning — do not advertise face galleries |

## Deployer obligations (roughly ordered by priority)

1. **Obtain explicit consent before calling `observe()` on any identifiable face.** The library does not police this; your application must. Common failure: ingesting every face a camera sees in a semi-public space (lobby, cafe) without signage and a lawful basis.

2. **Use `forget()` — not delete + no prune — when erasure is requested.**
   - `forget(entity_id)` runs `delete_entity_cascade` AND calls `prune_versions(now)` so LanceDB's version history doesn't retain the vectors. Without prune, old versions keep biometric data for 7 days (the default retention).
   - This is verified by the `test_forget_physically_removes_vectors_from_version_history` test in the library; operators get compliance for free by using the facade method.
   - After `forget`, attempting to restore the entity from backup re-materializes regulated data you promised to erase. **Do not build a restore-from-backup path for forgotten entities.**

3. **Default to `consent_required=True` for face observations.** (Wave 4.1 facade plumbing.) Bypassing this flag is a code-level decision you must justify in your privacy review.

4. **Never put raw embeddings or image bytes in the EventLog.** The library already doesn't — EventLog records `counts_only` for `forget`, opaque IDs for other ops. Your application must not introduce a log surface that does.

5. **Pick a jurisdiction profile.** vemem has no `compliance_profile` config as of v0; this is scheduled for v0.1. Until then, treat defaults as "none" and apply:
   - EU users → GDPR defaults above
   - Illinois users → BIPA: stricter consent, no `grace_days>0`, no restore-from-backup, written retention policy
   - California users → CCPA: provide a "do not sell" surface even if you don't sell (flag is cheap to add)

6. **Don't host a public face-search demo.** A website that accepts visitor faces into a shared gallery makes you operating Clearview-adjacent territory — it's what attracts regulator attention, not the library itself. Keep demos local-only or user-only.

7. **Backup policy.** Regulators accept "beyond use" backups that age out on normal rotation. Do not add surgical backup-restore for forgotten entities. Standard rotation (30–90 days) is fine; verify it actually rotates.

## What the library does NOT provide (and what to layer on top)

| Gap | v0.1+ intent |
|---|---|
| Consent capture UI | Your app |
| Jurisdiction profile enum | `compliance_profile={"gdpr", "bipa", "ccpa", "none"}` config (roadmap) |
| Cryptographic erasure (ISO/IEC Renewable Biometric References) | Per-entity encryption keys; `forget()` destroys the key. Not v0. |
| Data-subject request workflow (intake, identity verification, SLA tracking) | Your app |
| Audit-log retention automation | Partial: EventLog has `prune_events(older_than=...)`, default retention 1 year / 5 years for `forget` events. You schedule the prune. |

## Reporting

If you believe a deployer is misusing vemem (harvesting faces without consent, running a public search service, etc.) you can open a non-security issue on the repo, or contact the author via the email in `pyproject.toml`. The library author has no control over deployments — your recourse is a report to the relevant data-protection authority.

## One-page summary to paste into your privacy review

> vemem is a self-hosted Python library that stores face embeddings and bi-temporal metadata per entity. Erasure uses `forget()` + LanceDB version prune, which physically removes vectors from version history (verified by the library's test suite). Data subjects can export their data via `export(..., include_embeddings=True)` and restrict processing via `restrict()`. The library does not include consent capture UI, jurisdiction profile enforcement, or cryptographic erasure — those are the deployer's responsibility.

## References

- Spec §7 — non-negotiable compliance rules
- Spec §4.5 — forget semantics and prune requirement
- Spec §4.8 — export semantics (Art. 20)
- [GDPR Art. 17](https://gdpr-info.eu/art-17-gdpr/) — right to erasure
- [GDPR Art. 18](https://gdpr-info.eu/art-18-gdpr/) — right to restriction
- [GDPR Art. 20](https://gdpr-info.eu/art-20-gdpr/) — right to portability
- [Illinois BIPA](https://www.ilga.gov/legislation/ilcs/ilcs3.asp?ActID=3004&ChapterID=57)
