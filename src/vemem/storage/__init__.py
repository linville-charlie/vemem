"""Storage Protocol + LanceDB backend.

Defines the Store interface and ships a LanceDB-backed default. Storage is
responsible for persisting immutable Observations/Embeddings, append-only
Bindings (bi-temporal), and the EventLog, plus performing prune-on-forget.
"""
