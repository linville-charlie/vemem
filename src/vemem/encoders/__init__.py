"""Encoder Protocol + concrete adapters.

Encoders produce embeddings from image crops. Encoder version is part of the
encoder_id; upgrading an encoder produces new rows, never overwrites — see
docs/spec/identity-semantics.md §3.1a.
"""
