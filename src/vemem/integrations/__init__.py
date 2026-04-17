"""First-party integrations for external agent frameworks.

Each subpackage glues vemem into a specific host. Current members:

- ``vemem.integrations.openclaw`` — automatic image-understanding provider
  for `openclaw <https://openclaw.dev>`_ via a persistent HTTP sidecar
  plus a drop-in TypeScript plugin.

These are supported alongside the core library: their behavior is covered
by tests, changelog entries call out breaking changes, and bug reports are
welcome at https://github.com/linville-charlie/vemem/issues.
"""
