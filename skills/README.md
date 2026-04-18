# Agent skills

[`vemem/`](./vemem/) is an [Agent Skills](https://agentskills.io) package conforming to the open standard Anthropic released. It works out of the box in every skills-compatible host — currently 35+ products including Claude, Claude Code, Cursor, GitHub Copilot, OpenAI Codex, Gemini CLI, Goose, OpenHands, Hermes Agent, OpenClaw, Letta, and Roo Code.

## Install

Symlink (or copy) `skills/vemem/` into your host's skills directory:

| Host | Location |
|---|---|
| Claude.ai (Claude API) | upload via [agent-skills UI](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) |
| Claude Code | `~/.claude/skills/vemem` |
| Hermes Agent | `~/.hermes/skills/vemem` |
| OpenClaw | `~/.openclaw/skills/vemem` |
| Cursor | `.cursor/skills/vemem` (project) or the global config path |
| Goose, OpenHands, Gemini CLI, others | see each host's skills docs |

Example for Claude Code:

```bash
mkdir -p ~/.claude/skills
ln -s "$(pwd)/skills/vemem" ~/.claude/skills/vemem
```

## Verify

In any skills-compatible host, the agent should auto-discover vemem by description when the user mentions faces, identities, or visual memory. You can also invoke the skill explicitly, e.g. `/vemem` in hosts that support slash-invocation.

## What the skill does

It loads vemem's usage patterns into the agent's context at the moment they're needed: which op to call, common patterns, correction flow, privacy semantics, troubleshooting. It does NOT install the library itself — that's still `pip install vemem`. The skill just teaches the agent how to use what you've installed.

## Contributing

Skill improvements are welcome via PR against `skills/vemem/SKILL.md`. Keep the body under 500 lines and split long reference material into `skills/vemem/references/` files per the [Agent Skills spec](https://agentskills.io/specification).
