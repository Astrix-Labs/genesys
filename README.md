<!-- mcp-name: io.github.rishimeka/genesys-memory -->
[![PyPI](https://img.shields.io/pypi/v/genesys-memory)](https://pypi.org/project/genesys-memory/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/genesys-memory)](https://pypi.org/project/genesys-memory/)
[![CI](https://github.com/rishimeka/genesys/actions/workflows/ci.yml/badge.svg)](https://github.com/rishimeka/genesys/actions/workflows/ci.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

# Genesys

**The intelligence layer for AI memory.**

> Genesys doesn't just remember what happened; it remembers why. A scoring engine + causal graph + lifecycle manager for AI agent memory. Speaks MCP natively.
<img width="1512" height="827" alt="image" src="https://github.com/user-attachments/assets/d152aa07-a852-4b8e-9f98-942d0bebd497" />

## What is this

Genesys is a scoring engine, causal graph, and lifecycle manager for AI memory. Memories are scored by a multiplicative formula (relevance × connectivity × reactivation), connected in a causal graph, and actively forgotten when they become irrelevant.

This package (`genesys-memory`) is the core library: an in-memory causal graph engine with optional JSON persistence, plus a stdio MCP server. It has no database dependency and no REST API. A hosted product built on top of this library — with Postgres, additional storage backends, and a REST/HTTP MCP API — is available separately at `genesys-api.astrixlabs.ai`; it is not part of this package.

## Why

- **Flat memory doesn't scale.** Dumping everything into a vector store gives you recall with zero understanding. The 500th memory buries the 5 that matter.
- **No forgetting = no intelligence.** Real memory systems forget. Without active pruning, your AI drowns in stale context.
- **No causal reasoning.** Vector similarity can't answer "why did I choose X?" — you need a graph.

Your AI remembers everything but understands nothing. Genesys fixes that.

## Quick Start

Install the package. The base install has zero database dependencies — state lives in memory and is optionally persisted to a JSON file.

```bash
pip install genesys-memory
```

Optional extras:

```bash
pip install 'genesys-memory[openai]'      # OpenAI embeddings
pip install 'genesys-memory[local]'       # Local embeddings (sentence-transformers, no API key)
pip install 'genesys-memory[anthropic]'   # LLM-based causal inference (consolidation, contradiction detection)
```

Run the stdio MCP server directly:

```bash
python3 -m genesys_memory
```

### From source

```bash
git clone https://github.com/rishimeka/genesys.git
cd genesys
pip install -e '.[dev]'
pytest tests/
```

## Connect to your AI

### Claude Code

```bash
claude mcp add genesys -- python -m genesys_memory
```

### Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "genesys": {
      "command": "python",
      "args": ["-m", "genesys_memory"]
    }
  }
}
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `memory_store` | Store a new memory, optionally linking to related memories |
| `memory_recall` | Recall memories by natural language query (vector + graph) |
| `memory_search` | Search memories with filters (status, date range, keyword) |
| `memory_traverse` | Walk the causal graph from a given memory node |
| `memory_explain` | Explain why a memory exists and its causal chain |
| `memory_stats` | Get memory system statistics |
| `pin_memory` | Pin a memory so it's never forgotten |
| `unpin_memory` | Unpin a previously pinned memory |
| `delete_memory` | Permanently delete a memory |
| `list_core_memories` | List core memories, optionally filtered by category |
| `set_core_preferences` | Set user preferences for core memory categories |

## How it works

Every memory is scored by three forces multiplied together:

```
decay_score = relevance × connectivity × reactivation
```

- **Relevance** decays over time. Old memories fade unless reinforced.
- **Connectivity** rewards memories with many causal links. Hub memories survive.
- **Reactivation** boosts memories that keep getting recalled. Frequency matters.

Because the formula is multiplicative, a memory must score on *all three* axes to survive. A highly connected but never-accessed memory still decays. A frequently recalled but causally orphaned memory still fades.

```
STORE → ACTIVE → DORMANT → FADING → PRUNED
           ↑                    │
           └── reactivation ────┘
                                  (only if score=0, orphan, not pinned)
```

Memories can also be promoted to **core** status — structurally important memories that are auto-pinned and never pruned.

## Benchmark Results

We've run internal evaluations against the [LoCoMo](https://arxiv.org/abs/2402.06397) long-conversation memory benchmark during development. These are self-reported, run with our own harness (category 5 — adversarial questions with disputed ground truth — excluded), and not independently reproduced, so treat them as directional rather than a verified claim. Reproduction scripts are in [`benchmarks/`](benchmarks/) if you want to run your own numbers.

## Storage backend

This package ships one storage backend: an in-memory causal graph (`storage/memory.py`) with optional JSON persistence via `GENESYS_PERSIST_PATH`. No database is required.

Additional backends — Postgres/pgvector, FalkorDB, MongoDB, and an Obsidian vault adapter — along with a REST API, OAuth, and multi-user auth, are part of the hosted product at `genesys-api.astrixlabs.ai` and are not included in this repo.

Want a different storage backend for the open-source library? Implement the provider protocols in [`storage/base.py`](src/genesys_memory/storage/base.py) and bring your own.

## Configuration

Copy `.env.example` to `.env` and set:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Unless `GENESYS_EMBEDDER=local` | Embeddings |
| `ANTHROPIC_API_KEY` | No | Enables LLM-based causal inference (consolidation, contradiction detection). Off by default — without it, causal edges only come from edges the caller explicitly declares in `memory_store` plus cosine-similarity linking. |
| `GENESYS_EMBEDDER` | No | `openai` (default) or `local` (sentence-transformers, no API key) |
| `GENESYS_PERSIST_PATH` | No | JSON file path to persist state across restarts (in-memory otherwise) |
| `GENESYS_USER_ID` | No | Default user ID for single-tenant mode |

See [`.env.example`](.env.example) for all options.

## Built by

Genesys is built by [Rishi Meka](https://github.com/rishimeka) at [Astrix Labs](https://astrixlabs.ai). It came out of frustration with re-explaining project context to Claude every session. The goal is the intelligence layer between your LLM and your memory — fully open source.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[AGPL-3.0-or-later](LICENSE)

> **Note:** Genesys releases prior to v0.3.6 were documented as Apache 2.0 in error. The LICENSE file has always contained the AGPLv3 text. From v0.3.6 onward, all documentation correctly references AGPL-3.0-or-later with a Contributor License Agreement.
