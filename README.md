# JARVIS Knowledge Core 🤖⚡

A JARVIS-style assistant UI backed by a **temporal knowledge graph** — every conversation becomes structured, time-aware memory.

![stack](https://img.shields.io/badge/stack-Graphiti%20%2B%20FalkorDB%20%2B%20fastembed-00e5ff)

## Architecture

```
┌─────────────────────┐     ┌──────────────────────────────┐
│  JARVIS HUD (web)   │────▶│  FastAPI server (:8630)      │
│  arc-reactor canvas │     ├──────────────────────────────┤
│  live graph physics │     │  Graphiti temporal KG        │
└─────────────────────┘     │  ├─ extraction: gemini-flash │
                            │  ├─ chat: stealth/ox-alpha   │
                            │  └─ embeddings: fastembed    │
                            │      (local, ONNX, $0)       │
                            ├──────────────────────────────┤
                            │  FalkorDB Lite (embedded)    │
                            │  zero containers, zero Java  │
                            └──────────────────────────────┘
```

## Why this stack

| Choice | Reason |
|---|---|
| **Graphiti** | Temporal knowledge graph — facts carry `valid_at`/`invalid_at`, so memory *evolves* instead of contradicting itself |
| **FalkorDB Lite** | Embedded graph DB via redislite — no Docker needed (Kuzu segfaults on some hosts) |
| **fastembed (bge-small)** | Local ONNX embeddings — $0 cost, ~ms latency |
| **stealth/ox-alpha** | Free 1M-context reasoning model for chat replies |
| **gemini-2.5-flash** | Strict-schema JSON extraction (ox-alpha fails structured output) |

## Quick start

```bash
./run.sh                # default port 8630
./run.sh 9000           # custom port
```

Open http://127.0.0.1:8630/ — type to chat (auto-ingests), or prefix with `remember:` to store facts directly.

```bash
curl -X POST localhost:8630/ingest -H 'Content-Type: application/json' \
  -d '{"text":"Alice works at Acme Corp","source":"notes"}'

curl -X POST localhost:8630/search -H 'Content-Type: application/json' \
  -d '{"query":"Where does Alice work?"}'
```

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | HUD |
| `/chat` | POST | Chat grounded in graph memory (streams) |
| `/ingest` | POST | Extract entities/facts into the graph |
| `/search` | POST | Hybrid semantic+BM25+graph fact search |
| `/graph` | GET | Nodes+edges snapshot for visualization |
| `/stats` | GET | Entity/edge/episode counts |

## Setup

Requires an OpenRouter key:

```bash
echo "OPENROUTER_API_KEY=sk-or-..." > .env   # chmod 600!
```

## License

MIT
