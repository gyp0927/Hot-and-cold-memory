# Adaptive Memory

<p align="center">
  <b>Agent Memory System with Frequency-Driven Hot/Cold Tiering</b>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/release/python-3110/">
    <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  </a>
  <a href="https://fastapi.tiangolo.com/">
    <img src="https://img.shields.io/badge/FastAPI-0.115%2B-green" alt="FastAPI">
  </a>
  <a href="https://qdrant.tech/">
    <img src="https://img.shields.io/badge/Qdrant-VectorDB-orange" alt="Qdrant">
  </a>
  <img src="https://img.shields.io/badge/license-MIT-yellow" alt="License">
</p>

---

## What is this?

A memory system for AI agents. Like human memory, it has **short-term** (hot) and **long-term** (cold) storage:

- **Short-term** keeps full text for fast recall
- **Long-term** compresses old memories to save space
- Memories automatically move between layers based on how often they're accessed

## Why not just use a vector database?

| Plain Vector DB | Adaptive Memory |
|----------------|-----------------|
| Everything stored the same way | Hot memories fast, cold memories compressed |
| No concept of "forgetting" | Old unused memories naturally fade |
| Linear scaling cost | Compression saves 60-80% storage |
| No access history | Frequency tracking enables smart routing |

## Quick Start

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# Edit .env and add your OpenAI API key

# 3. Start
python -m adaptive_memory.api.main
```

Service runs at `http://localhost:8000`

## API Usage

### Write a memory

```bash
curl -X POST http://localhost:8000/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{
    "content": "User prefers Python over JavaScript",
    "memory_type": "fact",
    "source": "conversation_001",
    "importance": 0.8
  }'
```

**Response:**
```json
{
  "memory_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "success",
  "tier": "hot"
}
```

### Retrieve relevant memories

```bash
curl -X POST http://localhost:8000/api/v1/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "What language does the user like?", "top_k": 5}'
```

**Response:**
```json
{
  "memories": [
    {
      "memory_id": "550e8400-e29b-41d4-a716-446655440000",
      "content": "User prefers Python over JavaScript",
      "score": 0.92,
      "tier": "hot",
      "memory_type": "fact"
    }
  ],
  "routing_strategy": "hot_only",
  "total_latency_ms": 45.2
}
```

### List memories

```bash
# All facts
curl "http://localhost:8000/api/v1/memories?memory_type=fact&limit=20"

# From a specific conversation
curl "http://localhost:8000/api/v1/memories?source=conversation_001"
```

## How It Works

```
Agent writes memory
        |
        v
+-------------------+     +------------------+
|  Frequency Check  |---->|   Short-term     |
|  (hot topic?)     |     |   (Hot Tier)     |
+-------------------+     |  Full text       |
        | no              |  < 100ms recall  |
        v                 +------------------+
+-------------------+              |
|   Long-term       |<-------------+
|   (Cold Tier)     |   Auto-migration
|   Compressed      |   by frequency
+-------------------+
        ^
        |
Agent queries memory
        |
        v
+-------------------+     +------------------+
|  Router decides   |---->|  Query Hot Only  |
|  HOT / COLD / BOTH|     |  (freq >= 0.7)   |
+-------------------+     +------------------+
                                   |
                          +------------------+
                          | Query Both Tiers |
                          | (0.25 < freq)    |
                          +------------------+
                                   |
                          +------------------+
                          | Query Cold Only  |
                          | (freq <= 0.25)   |
                          +------------------+
```

### Memory lifecycle

| Stage | Storage | Latency | Trigger |
|-------|---------|---------|---------|
| New / Hot | Full text + embedding | < 100ms | Frequency >= 0.7 or 50+ accesses |
| Cooling | Same | Same | No access for 3+ days |
| Cold | LLM-compressed summary | ~200ms | Frequency drops below 0.25 |
| Forgotten | Still stored but rarely queried | - | Only accessed if explicitly searched |

### Decay curve

Default half-life: **72 hours**

| Time since last access | Remaining score |
|------------------------|-----------------|
| 0h | 100% |
| 24h | 79% |
| 48h | 63% |
| 72h | 50% |
| 1 week | 31% |

But with **minimum score protection** (`log(access_count) / 6`), a memory with 50 historical accesses can never drop below 0.65 regardless of decay.

## Memory Types

| Type | Use case | Example |
|------|----------|---------|
| `observation` | Something the agent noticed | "User closed the tab without saving" |
| `fact` | Factual knowledge about user | "User prefers Python over JavaScript" |
| `reflection` | Agent's own reasoning | "User seems frustrated when responses are too verbose" |
| `summary` | Condensed from multiple memories | "User is a backend engineer who likes clean code" |

## Configuration

Key settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `DECAY_HALF_LIFE_HOURS` | 72 | How fast memories fade without access |
| `HOT_ACCESS_COUNT_THRESHOLD` | 50 | Cumulative accesses to force hot tier |
| `HOT_TO_COLD_THRESHOLD` | 0.25 | Score below which memories migrate to cold |
| `COLD_TO_HOT_THRESHOLD` | 0.7 | Score above which memories promote to hot |
| `HOT_TIER_CAPACITY` | 10000 | Max short-term memories before eviction |
| `COMPRESSION_MODEL` | gpt-4o-mini | LLM for summarizing cold memories |

## Architecture

```
┌─────────────────────────────────────────┐
│           Agent (your code)             │
└─────────────────┬───────────────────────┘
                  │ HTTP
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
┌────────┐  ┌──────────┐  ┌──────────┐
│ POST   │  │  POST    │  │   GET    │
│/memories│  │/retrieve │  │ /memories│
└────┬───┘  └────┬─────┘  └────┬─────┘
     │           │             │
     ▼           ▼             ▼
┌─────────────────────────────────────────┐
│         MemoryPipeline                  │
│  - Generate embedding                   │
│  - Check topic frequency                │
│  - Route to Hot or Cold tier            │
└─────────────────────────────────────────┘
     │                           │
     ▼                           ▼
┌──────────┐            ┌──────────────┐
│ Hot Tier │            │  Cold Tier   │
│(Qdrant + │            │(Qdrant +     │
│  local   │            │  local store)│
│  store)  │            │              │
│          │            │  LLM-compressed
│ Full text│            │  summaries   │
│ < 100ms  │            │ ~200ms       │
└──────────┘            └──────────────┘
     │                           │
     └───────────┬───────────────┘
                 ▼
        ┌─────────────┐
        │  PostgreSQL │
        │  (metadata) │
        └─────────────┘
```

## Tech Stack

- **API**: FastAPI, Uvicorn
- **Vector DB**: Qdrant (local or server)
- **Metadata DB**: PostgreSQL via async SQLAlchemy (SQLite works too)
- **Cache**: Redis or in-memory LRU
- **Embeddings**: OpenAI or sentence-transformers
- **Compression**: OpenAI API (gpt-4o-mini)
- **Monitoring**: Prometheus metrics

## Running Tests

```bash
pytest -q
```

## License

MIT
