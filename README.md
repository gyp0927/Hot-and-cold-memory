# Hot-and-cold-memory
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
| No concept of "forgetting" | Old unused memories physically deleted |
| Linear scaling cost | Compression saves 60-80% storage |
| No access history | Frequency tracking enables smart routing |
| Single retrieval mode | Hybrid vector + keyword search |

## Quick Start

```bash
# 1. Install
pip install -e "."

# 2. Configure
cp .env.example .env
# Edit .env and add your OpenAI API key

# 3. Start
python -m hot_and_cold_memory.api.main
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

### Retrieve with hybrid search (vector + keyword)

```bash
curl -X POST http://localhost:8000/api/v1/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "Python programming", "top_k": 5, "use_hybrid": true}'
```

Hybrid search fuses vector similarity and keyword matching using Reciprocal Rank Fusion (RRF), improving recall when exact terms matter.

### List memories

```bash
# All facts
curl "http://localhost:8000/api/v1/memories?memory_type=fact&limit=20"

# From a specific conversation
curl "http://localhost:8000/api/v1/memories?source=conversation_001"
```

### Get related memories (association graph)

```bash
curl "http://localhost:8000/api/v1/memories/550e8400-e29b-41d4-a716-446655440000/related"
```

**Response:**
```json
{
  "memory_id": "550e8400-e29b-41d4-a716-446655440000",
  "related": [
    {
      "memory_id": "660e8400-e29b-41d4-a716-446655440001",
      "content": "User uses Django for web projects",
      "tier": "hot",
      "link_type": "coaccess",
      "strength": 1.2
    }
  ]
}
```

Memories retrieved together automatically form coaccess links. Strength increases with repeated co-retrieval.

## Core Features

### 1. Auto-Importance Scoring

Memories are automatically scored for importance on ingestion (no manual tagging needed):

- **Rule-based**: Chinese keyword signals (preferences, identity, health, goals, etc.)
- **Memory type multipliers**: `fact` and `summary` get higher base scores
- **Optional LLM fallback**: For ambiguous content near the threshold

| Signal Type | Example | Score Impact |
|-------------|---------|-------------|
| Preference | "User likes Python, dislikes JavaScript" | +0.15 |
| Identity | "Name is Zhang San, backend engineer" | +0.15 |
| Health | "Severe peanut allergy" | +0.15 |
| Goal | "Planning to switch jobs in 3 months" | +0.12 |
| Date | "Wedding anniversary on June 1" | +0.10 |
| Small talk | "Weather is nice today" | ~0.2 |

### 2. True Forgetting (TTL + Active Deletion)

Old, unimportant memories are physically deleted, not just hidden:

- Cold memories with `importance < 0.2` that haven't been accessed in 30+ days are removed
- Low-importance memories get an `expires_at` timestamp when compressed to cold
- High-importance memories (score >= 0.6) are protected from demotion and never expire

### 3. Hybrid Search (Vector + Keyword)

Fuses two retrieval signals for better recall:

- **Vector similarity**: Semantic meaning matching
- **Keyword search**: Exact term matching in content (cross-database `ILIKE`)
- **RRF fusion**: Reciprocal Rank Fusion combines ranked lists without score calibration

Enable per-query with `"use_hybrid": true` or globally via `ENABLE_HYBRID_SEARCH`.

### 4. Memory Consolidation (Deduplication + Merging)

Detects and merges semantically duplicate memories:

1. Compute pairwise embedding cosine similarity
2. Pairs above `CONSOLIDATION_SIMILARITY_THRESHOLD` (default 0.92) are merged
3. LLM merges content while preserving all important facts
4. Result inherits max importance, max access count, and combined tags

### 5. Memory Association Graph

Memories retrieved together in the same query automatically form coaccess links:

- `POST /retrieve` returning N memories creates C(N,2) links
- Link strength accumulates on repeated co-retrieval
- Query via `GET /memories/{id}/related`
- Links are bidirectional (reverse link detected and updated)

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
        |
        | True Forgetting
        v
   [Physically deleted]
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
                                   |
                          +------------------+
                          | Hybrid Fusion    |
                          | (optional RRF)   |
                          +------------------+
```

### Memory lifecycle

| Stage | Storage | Latency | Trigger |
|-------|---------|---------|---------|
| New / Hot | Full text + embedding | < 100ms | Frequency >= 0.7 or 50+ accesses |
| Cooling | Same | Same | No access for 3+ days |
| Cold | LLM-compressed summary | ~200ms | Frequency drops below 0.25 |
| Forgotten | **Deleted** | - | Cold + old + unimportant |

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
| `ENABLE_AUTO_IMPORTANCE` | true | Auto-score importance on ingestion |
| `ENABLE_FORGETTING` | true | Physically delete old unimportant memories |
| `FORGET_MIN_IMPORTANCE` | 0.2 | Importance threshold for forgetting |
| `FORGET_MIN_DAYS_SINCE_ACCESS` | 30 | Minimum age before a memory can be forgotten |
| `ENABLE_HYBRID_SEARCH` | true | Enable vector + keyword fusion |
| `HYBRID_RRF_K` | 60 | RRF constant for hybrid ranking |
| `ENABLE_CONSOLIDATION` | true | Enable deduplication and merging |
| `CONSOLIDATION_SIMILARITY_THRESHOLD` | 0.92 | Cosine similarity threshold for merging |
| `ENABLE_ASSOCIATIONS` | true | Enable automatic coaccess link creation |

## Architecture

```
 +--------------------------------------------------------------------------------+
 |                               Agent (your code)                                |
 +----------------+----------------+----------------+-----------------------------+
                  | HTTP           | HTTP           | HTTP
        +---------v------+ +-------v--------+ +-----v---------+
        | POST /memories | | POST /retrieve | | GET /memories |
        +---------+------+ +-------+--------+ +----+----------+
                  |                |               |
                  v                v               v
 +--------------------------------------------------------------------------------+
 |                           MemoryPipeline / Retriever                           |
 |  - Generate embedding                                                          |
 |  - Auto-importance scoring                                                     |
 |  - Frequency-driven routing (hot / cold / both)                                |
 |  - Hybrid search (vector + keyword RRF fusion)                                 |
 |  - Coaccess link creation                                                      |
 +----------------+----------------+----------------+-----------------------------+
                  |                |                |
        +---------v------+ +-------v--------+ +-----v----------+
        |   Hot Tier     | |   Cold Tier    | | Consolidation  |
        | (Qdrant +      | | (Qdrant +      | | Engine         |
        |  local store)  | |  local store)  | | - Deduplicate  |
        |                | |                | | - Merge (LLM)  |
        | Full text      | | LLM-compressed | |                |
        | < 100ms        | | summaries      | |                |
        +--------+-------+ +--------+-------+ +----------------+
                 |                 |
                 +--------+--------+
                          |
          +---------------v------------------+
          |       PostgreSQL (metadata)      |
          |  - memories (tier, importance,   |
          |    compressed, expires_at)       |
          |  - topic_clusters                |
          |  - access_logs                   |
          |  - migration_logs                |
          |  - memory_links (associations)   |
          +----------------------------------+
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
