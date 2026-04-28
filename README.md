# Adaptive Memory - Agent Memory System with Frequency-Driven Tiered Storage

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/FastAPI-0.115%2B-green?style=flat-square&logo=fastapi" />
  <img src="https://img.shields.io/badge/Qdrant-VectorDB-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" />
</p>

<p align="center">
  <b>Adaptive Agent Memory with Frequency-Driven Hot/Cold Tiering</b>
</p>

> 将操作系统的"热数据/冷数据"分层概念引入 Agent 记忆系统。高频记忆保持原文快速检索，低频记忆经 LLM 压缩为摘要节省存储，记忆根据访问频率自动在两层之间迁移。

---

## 核心特性

- **热冷分层记忆**
  - **Short-term (Hot)**：高频记忆保持原文存储，低延迟检索 (< 100ms)
  - **Long-term (Cold)**：低频记忆经 LLM 压缩为摘要，节省 60%~80% 存储

- **记忆类型支持**
  - `observation` — 观察（Agent 感知到的信息）
  - `fact` — 事实（确定的知识）
  - `reflection` — 反思（Agent 的推理和总结）
  - `summary` — 摘要（压缩后的长期记忆）

- **频率驱动路由**
  - 根据话题频率自动决定检索策略：Hot Only / Cold Only / Both
  - 累计访问次数达到阈值也进入 Hot（不依赖分数衰减）

- **智能记忆巩固**
  - 低频 Hot 记忆自动压缩迁移到 Cold（长期记忆）
  - 高频 Cold 记忆自动解压提升回 Hot（短期记忆）

- **语义聚类**
  - 相似查询自动归为一类话题
  - 话题级别频率追踪，而非单条记忆

---

## 快速开始

### 1. 安装依赖

```bash
pip install -e ".[dev]"
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，填入你的 OpenAI API Key：

```bash
cp .env.example .env
# 编辑 .env，设置 LLM_API_KEY
```

### 3. 启动服务

```bash
python -m adaptive_memory.api.main
```

服务启动在 `http://localhost:8000`

---

## API 使用

### 写入记忆

```bash
curl -X POST http://localhost:8000/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{
    "content": "用户喜欢Python编程语言",
    "memory_type": "fact",
    "source": "conversation_123",
    "importance": 0.8,
    "tags": ["user_preference", "programming"]
  }'
```

### 检索记忆

```bash
curl -X POST http://localhost:8000/api/v1/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "用户喜欢什么编程语言？",
    "top_k": 5
  }'
```

### 列出记忆

```bash
curl "http://localhost:8000/api/v1/memories?memory_type=fact&limit=10"
```

---

## 架构

```
Agent 调用
    │
    ▼
POST /memories  ──→  MemoryPipeline  ──→  Embedding
    │                                         │
    │                                         ▼
    │                              Frequency Tracker (话题频率)
    │                                         │
    │                              Hot? ──→  Short-term Memory (完整原文)
    │                              Cold? ──→  Long-term Memory (压缩摘要)
    │
    ▼
POST /retrieve  ──→  UnifiedRetriever  ──→  FrequencyRouter
    │                                         │
    │                              HOT_ONLY / COLD_ONLY / BOTH
    │                                         │
    │                              Hot Tier + Cold Tier 并行检索
    │                                         │
    │                              ResultRanker 合并排序
    │                                         │
    ▼                              返回相关记忆列表
  记忆结果
```

---

## 关键配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `DECAY_HALF_LIFE_HOURS` | 72 | 记忆分数半衰期（小时） |
| `HOT_ACCESS_COUNT_THRESHOLD` | 50 | 累计访问次数达到此值进入 Hot |
| `HOT_TO_COLD_THRESHOLD` | 0.25 | 分数低于此值从 Hot 降到 Cold |
| `COLD_TO_HOT_THRESHOLD` | 0.7 | 分数高于此值从 Cold 升到 Hot |
| `HOT_TIER_CAPACITY` | 10000 | Hot 层最大记忆数 |

---

## 技术栈

- **API**: FastAPI + Uvicorn
- **向量数据库**: Qdrant
- **元数据数据库**: PostgreSQL / SQLite (async SQLAlchemy)
- **缓存**: Redis / 内存 LRU
- **嵌入**: OpenAI / sentence-transformers
- **LLM**: OpenAI API (压缩摘要)
- **监控**: Prometheus

---

## License

MIT
