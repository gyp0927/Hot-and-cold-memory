# Adaptive RAG - 频率驱动热冷分层检索系统

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/FastAPI-0.115%2B-green?style=flat-square&logo=fastapi" />
  <img src="https://img.shields.io/badge/Qdrant-VectorDB-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" />
</p>

<p align="center">
  <b>Self-organizing Knowledge Base with Frequency-Driven Tiered Retrieval</b>
</p>

> 将操作系统的"热数据/冷数据"分层概念引入 RAG，让知识库根据访问频率自动优化存储结构。高频内容保持原文低延迟检索，低频内容经 LLM 压缩节省存储，新文档根据话题热度智能分流。

---

## 核心特性

- **热冷分层存储**
  - **Hot Tier**：高频 chunk 保持原文存储，低延迟检索 (< 100ms)
  - **Cold Tier**：低频 chunk 经 LLM 压缩为摘要，节省 60%~80% 存储
  - **Cold Raw**：新文档的冷话题内容直接存原始文本，跳过压缩以节省 LLM 成本

- **智能摄入分流**
  - 上传文档时，根据话题历史频率自动决定每个 chunk 的去向
  - 热话题 chunk → Hot Tier（完整嵌入 + 原文）
  - 冷话题 chunk → Cold Tier Raw（原始文本，后续再压缩）

- **频率驱动路由**
  - 根据查询话题频率自动路由到 Hot / Cold / Both 层级
  - 高频话题 → Hot Only，低频话题 → Cold Only，中频 → 双层并行查询

- **语义查询聚类**
  - "苹果公司营收"和"Apple revenue"自动归为同一话题
  - 基于 embedding 相似度的动态聚类 (cosine >= 0.85)
  - 超大聚类自动分裂，7 天未活跃聚类自动清理

- **时间衰减算法**
  - 指数衰减，半衰期 24 小时
  - 近期访问权重更高，旧访问自动 fade out

- **自动冷热迁移**
  - 基于频率阈值自动触发 Hot <-> Cold 迁移
  - 批量压缩：单次 LLM 调用处理多个 chunk，成本降低 ~10x
  - 支持非高峰窗口限制（默认 2:00-5:00）+ 手动触发
  - 热层容量超限自动淘汰最冷 chunk

- **多格式文档支持**
  - 文本 (.txt, .md, .json, .csv)、PDF (.pdf)、Word (.docx)、图片 OCR (.png, .jpg, .jpeg, .webp 等)

- **多模型兼容**
  - 嵌入模型：OpenAI / sentence-transformers (免费本地)
  - LLM：任意兼容 OpenAI 或 Anthropic API 格式的服务商（Kimi / DeepSeek / 通义千问 / OpenAI / 智谱）

- **多级缓存**
  - 嵌入缓存：2000 条 LRU 缓存，避免重复计算相同文本的 embedding
  - 查询缓存：5 秒 TTL 缓存，防止 UI 轮询或快速重提交导致的冗余检索
  - 内存/Redis 双后端

- **质量保障**
  - 解压缩时进行 cosine similarity 验证，低质量扩张自动回退到压缩文本
  - 热层结果评分加权 (+5%)，冷层摘要评分微调 (-5%)

---

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                        用户查询                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
              ┌────────▼────────┐
              │  UnifiedRetriever│  ← 5秒查询缓存
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  FrequencyRouter │  ← 根据话题频率决策路由策略
              │  频率路由层      │
              └────────┬────────┘
                       │
          ┌────────────┼────────────┐
          │            │            │
    ┌─────▼─────┐ ┌───▼────┐ ┌────▼─────┐
    │  Hot Tier  │ │Cold Tier│ │   Both   │
    │  原文存储   │ │摘要存储 │ │  双层查询 │
    │  低延迟     │ │高压缩   │ │          │
    └─────┬─────┘ └───┬────┘ └────┬─────┘
          │           │           │
          └───────────┼───────────┘
                      │
              ┌───────▼───────┐
              │  ResultRanker  │  ← 热层加权、去重、截断
              │  结果合并重排   │
              └───────┬───────┘
                      │
              ┌───────▼───────┐
              │ 更新频率计数器 │ ← 异步记录访问，触发冷热迁移
              └───────────────┘
```

### 数据流

```
新文档上传
    │
    ▼
┌─────────────────┐
│  分块 + 嵌入     │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
 热话题     冷话题
    │         │
    ▼         ▼
 Hot Tier  Cold Tier Raw
 (原文)     (原文，未压缩)
    │         │
    │    ┌────┘
    │    ▼
    │  ┌───────────────────────┐
    │  │  定时迁移任务 (每小时)  │
    │  │  - Hot->Cold: 压缩低频  │
    │  │  - Cold Raw->Cold: 压缩  │
    │  │  - Cold->Hot: 解压高频  │
    │  └───────────────────────┘
    │         │
    ▼         ▼
用户查询 -> FrequencyRouter -> Hot/Cold/Both -> 返回结果
    │
    └─ 异步更新频率分数 (fire-and-forget)
```

---

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| API 框架 | FastAPI + Uvicorn | 异步高性能，自动 OpenAPI 文档 |
| 向量数据库 | Qdrant (本地模式) | 无需 Docker，文件级持久化 |
| 元数据存储 | PostgreSQL / SQLite + SQLAlchemy | 生产用 PostgreSQL (asyncpg)，开发可用 SQLite |
| 缓存 | 内存缓存 / Redis | 本地开发用内存，生产可选 Redis |
| 嵌入模型 | OpenAI / sentence-transformers | 本地模型免费，支持 CPU/GPU |
| LLM | 任意兼容 OpenAI/Anthropic | 统一客户端自动识别格式 |
| 监控 | Prometheus + OpenTelemetry | 指标采集和分布式追踪 |
| 文档存储 | 本地文件系统 | 按 UUID 前两位分片，避免单目录过载 |
| 前端 | 纯 HTML/CSS/JS | 单文件，零依赖 |

---

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/gyp0927/adaptive-rag.git
cd adaptive-rag
```

### 2. 安装依赖

```bash
# Python 3.11+
pip install -e ".[local]"
```

### 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 LLM API Key：

```bash
# === 使用 Kimi Code ===
LLM_BASE_URL=https://api.kimi.com/coding
LLM_API_KEY=sk-kimi-your-key
COMPRESSION_MODEL=kimi-for-coding
DECOMPRESSION_MODEL=kimi-for-coding

# === 或使用 DeepSeek (更便宜) ===
# LLM_BASE_URL=https://api.deepseek.com/v1
# LLM_API_KEY=sk-your-deepseek-key
# COMPRESSION_MODEL=deepseek-chat
# DECOMPRESSION_MODEL=deepseek-chat

# === 或使用 OpenAI ===
# LLM_BASE_URL=https://api.openai.com/v1
# LLM_API_KEY=sk-your-openai-key
# COMPRESSION_MODEL=gpt-4o-mini
# DECOMPRESSION_MODEL=gpt-4o
```

### 4. 启动服务

```bash
# 启动 API 服务（首次启动自动初始化数据库和向量存储）
uvicorn adaptive_rag.api.main:app --host 0.0.0.0 --port 8000
```

### 5. 打开前端

浏览器访问 http://localhost:8000

---

## 前端界面

内置可视化前端，无需额外安装：

| 功能 | 说明 |
|------|------|
| 文档上传 | 支持 txt/md/pdf/docx/png/jpg，自动提取文本 |
| 智能查询 | 输入问题，系统自动路由到热层/冷层 |
| 系统监控 | 实时显示 Hot/Cold chunk 数量、查询次数 |
| 手动迁移 | 一键触发冷热数据迁移 |
| 系统日志 | 实时记录所有操作和错误 |

### 截图

```
┌─────────────────────────────────────────────────────────────┐
│  Adaptive RAG - 频率驱动热冷分层系统          [在线]         │
├──────────────┬──────────────────────────────────────────────┤
│  文档上传    │  系统监控                                    │
│  [选择文件]  │  ┌─────┐ ┌─────┐ ┌─────┐                    │
│  [粘贴文本]  │  │ 5   │ │ 2   │ │ 12  │                    │
│  [上传文档]  │  │ Hot │ │ Cold│ │Query│                    │
├──────────────┼──────────────────────────────────────────────┤
│  智能查询    │  查询结果                                    │
│  [查询输入]  │  策略: 热层优先 | 频率: 0.85                 │
│  [执行查询]  │  ┌─────────────────────────────────────┐    │
│              │  │ HOT  #1  相似度: 0.92                │    │
│              │  │ 自适应RAG根据查询频率自动分层...      │    │
│              │  │ 访问: 8次 | 频率: 1.000              │    │
│              │  └─────────────────────────────────────┘    │
└──────────────┴──────────────────────────────────────────────┘
```

---

## API 接口

### 查询

```bash
# 智能查询（自动路由）
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是自适应RAG",
    "top_k": 5,
    "tier": null,
    "decompress": false
  }'

# 强制查询热层
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是自适应RAG",
    "top_k": 5,
    "tier": "hot"
  }'

# 查询并解压缩冷层结果
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是自适应RAG",
    "top_k": 5,
    "decompress": true
  }'
```

响应示例：

```json
{
  "chunks": [
    {
      "chunk_id": "xxx",
      "document_id": "xxx",
      "content": "自适应RAG是一种新型的检索增强生成架构...",
      "score": 0.92,
      "tier": "hot",
      "is_decompressed": false,
      "access_count": 8,
      "frequency_score": 1.0
    }
  ],
  "routing_strategy": "hot_only",
  "hot_results_count": 3,
  "cold_results_count": 0,
  "total_latency_ms": 45.2,
  "topic_frequency": 0.85
}
```

### 文档管理

```bash
# 上传文档
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@document.pdf" \
  -F "title=我的文档"

# 测试文本提取（不存储）
curl -X POST "http://localhost:8000/api/v1/documents/extract-test" \
  -F "file=@document.pdf"

# 列出文档
curl "http://localhost:8000/api/v1/documents?limit=20&offset=0"

# 获取文档详情（含所有 chunk 内容）
curl "http://localhost:8000/api/v1/documents/{document_id}"

# 删除文档
curl -X DELETE "http://localhost:8000/api/v1/documents/{document_id}"
```

### 管理操作

```bash
# 手动触发迁移周期
curl -X POST "http://localhost:8000/api/v1/admin/migrate"

# 健康检查
curl "http://localhost:8000/health"

# 就绪检查
curl "http://localhost:8000/ready"
```

---

## 核心概念

### 频率分数计算

```
frequency_score = decay_engine.apply_decay(
    base_score=metadata.frequency_score,
    last_accessed=metadata.last_accessed_at,
    access_count=metadata.access_count,
)
```

- **访问次数**: 累计访问次数
- **时间衰减**: 指数衰减，半衰期 24 小时
- **话题热度**: 查询聚类的聚合频率

### 路由策略

| 话题频率 | 路由策略 | 说明 |
|----------|----------|------|
| >= 0.7 | `HOT_ONLY` | 高频话题，只查热层（最低延迟） |
| <= 0.3 | `COLD_ONLY` | 低频话题，只查冷层（避免浪费热层查询） |
| 0.3 ~ 0.7 | `BOTH` | 中频话题，双层并行查询后合并重排 |

路由决策在查询时实时计算，无需人工干预。

### 冷热迁移阈值

| 方向 | 阈值 | 触发条件 |
|------|------|----------|
| Hot -> Cold | frequency <= 0.3 | 低频 chunk 经 LLM 压缩为摘要，批量压缩降低成本 ~10x |
| Cold -> Hot | frequency >= 0.7 | 高频 chunk 经 LLM 解压恢复原文 |
| 热层容量 | > 10000 chunks | 自动淘汰最冷的 10% chunk 到冷层 |

### 分块策略

通过 `CHUNK_STRATEGY` 环境变量配置：

| 策略 | 配置值 | 说明 |
|------|--------|------|
| **递归分块** | `recursive` (默认) | 按段落 -> 句子 -> 词自然边界切分，保留语义完整性 |
| **固定大小** | `fixed` | 等长切分，简单快速 |
| **LLM 语义分块** | `llm` | 调用 LLM 按语义主题分段，质量最高但需消耗 token，失败自动回退到递归分块 |

---

## 配置说明

### 嵌入模型选择

```bash
# 免费本地模型 (推荐开发使用)
EMBEDDING_PROVIDER=sentence-transformers
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# 中文优化
# LOCAL_EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
# EMBEDDING_DIMENSION=1024

# OpenAI 付费模型
# EMBEDDING_PROVIDER=openai
# EMBEDDING_MODEL=text-embedding-3-small
# EMBEDDING_DIMENSION=1536
```

### LLM 模型选择

支持任意兼容 OpenAI 或 Anthropic API 格式的服务商：

| 服务商 | Base URL | 压缩模型 | 解压模型 | 成本 |
|--------|----------|----------|----------|------|
| **Kimi Code** | `https://api.kimi.com/coding` | `kimi-for-coding` | `kimi-for-coding` | 中等 |
| **DeepSeek** | `https://api.deepseek.com/v1` | `deepseek-chat` | `deepseek-chat` | 最低 |
| **通义千问** | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-turbo` | `qwen-plus` | 低 |
| **OpenAI** | `https://api.openai.com/v1` | `gpt-4o-mini` | `gpt-4o` | 高 |

### 分块配置

```bash
# 分块策略
CHUNK_STRATEGY=recursive      # recursive / llm
CHUNK_SIZE=512                # 目标 chunk 大小
CHUNK_OVERLAP=50              # 相邻 chunk 重叠字符数
```

### 数据库切换（开发 vs 生产）

```bash
# 开发：SQLite（零配置）
METADATA_DB_URL=sqlite+aiosqlite:///./data/adaptive_rag.db

# 生产：PostgreSQL
METADATA_DB_URL=postgresql+asyncpg://rag:rag_password@localhost:5432/adaptive_rag
```

---

## 项目结构

```
adaptive-rag/
├── src/adaptive_rag/
│   ├── api/                      # FastAPI REST 接口 + 前端
│   │   ├── main.py               # 应用工厂，服务生命周期管理
│   │   ├── routers/
│   │   │   ├── query.py          # 查询接口
│   │   │   ├── documents.py      # 文档上传/列表/详情/删除
│   │   │   ├── admin.py          # 手动触发迁移
│   │   │   └── health.py         # 健康/就绪探针
│   │   ├── schemas/
│   │   │   ├── query.py          # 查询请求/响应模型
│   │   │   └── document.py       # 文档请求/响应模型
│   │   └── static/
│   │       └── index.html        # 可视化前端
│   ├── core/                     # 基础设施
│   │   ├── config.py             # Pydantic Settings 配置管理
│   │   ├── logging.py            # structlog 结构化日志
│   │   ├── exceptions.py         # 自定义异常层级
│   │   └── llm_client.py         # 统一 LLM 客户端（OpenAI/Anthropic 格式）
│   ├── storage/                  # 存储层
│   │   ├── vector_store/         # Qdrant 向量存储（本地/远程）
│   │   ├── metadata_store/       # SQLAlchemy 元数据（PostgreSQL/SQLite）
│   │   ├── document_store/       # 本地文件存储（UUID 分片）
│   │   └── cache/                # 内存/Redis 缓存
│   ├── ingestion/                # 文档摄入流水线
│   │   ├── chunker.py            # 分块：递归 / 固定 / LLM 语义
│   │   ├── embedder.py           # 嵌入生成 + 2000 条 LRU 缓存
│   │   ├── pipeline.py           # 摄入编排：分流 -> 存储 -> 容量检查
│   │   └── extractors/           # PDF / DOCX / 图片 / 文本提取
│   ├── tiers/                    # 热冷分层核心
│   │   ├── hot_tier.py           # 热层：原文 + 完整嵌入
│   │   ├── cold_tier.py          # 冷层：压缩摘要 + 摘要嵌入 / 原始文本
│   │   ├── compression.py        # LLM 压缩引擎（单条 + 批量 ~10x）
│   │   └── decompression.py      # LLM 解压引擎 + cosine 质量验证
│   ├── frequency/                # 频率追踪
│   │   ├── tracker.py            # 频率分数计算与批量查询
│   │   ├── decay.py              # 时间衰减引擎
│   │   └── clustering.py         # 查询语义聚类
│   ├── retrieval/                # 检索层
│   │   ├── router.py             # 频率驱动路由 + fire-and-forget 记录
│   │   ├── ranker.py             # 热层加权 + 去重 + 截断
│   │   └── retriever.py          # 统一检索入口 + 5秒 TTL 查询缓存
│   └── migration/                # 冷热迁移
│       ├── engine.py             # 迁移引擎：批量压缩 + 非高峰限制
│       ├── policies.py           # 阈值策略
│       └── scheduler.py          # 定时调度器
├── scripts/
│   └── init_db.py                # 数据库初始化脚本
├── tests/                        # 单元测试 + 集成测试
├── pyproject.toml                # 项目依赖
├── docker-compose.yml            # 生产基础设施（Qdrant + PostgreSQL + Redis）
└── README.md                     # 本文件
```

---

## Docker 部署（生产）

```bash
# 启动基础设施
docker-compose up -d

# 启动 API
uvicorn adaptive_rag.api.main:app --host 0.0.0.0 --port 8000
```

`docker-compose.yml` 包含：
- Qdrant（向量数据库）
- PostgreSQL（元数据存储）
- Redis（分布式缓存）

---

## 图片 OCR 支持

如需支持图片文字提取，需安装 Tesseract OCR：

```bash
# macOS
brew install tesseract tesseract-lang

# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim

# Windows
# 下载: https://github.com/UB-Mannheim/tesseract/wiki
# 安装后添加到 PATH
```

验证安装：

```bash
tesseract --version
```

---

## 相关概念

本项目实现了以下学术论文/概念中的思想：

- **Tiered Memory RAG** / **Hot-Cold Context Management**
- 帕累托分布优化：80% 查询集中在 20% 内容上
- 自组织知识库 (Self-organizing Knowledge Base)

---

## License

MIT License - 详见 [LICENSE](LICENSE)

---

<p align="center">
  Made with ❤️ for efficient RAG systems
</p>
