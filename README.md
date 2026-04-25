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

> 将操作系统的"热数据/冷数据"分层概念引入 RAG，让知识库根据访问频率自动优化存储结构。

---

## 核心特性

- **热冷分层存储**
  - **Hot Tier**：高频 chunk 保持原文存储，低延迟检索 (< 100ms)
  - **Cold Tier**：低频 chunk 经 LLM 压缩为摘要，节省 60%~80% 存储

- **频率驱动路由**
  - 根据查询话题频率自动路由到 Hot/Cold/Both 层级
  - 高频话题 → Hot Only，低频话题 → Cold Only，未知话题 → Both

- **语义查询聚类**
  - "苹果公司营收"和"Apple revenue"自动归为同一话题
  - 基于 embedding 相似度的动态聚类 (cosine ≥ 0.85)

- **时间衰减算法**
  - 指数衰减，半衰期 24 小时
  - 近期访问权重更高，旧访问自动 fade out

- **自动冷热迁移**
  - 基于频率阈值自动触发 Hot↔Cold 迁移
  - 支持定时任务 + 手动触发

- **多格式文档支持**
  - 文本 (.txt, .md)、PDF (.pdf)、Word (.docx)、图片 OCR (.png, .jpg)

- **多模型兼容**
  - 嵌入模型：OpenAI / sentence-transformers (免费本地)
  - LLM：OpenAI / DeepSeek / 通义千问 / Kimi / 智谱 (任意兼容 OpenAI/Anthropic 格式)

- **可视化前端**
  - 内置 Web UI：文档上传、智能查询、系统监控、实时日志

---

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                        用户查询                              │
└──────────────────────┬──────────────────────────────────────┘
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
              │  结果合并+重排   │
              │  ResultRanker  │
              └───────┬───────┘
                      │
              ┌───────▼───────┐
              │  LLM 生成答案  │
              └───────┬───────┘
                      │
              ┌───────▼───────┐
              │ 更新频率计数器 │ ← 记录访问，触发冷热迁移
              └───────────────┘
```

### 数据流

```
新文档上传 → Hot Tier (frequency_score=1.0)
                ↓
    ┌───────────────────────┐
    │  定时迁移任务 (每小时)  │
    │  - Hot→Cold: 压缩低频  │
    │  - Cold→Hot: 解压高频  │
    └───────────────────────┘
                ↓
用户查询 → FrequencyRouter → Hot/Cold/Both → 返回结果
                ↓
    异步更新频率分数 (fire-and-forget)
```

---

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| API 框架 | FastAPI + Uvicorn | 异步高性能，自动 OpenAPI 文档 |
| 向量数据库 | Qdrant (本地模式) | 无需 Docker，文件级持久化 |
| 元数据存储 | SQLite + SQLAlchemy | 零配置，开发友好 |
| 缓存 | 内存缓存 / Redis | 支持本地和分布式 |
| 嵌入模型 | sentence-transformers | all-MiniLM-L6-v2 (免费，384维) |
| LLM | 任意兼容 OpenAI/Anthropic | Kimi / DeepSeek / 通义千问 / OpenAI |
| 前端 | 纯 HTML/CSS/JS | 单文件，零依赖 |

---

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/adaptive-rag.git
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
# 初始化数据库
python scripts/init_db.py

# 启动 API 服务
uvicorn adaptive_rag.api.main:app --host 0.0.0.0 --port 8000
```

### 5. 打开前端

浏览器访问 http://localhost:8000

---

## 前端界面

内置可视化前端，无需额外安装：

| 功能 | 说明 |
|------|------|
| 📄 文档上传 | 支持 txt/md/pdf/docx/png/jpg，自动提取文本 |
| 🔍 智能查询 | 输入问题，系统自动路由到热层/冷层 |
| 📊 系统监控 | 实时显示 Hot/Cold chunk 数量、查询次数 |
| 🔄 手动迁移 | 一键触发冷热数据迁移 |
| 📝 系统日志 | 实时记录所有操作和错误 |

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

### 文档上传

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@document.pdf" \
  -F "title=我的文档"
```

### 智能查询

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是自适应RAG",
    "top_k": 5,
    "tier": null
  }'
```

响应示例：

```json
{
  "chunks": [
    {
      "chunk_id": "xxx",
      "content": "自适应RAG是一种新型的检索增强生成架构...",
      "score": 0.92,
      "tier": "hot",
      "is_decompressed": false,
      "access_count": 8,
      "frequency_score": 1.0
    }
  ],
  "routing_strategy": "hot_first",
  "hot_results_count": 3,
  "cold_results_count": 0,
  "total_latency_ms": 45.2,
  "topic_frequency": 0.85
}
```

### 手动触发迁移

```bash
curl -X POST "http://localhost:8000/api/v1/admin/migrate"
```

---

## 核心概念

### 频率分数计算

```
frequency_score = 0.4 * log(1 + access_count)
                + 0.3 * recency_decay
                + 0.3 * cluster_popularity
```

- **访问次数** (40%): 累计访问次数的对数
- **时间衰减** (30%): 指数衰减，半衰期 24 小时
- **话题热度** (30%): 查询聚类的聚合频率

### 路由策略

| 话题频率 | 路由策略 | 说明 |
|----------|----------|------|
| ≥ 0.7 | HOT_ONLY | 高频话题，只查热层 |
| ≤ 0.3 | HOT_FIRST | 低频/未知话题，先查热层 |
| 0.3~0.7 | BOTH | 中频话题，双层查询合并 |

### 冷热迁移阈值

| 方向 | 阈值 | 触发条件 |
|------|------|----------|
| Hot → Cold | frequency ≤ 0.3 | 低频 chunk 经 LLM 压缩为摘要 |
| Cold → Hot | frequency ≥ 0.7 | 高频 chunk 经 LLM 解压恢复原文 |

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

| 服务商 | Base URL | 压缩模型 | 解压模型 | 价格 |
|--------|----------|----------|----------|------|
| **Kimi Code** | `https://api.kimi.com/coding` | `kimi-for-coding` | `kimi-for-coding` | 中等 |
| **DeepSeek** | `https://api.deepseek.com/v1` | `deepseek-chat` | `deepseek-chat` | 最低 |
| **通义千问** | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-turbo` | `qwen-plus` | 低 |
| **OpenAI** | `https://api.openai.com/v1` | `gpt-4o-mini` | `gpt-4o` | 高 |

---

## 项目结构

```
adaptive-rag/
├── src/adaptive_rag/
│   ├── api/                  # FastAPI REST 接口 + 前端
│   │   ├── main.py           # 应用工厂
│   │   ├── routers/          # 路由: query, documents, admin, health
│   │   ├── schemas/          # Pydantic 模型
│   │   └── static/
│   │       └── index.html    # 可视化前端
│   ├── core/                 # 配置、日志、异常、LLM客户端
│   ├── storage/              # 存储层
│   │   ├── vector_store/     # Qdrant 向量存储
│   │   ├── metadata_store/   # SQLite 元数据
│   │   ├── document_store/   # 本地文件存储
│   │   └── cache/            # 内存/Redis 缓存
│   ├── ingestion/            # 文档摄入
│   │   ├── chunker.py        # 递归分块
│   │   ├── embedder.py       # 嵌入生成 (OpenAI / 本地)
│   │   ├── pipeline.py       # 摄入流程
│   │   └── extractors/       # PDF / DOCX / 图片 / 文本提取
│   ├── tiers/                # 热冷分层核心
│   │   ├── hot_tier.py       # 热层实现
│   │   ├── cold_tier.py      # 冷层实现
│   │   ├── compression.py    # LLM 压缩引擎
│   │   └── decompression.py  # LLM 解压引擎
│   ├── frequency/            # 频率追踪
│   │   ├── tracker.py        # 频率分数计算
│   │   ├── decay.py          # 时间衰减引擎
│   │   └── clustering.py     # 查询聚类
│   ├── retrieval/            # 检索路由
│   │   ├── router.py         # 频率驱动路由
│   │   └── ranker.py         # 结果融合重排
│   └── migration/            # 冷热迁移
│       ├── engine.py         # 迁移引擎
│       └── policies.py       # 阈值策略
├── scripts/
│   └── init_db.py            # 数据库初始化
├── pyproject.toml
├── docker-compose.yml        # Qdrant + PostgreSQL + Redis (可选)
└── README.md
```

---

## Docker 部署 (可选)

如需生产级部署（独立 Qdrant + PostgreSQL + Redis）：

```bash
# 启动基础设施
docker-compose up -d

# 初始化数据库
python scripts/init_db.py

# 启动 API
uvicorn adaptive_rag.api.main:app --host 0.0.0.0 --port 8000
```

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
