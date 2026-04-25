# 嵌入模型配置指南

本项目支持多种嵌入模型，从付费的 OpenAI API 到完全免费的本地模型。

## 快速选择

| 需求 | 推荐模型 | 维度 | 成本 |
|------|----------|------|------|
| 最快、最低资源 | all-MiniLM-L6-v2 | 384 | 免费 |
| 平衡质量/速度 | all-mpnet-base-v2 | 768 | 免费 |
| 中文场景 | bge-large-zh-v1.5 | 1024 | 免费 |
| 多语言+混合搜索 | BGE-M3 | 1024 | 免费 |
| 简单省事 | text-embedding-3-small | 1536 | 付费 |

## 使用免费的本地模型

### 1. 安装依赖

```bash
pip install -e ".[local]"
# 或者单独安装
pip install sentence-transformers
```

### 2. 修改 `.env` 配置

```bash
# 使用免费的本地模型
EMBEDDING_PROVIDER=sentence-transformers
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
LOCAL_EMBEDDING_DEVICE=cpu
```

### 3. 模型说明

#### all-MiniLM-L6-v2 (推荐入门)
- **维度**: 384
- **参数**: 22M
- **速度**: ~14,000 句/秒 (CPU)
- **特点**: 极快、资源占用低
- **适用**: 原型开发、高吞吐场景

```bash
EMBEDDING_DIMENSION=384
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

#### all-mpnet-base-v2 (推荐生产)
- **维度**: 768
- **参数**: 110M
- **速度**: ~4,000 句/秒 (CPU)
- **特点**: 质量更好，速度仍很快
- **适用**: 生产环境

```bash
EMBEDDING_DIMENSION=768
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
```

#### bge-large-zh-v1.5 (中文场景)
- **维度**: 1024
- **参数**: 326M
- **特点**: 中文优化，C-MTEB 排行榜第一
- **适用**: 中文知识库

```bash
EMBEDDING_DIMENSION=1024
LOCAL_EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
```

#### BGE-M3 (多语言高级)
- **维度**: 1024
- **参数**: 568M
- **特点**: 支持 100+ 语言，支持稀疏向量
- **适用**: 多语言混合知识库

```bash
EMBEDDING_DIMENSION=1024
LOCAL_EMBEDDING_MODEL=BAAI/bge-m3
```

### 4. 使用 GPU 加速 (可选)

如果有 NVIDIA GPU:

```bash
# 先安装 PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 配置使用 GPU
LOCAL_EMBEDDING_DEVICE=cuda
```

速度提升约 4-8 倍。

### 5. 首次运行

首次使用某个模型时会自动下载，下载后缓存到本地：

```
~/.cache/torch/sentence_transformers/
```

BGE-M3 约 2.2GB，all-MiniLM-L6-v2 仅约 80MB。

## 继续使用 OpenAI (默认)

```bash
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
LLM_API_KEY=sk-your-key
```

## 模型对比 (MTEB 排行榜)

| 模型 | 平均分数 | 检索分数 | 聚类分数 |
|------|----------|----------|----------|
| text-embedding-3-large | 64.6 | 55.4 | 49.0 |
| BGE-M3 | 63.2 | 55.1 | 46.9 |
| bge-large-zh-v1.5 | 64.5 | - | - |
| all-mpnet-base-v2 | 57.1 | 43.8 | 43.7 |
| all-MiniLM-L6-v2 | 51.1 | 38.1 | 39.2 |

**结论**: 免费模型中 BGE-M3 和 bge-large-zh 的质量接近甚至超过 OpenAI，且完全免费。
