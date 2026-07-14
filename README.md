# RAG Agent

具备记忆 + 知识库 + 自动评估能力的 RAG（检索增强生成）Agent。

## 架构

```
User Query → Agent 编排层
                ├── 记忆层（短期会话 + 长期用户事实）
                ├── 知识库层（文档加载 → 分块 → Chroma 向量库）
                ├── 生成层（OpenAI 兼容 LLM，支持降级）
                └── 评估层（Faithfulness / Answer Relevance / Context Precision）
```

## 快速开始

```bash
# 1. 安装依赖
uv sync

# 2. 配置 LLM（复制模板并填写你的 token）
cp .env.example .env
# 编辑 .env，填入 AI_STUDIO_TOKEN

# 3. 运行端到端验证
uv run python main.py
```

## 运行测试

```bash
# 安装 dev 依赖并运行全部测试
uv sync --group dev
uv run pytest tests/ -v
```

测试使用 `MockLLMClient` 和内存/临时向量库，无需配置真实 LLM 即可离线运行。

## 启动 API 服务

```bash
# 启动 FastAPI 服务（默认端口 8000）
uv run python -m rag_agent.api
```

核心端点：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| POST | `/chat` | 单轮对话 |
| POST | `/chat/stream` | 流式对话（SSE） |
| POST | `/documents` | 上传/添加文档 |
| DELETE | `/documents/{doc_id}` | 删除文档 |
| GET | `/memory/{user_id}` | 查看用户长期记忆 |
| GET | `/evaluations/reports` | 失败案例报告 |

示例：

```bash
# 上传文档
curl -X POST "http://localhost:8000/documents" \
  -F "file=@docs/guide.md" \
  -F 'metadata={"tag":"help"}'

# 对话
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u1","question":"RAG 是什么？"}'

# 流式对话
curl -N -X POST "http://localhost:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u1","question":"RAG 是什么？"}'
```

## 配置

在 `.env` 文件中配置 LLM 接入：

```env
AI_STUDIO_TOKEN=your_token_here
OPENAI_BASE_URL=https://idealab.alibaba-inc.com/api/openai/v1
OPENAI_MODEL=qwen3.6-plus
```

更多配置项见 `rag_agent/config.py`，例如：

```env
# 存储路径
KB_STORE_PATH=data/kb
MEMORY_STORE_PATH=data/memory
EVAL_DB_PATH=data/eval/evaluations.db

# Agent 行为
AGENT_TOP_K=5
AGENT_MAX_TURNS=6

# 记忆策略
MEMORY_DEDUP_THRESHOLD=0.92
MEMORY_MAX_FACTS_PER_USER=100

# 评估阈值
EVAL_FAILURE_THRESHOLD=0.6
```

不配置 LLM 也可运行，会自动降级为 Mock 模式。

## 模块说明

| 模块 | 目录 | 功能 |
|---|---|---|
| 配置 | `rag_agent/config.py` | Pydantic Settings 集中管理所有环境变量与默认参数 |
| 知识库 | `rag_agent/knowledge/` | 多格式文档加载（txt/md/pdf/url）、分块、Chroma 向量存储与语义检索 |
| 记忆 | `rag_agent/memory/` | 短期会话记忆 + 长期用户事实（向量库存储，支持去重与容量限制） |
| 评估 | `rag_agent/evaluation/` | RAGAS 风格指标（Faithfulness / Relevance / Precision）+ 规则检查 |
| 检索增强 | `rag_agent/retrieval/` | Query 改写（指代消解 + 口语化），支持自定义 Transformer |
| 缓存 | `rag_agent/cache/` | Semantic Cache，按 query 意图相似度复用答案，跳过 LLM |
| Agent | `rag_agent/agent.py` | 编排记忆 → 知识库 → LLM → 评估的全链路，支持同步/异步/流式 |
| LLM | `rag_agent/llm.py` | OpenAI 兼容客户端，支持 .env 配置、Mock 降级、异步与流式 |
| API | `rag_agent/api.py` | FastAPI 服务入口 |

## 文档

- [产品需求](docs/PRD.md)
- [技术设计](docs/Technical_Design.md)
- [整体架构](docs/Architecture.md)
- [知识库模块](docs/Knowledge_Base.md)
- [记忆模块](docs/Memory_Module.md)
- [评估模块](docs/Evaluation_Module.md)
- [检索增强模块](docs/Retrieval_Module.md)
- [缓存模块](docs/Cache_Module.md)
- [安全护栏模块](docs/Guardrails_Module.md)
- [优化路线图](docs/Optimization_Roadmap.md)

## 技术栈

Python 3.10+ · sentence-transformers · Chroma · OpenAI SDK · FastAPI · pydantic-settings · pymupdf · uv
# RAG Agent

具备记忆 + 知识库 + 自动评估能力的 RAG（检索增强生成）Agent。

## 架构

```
User Query → Agent 编排层
                ├── 记忆层（短期会话 + 长期用户事实）
                ├── 知识库层（文档加载 → 分块 → Chroma 向量库）
                ├── 生成层（OpenAI 兼容 LLM，支持降级）
                └── 评估层（Faithfulness / Answer Relevance / Context Precision）
```

## 快速开始

```bash
# 1. 安装依赖
uv sync

# 2. 配置 LLM（复制模板并填写你的 token）
cp .env.example .env
# 编辑 .env，填入 AI_STUDIO_TOKEN

# 3. 运行端到端验证
uv run python main.py
```

## 配置

在 `.env` 文件中配置 LLM 接入：

```env
AI_STUDIO_TOKEN=your_token_here
OPENAI_BASE_URL=https://idealab.alibaba-inc.com/api/openai/v1
OPENAI_MODEL=qwen3.6-plus
```

不配置 LLM 也可运行，会自动降级为 Mock 模式。

## 模块说明

| 模块 | 目录 | 功能 |
|---|---|---|
| 知识库 | `rag_agent/knowledge/` | 多格式文档加载（txt/md/pdf/url）、分块、Chroma 向量存储与语义检索 |
| 记忆 | `rag_agent/memory/` | 短期会话记忆 + 长期用户事实（向量库存储，支持去重与容量限制） |
| 评估 | `rag_agent/evaluation/` | RAGAS 风格指标（Faithfulness / Relevance / Precision）+ 规则检查 |
| Agent | `rag_agent/agent.py` | 编排记忆 → 知识库 → LLM → 评估的全链路 |
| LLM | `rag_agent/llm.py` | OpenAI 兼容客户端，支持 .env 配置与 Mock 降级 |

## 文档

- [产品需求](docs/PRD.md)
- [技术设计](docs/Technical_Design.md)
- [整体架构](docs/Architecture.md)
- [知识库模块](docs/Knowledge_Base.md)
- [记忆模块](docs/Memory_Module.md)
- [安全护栏模块](docs/Guardrails_Module.md)

## 技术栈

Python 3.14+ · sentence-transformers · Chroma · OpenAI SDK · pymupdf · uv
