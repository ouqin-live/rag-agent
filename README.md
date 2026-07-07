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

## 技术栈

Python 3.14+ · sentence-transformers · Chroma · OpenAI SDK · pymupdf · uv
