# RAG Agent 架构说明

本文档描述 `rag-agent` 的整体架构、模块职责与核心数据流。

> 对应版本：v0.3.0+（LangGraph 全链路）

## 1. 总体架构

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│   API 层                                 │  ← rag_agent/api.py
│   FastAPI / REST / SSE                  │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│   Agent 编排层                           │  ← rag_agent/agent.py
│   LangGraph 状态图驱动全流程              │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│   LangGraph 工作流（rag_agent/graph/）   │
│                                         │
│   input_guardrail → cache_lookup        │
│   → route → transform_query             │
│   → retrieve → generate                 │
│   → output_guardrail → self_correction  │
│   → remember → evaluate                 │
└────────┬────────────────────────────────┘
         │
    ┌────┴────┬────────────┬────────────┐
    ▼         ▼            ▼            ▼
┌───────┐ ┌────────┐  ┌─────────┐  ┌──────────┐
│ 记忆层 │ │ 知识库层 │  │ 安全护栏 │  │ 评估层    │
└───────┘ └────────┘  └─────────┘  └──────────┘
```

Agent 的整个问答流程由一张 LangGraph 状态图驱动，每个步骤是一个节点，节点间通过边连接。图结构可配置、可观测、便于扩展。

## 2. 各层职责与对应文件

### 2.1 API 层

**文件**：`rag_agent/api.py`

基于 FastAPI 暴露 REST 服务：

- `POST /chat`：单轮/多轮对话
- `POST /chat/stream`：Server-Sent Events 流式对话
- `POST /documents`：文件上传或本地路径入库
- `DELETE /documents/{doc_id}`：删除文档
- `GET /memory/{user_id}`：查看用户长期记忆
- `GET /evaluations/reports`：失败案例报告

启动方式：

```bash
uv run python -m rag_agent.api
```

### 2.2 Agent 编排层

**文件**：`rag_agent/agent.py`

`Agent` 类负责创建 LangGraph 图并调度执行。核心方法：

| 方法 | 说明 |
|---|---|
| `Agent.chat(user_id, question)` | 同步单轮对话 |
| `Agent.achat(user_id, question)` | 异步单轮对话 |
| `Agent.achat_stream(user_id, question)` | 异步流式生成 |

每次调用将用户问题注入 LangGraph 状态图，图自动执行完整链路。图执行完毕后，Agent 做后处理：更新短期记忆、写入语义缓存。

### 2.3 LangGraph 工作流

**文件**：`rag_agent/graph/`

状态图是 Agent 的核心引擎，定义了 10 个节点：

| 节点 | 文件 | 职责 |
|---|---|---|
| `input_guardrail` | `nodes.py` | 输入护栏：Prompt Injection + PII 检测 |
| `cache_lookup` | `nodes.py` | 语义缓存查询，命中时短接结束 |
| `route` | `nodes.py` | 查询路由：选择 KB / LTM / 工具 |
| `transform_query` | `nodes.py` | Query Rewriting：指代消解与标准化 |
| `retrieve` | `nodes.py` | 检索：KB 混合检索 + LTM 召回 + 工具调用 |
| `generate` | `nodes.py` | LLM 生成答案 |
| `output_guardrail` | `nodes.py` | 输出护栏：毒性审核 |
| `self_correction` | `nodes.py` | 自我修正：Faithfulness 评估 + query 重写 |
| `remember` | `nodes.py` | 提取事实并写入长期记忆 |
| `evaluate` | `nodes.py` | 评估打分并持久化 |

**状态定义**：[`state.py`](../rag_agent/graph/state.py) 用 TypedDict 定义 GraphState，包含路由决策、检索结果、生成答案、修正状态、缓存标记、评估结果等字段。

**图组装**：[`graph.py`](../rag_agent/graph/graph.py) 用 `StateGraph` 将节点和边组装成可执行图。

完整图结构：

```
START
  → input_guardrail ──(blocked)→ END
  → cache_lookup ──(hit)→ END
  → route → transform_query → retrieve → generate
  → output_guardrail → self_correction
    ├──(needs correction)→ retrieve（循环）
    └──(done)→ remember → evaluate → END
```

### 2.4 知识库层

**文件**：`rag_agent/knowledge/`

| 组件 | 文件 | 职责 |
|---|---|---|
| 抽象层 | `base.py` | `Document` / `Chunk` / `VectorStore` 接口 |
| 加载器 | `loader.py` | 支持 txt、md、pdf、url 的多格式加载 |
| 分块器 | `chunker.py` | `FixedSizeChunker` / `RecursiveChunker` / `SemanticChunker` |
| 重排序 | `reranker.py` | `EmbeddingReranker` / `CrossEncoderReranker` |
| 向量存储 | `chroma_store.py` | `ChromaVectorStore`：HNSW 索引，自动持久化 |
| 兼容存储 | `store.py` | `LocalVectorStore`：SQLite + numpy，零额外依赖 |
| 编排入口 | `kb.py` | `KnowledgeBase`：加载 → 分块 → embedding → 入库 → 检索 |

检索流程：

```
query
  │
  ├──► Dense 向量检索（Chroma HNSW）
  ├──► BM25 关键词检索
  ├──► RRF 融合排序
  └──► 可选 EmbeddingReranker 精排
```

### 2.5 记忆层

**文件**：`rag_agent/memory/`

三层记忆架构：

```
ShortTermMemory（当前会话最近 N 轮）
         │
         ▼ 超出限制时归档
MediumTermMemory（本次会话摘要）
         │
         ▼ 每轮自动提取
LongTermMemory（跨会话用户事实/偏好，Chroma 持久化）
```

| 组件 | 文件 | 职责 |
|---|---|---|
| 短期记忆 | `short_term.py` | 内存中保留最近 N 轮完整对话 |
| 中期记忆 | `medium_term.py` | 会话摘要，旧轮次超出限制时由 LLM 压缩 |
| 事实提取 | `extractor.py` | LLM + 规则双路提取用户偏好/事实 |
| 长期记忆 | `long_term.py` | 基于 Chroma 持久化用户事实 |

### 2.6 安全护栏

**文件**：`rag_agent/guardrails.py`

- **输入护栏**：Prompt Injection 检测（14 组正则）+ PII 检测与脱敏
- **输出护栏**：敏感内容/毒性审核（5 类敏感词）
- **置信度门控**：检索分数过低时主动记录警告

护栏已接入 LangGraph 图的 `input_guardrail` 和 `output_guardrail` 节点。默认 WARN 模式，可切换为硬拦截。

### 2.7 生成层

**文件**：`rag_agent/llm.py`

- `OpenAICompatibleClient`：兼容 OpenAI API
- `MockLLMClient`：无 LLM 服务时的确定性降级
- `ResilientLLMClient`：指数退避重试 + Fallback 模型切换（`rag_agent/resilience.py`）

### 2.8 评估层

**文件**：`rag_agent/evaluation/`

| 组件 | 文件 | 职责 |
|---|---|---|
| 抽象层 | `base.py` | `EvaluationResult` / `BaseMetric` |
| 指标 | `metrics.py` | Faithfulness / Answer Relevance / Context Precision |
| 规则 | `rules.py` | 空回答、长度、拒绝、敏感词、明显幻觉检查 |
| 评估器 | `evaluator.py` | 组合指标与规则，SQLite 持久化 |
| 报告 | `report.py` | 失败案例文本报告与 CSV 导出 |

评估已接入 LangGraph 图的 `evaluate_node`，每次回答后自动执行。

### 2.9 Embedding 层

**文件**：`rag_agent/embedder.py`

- `SentenceTransformerEmbedder`：优先加载 `BAAI/bge-small-zh-v1.5`
- `FallbackEmbedding`：离线降级，维度自动与真实模型对齐

### 2.10 配置层

**文件**：`rag_agent/config.py`

基于 Pydantic Settings 集中管理所有参数，支持 `.env` 文件或环境变量覆盖。

## 3. 核心数据流

### 3.1 一次对话的完整流程（图内）

```
用户提问 "RAG 是什么？"
   │
   ├──► input_guardrail：检查 Prompt Injection / PII
   │      └── 拦截 → 返回安全提示，流程结束
   │
   ├──► cache_lookup：查询语义缓存
   │      └── 命中 → 返回缓存答案，流程结束
   │
   ├──► route：QueryRouter 决定数据源
   │      └── use_knowledge_base=True
   │
   ├──► transform_query：QueryTransformer 改写
   │      └── "RAG 是什么？" → 保持不变（首轮无指代）
   │
   ├──► retrieve：并行检索
   │      ├── KnowledgeBase.hybrid_search → Top-K chunks
   │      └── LongTermMemory.recall → 用户相关事实
   │
   ├──► generate：LLM 基于上下文生成答案
   │
   ├──► output_guardrail：检查输出毒性
   │      └── 命中 → 替换为安全回答
   │
   ├──► self_correction：Faithfulness 评估
   │      ├── 分数不足 → 重写 query → 回到 retrieve（循环）
   │      └── 通过 → 继续
   │
   ├──► remember：提取本轮事实 → LongTermMemory.remember()
   │
   └──► evaluate：评估打分 → 写入 SQLite
```

### 3.2 图外后处理

```
图执行完毕
   │
   ├──► ShortTermMemory.add(user, question)
   ├──► ShortTermMemory.add(assistant, answer)
   ├──► 旧轮次超出限制 → MediumTermMemory.update()
   │
   └──► SemanticCache.store()：写入缓存供下次复用
```

## 4. 关键设计决策

| 决策 | 说明 |
|---|---|
| **LangGraph 统一编排** | 全流程由一张状态图驱动，节点可独立测试、可观测、可扩展 |
| **本地优先** | Chroma 本地向量库 + sentence-transformers 本地 embedding，离线可用 |
| **降级无处不在** | embedding、LLM、评估指标均可在离线时降级，保证核心流程不中断 |
| **模块解耦** | 知识库/记忆/评估/护栏均通过抽象接口注入图节点，便于替换 |
| **持久化** | 向量库、长期记忆、评估结果均持久化到 `data/` 目录，重启不丢失 |
| **配置外部化** | 所有可调参数通过 `rag_agent/config.py` + `.env` 管理 |

## 5. 运行入口

### 5.1 端到端验证

```bash
uv run python main.py
```

支持 `--cases` 参数选择性运行：`uv run python main.py --cases kb,memory,guardrails`

### 5.2 API 服务

```bash
uv run python -m rag_agent.api
```

服务启动后访问 `http://localhost:8000/docs` 查看 Swagger 文档。

### 5.3 图可视化

```bash
uv run python -m rag_agent.graph.graph
```

输出 LangGraph 状态图的 ASCII 和 Mermaid 两种视图。

## 6. 模块文档索引

| 模块 | 文档 |
|---|---|
| Agentic / LangGraph 工作流 | `docs/Agentic_Module.md` |
| 知识库 | `docs/Knowledge_Base.md` |
| 记忆系统 | `docs/Memory_Module.md` |
| 语义缓存 | `docs/Cache_Module.md` |
| 检索增强（Query Rewriting） | `docs/Retrieval_Module.md` |
| 安全护栏 | `docs/Guardrails_Module.md` |
| 自动评估 | `docs/Evaluation_Module.md` |
| 优化路线图 | `docs/Optimization_Roadmap.md` |
| LangGraph 迁移方案 | `docs/LangGraph_Migration_Plan.md` |
# RAG Agent 架构说明

本文档描述 `rag-agent` 的整体架构、模块职责与核心数据流。

> 对应版本：v0.2.0+

## 1. 总体架构

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│   API 层（可选）                          │  ← rag_agent/api.py
│   FastAPI / REST / SSE                  │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│   Agent 编排层                           │  ← rag_agent/agent.py
│  （语义缓存 → 路由 → 检索 → 生成 → 评估）  │
└────────┬────────────────────────────────┘
         │
    ┌────┴────┬────────────┬────────────┐
    ▼         ▼            ▼            ▼
┌───────┐ ┌────────┐  ┌─────────┐  ┌──────────┐
│ 记忆层 │ │ 知识库层 │  │ 生成层   │  │ 评估层    │
└───────┘ └────────┘  └─────────┘  └──────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│   Agentic 层（可选）                     │  ← rag_agent/agentic/
│  （查询路由 / ReAct / 工具调用）           │
└─────────────────────────────────────────┘
```

## 2. 各层职责与对应文件

### 2.1 API 层（可选）

**文件**：`rag_agent/api.py`

基于 FastAPI 暴露 REST 服务，支持：

- `POST /chat`：单轮/多轮对话
- `POST /chat/stream`：Server-Sent Events 流式对话
- `POST /documents`：文件上传或本地路径入库
- `DELETE /documents/{doc_id}`：删除文档
- `GET /memory/{user_id}`：查看用户长期记忆
- `GET /evaluations/reports`：失败案例报告

启动方式：

```bash
uv run python -m rag_agent.api
```

### 2.2 Agent 编排层

**文件**：`rag_agent/agent.py`

统一调度所有模块，提供同步与异步两种调用方式：

| 方法 | 说明 |
|---|---|
| `Agent.chat(user_id, question)` | 同步单轮对话 |
| `Agent.achat(user_id, question)` | 异步单轮对话 |
| `Agent.achat_stream(user_id, question)` | 异步流式生成 |

单次对话的完整流程（高级 RAG 模式，默认）：

1. **语义缓存查找**：命中则直接返回缓存答案
2. **查询改写**：`QueryTransformer` 进行指代消解与问题标准化
3. **长期记忆召回**：根据问题检索用户历史事实
4. **知识库混合检索**：Dense + BM25 + RRF 融合，可选 Cross-Encoder 精排
5. **Prompt 构建**：系统提示 + 用户事实 + 参考资料 + 中期摘要 + 短期对话历史
6. **LLM 生成**：优先调用真实 LLM，失败则模板降级
7. **短期记忆更新**：保留最近 N 轮对话，超限时归档到中期记忆
8. **事实提取与写入**：从本轮对话提取用户偏好/事实，存入长期记忆
9. **自动评估**：对回答打分并持久化

当开启 `AGENTIC_ENABLED=true` 时，步骤 2~6 替换为 Agentic 流程：

1. **查询路由**：`QueryRouter` 决定使用知识库、长期记忆、计算器、时间工具等
2. **查询改写**：`QueryTransformer` 进行指代消解与问题标准化
3. **ReAct 循环**：检索 → 生成 → 反思 →（修正并再次检索），最多迭代 `AGENTIC_MAX_ITERATIONS` 次
4. **工具调用**：根据路由结果调用计算器、时间等工具

Agentic 流程同样复用语义缓存、记忆、事实提取、评估等后处理环节。

### 2.3 Agentic 层

**文件**：`rag_agent/agentic/`

在原有高级 RAG 流程之上提供动态决策与自我修复能力：

| 组件 | 文件 | 职责 |
|---|---|---|
| 路由 | `router.py` | `RuleBasedRouter` / `LLMQueryRouter`，选择数据源/工具 |
| 工具 | `tools.py` | `CalculatorTool` / `DatetimeTool`，可扩展 |
| 自我修正 | `self_correction.py` | `SelfCorrector`，基于忠实度触发查询重写 |
| ReAct 循环 | `react.py` | `ReactLoop`，协调路由、检索、生成、反思 |

默认关闭，通过 `AGENTIC_ENABLED=true` 开启。详见 `docs/Agentic_Module.md`。

### 2.4 知识库层

**文件**：`rag_agent/knowledge/*.py`

| 组件 | 文件 | 职责 |
|---|---|---|
| 抽象层 | `base.py` | `Document` / `Chunk` / `VectorStore` 接口 |
| 加载器 | `loader.py` | 支持 txt、md、pdf、url 的多格式加载 |
| 分块器 | `chunker.py` | `FixedSizeChunker` / `RecursiveChunker` / `SemanticChunker` |
| 重排序 | `reranker.py` | `EmbeddingReranker` / `CrossEncoderReranker` |
| 向量存储 | `chroma_store.py` | `ChromaVectorStore`：HNSW 索引，自动持久化（默认） |
| 兼容存储 | `store.py` | `LocalVectorStore`：SQLite + numpy，零额外依赖 |
| 编排入口 | `kb.py` | `KnowledgeBase`：加载 → 分块 → embedding → 入库 → 检索 |

检索流程：

```
query
  │
  ├──► Dense 向量检索（Chroma HNSW）
  │
  ├──► BM25 关键词检索（内部维护）
  │
  ├──► RRF 融合排序
  │
  └──► 可选 Cross-Encoder / Embedding 精排
```

### 2.5 记忆层

**文件**：`rag_agent/memory/*.py`

| 组件 | 文件 | 职责 |
|---|---|---|
| 短期记忆 | `short_term.py` | 内存中保留最近 N 轮完整对话 |
| 中期记忆 | `medium_term.py` | 会话摘要，旧轮次超出限制时由 LLM 压缩 |
| 事实提取 | `extractor.py` | LLM + 规则双路提取用户偏好/事实 |
| 长期记忆 | `long_term.py` | 基于 Chroma 持久化用户事实，支持 remember/recall/forget |

三层记忆架构：

```
ShortTermMemory（当前会话最近 N 轮）
         │
         ▼ 超出限制时归档
MediumTermMemory（本次会话摘要）
         │
         ▼ 每轮自动提取
LongTermMemory（跨会话用户事实/偏好）
```

### 2.6 生成层

**文件**：`rag_agent/llm.py`

- `OpenAICompatibleClient`：兼容 OpenAI API，含 IdeaLab `success=False` 错误处理
- `MockLLMClient`：无 LLM 服务时的确定性降级
- 同步/异步/流式三种生成接口：`generate` / `agenerate` / `agenerate_stream`
- `Agent._fallback_generate`：模板化兜底回答

### 2.7 评估层

**文件**：`rag_agent/evaluation/*.py`

| 组件 | 文件 | 职责 |
|---|---|---|
| 抽象层 | `base.py` | `EvaluationResult` / `BaseMetric` |
| 指标 | `metrics.py` | Faithfulness / Answer Relevance / Context Precision（LLM + fallback） |
| 规则 | `rules.py` | 空回答、长度、拒绝、敏感词、明显幻觉检查 |
| 评估器 | `evaluator.py` | 组合指标与规则，按严重程度加权扣分，SQLite 持久化 |
| 报告 | `report.py` | 失败案例文本报告与 CSV 导出 |

### 2.8 Embedding 层

**文件**：`rag_agent/embedder.py`

- `SentenceTransformerEmbedder`：优先加载 `BAAI/bge-small-zh-v1.5`
- `FallbackEmbedding`：离线/无网络时的确定性字符随机投影
- **关键设计**：fallback 维度自动与真实模型维度对齐，避免维度不一致崩溃
- 支持异步编码 `aencode()`，通过线程池执行

### 2.9 配置层

**文件**：`rag_agent/config.py`

基于 Pydantic Settings 集中管理：

- 存储路径（KB / Memory / Eval）
- Embedding 模型与 LLM 参数
- Agent 行为参数（top_k、max_turns、system_prompt）
- 记忆策略阈值与评估阈值

所有参数均可通过 `.env` 文件或环境变量覆盖，运行前自动加载。

## 3. 核心数据流

### 3.1 一次对话的数据流

```python
agent.chat("u1", "RAG 是什么？")
# 或异步版本
await agent.achat("u1", "RAG 是什么？")
```

```
u1 的问题
   │
   ├──► SemanticCache.lookup("u1", "RAG 是什么？")
   │      └── 命中则直接返回缓存答案，跳过后续所有步骤
   │
   ├──► 若 AGENTIC_ENABLED=true：
   │      ├──► QueryRouter.route("RAG 是什么？", history)
   │      │      └── 返回 ["knowledge_base"]（或 calculator/datetime 等）
   │      │
   │      ├──► QueryTransformer.rewrite("RAG 是什么？", history)
   │      │      └── 返回改写后的检索 query
   │      │
   │      └──► ReactLoop.run：检索 → 生成 → 反思 →（修正并再次检索）
   │             └── 返回 answer
   │
   └──► 否则（高级 RAG 模式）：
          ├──► QueryTransformer.rewrite("RAG 是什么？", history)
          │      └── 返回改写后的检索 query（指代消解、口语化改写）
          │
          ├──► LongTermMemory.recall("u1", "RAG 是什么？")
          │      └── 返回 ["用户偏好中文回答", "用户喜欢简洁解释"]
          │
          ├──► KnowledgeBase.hybrid_search("RAG 是什么？")
          │      └── 返回 Top-K chunks（Dense + BM25 + RRF + 可选精排）
          │
          ├──► 构建 Prompt（系统提示 + 用户事实 + 中期摘要 + 参考资料 + 历史）
          │
          └──► LLM.generate(prompt) → answer
   │
   ├──► ShortTermMemory.add(user, question)
   └──► ShortTermMemory.add(assistant, answer)
   │
   ├──► 旧轮次超出限制 → MediumTermMemory.update()
   │
   ├──► MemoryExtractor.extract(question, answer)
   │      └── 提取新事实 → LongTermMemory.remember()
   │
   └──► Evaluator.evaluate(question, answer, contexts)
          └── 写入 SQLite
```

### 3.2 知识库入库数据流

```python
kb.add_document("docs/guide.md")
```

```
loader.load() → Document
   │
   ▼
chunker.chunk() → List[Chunk]
   │
   ▼
embedder.encode() → embeddings
   │
   ▼
ChromaVectorStore.upsert() → HNSW 索引 + 自动持久化
```

## 4. 关键设计决策

| 决策 | 说明 |
|---|---|
| **本地优先** | 默认使用 Chroma 本地向量库， sentence-transformers 本地 embedding，离线可用 |
| **降级无处不在** | embedding、LLM、评估指标均可在离线时降级，保证核心流程不中断 |
| **模块解耦** | KnowledgeBase / Memory / Evaluator / LLM 均通过抽象接口交互，便于替换 |
| **持久化** | 向量库、长期记忆、评估结果均持久化到 `data/` 目录，重启不丢失 |
| **增量更新** | 同一文档重新入库时，先删除旧 chunks 再插入新 chunks |
| **配置外部化** | 所有可调参数通过 `rag_agent/config.py` + `.env` 管理，避免硬编码 |
| **同步/异步并存** | 核心接口同时提供同步与异步版本，便于脚本与 Web 服务两种场景 |

## 5. 运行入口

### 5.1 端到端验证脚本

```bash
uv run python main.py
```

`main.py` 演示：知识库加载、用户偏好表达、多轮对话、跨会话长期记忆召回、自动评估打分、失败案例报告生成。

### 5.2 API 服务

```bash
uv run python -m rag_agent.api
```

服务启动后访问 `http://localhost:8000/docs` 查看自动生成的 Swagger 文档。

## 6. 常见扩展方向

| 扩展方向 | 当前状态 | 说明 |
|---|---|---|
| 接入真实 LLM | ✅ 已支持 | 配置 `AI_STUDIO_TOKEN` / `OPENAI_API_KEY` 即可 |
| 混合检索（BM25 + Dense） | ✅ 已支持 | `KnowledgeBase.hybrid_search()` |
| 暴露 HTTP API（FastAPI） | ✅ 已支持 | `rag_agent/api.py` |
| 重排序模型 | ✅ 已支持 | `EmbeddingReranker` / `CrossEncoderReranker` |
| 语义分块 | ✅ 已支持 | `SemanticChunker` |
| 中期记忆摘要 | ✅ 已支持 | `MediumTermMemory` |
| Query 改写 | ✅ 已支持 | `RewritingTransformer`，解决指代消解与口语化 |
| 语义缓存 | ✅ 已支持 | `SemanticCache`，按意图相似度复用答案，跳过 LLM |
| 异步与流式 | ✅ 已支持 | `achat` / `achat_stream` |
| 统一配置管理 | ✅ 已支持 | `rag_agent/config.py` |
| Agentic RAG / 工具调用 | ✅ 已支持 | ReAct / self-reflection / 查询路由 / 工具调用，见 `docs/Agentic_Module.md` |
| HyDE / Multi-Query / Step-back | ⏳ 待实现 | 见 `docs/Optimization_Roadmap.md` |
| 缓存层 | ⏳ 待实现 | embedding / 检索结果 / 响应缓存 |
| 可观测性 | ⏳ 待实现 | tracing / metrics |
