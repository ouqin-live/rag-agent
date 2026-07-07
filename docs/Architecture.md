# RAG Agent 架构说明

本文档描述 `rag-agent` 的整体架构、模块职责与核心数据流。

## 1. 总体架构

```
User Query
    │
    ▼
┌─────────────────┐
│   Agent 编排层   │  ← rag_agent/agent.py
│  (记忆 → KB →    │
│   生成 → 评估)   │
└────────┬────────┘
         │
    ┌────┴────┬────────────┬────────────┐
    ▼         ▼            ▼            ▼
┌───────┐ ┌────────┐  ┌─────────┐  ┌──────────┐
│ 记忆层 │ │ 知识库层 │  │ 生成层   │  │ 评估层    │
└───────┘ └────────┘  └─────────┘  └──────────┘
```

## 2. 各层职责与对应文件

### 2.1 Agent 编排层

**文件**：`rag_agent/agent.py`

统一调度所有模块，单次 `chat(user_id, question)` 的完整流程：

1. **长期记忆召回**：根据问题检索用户历史事实
2. **知识库检索**：召回相关文档 chunks
3. **Prompt 构建**：系统提示 + 用户事实 + 参考资料 + 短期对话历史
4. **LLM 生成**：优先调用真实 LLM，失败则模板降级
5. **短期记忆更新**：保留最近 N 轮对话
6. **事实提取与写入**：从本轮对话提取用户偏好/事实，存入长期记忆
7. **自动评估**：对回答打分并持久化

### 2.2 知识库层

**文件**：`rag_agent/knowledge/*.py`

| 组件 | 文件 | 职责 |
|---|---|---|
| 抽象层 | `base.py` | `Document` / `Chunk` / `VectorStore` 接口 |
| 加载器 | `loader.py` | 支持 txt、md、pdf、url 的多格式加载 |
| 分块器 | `chunker.py` | `FixedSizeChunker`（句子边界） / `RecursiveChunker` |
| 向量存储 | `store.py` | `LocalVectorStore`：SQLite 存元数据 + numpy 存向量 |
| 编排入口 | `kb.py` | `KnowledgeBase`：加载 → 分块 → embedding → 入库 → 检索 |

### 2.3 记忆层

**文件**：`rag_agent/memory/*.py`

| 组件 | 文件 | 职责 |
|---|---|---|
| 短期记忆 | `short_term.py` | 内存中保留最近 N 轮对话，按轮次淘汰 |
| 事实提取 | `extractor.py` | 基于规则从对话中提取用户偏好/事实 |
| 长期记忆 | `long_term.py` | 复用 `LocalVectorStore` 持久化用户事实，支持 remember/recall/forget |

### 2.4 生成层

**文件**：`rag_agent/llm.py`

- `OpenAICompatibleClient`：兼容 OpenAI API，含 IdeaLab `success=False` 错误处理
- `MockLLMClient`：无 LLM 服务时的确定性降级
- `Agent._fallback_generate`：模板化兜底回答

### 2.5 评估层

**文件**：`rag_agent/evaluation/*.py`

| 组件 | 文件 | 职责 |
|---|---|---|
| 抽象层 | `base.py` | `EvaluationResult` / `BaseMetric` |
| 指标 | `metrics.py` | Faithfulness / Answer Relevance / Context Precision（LLM + fallback） |
| 规则 | `rules.py` | 空回答、长度、拒绝、敏感词、明显幻觉检查 |
| 评估器 | `evaluator.py` | 组合指标与规则，SQLite 持久化 |
| 报告 | `report.py` | 失败案例文本报告与 CSV 导出 |

### 2.6 Embedding 层

**文件**：`rag_agent/embedder.py`

- `SentenceTransformerEmbedder`：优先加载 `BAAI/bge-small-zh-v1.5`
- `FallbackEmbedding`：离线/无网络时的确定性字符随机投影
- **关键设计**：fallback 维度自动与真实模型维度对齐，避免维度不一致崩溃

## 3. 核心数据流

### 3.1 一次对话的数据流

```python
agent.chat("u1", "RAG 是什么？")
```

```
u1 的问题
   │
   ├──► LongTermMemory.recall("u1", "RAG 是什么？")
   │      └── 返回 ["用户偏好中文回答", "用户喜欢简洁解释"]
   │
   ├──► KnowledgeBase.search("RAG 是什么？")
   │      └── 返回 Top-K chunks
   │
   ├──► 构建 Prompt（系统提示 + 用户事实 + 参考资料 + 历史）
   │
   ├──► LLM.generate(prompt) → answer
   │
   ├──► ShortTermMemory.add(user, question)
   └──► ShortTermMemory.add(assistant, answer)
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
LocalVectorStore.add() → SQLite + numpy
```

## 4. 关键设计决策

| 决策 | 说明 |
|---|---|
| **本地优先** | 默认使用 SQLite + numpy，零外部向量数据库依赖，离线可用 |
| **降级无处不在** | embedding、LLM、评估指标均可在离线时降级，保证核心流程不中断 |
| **模块解耦** | KnowledgeBase / Memory / Evaluator / LLM 均通过抽象接口交互，便于替换 |
| **持久化** | 向量库、长期记忆、评估结果均持久化到 `data/` 目录，重启不丢失 |
| **增量更新** | 同一文档重新入库时，先删除旧 chunks 再插入新 chunks |

## 5. 运行入口

```bash
uv run python main.py
```

当前 `main.py` 是完整的端到端验证脚本，会依次演示：知识库加载、用户偏好表达、多轮对话、跨会话长期记忆召回、自动评估打分、失败案例报告生成。

## 6. 常见扩展方向

- 接入真实 LLM（配置 `OPENAI_API_KEY`）
- 添加混合检索（BM25 + Dense）
- 暴露 HTTP API（FastAPI）
- 接入更复杂的记忆摘要与事实提取
