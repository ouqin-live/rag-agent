# 检索增强模块架构说明与优化方向

## 1. 当前架构

检索增强模块位于 `rag_agent/retrieval/`，负责在检索前对用户问题进行预处理，提升召回质量。

```
User Query
   │
   ▼
┌─────────────────────────────┐
│   QueryTransformer 链       │
│  (Rewrite → [HyDE] →       │
│   [Multi-Query])            │
└────────┬────────────────────┘
         ▼
KnowledgeBase.hybrid_search(transformed_queries)
```

## 2. 各组件职责与对应文件

### 2.1 抽象层 `query_transform.py`

| 组件 | 职责 |
|---|---|
| `QueryTransformer` | 抽象基类，定义 `atransform(query, history) -> list[str]` |
| `IdentityTransformer` | 空操作实现，直接返回原 query |
| `RewritingTransformer` | 基于对话历史的指代消解与口语化改写 |

### 2.2 RewritingTransformer

**工作原理**：

1. 接收用户 query 和对话历史
2. 用 LLM 将指代词（"它"、"那个"）替换为上文的具体概念
3. 将口语化表达改写为标准检索句
4. 返回改写后的问题，用于向量检索

**关键设计**：

- 通过 `max_history_turns` 控制注入的上下文轮数（默认 3），控制 token 成本
- 改写失败时自动降级为原 query
- 改写结果与原 query 相同时不输出日志，避免噪声

**Prompt**：

```
请把以下用户问题改写成一个适合向量检索的标准问题。

要求：
1. 消除指代词（如"它"、"那个"、"这个"），用上文提到的具体概念替换。
2. 补充必要的上下文，但只基于对话历史，不要添加文档中没有的信息。
3. 保持原意，不要扩展问题范围。
4. 只返回改写后的问题，不要解释、不要输出多余内容。

对话历史：...
当前问题：那它怎么减少幻觉？
→ RAG（检索增强生成）是如何减少幻觉的？
```

## 3. 核心数据流

### 3.1 在 Agent 中的位置

```
用户提问
   │
   ├──► SemanticCache.lookup() ── 缓存命中则直接返回
   │
   ├──► QueryTransformer.transform(question, history)
   │      └── 返回改写后的检索 query 列表
   │
   ├──► LongTermMemory.recall(rewritten_query)
   │
   ├──► KnowledgeBase.hybrid_search(rewritten_query)
   │      └── 返回 Top-K chunks
   │
   └──► 后续生成与评估
```

### 3.2 同步与异步

- `QueryTransformer.transform()` — 同步，内部 `asyncio.run()` 调用异步 LLM
- `QueryTransformer.atransform()` — 异步，直接调用 `llm.agenerate()`
- Agent 的 `chat()` 走同步路径，`achat()` / `achat_stream()` 走异步路径

## 4. 配置项

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `QUERY_TRANSFORM_ENABLED` | `true` | 是否启用 query 改写 |
| `QUERY_TRANSFORM_MAX_HISTORY_TURNS` | `3` | 注入改写 prompt 的对话轮数 |

通过 `.env` 或 `rag_agent/config.py` 修改。

## 5. 改写效果示例

| 原问题 | 改写后检索 query |
|---|---|
| 那它怎么减少幻觉？ | RAG（检索增强生成）是如何减少幻觉的？ |
| RAG 是什么？ | 什么是RAG？ |
| 请解释 Faithfulness 指标，尽量简洁。 | Faithfulness 指标是什么？ |

## 6. 当前局限

### 6.1 改写策略单一

仅实现了 Query Rewriting，缺少：

- **HyDE**：LLM 先写假设答案，用答案做向量检索
- **Multi-Query**：一个问题生成多角度查询，合并结果
- **Step-back Prompting**：先检索上位概念，再检索细节

### 6.2 多 query 融合机制

当前 `transform()` 返回 `list[str]`，但 Agent 只取第一个 query 做检索。多 query 结果的去重、RRF 融合尚未实现。

### 6.3 一次性改写

每次检索都独立改写，不参考之前的改写结果。在多轮对话中，可能重复改写相同概念。

### 6.4 LLM 依赖

改写完全依赖 LLM，离线时退回 `IdentityTransformer`（不改写）。没有纯规则的降级改写方案。

## 7. 可优化方向

### 7.1 HyDE（Hypothetical Document Embedding）

让 LLM 先生成一段假设答案，再用假设答案做向量检索。适用于：

- 问题太短，embedding 信息量不足
- 用户用词与文档用词 gap 大

```python
class HydeTransformer(QueryTransformer):
    async def atransform(self, query, history=None):
        hypothetical = await self.llm.agenerate(
            "请根据你的知识，写一段回答：" + query
        )
        return [hypothetical]
```

### 7.2 Multi-Query Retrieval

一个问题生成 3 个角度查询：

```python
class MultiQueryTransformer(QueryTransformer):
    async def atransform(self, query, history=None):
        responses = await self.llm.agenerate(
            f"请从3个不同角度改写：{query}，返回JSON数组"
        )
        return json.loads(responses)
```

检索时合并结果并去重。

### 7.3 改写缓存

对近期改写过的 query 做缓存，避免重复调用 LLM。key 为 `(user_id, query)`。

### 7.4 规则改写降级

离线时用正则做基础的指代消解，减少对 LLM 的完全依赖。

### 7.5 条件改写

只在检测到指代词、口语化表达时改写，否则跳过。减少不必要的 LLM 调用。

检测规则示例：

- 包含 "它"、"那个"、"这个"、"这玩意儿"
- 问题长度 < 5 字符
- 以 "？" 结尾且无上下文（首轮对话）

## 8. 优先级建议

| 优化方向 | 优先级 | 说明 |
|---|---|---|
| Query Rewriting | ✅ 已落地 | 指代消解 + 口语化改写 |
| 条件改写 | P1 | 减少不必要的 LLM 调用 |
| HyDE | P1 | 对短 query 效果显著 |
| Multi-Query Retrieval | P2 | 需要结果融合机制配合 |
| 改写缓存 | P2 | 简单但有效 |
| 规则改写降级 | P2 | 离线可用 |
| Step-back Prompting | P3 | 需先用评估数据验证收益 |
