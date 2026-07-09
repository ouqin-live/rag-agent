# 缓存模块架构说明与优化方向

## 1. 当前架构

缓存模块位于 `rag_agent/cache/`，通过语义相似度复用历史回答，跳过 retrieval + LLM 调用，降低成本和延迟。

```
User Query
   │
   ▼
┌─────────────────────────────┐
│   SemanticCache.lookup()    │
│   (query embedding 余弦比对) │
└────────┬────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
  命中        未命中
   │          │
跳过后续    走正常流程
直接返回    写入缓存
```

## 2. 各组件职责与对应文件

### 2.1 语义缓存 `semantic_cache.py`

**核心类**：`SemanticCache`

| 方法 | 职责 |
|---|---|
| `lookup(query, user_id)` | 查缓存，返回相似 query 的历史回答或 `None` |
| `store(query, user_id, answer, ...)` | 写入缓存 |
| `clear(user_id=None)` | 清空某用户或全部缓存 |

**关键设计**：

- **语义命中**：用 embedding 的余弦相似度判断意图相似性，不是字符串完全匹配
- **用户隔离**：每用户独立缓存桶，避免跨用户记忆泄露
- **容量控制**：每用户最多保留 `max_entries_per_user` 条（默认 100），超出淘汰最旧
- **TTL 过期**：支持 `ttl_seconds` 设置缓存有效期，`None` 表示永不过期
- **内存存储**：当前为进程内存，服务重启后清空

## 3. 核心数据流

### 3.1 缓存命中流程

```
用户提问 "RAG 究竟是什么？"
   │
   ▼
SemanticCache.lookup("RAG 究竟是什么？", user_id)
   │
   ├──► embedder.encode(query) → query_embedding
   │
   ├──► 遍历该用户缓存条目，计算与每个 query_embedding 的余弦相似度
   │      - "RAG 是什么？" → score=0.934 ← 命中！
   │
   ├──► 返回缓存的 answer、contexts、long_term_facts、evaluation
   │
   ├──► 更新 ShortTermMemory
   │
   └──► 返回 ChatResponse（cache_hit=True）
```

### 3.2 缓存写入流程

```
正常回答完成后
   │
   ▼
SemanticCache.store(
    query="RAG 是什么？",
    user_id=user_id,
    answer="RAG 是...",
    contexts=[...],
    long_term_facts=[...],
    evaluation=eval_result,
)
   │
   ▼
保存 query_embedding + 完整应答 → 下次查询可用
```

### 3.3 在 Agent 中的位置

```
用户提问
   │
   ├──► SemanticCache.lookup()     ← 第一步：查缓存
   │      ├── 命中 → 直接返回（跳过后 7 步）
   │      └── 未命中 → 继续
   │
   ├──► QueryTransformer
   ├──► LongTermMemory.recall()
   ├──► KnowledgeBase.hybrid_search()
   ├──► LLM.generate()
   ├──► ShortTermMemory.update()
   ├──► MemoryExtractor → LongTermMemory.remember()
   ├──► Evaluator.evaluate()
   └──► SemanticCache.store()      ← 最后一步：写入缓存
```

## 4. 配置项

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `SEMANTIC_CACHE_ENABLED` | `true` | 是否启用语义缓存 |
| `SEMANTIC_CACHE_THRESHOLD` | `0.92` | 余弦相似度命中阈值 |
| `SEMANTIC_CACHE_TTL_SECONDS` | `None` | 缓存 TTL，`None` 为永不过期 |

通过 `.env` 或 `rag_agent/config.py` 修改。

> 建议：如果知识库频繁更新，设置 `SEMANTIC_CACHE_TTL_SECONDS=3600` 确保缓存不会返回过期信息。

## 5. 缓存效果示例

```
User: RAG 是什么？
  → 正常走全程（retrieval + LLM + evaluation）

User: RAG 究竟是什么？
  ⚡ 缓存命中 (score=0.934, 原缓存: RAG 是什么？)
  → 直接返回，跳过 3 次 LLM 调用
```

## 6. 当前局限

### 6.1 内存存储

缓存只存在于进程中，服务重启丢失。适合单进程部署，不适合多 worker 共享。

### 6.2 单一缓存策略

当前只有语义缓存（query → 回答），缺少：

- **Embedding 缓存**：缓存 `embedder.encode(text)` 的结果。同一个 chunk 或 query 反复出现时避免重复计算向量。虽然 SemanticCache 复用了回答，但每次 lookup 仍然要 `encode(query)`，不会省这一步
- **检索结果缓存**：热门 query 每次都重新检索
- **LLM 响应缓存**：相同 prompt 重新调用

### 6.3 无相似度反馈

缓存命中后不做区分：相似度 0.93 和 0.99 的条目处理方式相同。可对低分命中做标记或降低置信度。

### 6.4 无知识库版本感知

知识库更新后，旧缓存可能包含过时的答案。当前的 TTL 机制只能靠时间窗口，不够精确。

### 6.5 缓存命中不提取新事实

命中时直接使用缓存的 `long_term_facts`，不重新提取。如果缓存条目中的事实与当前表述略有差异，不会更新。

## 7. 可优化方向

### 7.1 Embedding 缓存（文本 → 向量）

缓存 `embedder.encode(text)` 的结果。与 SemanticCache 不同：
- SemanticCache 缓存的是 **query → 完整回答**，命中时跳过 LLM + 检索
- Embedding 缓存缓存的是 **文本 → 向量**，命中时跳过 `encode()` 计算

用 LRU 缓存 key 为 `sha256(text)`，缓存编码后的向量：

```python
class EmbeddingCache:
    def __init__(self, maxsize=10000):
        self._cache = lru_cache(maxsize=maxsize)(self._encode)

    def get(self, text): ...
```

收益：
- 文档入库时相同 chunk 不重复编码
- SemanticCache lookup 时相同 query 不重复计算 embedding

### 7.2 检索结果缓存

key 为 `(user_id, query_embedding_hash, top_k)`，value 为 retrieval results，带 TTL。

注意：知识库更新时需要失效相关缓存。

### 7.3 持久化

将缓存条目持久化到 SQLite 或 Redis：

```python
class PersistentSemanticCache(SemanticCache):
    def __init__(self, db_path, ...):
        self._conn = sqlite3.connect(db_path)
        # 存储 embedding 为 binary blob
```

收益：服务重启后缓存不丢。

### 7.4 知识库版本感知

在知识库 `add_document` / `remove_document` 时生成版本号，缓存条目关联版本号，版本不匹配时失效。

### 7.5 分层缓存

冷热分离，高频 query 在内存，低频 query 在磁盘/Redis。参考 L1/L2 cache 设计。

### 7.6 统计与监控

记录缓存命中率、平均相似度分数，为优化阈值提供数据支撑。

## 8. 优先级建议

| 优化方向 | 优先级 | 说明 |
|---|---|---|
| Semantic Cache | ✅ 已落地 | 语义相似度命中，含 context 直接复用 |
| 检索结果缓存 | P2 | SemanticCache 已覆盖主路径；仅在同 query 不同 prompt 场景需要单独复用检索结果 |
| 持久化 | P2 | 服务重启缓存不丢 |
| 统计与监控 | P2 | 命中率、平均相似度 |
| 知识库版本感知 | P2 | 知识库更新时自动失效 |
| Embedding 缓存 | P2 | 需完全匹配，单用户场景收益低，多用户/批量入库时才有价值 |
| 分层缓存 | P3 | 大规模时才需要 |
