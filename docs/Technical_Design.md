# 技术设计文档：RAG Agent 记忆 + 知识库 + 自动评估能力

> 对应 PRD：`docs/PRD.md`  
> 版本：v0.2.0  
> 状态：草案

## 1. 设计目标

将现有 `main.py` 中的硬编码 RAG 流程拆分为可独立演进、可替换、可持久化的模块：

- **知识库（Knowledge Base）**：支持多源文档接入、持久化向量存储、增量更新。
- **记忆（Memory）**：短期会话上下文 + 长期用户事实/偏好，持久化并支持检索。
- **自动评估（Evaluation）**：基于 RAGAS 指标与规则，对每次问答打分并持久化。
- **编排（Agent）**：统一调度检索、记忆、生成、评估，保留离线降级能力。

## 2. 总体架构

```
┌─────────────────────────────────────────────────────────────┐
│                         Agent                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  ShortTerm  │  │  LongTerm   │  │     Evaluator       │  │
│  │  Memory     │  │  Memory     │  │  (RAGAS + Rules)    │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         │                │                    │             │
│         └────────────────┴────────────────────┘             │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Generation Layer                         │   │
│  │  Prompt = system + long-term memory + retrieved KB   │   │
│  │           + medium-term summary + short-term history │   │
│  │           + question                                 │   │
│  │  LLM generate with fallback                         │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Agentic Layer（可选）                     │   │
│  │  Query Router → Query Transformer → ReAct Loop       │   │
│  │  → Tool Invocation → Self-Correction                 │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Knowledge Base                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │   Loaders    │  │   Chunkers   │  │   Vector Store   │   │
│  │  txt/md/pdf  │  │  fixed/      │  │  local numpy/sql │   │
│  │  /url        │  │  recursive   │  │  optional remote │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 3. 模块设计

### 3.1 知识库模块（`rag_agent/knowledge/`）

#### 职责
- 加载多源文档
- 分块与向量化
- 持久化存储
- 检索（Dense / Hybrid / Metadata filter）

#### 核心类

```python
class Document:
    id: str
    content: str
    metadata: dict
    source: str

class BaseLoader(ABC):
    @abstractmethod
    def load(self, source: str) -> list[Document]: ...

class TextLoader(BaseLoader): ...
class MarkdownLoader(BaseLoader): ...
class PdfLoader(BaseLoader): ...
class UrlLoader(BaseLoader): ...

class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]: ...

class FixedSizeChunker(BaseChunker): ...
class RecursiveChunker(BaseChunker): ...

class Chunk:
    id: str
    text: str
    doc_id: str
    metadata: dict
    embedding: np.ndarray | None

class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk]): ...
    @abstractmethod
    def delete_by_doc(self, doc_id: str): ...
    @abstractmethod
    def search(self, query: str, top_k: int = 5, filters: dict | None = None) -> list[RetrievalResult]: ...
    @abstractmethod
    def persist(self): ...

class LocalVectorStore(VectorStore):
    """基于 numpy + JSON/SQLite 的本地向量库，零外部依赖"""

class KnowledgeBase:
    def __init__(self, store: VectorStore, chunker: BaseChunker, embedder: BaseEmbedder):
        ...

    def add_document(self, source: str, loader: str | BaseLoader = "auto", metadata: dict | None = None):
        """加载 → 分块 → 向量化 → 入库 → 持久化"""

    def remove_document(self, doc_id: str):
        """按 doc_id 删除文档及其全部 chunk"""

    def search(self, query: str, top_k: int = 5, filters: dict | None = None) -> list[RetrievalResult]:
        """支持元数据过滤"""
```

#### 持久化方案
- 默认：`LocalVectorStore` 使用 SQLite 存储 chunk 元数据 + 二进制文件存储归一化向量。
- 可选：通过环境变量切换 `MilvusVectorStore` / `ElasticsearchVectorStore`。
- 离线： embedding 降级到 `FallbackEmbedding`，向量库仍可工作。

#### 增量更新策略
1. 文档入库时生成 `doc_id = hash(source + 内容)`。
2. 若 `doc_id` 已存在：先删除旧 chunks，再插入新 chunks。
3. 提供 `rebuild()` 方法用于全量重建索引。

---

### 3.2 记忆模块（`rag_agent/memory/`）

#### 职责
- 短期记忆：保存当前会话的 N 轮对话，直接注入 prompt。
- 长期记忆：跨会话提取并检索用户事实/偏好。

#### 核心类

```python
class Message:
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime

class ShortTermMemory:
    def __init__(self, max_turns: int = 6):
        ...

    def add(self, role: str, content: str):
        ...

    def get_messages(self) -> list[Message]:
        """返回最近 N 轮，超出则自动丢弃"""

    def clear(self):
        ...

class MemoryFact:
    id: str
    user_id: str
    content: str
    source_turn: int  # 来源会话轮次
    created_at: datetime
    embedding: np.ndarray | None

class LongTermMemory:
    def __init__(self, store: VectorStore):
        """复用 VectorStore 存储事实 embedding"""

    def remember(self, user_id: str, fact: str, source_turn: int | None = None):
        """写入一条事实；自动去重（相似度阈值）"""

    def recall(self, user_id: str, query: str, top_k: int = 3) -> list[MemoryFact]:
        """检索与当前问题相关的用户事实"""

    def forget(self, user_id: str, fact_id: str):
        """显式删除某条记忆"""
```

#### 记忆写入策略
1. 每次对话后，Agent 调用 `MemoryExtractor`（基于规则 + 轻量 LLM prompt）从用户/助手消息中提取事实。
2. 生成 embedding，与已有事实计算余弦相似度。
3. 相似度 > 0.92：合并或跳过；相似度 0.80~0.92：更新旧事实；< 0.80：新增。
4. 支持 `max_facts_per_user` 上限，超出时按时间淘汰。

#### 短期记忆注入 prompt 的格式

```
[对话历史]
User: ...
Assistant: ...
User: {current_question}
```

#### 长期记忆注入 prompt 的格式

```
[用户相关信息]
- 用户偏好中文回答
- 用户是高级开发者
```

---

### 3.3 评估模块（`rag_agent/evaluation/`）

#### 职责
- 对每次问答进行多维度评分。
- 支持 RAGAS 指标与自定义规则。
- 持久化评估结果，支持失败案例聚合。

#### 核心类

```python
class EvaluationResult:
    question: str
    answer: str
    contexts: list[str]
    scores: dict[str, float]  # {"faithfulness": 0.85, "answer_relevance": 0.92}
    passed_rules: list[str]
    failed_rules: list[str]
    overall_score: float
    timestamp: datetime

class BaseMetric(ABC):
    @abstractmethod
    def score(self, question: str, answer: str, contexts: list[str]) -> float: ...

class FaithfulnessMetric(BaseMetric): ...
class AnswerRelevanceMetric(BaseMetric): ...
class ContextPrecisionMetric(BaseMetric): ...
class ContextRecallMetric(BaseMetric): ...

class RuleChecker:
    def check(self, question: str, answer: str, contexts: list[str]) -> tuple[list[str], list[str]]:
        """返回 (passed_rules, failed_rules)"""

class Evaluator:
    def __init__(self, metrics: list[BaseMetric], rules: RuleChecker | None = None):
        ...

    def evaluate(self, question: str, answer: str, contexts: list[str]) -> EvaluationResult:
        ...
```

#### 指标实现策略

| 指标 | 实现方式 | 离线 fallback |
|---|---|---|
| Faithfulness | LLM 判断 answer 中的每个 claim 是否能在 contexts 中找到支持 | 规则：答案中是否出现上下文未包含的实体/数字 |
| Answer Relevance | LLM 判断 answer 是否直接回应 question | 关键词重叠度 |
| Context Precision | LLM 判断每个 context 是否与问题相关 | 问题与上下文 TF-IDF 相似度 |
| Context Recall | 需要 ground truth，先不做自动计算，预留接口 | - |

#### 规则检查示例
- 答案为空
- 答案中出现 "我不知道" / "无法回答"
- 答案长度异常（过短/过长）
- 答案中包含敏感词

#### 评估结果持久化
- 使用 SQLite 表 `evaluations` 存储。
- 按 `overall_score < threshold` 标记为失败案例。
- 提供 `ReportGenerator` 导出最近 N 条低分记录。

---

### 3.4 编排模块（`rag_agent/agent.py`）

#### 职责
- 统一调度记忆、知识库、生成、评估。
- 保留离线降级与错误处理。

#### 核心类

```python
class AgentConfig:
    knowledge_base: KnowledgeBase
    short_term_memory: ShortTermMemory
    long_term_memory: LongTermMemory | None
    evaluator: Evaluator | None
    llm_client: OpenAI
    model: str
    fallback_enabled: bool = True

class Agent:
    def __init__(self, config: AgentConfig):
        ...

    def chat(self, user_id: str, question: str) -> ChatResponse:
        # -1. 语义缓存查找
        cached = self._lookup_semantic_cache(user_id, question)
        if cached is not None:
            return cached

        if self.config.agentic_enabled:
            # Agentic 模式：路由 → 查询改写 → ReAct 循环
            result = self._react_loop.run(user_id, question, history)
            answer = result.answer
            contexts = [r.text for r in result.contexts]
        else:
            # 高级 RAG 模式：查询改写 → 记忆召回 → 知识库检索 → LLM 生成
            search_query = self.config.query_transformer.transform(question, history)[0]
            long_term_facts = []
            if self.config.long_term_memory:
                long_term_facts = self.config.long_term_memory.recall(user_id, search_query)
            kb_results = self.config.knowledge_base.hybrid_search(search_query)
            contexts = [r.text for r in kb_results]
            messages = build_prompt(
                system="你是一个严谨的 RAG 助手...",
                long_term_facts=long_term_facts,
                medium_term_summary=self.config.medium_term_memory.get_summary(),
                short_term_history=self.config.short_term_memory.get_messages(),
                contexts=contexts,
                question=question,
            )
            try:
                answer = self._llm_generate(messages)
            except Exception as e:
                if self.config.fallback_enabled:
                    answer = self._fallback_generate(question, contexts)
                else:
                    raise

        # 短期记忆更新
        self.config.short_term_memory.add("user", question)
        self.config.short_term_memory.add("assistant", answer)

        # 长期记忆提取与写入
        if self.config.long_term_memory:
            facts = extract_facts(question, answer)
            for f in facts:
                self.config.long_term_memory.remember(user_id, f)

        # 自动评估
        eval_result = None
        if self.config.evaluator:
            eval_result = self.config.evaluator.evaluate(question, answer, contexts)
            self._persist_evaluation(eval_result)

        return ChatResponse(
            answer=answer,
            contexts=contexts,
            evaluation=eval_result,
        )

    def reset_session(self):
        self.config.short_term_memory.clear()
```

## 4. 数据模型

### 4.1 向量库表结构（SQLite）

```sql
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    metadata TEXT,  -- JSON
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    text TEXT NOT NULL,
    metadata TEXT,  -- JSON
    embedding BLOB, -- np.float32 binary
    FOREIGN KEY (doc_id) REFERENCES documents(id)
);

CREATE INDEX idx_chunks_doc_id ON chunks(doc_id);
```

### 4.2 评估结果表结构

```sql
CREATE TABLE evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    contexts TEXT,  -- JSON list
    scores TEXT,    -- JSON dict
    overall_score REAL,
    failed_rules TEXT,  -- JSON list
    is_failure BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_evaluations_failure ON evaluations(is_failure, created_at);
```

### 4.3 长期记忆表结构

```sql
CREATE TABLE memory_facts (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    content TEXT NOT NULL,
    source_turn INTEGER,
    embedding BLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_memory_facts_user ON memory_facts(user_id);
```

## 5. Prompt 模板

### 5.1 系统提示

```
你是一个严谨的 RAG 助手。请仅根据提供的参考资料和已知用户信息回答问题，不要编造参考资料之外的信息。
如果参考资料不足以回答问题，请明确说明。
```

### 5.2 完整 Prompt 结构

```
[系统提示]

[用户长期记忆]
- {fact_1}
- {fact_2}

[参考资料]
- {context_1}
- {context_2}

[对话历史]
User: {prev_question}
Assistant: {prev_answer}

User: {current_question}
Assistant:
```

### 5.3 事实提取 Prompt

```
请从以下对话中提取用户的关键事实或偏好（如语言偏好、技术栈、角色等）。
每条事实用一句话表达，不要推断不存在的信息。

User: {question}
Assistant: {answer}

事实列表（每行一条，无则返回空）：
```

## 6. 降级与容错

| 场景 | 行为 |
|---|---|
| embedding 模型下载失败 | 使用 `FallbackEmbedding`（字符随机投影） |
| LLM API 鉴权/网络失败 | `ResilientLLMClient` 指数退避重试 → 切换 fallback 模型 → 最终降级到 `MockLLMClient` |
| Agentic 中大模型生成失败 | `ReactLoop` 返回基于参考资料/工具结果的兜底回答 |
| Agentic 中查询改写失败 | 回退到原始问题继续检索 |
| Agentic 中路由失败 | `LLMQueryRouter` 自动降级到 `RuleBasedRouter` |
| 向量库文件损坏 | 启动时尝试重建索引，失败则初始化空库 |
| 评估 LLM 失败 | 使用规则评分兜底，不阻断主流程 |
| 长期记忆存储失败 | 降级为仅使用短期记忆，主流程继续 |
| 知识库检索失败 | 捕获异常，降级为空上下文继续生成 |

## 7. 依赖调整

在 `pyproject.toml` 中新增依赖：

```toml
[project]
dependencies = [
    "openai>=2.44.0",
    "sentence-transformers>=5.6.0",
    "numpy>=1.26.0",
    "requests>=2.32.0",       # URL 加载
    "pymupdf>=1.24.0",        # PDF 解析
    "tiktoken>=0.8.0",        # token 计数与摘要
    "ragas>=0.1.0",           # 自动评估（可选，优先自研指标）
]
```

> 注：为保持项目轻量，RAGAS 仅作为可选依赖；核心指标先自研实现，离线时可降级。

## 8. 目录结构

```
rag-agent/
├── main.py                         # 入口脚本（精简为示例）
├── pyproject.toml
├── docs/
│   ├── PRD.md
│   └── Technical_Design.md
├── rag_agent/
│   ├── __init__.py
│   ├── agent.py                    # Agent 编排
│   ├── agentic/                    # Agentic RAG：路由、ReAct、工具调用
│   │   ├── base.py
│   │   ├── router.py
│   │   ├── tools.py
│   │   ├── self_correction.py
│   │   └── react.py
│   ├── config.py                   # 配置加载
│   ├── embedder.py                 # Embedding 封装 + Fallback
│   ├── llm.py                      # LLM 客户端 + fallback
│   ├── resilience.py               # 重试、fallback、健康状态机
│   ├── knowledge/
│   │   ├── __init__.py
│   │   ├── base.py                 # Document / Chunk / VectorStore 抽象
│   │   ├── chunker.py              # 分块器
│   │   ├── loader.py               # 文档加载器
│   │   ├── store.py                # LocalVectorStore
│   │   └── kb.py                   # KnowledgeBase
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── short_term.py           # 短期记忆
│   │   ├── medium_term.py          # 中期会话摘要
│   │   ├── long_term.py            # 长期记忆
│   │   └── extractor.py            # 事实提取
│   ├── retrieval/                  # 查询改写
│   │   └── query_transform.py
│   ├── cache/                      # 语义缓存
│   │   └── semantic_cache.py
│   └── evaluation/
│       ├── __init__.py
│       ├── base.py                 # 评估抽象
│       ├── metrics.py              # 指标实现
│       ├── rules.py                # 规则检查
│       └── report.py               # 报告生成
└── data/
    ├── kb/                         # 知识库持久化数据
    ├── memory/                     # 长期记忆数据
    └── eval/                       # 评估结果数据
```

## 9. 实现路线图

| 阶段 | 任务 | 产出 |
|---|---|---|
| Phase 1 | 重构知识库模块 | `KnowledgeBase` + `LocalVectorStore` + 多格式加载 |
| Phase 2 | 引入记忆模块 | `ShortTermMemory` + `LongTermMemory` |
| Phase 3 | 引入评估模块 | `Evaluator` + 自研 RAGAS 指标 + 持久化 |
| Phase 4 | Agent 编排与 CLI | `Agent.chat()` + 配置化 + 离线降级验证 |
| Phase 5 | Agentic RAG 与工具调用 | `rag_agent/agentic/` + 查询路由 + ReAct + 自我修正 |
| Phase 6 | 反馈闭环 | 失败案例报告、检索策略调优 |

## 10. 测试策略

### 10.1 单元测试
- `LocalVectorStore`：增删查、持久化、元数据过滤
- `Chunker`：分块边界、overlap
- `Evaluator`：各指标在已知样本上的分数符合预期
- `Memory`：短期记忆轮次淘汰、长期记忆去重

### 10.2 集成测试
- 完整 `Agent.chat()` 流程：提问 → 检索 → 生成 → 评估
- 离线模式：断开网络后仍能跑通
- 增量更新：同文档修改后向量库正确更新

### 10.3 验收测试
- 多轮对话能引用历史上下文
- 长期记忆跨会话生效
- 低分回答被标记并生成报告

## 11. 关键配置项

| 配置项 | 环境变量 | 默认值 | 说明 |
|---|---|---|---|
| 向量库路径 | `KB_STORE_PATH` | `data/kb` | 知识库持久化目录 |
| 记忆库路径 | `MEMORY_STORE_PATH` | `data/memory` | 长期记忆目录 |
| 评估库路径 | `EVAL_STORE_PATH` | `data/eval/evaluations.db` | 评估 SQLite 路径 |
| 最大短期记忆轮数 | `MAX_SHORT_TERM_TURNS` | `6` | 保留最近 6 轮 |
| 单用户最大长期记忆数 | `MAX_LONG_TERM_FACTS` | `100` | 超出则淘汰旧记忆 |
| 事实去重阈值 | `MEMORY_DEDUP_THRESHOLD` | `0.92` | 余弦相似度 |
| 失败案例阈值 | `EVAL_FAILURE_THRESHOLD` | `0.6` | overall_score 低于此值标记 |
| 离线模式 | `RAG_OFFLINE` | `false` | 强制使用降级策略 |
