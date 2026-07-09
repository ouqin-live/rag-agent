# RAG Agent 优化路线图

> 状态：基于当前 v0.2.0 实现与市场主流 RAG/Agent 方案对齐分析  
> 目标：把当前“端到端验证脚本”逐步升级为可生产运行、可观测、可自我进化的 Agent 系统

---

## 1. 当前状态总览

### 1.1 已具备的优势

| 能力 | 实现位置 | 说明 |
|---|---|---|
| 混合检索 | `rag_agent/knowledge/kb.py` | Dense + BM25 + RRF 融合 |
| 重排序 | `rag_agent/knowledge/reranker.py` | EmbeddingReranker / CrossEncoderReranker |
| 语义分块 | `rag_agent/knowledge/chunker.py` | SemanticChunker 按话题边界切分 |
| 三层记忆 | `rag_agent/memory/` | 短期会话 + 中期摘要 + 长期事实 |
| LLM 事实提取 | `rag_agent/memory/extractor.py` | LLMMemoryExtractor，离线降级规则提取 |
| 自动评估 | `rag_agent/evaluation/` | Faithfulness / Relevance / Precision + 规则加权扣分 |
| 降级体系 | 全模块 | embedding、LLM、评估指标均支持 fallback |

### 1.2 关键差距

- 无测试体系，一个测试文件都没有
- 无服务化接口，只有 `main.py` 演示脚本
- 全同步调用，无异步 / 流式能力
- 缺少统一配置管理，大量阈值硬编码
- Python 版本要求 `>=3.14`，限制采用
- 无缓存、无可观测、无用户反馈闭环
- 缺少 Query 改写、Agentic 检索、工具调用等进阶能力

---

## 2. 优化任务清单（按优先级）

### P0 — 基础工程化（必须先做）

这些项决定项目能否从“脚本”变成“可维护、可部署的系统”。

#### P0-1 建立测试体系 ⏳ 待实现

- **目标**：为所有核心模块补单元测试和集成测试
- **范围**：
  - 单元测试：`chunker`、`LocalVectorStore`、`ChromaVectorStore`、`LongTermMemory` 去重、`Evaluator` 打分
  - 集成测试：完整 `Agent.chat()` 链路，覆盖真实 LLM / Mock 降级 / 离线模式
  - 回归测试：同文档增量更新、跨会话长期记忆召回
- **产出**：新增 `tests/` 目录，`pytest` 可一键运行

#### P0-2 提供服务化接口 ✅ 已实现

- **目标**：暴露 REST API，支持对话、文档管理、记忆查看、评估报告
- **实现文件**：`rag_agent/api.py`
- **已验证端点**：`/health`、`/chat`、`/chat/stream`、`/documents`、`/memory/{user_id}`、`/evaluations/reports`
- **建议实现**：基于 FastAPI
- **核心端点**：
  - `POST /chat` — 单轮/多轮对话
  - `POST /chat/completions` — OpenAI-compatible 接口（可选）
  - `POST /documents` — 文档入库
  - `DELETE /documents/{doc_id}` — 删除文档
  - `GET /memory/{user_id}` — 查看用户长期记忆
  - `GET /evaluations/reports` — 失败案例报告
- **产出**：新增 `rag_agent/api/` 或 `api.py`

#### P0-3 引入异步支持 ✅ 已实现

- **目标**：把 LLM 调用和 embedding 调用改为异步，提升并发能力
- **实现文件**：`rag_agent/llm.py`、`rag_agent/embedder.py`、`rag_agent/agent.py`
- **新增方法**：
  - `BaseLLMClient.agenerate()` / `agenerate_stream()`
  - `BaseEmbedder.aencode()`
  - `Agent.achat()` / `achat_stream()`
- **改造点**：
  - `BaseLLMClient` 增加 `agenerate()`
  - `BaseEmbedder` 增加 `aencode()`
  - `Agent` 增加 `achat()`
  - FastAPI 端点默认走异步链路
- **产出**：保留同步接口作为兼容层

#### P0-4 统一配置管理 ✅ 已实现

- **目标**：用 Pydantic Settings 集中管理所有环境变量与默认参数
- **实现文件**：`rag_agent/config.py`
- **已接入模块**：`embedder.py`、`llm.py`、`agent.py`、`short_term.py`、`long_term.py`、`evaluator.py`
- **覆盖项**：
  - 向量库/记忆/评估存储路径
  - embedding 模型名、LLM 模型名、温度
  - `top_k`、`max_turns`、`failure_threshold`、`dedup_threshold`
  - 重排序开关、离线模式开关
- **当前硬编码位置示例**：
  - `rag_agent/evaluation/evaluator.py` 中 `failure_threshold=0.6`
  - `rag_agent/memory/short_term.py` 中 `max_turns=6`
  - `rag_agent/agent.py` 中 `top_k=5`
- **产出**：新增 `rag_agent/config.py`

#### P0-5 放宽 Python 版本要求 ✅ 已实现

- **目标**：从 `>=3.14` 降到 `>=3.10`
- **实现文件**：`pyproject.toml`
- **说明**：代码使用 `from __future__ import annotations`，3.10 即可兼容现有类型注解语法
- **原因**：3.14 尚未普及，严重限制项目采用范围
- **产出**：更新 `pyproject.toml`

---

### P1 — 效果与体验提升（做完 P0 后推进）

这些项能显著改善回答质量、延迟、成本和用户体验。

#### P1-1 Query 理解与改写 ✅ 部分实现

- **目标**：在检索前对用户问题进行预处理，提升召回质量
- **已落地**：
  - **Query Rewriting**：基于对话历史的指代消解与口语化改写
  - 实现文件：`rag_agent/retrieval/query_transform.py`
  - 已接入：`Agent.chat()` / `achat()` / `achat_stream()` 在检索前自动调用
  - 配置项：`QUERY_TRANSFORM_ENABLED` / `QUERY_TRANSFORM_MAX_HISTORY_TURNS`
  - 效果示例：`"那它怎么减少幻觉？"` → `"RAG（检索增强生成）是如何减少幻觉的？"`
- **待实现**：
  - **HyDE**：让 LLM 生成假设答案，再用假设答案做向量检索
  - **Multi-Query Retrieval**：一个问题生成多角度查询，合并结果
  - **Step-back Prompting**：先检索抽象概念，再检索细节
- **接入点**：`Agent` 检索前统一调用 `QueryTransformer`
- **产出**：新增 `rag_agent/retrieval/query_transform.py`

#### P1-2 引入多级缓存（含 Semantic Cache）✅ 部分实现

- **目标**：降低重复 embedding 和 LLM 调用成本，相同意图问题直接命中
- **已落地**：
  - **Semantic Cache**：基于 query embedding 余弦相似度命中，命中时直接返回缓存答案，跳过 retrieval + LLM
  - 实现文件：`rag_agent/cache/semantic_cache.py`
  - 已接入：`Agent.chat()` / `achat()` / `achat_stream()` 在检索前优先查缓存，生成后自动写入缓存
  - 配置项：`SEMANTIC_CACHE_ENABLED`（默认 true）、`SEMANTIC_CACHE_THRESHOLD`（默认 0.92）、`SEMANTIC_CACHE_TTL_SECONDS`
  - 用户隔离：按 `user_id` 分桶，避免记忆泄露
  - 容量控制：每用户最多 100 条，超出淘汰最旧
- **待实现**：
  - **LLM 响应缓存**：相同 prompt 直接命中（对确定性问题有效）
  - 持久化：当前为内存缓存，可选 Redis / SQLite 持久化
  - **Embedding 缓存**：文本 → vector，需完全匹配，多用户场景才明显
  - **检索结果缓存**：SemanticCache 已含 context 复用，仅特殊场景需要
- **产出**：新增 `rag_agent/cache/`

#### P1-3 流式生成

- **目标**：让 LLM 输出逐字/逐句返回，提升交互体验
- **改造点**：
  - `BaseLLMClient` 增加 `generate_stream()`
  - FastAPI 端点返回 `StreamingResponse`（SSE）
- **产出**：`POST /chat/stream` 端点

#### P1-4 Parent Document Retrieval

- **目标**：检索用小 chunk 提升精度，生成时注入父级上下文避免断裂
- **实现方式**：
  - `Chunk` 增加 `parent_id` / `section_title` 字段
  - 检索命中 chunk 后，根据 `parent_id` 把同段落/父文档片段一起注入 prompt
- **产出**：扩展 `KnowledgeBase.search()` 或新增 `search_with_parent_context()`

#### P1-5 结构化输出与 Function Calling

- **目标**：让 LLM 输出可被程序稳定解析，支持工具调用
- **改造点**：
  - `BaseLLMClient.generate` 支持 `response_format={"type": "json_object"}`
  - 支持 `tools` 参数传入和 `tool_calls` 解析
- **应用场景**：事实提取、评估指标、Agent 工具调用
- **产出**：升级 `rag_agent/llm.py`

#### P1-6 用户反馈闭环

- **目标**：收集人工反馈，校准自动评估，驱动持续优化
- **改造点**：
  - `evaluations` 表新增 `user_feedback` 字段（👍 / 👎 / None）
  - 新增 `POST /evaluations/{id}/feedback` 接口
  - 失败案例报告按用户反馈加权
- **产出**：扩展 `rag_agent/evaluation/evaluator.py`

#### P1-7 可观测性建设（Trace 级别）

- **目标**：建立结构化日志、链路追踪和核心指标监控，定位到具体 Tool Call / 检索步骤的问题
- **内容**：
  - 结构化 JSON 日志替代纯文本日志
  - **Trace 级别链路追踪**：每次 `Agent.chat()` 生成一个 trace，每个子步骤生成 span
    - `retrieve.long_term_memory`
    - `retrieve.knowledge_base`
    - `rerank`
    - `llm.generate`
    - `memory.extract`
    - `evaluate`
  - 在 span 中记录：输入参数、输出摘要、耗时、token 用量、是否命中缓存、是否降级
  - **定位幻觉/超时**：通过 trace 可看到是哪一步引入上下文外信息（如 `llm.generate` 输入的 contexts 不足）或哪一步超时（如 `llm.generate` / `retrieve`）
  - Metrics：检索命中率、LLM 调用成功率、平均延迟、评估分数分布
- **可选工具**：
  - **LangSmith**：与 OpenAI SDK 集成最方便，适合快速落地
  - **Arize Phoenix**：开源，支持 LLM trace 与 evaluation 可视化
  - **OpenTelemetry**：最通用，可接入任意 APM（Jaeger / Grafana Tempo）
- **产出**：新增 `rag_agent/observability.py`

#### P1-8 容错与重试机制

- **目标**：提升 LLM 调用和工具调用的稳定性，单点失败时不中断主流程
- **内容**：
  - **指数退避重试**：对 LLM 网络超时 / 限流错误实现 `backoff` 重试（如 1s → 2s → 4s）
  - **Fallback LLM 切换**：主模型失败时自动切到备用模型（如主模型 `qwen3.6-plus` → 备用 `qwen-turbo` → Mock）
  - **降级状态机**：记录当前可用模型列表，健康检查失败后自动降级
  - **工具调用失败处理**：工具调用超时或返回错误时，Agent 可捕获异常并尝试修复或换工具
- **配置项**：
  - `LLM_MAX_RETRIES=3`
  - `LLM_RETRY_BACKOFF=2.0`
  - `LLM_FALLBACK_MODELS=qwen-turbo,gpt-3.5-turbo`
- **产出**：扩展 `rag_agent/llm.py`，新增 `rag_agent/resilience.py`
- **状态**：✅ 已完成
- **实现摘要**：
  - 新增 `rag_agent/resilience.py`：提供 `RetryConfig`、`with_retry`、
    `ModelHealthState`、`ResilientLLMClient`
  - 指数退避重试：默认 3 次重试，退避因子 2.0（1s → 2s → 4s），
    自动捕获 `APITimeoutError`、`RateLimitError`、`APIError` 等可重试异常
  - Fallback 模型切换：`get_llm_client()` 返回 `ResilientLLMClient`，
    主模型失败后依次尝试 `LLM_FALLBACK_MODELS` 中配置的备用模型，
    最终降级到 `MockLLMClient`
  - 健康状态机：`ModelHealthState` 记录每个模型的连续失败次数，
    超过 `LLM_HEALTH_FAILURE_THRESHOLD` 后自动跳过该模型；成功后重置计数
  - 检索容错：`Agent.chat/achat/achat_stream` 中 `hybrid_search` 失败时被捕获，
    降级为空上下文继续生成，避免单点失败中断主流程
  - 配置项已加入 `rag_agent/config.py` 和 `.env.example`

---

### P2 — 进阶 Agent 能力（长期演进）

这些项把项目从“高级 RAG”推向“Agent 系统”。

#### P2-1 Agentic RAG（含 Self-Correction Loop）

- **目标**：让 Agent 能主动判断是否需要补充检索、改写问题、使用工具，并在失败时自我修复
- **可选方向**：
  - **ReAct 循环**：检索 → 生成 → 反思 → 必要时再检索
  - **Self-RAG / Corrective RAG**：生成后自我评估，检索不足时修正
  - **Self-Correction Loop**：工具调用失败 / 答案质量低时，自动重写 query、换工具或补充检索
    - 例如：计算器返回错误 → 修正表达式再试一次
    - 例如：Faithfulness 分数低 → 重新检索更相关的上下文
  - **查询路由**：根据问题类型选择 KB、长期记忆、Web Search、计算器等
- **产出**：新增 `rag_agent/agentic/` 模块
- **状态**：✅ 已完成（基础能力落地并补齐与原流程的对齐）
- **实现摘要**：
  - 新增 `rag_agent/agentic/` 模块：
    - `base.py`：定义 `AgenticContext`、`BaseTool`、`ToolResult`
    - `router.py`：规则路由 `RuleBasedRouter` + LLM 路由 `LLMQueryRouter`，
      支持自动选择 `knowledge_base`、`long_term_memory`、`calculator`、`datetime`
    - `tools.py`：内置安全计算器 `CalculatorTool`、时间工具 `DatetimeTool`
    - `self_correction.py`：基于 `FaithfulnessMetric` 的 `SelfCorrector`，
      分数低于阈值时自动重写查询并补充检索；
      优先复用 `Evaluator` 中名为 `faithfulness` 的指标，保证与最终评估标准一致
    - `react.py`：`ReactLoop` 实现 路由 → 查询改写 → 检索 → 生成 → 反思 → 修正 循环
  - `Agent` 支持 `agentic_enabled` 模式：启用后走 ReAct / Self-Correction 流程；
    未启用时保持原有高级 RAG 流程不变
  - 与原高级 RAG 流程对齐：
    - Agentic 流程前置语义缓存命中，避免重复执行 ReAct 循环
    - `QueryTransformer` 传入原始 `Message` 列表，支持 `RewritingTransformer` 指代消解
    - Prompt 中注入中期记忆摘要
    - 大模型生成失败时返回基于参考资料/工具结果的兜底回答
    - `achat_stream` 支持 Agentic 模式（执行 ReAct 后模拟流式输出）
  - 新增模块文档 `docs/Agentic_Module.md`
  - 配置项已加入 `rag_agent/config.py` 和 `.env.example`：
    `AGENTIC_ENABLED`、`AGENTIC_MAX_ITERATIONS`、
    `AGENTIC_FAITHFULNESS_THRESHOLD`、`AGENTIC_USE_LLM_ROUTER`

#### P2-2 工具调用生态

- **目标**：让 Agent 不局限于知识库，可调用外部工具
- **候选工具**：
  - Web Search（Serper / DuckDuckGo）
  - 计算器 / Python 代码执行器
  - 日期/时间工具
  - 企业 API 查询工具
- **产出**：定义 `BaseTool` 接口，支持 OpenAI function calling 格式

#### P2-3 更强的安全与护栏

- **目标**：降低 prompt injection、PII 泄露、有害内容风险
- **内容**：
  - Prompt Injection 检测（轻量分类器或规则）
  - PII 检测与脱敏
  - 输出毒性/敏感内容审核
  - 检索置信度过低时主动拒绝回答
- **产出**：扩展 `rag_agent/evaluation/rules.py` 或新增 `rag_agent/guardrails.py`

#### P2-4 记忆系统进阶

- **目标**：让长期记忆更智能、更可靠
- **方向**：
  - **重要性评分**：高频事实 / 用户确认事实加权
  - **时间衰减**：旧事实召回 score 递减
  - **冲突检测**：用户前后矛盾时标记或取最新
  - **跨会话摘要持久化**：把 `MediumTermMemory` 摘要写入长期记忆
- **产出**：扩展 `rag_agent/memory/long_term.py`

#### P2-5 文档解析增强

- **目标**：支持更复杂的文档结构
- **方向**：
  - Markdown 标题感知分块（保留 H1/H2 层级）
  - PDF 表格提取
  - 图片 OCR
  - URL 用 `trafilatura` / `playwright` 替代正则清洗
- **产出**：扩展 `rag_agent/knowledge/loader.py` 和 `rag_agent/knowledge/chunker.py`

#### P2-6 成本与配额管理

- **目标**：对外开放服务时控制成本和滥用
- **内容**：
  - Token 用量统计
  - 按 API Key / 用户的 rate limiting
  - Embedding / LLM 调用成本估算
- **产出**：新增中间件或拦截层

#### P2-7 CI/CD 与代码质量

- **目标**：建立自动化质量门禁
- **内容**：
  - GitHub Actions：pytest、ruff、black、mypy
  - `pre-commit` 钩子
  - 测试覆盖率报告
- **产出**：`.github/workflows/`、`.pre-commit-config.yaml`

---

## 3. 推荐落地顺序

按依赖关系和投入产出比，建议分阶段推进：

| 阶段 | 任务 | 产出 |
|---|---|---|
| **阶段 1：工程基础** | P0-1 测试 + P0-4 配置 + P0-5 Python 版本 | 可放心改代码、可配置运行（P0-4/P0-5 已完成） |
| **阶段 2：服务化** | P0-2 FastAPI + P0-3 异步 + P1-3 流式 | 项目变成可部署服务（P0-2/P0-3 已完成） |
| **阶段 3：质量与成本** | P1-1 Query 改写 + P1-2 语义缓存 + P1-4 Parent Document | 回答质量与延迟双提升，成本下降 30%+ |
| **阶段 4：稳定性** | P1-8 容错重试 + Fallback LLM | 生产环境单点失败自愈 |
| **阶段 5：观测与反馈** | P1-7 Trace 可观测 + P1-6 用户反馈 | 建立持续优化数据基础，可定位幻觉/超时根因 |
| **阶段 6：Agent 化** | P2-1 Agentic RAG（Self-Correction）+ P2-2 工具调用 | 从 RAG 升级为 Agent |
| **阶段 7：安全与治理** | P2-3 护栏 + P2-5 文档增强 + P2-6 成本配额 | 具备生产级服务能力 |

---

## 4. 关键文件索引

| 模块 | 当前核心文件 | 建议新增/重点改造 |
|---|---|---|
| Agent 编排 | `rag_agent/agent.py` | 增加 `achat`、Query 改写入口、工具调用 |
| LLM | `rag_agent/llm.py` | 异步、流式、JSON mode、function calling |
| Embedding | `rag_agent/embedder.py` | 异步、缓存 |
| 知识库 | `rag_agent/knowledge/kb.py` | Parent Document、Agentic 检索 |
| 向量存储 | `rag_agent/knowledge/chroma_store.py` | 保持接口，增强测试 |
| 分块 | `rag_agent/knowledge/chunker.py` | Markdown 标题感知、表格处理 |
| 加载器 | `rag_agent/knowledge/loader.py` | trafilatura、OCR |
| 记忆 | `rag_agent/memory/long_term.py` | 重要性、衰减、冲突检测 |
| 评估 | `rag_agent/evaluation/evaluator.py` | 用户反馈、趋势分析 |
| 配置 | `rag_agent/config.py` | 已落地，持续扩展新参数 |
| 服务 | `rag_agent/api.py` | 已落地，持续扩展新端点 |
| 检索增强 | `rag_agent/retrieval/query_transform.py` | 已落地 Query Rewriting |
| 缓存 | `rag_agent/cache/semantic_cache.py` | 已落地 Semantic Cache；待扩展 embedding/检索结果缓存 |
| 可观测 | 缺失 | 新增 `rag_agent/observability.py`，对接 LangSmith / Phoenix / OpenTelemetry |
| 容错 | 缺失 | 新增 `rag_agent/resilience.py`，指数退避 + Fallback LLM |
| 测试 | 缺失 | 新增 `tests/` |

---

## 5. 验收标准

完成阶段 2 后，项目应达到：

- [ ] `pytest` 全部通过
- [ ] `uv run python -m rag_agent.api` 可启动服务
- [ ] `POST /chat/stream` 能流式返回
- [ ] 断开网络后仍能离线跑通问答链路
- [ ] 所有关键参数可通过 `.env` 或配置文件修改

完成阶段 4 后，项目应达到：

- [ ] LLM 超时 / 限流时自动指数退避重试
- [ ] 主模型失败时可切换到 Fallback LLM
- [ ] 工具调用失败后有明确的错误捕获与反馈

完成阶段 5 后，项目应达到：

- [ ] 每次请求有 trace 记录各环节耗时
- [ ] 通过 trace 可定位到具体哪一步导致幻觉或超时
- [ ] 用户可提交 👍/👎 反馈
- [ ] 低分案例能自动生成带根因的报告

完成阶段 7 后，项目应达到：

- [ ] 支持 Agentic 多步检索与 Self-Correction Loop
- [ ] 支持至少 2 种外部工具
- [ ] 具备基础 prompt injection 和 PII 检测
