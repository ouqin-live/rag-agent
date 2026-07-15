# Agentic 模块说明

> 对应版本：v0.3.0+（LangGraph 全链路）  
> 相关代码：`rag_agent/graph/`、`rag_agent/agentic/`

本文档描述 `rag-agent` 的 Agentic RAG 能力，包括 LangGraph 状态图工作流、查询路由、自我修正、工具调用。

---

## 1. 定位

Agent 的**全部问答流程**由一张 LangGraph 状态图驱动。不再是「可选路径」，而是唯一编排引擎。

```
用户问题
    │
    ▼
┌─────────────────────────────────────────┐
│          LangGraph 状态图                 │
│                                         │
│  input_guardrail → cache_lookup         │
│  → route → transform_query → retrieve  │
│  → generate → output_guardrail         │
│  → self_correction → remember → evaluate│
└─────────────────────────────────────────┘
```

`rag_agent/agentic/` 中的路由、工具、自我修正等组件作为图的**节点依赖**注入，不再自行驱动循环。

---

## 2. 核心组件

### 2.1 LangGraph 工作流（`rag_agent/graph/`）

| 文件 | 类/函数 | 职责 |
|---|---|---|
| `state.py` | `GraphState` | TypedDict 状态定义，贯穿所有节点 |
| `nodes.py` | `make_nodes()` | 节点工厂，闭包注入依赖生成 10 个节点 |
| `graph.py` | `build_agentic_graph()` | 用 StateGraph 组装节点和边，编译成可执行图 |
| `agent.py` | `LangGraphAgent` | 封装图调用，提供 sync/async `chat` 接口 |

### 2.2 Agentic 组件（`rag_agent/agentic/`）

| 文件 | 类/函数 | 职责 | 被图节点使用 |
|---|---|---|---|
| `base.py` | `BaseTool` / `ToolResult` | 工具接口 | `retrieve_node` |
| `router.py` | `RuleBasedRouter` / `LLMQueryRouter` | 查询路由 | `route_node` |
| `tools.py` | `CalculatorTool` / `DatetimeTool` | 内置工具 | `retrieve_node` |
| `self_correction.py` | `SelfCorrector` | 忠实度评估 + query 重写 | `self_correction_node` |

---

## 3. 图节点详解

### 3.1 input_guardrail — 输入护栏

检查 Prompt Injection 和 PII。拦截时设置 `guardrail_blocked=True`，图短接到 END。

### 3.2 cache_lookup — 语义缓存

调用 `SemanticCache.lookup()` 查询相似历史问题。命中时设置 `cache_hit=True`，图短接到 END。

### 3.3 route — 查询路由

`RuleBasedRouter` 根据问题类型决定启用哪些数据源：

| 问题类型 | 示例 | 路由结果 |
|---|---|---|
| 数学计算 | `1 + 2 等于多少` | `calculator` |
| 当前时间 | `现在几点` | `datetime` |
| 个人偏好 | `我喜欢的颜色是什么` | `long_term_memory` + `knowledge_base` |
| 一般知识 | `什么是 RAG` | `knowledge_base` |

开启 `AGENTIC_USE_LLM_ROUTER=true` 后切换为 LLM 路由，失败自动降级到规则路由。

### 3.4 transform_query — 查询改写

`QueryTransformer`（默认 `RewritingTransformer`）对问题做指代消解和标准化。例如：

- `"那它怎么减少幻觉？"` → `"RAG（检索增强生成）是如何减少幻觉的？"`

改写失败时自动回退到原始问题。

### 3.5 retrieve — 并行检索

根据路由结果并行执行：

- **知识库**：`KnowledgeBase.hybrid_search()`（Dense + BM25 + RRF）
- **长期记忆**：`LongTermMemory.recall()`
- **工具调用**：`CalculatorTool` / `DatetimeTool`

任一源失败不阻断其他源，降级为空结果继续。

### 3.6 generate — 答案生成

用系统提示 + 用户事实 + 参考资料 + 工具结果 + 对话历史构建 prompt，LLM 生成答案。失败时返回基于上下文的兜底回答。

### 3.7 output_guardrail — 输出护栏

检查 LLM 输出中的毒性/敏感内容（暴力、色情、仇恨言论等 5 类）。命中时替换为安全兜底回答。

### 3.8 self_correction — 自我修正

`SelfCorrector` 用 Faithfulness 指标评估答案质量：

- 分数 ≥ `AGENTIC_FAITHFULNESS_THRESHOLD`（默认 0.5）：通过，进入 remember
- 分数 < 阈值：LLM 生成更聚焦的补充查询，回到 retrieve 重新检索

循环最多 `AGENTIC_MAX_ITERATIONS` 次（默认 2）。

### 3.9 remember — 记忆存储

`LLMMemoryExtractor` 从本轮 Q&A 中提取用户事实/偏好，写入 `LongTermMemory`。LLM 提取失败时自动降级到规则提取。

### 3.10 evaluate — 评估打分

`Evaluator` 对回答做 Faithfulness / Answer Relevance / Context Precision 三维评分 + 规则检查，结果写入 SQLite。

---

## 4. 工具调用

当前内置工具：

- `CalculatorTool`：基于 AST 安全计算算术表达式
- `DatetimeTool`：返回当前日期时间

工具接口为 `BaseTool`，通过 `AgentConfig.tools` 字典注册：

```python
from rag_agent.agentic import CalculatorTool

agent = Agent(AgentConfig(
    ...
    tools={"calculator": CalculatorTool()},
))
```

---

## 5. 兜底与降级

| 场景 | 行为 |
|---|---|
| 路由失败（LLM 路由） | 降级到 `RuleBasedRouter` |
| 查询改写失败 | 回退到原始问题 |
| LLM 生成失败 | 返回基于参考资料/工具结果的兜底回答 |
| 检索失败 | 记录警告，继续用空上下文生成 |
| 护栏拦截 | 返回安全兜底提示（WARN 模式仅记录日志） |

---

## 6. 配置项

| 配置项 | 环境变量 | 默认值 | 说明 |
|---|---|---|---|
| 最大迭代次数 | `AGENTIC_MAX_ITERATIONS` | `2` | 自我修正最大轮数 |
| 忠实度阈值 | `AGENTIC_FAITHFULNESS_THRESHOLD` | `0.5` | 低于该阈值触发修正 |
| 使用 LLM 路由 | `AGENTIC_USE_LLM_ROUTER` | `false` | 是否使用 LLM 做查询路由 |

---

## 7. 后续扩展

| 方向 | 说明 |
|---|---|
| Web Search 工具 | 接入搜索引擎 |
| 代码执行器 | 安全的 Python / SQL 执行环境 |
| 多工具并行 | 一次调用多个工具并聚合结果 |
| 流式 Agentic | 图中支持 token 级流式输出 |
# Agentic 模块说明

> 对应版本：v0.2.0+  
> 相关代码：`rag_agent/agentic/`

本文档描述 `rag-agent` 的 Agentic RAG 能力，包括查询路由、ReAct 循环、自我修正、工具调用及其与原有高级 RAG 流程的关系。

---

## 1. 定位

Agentic RAG 是在原有高级 RAG 流程之上新增的一条可选路径：

- **原有流程**：查询改写 → 一次检索 → 生成 → 评估
- **Agentic 流程**：路由 → 查询改写 → ReAct 循环（检索 → 生成 → 反思 → 修正）→ 评估

Agentic 模式默认关闭，通过 `AGENTIC_ENABLED=true` 开启。开启后，`Agent.chat/achat/achat_stream` 会走 Agentic 流程；未开启时保持原有流程不变。

---

## 2. 核心组件

| 文件 | 类/函数 | 职责 |
|---|---|---|
| `base.py` | `AgenticContext` | 单轮 Agentic 流程的 mutable 状态 |
| `base.py` | `BaseTool` / `ToolResult` | 工具接口与工具结果封装 |
| `router.py` | `QueryRouter`（抽象） | 定义查询路由接口 |
| `router.py` | `RuleBasedRouter` | 基于规则的路由，无 LLM 依赖 |
| `router.py` | `LLMQueryRouter` | 基于 LLM JSON 输出的路由，失败自动降级 |
| `tools.py` | `CalculatorTool` | 安全算术表达式计算 |
| `tools.py` | `DatetimeTool` | 返回当前时间 |
| `self_correction.py` | `SelfCorrector` | 基于忠实度等指标判断是否修正，并重写查询 |
| `react.py` | `ReactLoop` | ReAct 循环协调器 |
| `react.py` | `ReactResult` | 循环最终结果 |

---

## 3. 查询路由（Query Routing）

根据问题类型自动选择数据源/工具：

| 问题类型 | 示例 | 路由结果 |
|---|---|---|
| 数学计算 | `1 + 2 等于多少` | `calculator` |
| 当前时间/日期 | `现在几点` | `datetime` |
| 个人偏好/历史 | `我喜欢的颜色是什么` | `long_term_memory` + `knowledge_base` |
| 一般知识 | `什么是 RAG` | `knowledge_base` |

默认使用 `RuleBasedRouter`。开启 `AGENTIC_USE_LLM_ROUTER=true` 后切换为 `LLMQueryRouter`，由 LLM 输出 JSON 路由决策；LLM 失败时自动降级到规则路由。

---

## 4. ReAct 循环

`ReactLoop` 每轮对话执行以下步骤：

```
用户问题
    │
    ├──► 路由：选择数据源/工具
    │
    ├──► 查询改写：使用 QueryTransformer（如 RewritingTransformer）
    │
    └──► ReAct 循环（最多 AGENTIC_MAX_ITERATIONS 次）
            │
            ├──► 检索证据（KB / LTM / 工具）
            │
            ├──► 生成答案（失败则返回兜底回答）
            │
            ├──► 反思：评估答案质量
            │
            └──► 若质量不足 → 重写查询 → 再次检索
```

### 4.1 详细流程与代码对应

`ReactLoop.run()`（`rag_agent/agentic/react.py`）的循环体与代码位置对应如下：

| 步骤 | 代码位置 | 说明 |
|---|---|---|
| 路由 | L114-L117 | `QueryRouter` 决定使用哪些数据源/工具 |
| 查询改写 | L119-L130 | `QueryTransformer` 改写原始问题；失败则回退到原问题 |
| 检索证据 | L136-L138 → `_retrieve` | 从知识库、长期记忆、工具中收集证据 |
| 生成答案 | L140-L141 → `_generate_answer` | LLM 基于证据生成答案；失败则返回兜底回答 |
| 反思 | L143-L147 → `SelfCorrector.check` | 评估答案 Faithfulness |
| 修正 | L153-L157 | 若未通过，使用重写后的查询并清空证据，进入下一轮 |

`search_query` 在每次修正后可能被 `SelfCorrector` 重写，因此循环内的检索 query 是动态变化的。

### 4.2 与主流 ReAct 的区别

本实现是**面向生产环境的简化版 ReAct**，保留了“检索 → 生成 → 反思 → 行动”的核心思想，但与学术定义和主流框架（如 LangChain ReAct Agent、LlamaIndex ReActAgent）存在差异：

| 维度 | 原始 ReAct / 主流框架 | 本项目实现 |
|---|---|---|
| 控制流 | LLM 通过 `Thought/Action/Observation` 文本轨迹决定下一步 | 循环结构由代码硬编码，LLM 只负责生成答案 |
| 反思方式 | LLM 自己输出反思结论 | 独立的 `SelfCorrector` 基于 Faithfulness 指标评估 |
| 动作空间 | 通用工具调用（搜索、计算、API 等） | 当前内置 `calculator`、`datetime`，按路由结果调用 |
| 停止条件 | LLM 输出 `Finish` 或达到最大步数 | 达到忠实度阈值或最大迭代次数 |
| 适用场景 | 探索性、多跳推理 | 结构化 RAG 问答，强调可控与兜底 |

这种设计牺牲了一定的 agent 自主性，但换来了更高的可控性、更低的 token 消耗和更稳定的生产表现。

### 4.3 反思与自我修正

`SelfCorrector` 使用 **Faithfulness（忠实度）** 指标评估答案：

- 分数 ≥ `AGENTIC_FAITHFULNESS_THRESHOLD`：答案通过，退出循环
- 分数 < 阈值：触发修正，LLM 生成更聚焦的补充查询，进入下一轮检索

`SelfCorrector` 会优先复用 `Evaluator` 中名为 `faithfulness` 的指标，保证自我修正与最终评估的尺子一致。

### 4.4 工具调用

当前内置工具：

- `CalculatorTool`：基于 AST 安全计算算术表达式
- `DatetimeTool`：返回当前日期时间

工具接口为 `BaseTool`，后续可通过 `AgentConfig.tools` 字典扩展 Web Search、代码执行器等。

---

## 5. 与原流程的关系

### 5.1 复用的能力

Agentic 流程完整复用了原有流程的以下能力：

- 语义缓存（Semantic Cache）查/存
- 短期/中期/长期记忆
- 事实提取与存储
- 自动评估（Evaluator）
- LLM 客户端的容错重试 / Fallback 模型切换（ResilientLLMClient）

### 5.2 新增的能力

- 查询路由
- 多轮检索与自我修正
- 工具调用

### 5.3 兜底与降级

- 路由失败：降级到规则路由
- 查询改写失败：回退到原始问题
- 大模型生成失败：返回基于参考资料/工具结果的兜底回答
- 检索失败：记录警告，继续用空上下文生成

---

## 6. 配置项

| 配置项 | 环境变量 | 默认值 | 说明 |
|---|---|---|---|
| 启用 Agentic | `AGENTIC_ENABLED` | `false` | 是否启用 Agentic RAG |
| 最大迭代次数 | `AGENTIC_MAX_ITERATIONS` | `2` | ReAct / 自我修正最大轮数 |
| 忠实度阈值 | `AGENTIC_FAITHFULNESS_THRESHOLD` | `0.5` | 低于该阈值触发修正 |
| 使用 LLM 路由 | `AGENTIC_USE_LLM_ROUTER` | `false` | 是否使用 LLM 做查询路由 |

---

## 7. 使用方式

### 通过环境变量

```bash
AGENTIC_ENABLED=true
AGENTIC_MAX_ITERATIONS=2
AGENTIC_FAITHFULNESS_THRESHOLD=0.5
AGENTIC_USE_LLM_ROUTER=false
```

### 通过代码

```python
from rag_agent.agent import Agent, AgentConfig

agent = Agent(AgentConfig(
    knowledge_base=kb,
    short_term_memory=stm,
    llm_client=llm,
    agentic_enabled=True,
))

response = agent.chat("user-1", "1 + 2 等于多少")
print(response.answer)
```

---

## 8. 后续扩展方向

| 方向 | 说明 |
|---|---|
| Web Search 工具 | 接入 Serper / DuckDuckGo 等搜索引擎 |
| 代码执行器 | 安全的 Python / SQL 执行环境 |
| 多工具并行 | 一次调用多个工具并聚合结果 |
| 更复杂的 Planner | 将复杂问题拆分为子问题分别求解 |
| 流式 Agentic | 在 ReAct 循环中支持真正的 token 级流式输出 |
