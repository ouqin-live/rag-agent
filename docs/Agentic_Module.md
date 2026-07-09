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

### 4.1 反思与自我修正

`SelfCorrector` 使用 **Faithfulness（忠实度）** 指标评估答案：

- 分数 ≥ `AGENTIC_FAITHFULNESS_THRESHOLD`：答案通过，退出循环
- 分数 < 阈值：触发修正，LLM 生成更聚焦的补充查询，进入下一轮检索

`SelfCorrector` 会优先复用 `Evaluator` 中名为 `faithfulness` 的指标，保证自我修正与最终评估的尺子一致。

### 4.2 工具调用

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
