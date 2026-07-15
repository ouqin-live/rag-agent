# LangGraph 渐进式迁移方案

## 目标

· 把当前自研的 ReAct 循环迁移为 LangGraph 状态图
· 保留现有记忆、知识库、评估、护栏等核心模块
· 让 Agent 主流程可配置、可观测、便于后续扩展多 Agent 协作

---

## 当前架构与问题

当前 Agentic 流程写在 rag_agent/agentic/react.py 的 ReactLoop 里：

· 路由 → 查询改写 → 检索 → 生成 → 反思 → 修正
· 循环结构是硬编码的，改流程要改代码
· 状态用 AgenticContext 手动传递

引入 LangGraph 后，这个循环可以变成一张图：

· 每个步骤是一个节点
· 节点之间的跳转用边表示
· 状态由 LangGraph 统一管理

---

## 迁移原则

· 不动现有模块内部实现
· 仅在现有模块外面包一层 LangGraph 节点
· 保留自研 ReactLoop 作为 fallback
· 通过配置开关切换两种模式
· 每阶段都要跑通 pytest tests/

---

## 核心改造点

### 1. 状态定义

新增 rag_agent/graph/state.py，定义 GraphState。

State 里放这些字段：

· question：用户原始问题
· search_query：改写后的检索 query
· contexts：检索到的知识库片段
· long_term_facts：召回的长期记忆
· tool_results：工具调用结果
· answer：LLM 生成的答案
· iteration：当前 ReAct 迭代次数
· route_decision：路由决策
· correction_score：自我修正分数
· needs_correction：是否需要再次检索
· cache_hit：是否命中语义缓存
· evaluation：评估结果

### 2. 节点拆分

新增 rag_agent/graph/nodes.py，把 ReactLoop 拆成以下节点：

· input_guardrail_node：输入护栏检查
· cache_lookup_node：语义缓存查询
· query_transform_node：Query Rewriting
· route_node：QueryRouter 路由决策
· retrieve_node：并行检索知识库、长期记忆、工具
· generate_node：LLM 生成答案
· output_guardrail_node：输出护栏检查
· self_correction_node：Faithfulness 评估与 query 重写
· remember_node：提取并存储长期记忆
· evaluate_node：评估本次回答质量

### 3. 边设计

主要三类边：

· 条件边：根据 route_node 结果决定检索哪些数据源
· 循环边：self_correction_node 判断需要修正时回到 retrieve_node
· 退出边：达到最大迭代次数或分数达标后结束

### 4. 图组装

新增 rag_agent/graph/graph.py，负责：

· 创建 StateGraph
· 添加节点
· 添加普通边和条件边
· 编译成可执行的图对象

### 5. 对外接口

新增 rag_agent/graph/agent.py 或复用 rag_agent/agent.py：

· LangGraphChatAgent：基于 LangGraph 的聊天入口
· 提供 chat、achat、achat_stream 方法
· 内部调用编译好的图

---

## 目录结构

新增目录：

· rag_agent/graph/
  · __init__.py
  · state.py
  · nodes.py
  · graph.py
  · agent.py

现有目录不变：

· rag_agent/agentic/：保留 ReactLoop 和 SelfCorrector
· rag_agent/knowledge/：保留知识库和向量存储
· rag_agent/memory/：保留三层记忆
· rag_agent/evaluation/：保留评估模块
· rag_agent/guardrails.py：保留安全护栏

---

## 迁移阶段

### 阶段一：LangGraph 基础骨架

目标：先把依赖和最小图跑起来，不改动现有 ReactLoop。

· uv add langgraph langchain-core
· 新建 rag_agent/graph/ 目录
· 定义 GraphState
· 实现最小图：retrieve_node → generate_node
· 写一个最小测试验证图能执行
· 此时只作为独立模块存在，不影响主流程

### 阶段二：完整 ReAct 循环

目标：把现有 ReactLoop 的能力完整搬到 LangGraph 上。

· 把 ReactLoop 的逻辑拆成 nodes.py 里的节点
· 实现 route_node，根据问题选择知识库、记忆或工具
· 实现条件边，让 retrieve_node 只调用被选中的数据源
· 实现 self_correction_node，用 Faithfulness 评分决定是否继续
· 实现循环边，分数不够时回到 retrieve_node
· 实现 generate_node，调用 LLM 生成答案
· 添加 LangGraphChatAgent 对外接口
· 通过 AGENTIC_USE_LANGGRAPH 开关和现有 ReactLoop 并行运行
· 跑通现有测试，确保两种模式行为一致

### 阶段三：接入完整链路

· 接入语义缓存
· 接入 Query Rewriting
· 接入输入/输出/置信度护栏
· 接入长期记忆召回与存储
· 接入评估模块

### 阶段四：增强能力

· 流程可视化
· 断点与人机协同
· 多 Agent 节点，例如专门的 Critic Agent

---

## 配置项

新增环境变量：

· AGENTIC_USE_LANGGRAPH：是否启用 LangGraph 模式
· LANGGRAPH_MAX_ITERATIONS：最大迭代次数
· LANGGRAPH_FAITHFULNESS_THRESHOLD：自我修正阈值

保留现有配置：

· AGENTIC_ENABLED
· AGENTIC_MAX_ITERATIONS
· AGENTIC_FAITHFULNESS_THRESHOLD
· AGENTIC_USE_LLM_ROUTER

---

## 风险与应对

### 风险一：调试变复杂

LangGraph 把流程拆成图，出问题要同时看代码、状态和图结构。

应对：

· 保留 ReactLoop 作为 fallback
· 通过配置开关切换两种实现
· 两套实现共用同一套测试用例

### 风险二：依赖变重

LangGraph 会引入 langchain 生态依赖，包体积变大。

应对：

· LangGraph 代码放在独立模块
· 非 LangGraph 模式不导入这些依赖
· 可选依赖组管理

### 风险三：异步兼容性

FastAPI 使用 achat，必须保证 LangGraph 图支持异步执行。

应对：

· LangGraph 原生支持 async
· 节点实现同时提供 sync 和 async 版本
· 流式输出通过图状态逐步返回

### 风险四：行为不一致

迁移后可能出现和原 ReactLoop 行为不一致的边界情况。

应对：

· 增加 LangGraph 模式专项测试
· 对比两套实现的输出结果
· 先让 LangGraph 模式可选，稳定后再设为默认

---

## 验收标准

· uv run pytest tests/ 全部通过
· LangGraph 模式能回答普通知识问题
· LangGraph 模式能调用 calculator 和 datetime 工具
· LangGraph 模式能触发自我修正
· 通过配置能切换回原有 ReactLoop

---

## 建议

· 先开 feature/langgraph 分支
· 不要一次性替换 main 流程
· 阶段一和阶段二先让图跑起来
· 阶段三再逐步接入护栏、缓存、记忆、评估
· 稳定后再更新 README 和 Optimization_Roadmap
