# 记忆模块架构说明与优化方向

## 1. 当前架构

当前记忆模块位于 `rag_agent/memory/`，采用**两层记忆**设计：

```
User Query
    │
    ├──► 短期记忆（ShortTermMemory）
    │      当前会话最近 N 轮完整对话
    │
    └──► 长期记忆（LongTermMemory）
           跨会话用户事实/偏好（向量库存储）
```

### 1.1 短期记忆 `ShortTermMemory`

**文件**：`rag_agent/memory/short_term.py`

- 以内存形式保存当前会话的 `Message` 列表
- 每条消息包含 `role`、`content`、`timestamp`
- 超过 `max_turns` 时，丢弃最旧的完整对话轮次
- 用于构建 prompt 中的近期对话历史

### 1.2 长期记忆 `LongTermMemory`

**文件**：`rag_agent/memory/long_term.py`

- 复用 `LocalVectorStore`（SQLite + numpy）做持久化
- 每个事实保存为带 embedding 的 chunk，按 `user_id` 过滤召回
- 支持 `remember`、`recall`、`forget`
- 去重策略：
  - 相似度 ≥ 0.92：跳过重复
  - 相似度 0.80 ~ 0.92：合并成更长/更新的事实
  - 相似度 < 0.80：新增
- 容量限制：默认每个用户最多 100 条事实，超出按时间淘汰最旧的

### 1.3 事实提取器 `RuleBasedMemoryExtractor`

**文件**：`rag_agent/memory/extractor.py`

- 基于正则规则从单轮对话中提取用户事实/偏好
- 覆盖中英文常见表达：
  - 偏好：`我喜欢...`、`I prefer...`
  - 身份：`我是...`、`I am a...`
  - 要求：`请用中文回答...`、`Please answer in...`
  - 能力：`我熟悉...`、`I am familiar with...`
- 过滤：过短/过长、以问号结尾、助手敷衍回答会被丢弃
- 在 `Agent.chat()` 每轮结束后自动执行，将新事实写入长期记忆

## 2. 数据流

```
用户提问
   │
   ├──► LongTermMemory.recall(user_id, question)
   │      └── 返回相关用户事实
   │
   ├──► 生成回答
   │
   ├──► ShortTermMemory.add(user, question)
   ├──► ShortTermMemory.add(assistant, answer)
   │
   └──► MemoryExtractor.extract(question, answer)
          └── 提取新事实
                └── LongTermMemory.remember(user_id, fact)
```

## 3. 当前局限

### 3.1 缺少中期记忆

理想的三层记忆架构应为：

| 层级 | 作用 | 当前状态 |
|---|---|---|
| 短期记忆 | 当前最近几轮完整对话 | ✅ 已实现 |
| 中期记忆 | 当前/近期会话摘要 | ❌ 缺失 |
| 长期记忆 | 跨会话用户事实 | ✅ 已实现 |

当前 `ShortTermMemory` 通过截断丢弃旧对话，没有摘要层，长会话中较早的上下文会完全丢失。

### 3.2 事实提取基于规则

`RuleBasedMemoryExtractor` 依赖正则匹配，无法处理：
- 暗示性偏好
- 复杂语义
- 否定/双重否定
- 上下文依赖的事实

### 3.3 短期记忆截断生硬

直接丢弃最旧轮次，而不是压缩成摘要，可能丢失关键上下文。

### 3.4 长期记忆写入时机单一

只在每轮对话结束后提取事实，不会主动总结多轮对话得出的综合信息。

## 4. 可优化方向

### 4.1 引入中期记忆（会话摘要）

新增 `MediumTermMemory` 模块：

```python
class MediumTermMemory:
    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client
        self.summary = ""

    def update(self, messages: list[Message]):
        """把旧对话合并进现有摘要"""
        prompt = f"""
当前会话摘要：{self.summary}
新增对话：{messages}
请更新摘要，保留关键信息，控制长度。
"""
        self.summary = self.llm_client.generate(prompt)
```

在 `Agent.chat()` 中，当短期记忆超过阈值时，把旧轮次归档到中期记忆：

```python
if len(self.config.short_term_memory) > threshold:
    old_turns = self.config.short_term_memory.archive_old_turns()
    self.config.medium_term_memory.update(old_turns)
```

Prompt 构建顺序：

```
[系统提示]
[长期记忆：用户事实]
[中期记忆：本次会话摘要]
[短期记忆：最近 N 轮完整对话]
[当前问题]
```

### 4.2 LLM-based 事实提取

用 LLM 替代规则提取器，提升复杂语义理解能力：

```python
prompt = """
请从以下对话中提取用户的关键事实和偏好。
只返回 JSON 数组，每条事实一行：
["事实1", "事实2"]

User: {question}
Assistant: {answer}
"""
```

离线时降级回 `RuleBasedMemoryExtractor`。

### 4.3 短期记忆智能淘汰

不仅按轮次截断，还可以：
- 按 token 数量截断
- 保留与当前问题最相关的历史轮次
- 对旧轮次做摘要后再淘汰

### 4.4 记忆重要性评分

为长期记忆事实引入重要性分数：
- 高频出现的事实提高权重
- 用户明确确认过的事实标记为重要
- 召回时按重要性+相似度排序

### 4.5 多会话长期记忆

当前长期记忆按 `user_id` 组织，可以扩展为：
- 按会话主题/标签分组
- 支持时间衰减，旧事实逐渐降低召回优先级
- 支持用户主动管理（列出、编辑、删除）

### 4.6 记忆冲突检测

当新事实与旧事实矛盾时（如用户先说喜欢中文，后说喜欢英文），自动检测并标记冲突，或选择最新的事实。

## 5. 优先级建议

| 优化方向 | 优先级 | 说明 |
|---|---|---|
 引入中期记忆 | P1 | 对长会话体验提升最明显 |
| LLM-based 事实提取 | P1 | 显著提升记忆质量 |
| 短期记忆按 token 截断 | P2 | 更精细的成本控制 |
| 记忆重要性评分 | P2 | 提升召回质量 |
| 多会话长期记忆 | P3 | 更复杂的用户管理场景 |
| 记忆冲突检测 | P3 | 高级功能，初期可不做 |
