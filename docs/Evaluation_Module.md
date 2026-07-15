# 评估模块架构说明与优化方向

## 1. 当前架构

评估模块位于 `rag_agent/evaluation/`，负责对每次问答自动打分并持久化。

```
question + answer + contexts
            │
            ▼
     ┌──────────────┐
     │   Evaluator   │
     └──────┬───────┘
            │
   ┌────────┼────────┐
   ▼        ▼        ▼
┌──────┐ ┌──────┐ ┌──────┐
│ 指标层 │ │ 规则层 │ │ 报告层 │
└──────┘ └──────┘ └──────┘
```

## 2. 各组件职责与对应文件

### 2.1 抽象层 `base.py`

- `EvaluationResult`：单次评估结果数据类（问题、回答、上下文、分数、规则结果、时间戳）
- `BaseMetric`：指标抽象接口，定义 `score(question, answer, contexts) -> float`

### 2.2 指标层 `metrics.py`

对回答进行多维度打分（0~1），当前实现三个指标：

| 指标 | 含义 | LLM 模式 | 离线 fallback |
|---|---|---|---|
| Faithfulness | 回答是否忠于检索上下文 | LLM 判断每条 claim 是否被上下文支持 | 回答中的数字/实体是否出现在上下文中 |
| Answer Relevance | 回答与问题的相关性 | LLM 0-10 分，除以 10 | 问题与回答的关键词 Jaccard 重叠度 |
| Context Precision | 检索结果中相关 chunk 占比 | LLM 逐条判断是否相关 | 问题与每个 chunk 的关键词重叠度 |

**关键设计**：`llm=None` 时直接走 fallback，不尝试 LLM 调用。传入真实 LLM 后自动升级，LLM 调用失败时降级回 fallback。

### 2.3 规则层 `rules.py`

检查回答是否违反基本规则，不依赖 LLM：

| 规则 | 名称 | 检测内容 |
|---|---|---|
| 空回答 | `not_empty` | 回答是否为空字符串 |
| 长度异常 | `length_ok` | 长度是否在 [5, 2000] 范围内 |
| 拒绝回答 | `no_refusal` | 是否包含"我不知道""无法回答" |
| 敏感词 | `no_sensitive` | 是否包含配置的敏感词（默认空） |
| 明显幻觉 | `no_obvious_hallucination` | 回答中的数字是否出现在检索上下文中 |

### 2.4 评估器 `evaluator.py`

`Evaluator` 串联指标与规则：

```python
evaluator = Evaluator(db_path="data/eval/evaluations.db")
result = evaluator.evaluate(question, answer, contexts)
```

一次 `evaluate` 完成：

1. 跑所有指标，收集分数
2. 跑所有规则，收集通过/未通过
3. 计算综合分：`avg(指标分数) - 0.3（任意规则未通过时惩罚）`
4. 写入 SQLite（`data/eval/evaluations.db`）

### 2.5 报告层 `report.py`

从 SQLite 读取评估记录，生成可读输出：

- `generate_text_report(threshold)`：Markdown 格式失败案例报告
- `export_csv(output_path)`：导出 CSV
- `get_failures(threshold, since)`：返回低分记录列表

## 3. 核心数据流

### 3.1 评估流程

```
question + answer + contexts
   │
   ├──► FaithfulnessMetric.score()     → 0.0 ~ 1.0
   ├──► AnswerRelevanceMetric.score()  → 0.0 ~ 1.0
   ├──► ContextPrecisionMetric.score() → 0.0 ~ 1.0
   │
   ├──► DefaultRuleChecker.check()
   │      ├── not_empty
   │      ├── length_ok
   │      ├── no_refusal
   │      ├── no_sensitive
   │      └── no_obvious_hallucination
   │
   ├──► 综合分 = avg(指标) - 0.3（有规则未通过）
   │
   └──► 写入 SQLite evaluations 表
```

### 3.2 在 Agent 中的位置

评估已集成到 LangGraph 图的 `evaluate_node`，每次回答后自动执行：

```
图内 evaluate_node（生成 + 记忆存储之后）：
   evaluator.evaluate(question, answer, contexts)
      └── 写入 SQLite evaluations 表
```

评估结果通过 `ChatResponse.evaluation` 返回。

## 4. 当前局限

### 4.1 离线指标精度有限

fallback 基于关键词/实体重叠，对语义理解弱。中文分词简单（按空格+中文字符切分），对复杂表达不够敏感。

### 4.2 指标覆盖不全

只有 3 个基础指标，缺少：
- Context Recall（需要 ground truth）
- Harmfulness（内容安全性）
- Conciseness（简洁性）

### 4.3 规则惩罚一刀切

任意规则未通过统一扣 0.3 分，不区分严重程度。空回答和长度轻微异常惩罚相同。

### 4.4 缺少人工标注对比

无法与人工标注的 ground truth 对比校准，评估的绝对分数参考意义有限。

### 4.5 报告缺乏可视化

只有文本报告和 CSV 导出，没有趋势图、分数分布直方图。

## 5. 可优化方向

### 5.1 引入 Context Recall 指标

需要用户提供 ground truth（标准答案），计算检索结果覆盖标准答案的程度：

```python
class ContextRecallMetric(BaseMetric):
    def score(self, question, answer, contexts, ground_truth=None):
        # 判断 ground_truth 中的每条信息是否能在 contexts 中找到
        ...
```

### 5.2 规则加权惩罚

不同规则设置不同扣分权重：

```python
RULE_WEIGHTS = {
    "not_empty": 0.5,      # 空回答严重扣分
    "length_ok": 0.05,     # 长度轻微扣分
    "no_refusal": 0.3,     # 拒绝回答中等扣分
    "no_sensitive": 0.5,   # 敏感词严重扣分
    "no_obvious_hallucination": 0.3,
}
```

### 5.3 LLM 判定替代关键词重叠

当 LLM 可用时，Context Precision 用 LLM 逐条判定效果更好：

```
对每条检索上下文判断是否与问题相关，输出 JSON：{"0": 1, "1": 0}
```

当前代码已支持 LLM 模式（`llm` 参数），只是运行时未启用。

### 5.4 引入人工标注校准

收集一组问答对，做人工标注（1-5 分），然后与自动评分做相关性分析：

```python
from scipy.stats import spearmanr
correlation = spearmanr(human_scores, auto_scores)
```

### 5.5 趋势监控

记录每次评估的时间序列，检测质量下降趋势：

```sql
SELECT date(created_at), avg(overall_score)
FROM evaluations
GROUP BY date(created_at)
ORDER BY date(created_at) DESC;
```

### 5.6 多轮评估

不仅是单轮问答，还可以评估 Agent 在多轮对话中的表现：
- 对话连贯性
- 长期记忆召回准确率
- 用户满意度模拟

### 5.7 评估适配器（Evaluator 可选组件）

让 `Evaluator` 支持可选组件，避免每次评估都跑所有指标：

```python
evaluator = Evaluator(
    metrics=[FaithfulnessMetric(llm=llm)],  # 只跑 faithfulness
    rules=DefaultRuleChecker(),
)
```

## 6. 优先级建议

| 优化方向 | 优先级 | 说明 |
|---|---|---|
| 规则加权惩罚 | P1 | 低实现成本，提升评分合理性 | ✅
| LLM 指标启用 | P1 | 已有代码基础，接入 LLM 即可升级精度 | ✅
| Context Recall | P2 | 需要标注数据，初期可人工标注少量 |
| 趋势监控 | P2 | 对持续运营有价值 |
| 人工标注校准 | P3 | 需要标注人力投入 |
| 多轮评估 | P3 | 需要更复杂的评估框架 |
| 评估适配器 | P3 | 性能优化，初期不需要 |
