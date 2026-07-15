# 安全护栏模块架构说明与优化方向

## 1. 当前架构

安全护栏模块位于 `rag_agent/guardrails.py`，为 RAG Agent 提供输入/输出保护和检索置信度门控，默认以非阻塞模式运行，命中风险时记录日志并返回警告，可通过配置切换为硬拦截。

```
用户输入
   │
   ▼
┌─────────────────────────────────────────────┐
│           Guardrails.check_input()          │
│  ├─ Prompt Injection 检测                   │
│  ├─ PII 检测                                │
│  └─ 可选：硬拦截 / 仅警告                     │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
      通过                  拦截/警告
        │                     │
        ▼                     ▼
   进入后续流程         返回安全提示或日志

LLM 输出
   │
   ▼
┌─────────────────────────────────────────────┐
│          Guardrails.check_output()          │
│  └─ 敏感内容 / 毒性审核                       │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
      通过                  拦截/警告

检索结果
   │
   ▼
┌─────────────────────────────────────────────┐
│      Guardrails.check_retrieval()           │
│  └─ 检索置信度门控（分数过低时警告）          │
└─────────────────────────────────────────────┘
```

## 2. 各组件职责

### 2.1 数据结构

| 类型 | 职责 |
|---|---|
| `GuardrailAction` | 护栏判定动作枚举：`ALLOW` / `WARN` / `BLOCK` |
| `GuardrailResult` | 单条检查结果：名称、动作、消息、附加详情 |
| `GuardrailsConfig` | 护栏配置总开关与各子模块开关 |
| `Guardrails.CheckResult` | 组合检查汇总：是否被拦截、拦截原因、结果列表 |

### 2.2 输入护栏

#### Prompt Injection 检测 `_detect_prompt_injection()`

基于正则模式匹配常见注入攻击：

| 攻击类型 | 示例模式 |
|---|---|
| 直接指令覆盖 | `ignore previous instructions` |
| 角色扮演越狱 | `you are now DAN`、`act as ...` |
| 提示泄露 | `show me your system prompt` |
| 分隔符攻击 | 长串 `-` / `=`、`<\|...\|>` 特殊令牌 |
| 编码/混淆注入 | `base64 decode`、十六进制转义 |

高风险模式（直接指令覆盖、提示泄露、分隔符攻击）在 `hard_block=True` 时直接 `BLOCK`。

#### PII 检测与脱敏 `_detect_pii()` / `_mask_pii()`

| PII 类型 | 示例 | 脱敏模板 |
|---|---|---|
| 中国身份证号 | `110101199001011234` | `***身份证***` |
| 中国手机号 | `13800138000` | `***手机号***` |
| 邮箱地址 | `user@example.com` | `***邮箱***` |
| 信用卡号 | 13-19 位数字 | `***卡号***` |
| IPv4 地址 | `192.168.1.1` | `***IP***` |

高风险 PII（身份证、手机号、信用卡号）在 `hard_block=True` 时直接 `BLOCK`。

### 2.3 输出护栏

#### 敏感内容 / 毒性审核 `_detect_toxicity()`

按类别维护敏感词表，命中后返回警告或拦截：

| 类别 | 示例词 |
|---|---|
| 暴力 | 杀死、炸弹、kill、bomb |
| 色情 | 色情、淫秽、porn、explicit |
| 仇恨言论 | 纳粹、种族歧视、nazi、genocide |
| 自残/自杀 | 自杀、自残、suicide、self-harm |
| 非法活动 | 黑客攻击、制作病毒、hack into、make virus |

### 2.4 检索置信度门控 `_check_retrieval_confidence()`

- 未检索到上下文时返回 `WARN`
- 提供 `scores` 且最高分低于阈值（默认 0.3）时返回 `WARN`
- 用于在知识库无法覆盖问题时主动拒绝回答，避免编造

### 2.5 统一入口 `Guardrails`

| 方法 | 职责 |
|---|---|
| `check_input(text, skip_pii=False)` | 输入护栏：Prompt Injection + PII |
| `check_output(text)` | 输出护栏：毒性审核 |
| `check_retrieval(contexts, scores)` | 检索置信度检查 |
| `check_prompt_injection(text)` | 单独检测 Prompt Injection |
| `check_pii(text)` | 单独检测 PII |
| `check_output_toxicity(text)` | 单独检测输出毒性 |
| `mask(text)` | 对文本中的 PII 进行脱敏替换 |

## 3. 核心数据流

### 3.1 输入检查流程

```
用户输入
   │
   ▼
Guardrails.check_input("用户输入")
   │
   ├──► Prompt Injection 检测
   │      ├── 命中高风险 + hard_block=True → BLOCK
   │      └── 否则 → WARN / ALLOW
   │
   ├──► PII 检测
   │      ├── 命中高风险 + hard_block=True → BLOCK
   │      └── 否则 → WARN / ALLOW
   │
   └──► 返回 CheckResult（blocked / blocked_by / results）
```

### 3.2 输出检查流程

```
LLM 输出
   │
   ▼
Guardrails.check_output("LLM 输出")
   │
   ├──► 毒性审核
   │      ├── 命中敏感词 + hard_block=True → BLOCK
   │      └── 否则 → WARN / ALLOW
   │
   └──► 返回 CheckResult
```

### 3.3 在 Agent 主流程中的位置

护栏已集成到 LangGraph 图中：

```
用户提问
   │
   ├──►（图内 input_guardrail_node）
   │      Guardrails.check_input()     ← 输入安全过滤
   │      ├── BLOCK → 图短接到 END，返回安全提示
   │      └── 通过 → 继续
   │
   ├──►（图内 cache_lookup → route → transform → retrieve → generate）
   │
   ├──►（图内 output_guardrail_node）
   │      Guardrails.check_output()    ← 输出安全审核
   │
   └──►（图内 self_correction → remember → evaluate）
```

置信度门控在 `retrieve_node` 内部执行，检索分数过低时记录警告。

## 4. 配置项

`GuardrailsConfig` 提供以下开关：

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `enabled` | `True` | 护栏总开关 |
| `prompt_injection_enabled` | `True` | 是否启用 Prompt Injection 检测 |
| `prompt_injection_hard_block` | `False` | 高风险注入是否直接拦截 |
| `pii_detection_enabled` | `True` | 是否启用 PII 检测 |
| `pii_hard_block` | `False` | 高风险 PII 是否直接拦截 |
| `output_toxicity_enabled` | `True` | 是否启用输出毒性审核 |
| `output_toxicity_hard_block` | `False` | 命中敏感词是否直接拦截 |
| `confidence_check_enabled` | `True` | 是否启用检索置信度门控 |
| `confidence_threshold` | `0.3` | 检索分数阈值 |
| `raise_on_block` | `False` | 硬拦截时是否抛出异常 |

> 当前默认全部为 **WARN 模式**（非阻塞），适合先观察误报率，确认稳定后再开启硬拦截。

## 5. 当前局限

### 5.1 基于规则，无模型审核

所有检测都依赖正则和关键词匹配，对变形、谐音、上下文隐晦表达鲁棒性差，无法识别语义层面的恶意输入。

### 5.2 PII 规则覆盖有限

仅覆盖身份证号、手机号、邮箱、信用卡号、IP 地址等常见类型，缺少姓名、地址、银行卡号精确校验（Luhn 算法）、护照号等。

### 5.3 敏感词表维护成本高

词表需要人工持续更新，容易出现漏检或误伤；中英文混合场景下匹配精度受大小写和分词影响。

### 5.4 无用户/场景分级策略

当前所有请求共用同一套规则，无法针对不同用户、不同业务场景配置差异化护栏策略。

### 5.5 置信度阈值单一

仅通过最高分与阈值比较，未考虑多个检索结果分数分布、问题类型、知识库覆盖度等因素。

### 5.6 未与评估模块联动

护栏判定结果未进入评估指标，无法量化护栏对安全性、误报率、用户体验的影响。

## 6. 可优化方向

### 6.1 接入 LLM 作为审核模型

用轻量级分类模型或调用 LLM 做语义级审核：

```python
class LLMGuardrail:
    def detect(self, text: str) -> GuardrailResult:
        prompt = f"判断以下输入是否包含注入/毒性/PII，返回 JSON：{text}"
        return parse(self.llm.generate(prompt))
```

收益：能识别变形攻击、隐喻、角色扮演越狱等规则无法覆盖的场景。

### 6.2 扩展 PII 检测能力

- 引入 `presidio` 等专用 PII 识别库
- 增加姓名、地址、护照号、车牌号等实体
- 对信用卡号做 Luhn 校验，降低误报
- 支持自定义业务敏感字段（如工号、订单号）

### 6.3 敏感词表动态管理

- 将词表外置到配置文件或数据库，支持热更新
- 引入 TF-IDF / embedding 相似度做语义扩展，减少纯关键词维护成本
- 增加白名单机制，避免正常技术词汇被误伤

### 6.4 分场景策略配置

支持按用户角色、API 端点、业务类型加载不同护栏配置：

```python
class GuardrailsPolicy:
    def get_config(self, user_role: str, endpoint: str) -> GuardrailsConfig:
        ...
```

### 6.5 检索置信度多因子判断

结合以下信号综合决策：

- 最高分、平均分、Top-K 分数差距
- 检索结果与问题的 embedding 相似度
- 历史问题覆盖度
- 用户是否要求创作类回答（可适当放宽）

### 6.6 与评估模块联动

将护栏结果纳入评估维度：

- 统计拦截率、误报率、漏报率
- 记录被拦截请求用于模型/规则迭代
- 在评估报告中展示安全指标

### 6.7 输入输出日志审计

对触发 WARN/BLOCK 的请求记录结构化日志，支持后续审计与分析：

```python
{
    "event": "guardrail_triggered",
    "rule": "prompt_injection",
    "action": "BLOCK",
    "timestamp": "...",
    "hashed_user_id": "...",
    "matched_patterns": [...]
}
```

## 7. 优先级建议

| 优化方向 | 优先级 | 状态 | 说明 |
|---|---|---|---|
| 规则化 Prompt Injection / PII / 毒性检测 | P0 | ✅ 已落地 | 基于正则和关键词，默认 WARN 模式 |
| 检索置信度门控 | P0 | ✅ 已落地 | 分数过低时主动拒绝 |
| PII 检测能力扩展 | P1 | 待实现 | 覆盖更多实体，降低误报 |
| 敏感词表动态管理 | P1 | 待实现 | 支持热更新与白名单 |
| 与评估模块联动 | P1 | 待实现 | 量化安全指标 |
| 接入 LLM 语义审核 | P2 | 待实现 | 提升对变形攻击的识别能力 |
| 分场景策略配置 | P2 | 待实现 | 支持差异化护栏策略 |
| 审计日志 | P2 | 待实现 | 结构化记录触发事件 |
