"""安全护栏：输入/输出保护、PII 检测、置信度门控（P2-3）。

提供的护栏类型：
- 输入护栏：Prompt Injection 检测、PII 脱敏
- 输出护栏：敏感内容/毒性审核
- 置信度门控：检索分数过低时主动拒绝回答

所有护栏默认为非阻塞模式（仅记录日志），可通过配置切换为硬拦截。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 公共数据结构
# ---------------------------------------------------------------------------

class GuardrailAction(Enum):
    """护栏判定后的动作。"""
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"


@dataclass
class GuardrailResult:
    """单条护栏检查结果。"""
    name: str
    action: GuardrailAction
    message: str = ""
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 输入护栏：Prompt Injection 检测
# ---------------------------------------------------------------------------

# 常见注入模式（大小写不敏感匹配）
_INJECTION_PATTERNS: list[tuple[str, str]] = [
    # 直接指令覆盖
    (r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|context|messages?)", "直接指令覆盖"),
    (r"forget\s+(everything|all|your\s+instructions?)", "要求遗忘指令"),
    (r"disregard\s+(previous\s+)?instructions?", "要求忽略指令"),
    (r"override\s+(system\s+)?prompt", "覆盖系统提示"),
    # 角色扮演越狱
    (r"you\s+are\s+now\s+DAN\b", "DAN 越狱"),
    (r"pretend\s+(you\s+are|to\s+be)\s+(a\s+)?", "伪装角色"),
    (r"act\s+as\s+(if\s+you\s+are|a\s+)?", "扮演角色"),
    (r"you\s+are\s+no\s+longer", "否认身份"),
    (r"from\s+now\s+on\s+you\s+(will|are)", "重定义行为"),
    # 提示泄露
    (r"(tell|show|reveal|print|output|display)\s+(me\s+)?(your\s+)?(system\s+prompt|instructions?|rules?)", "提示泄露"),
    (r"(what|show)\s+(is\s+)?(your\s+)?(system\s+)?(prompt|instructions?)", "询问系统提示"),
    # 分隔符攻击
    (r"[-=]{16,}", "可疑分隔符"),
    (r"<\|.*?\|>", "特殊令牌注入"),
    # 编码/混淆注入
    (r"base64\s*decode", "Base64 解码注入"),
    (r"\\x[0-9a-fA-F]{2}", "十六进制转义注入"),
]


def _detect_prompt_injection(text: str, hard_block: bool = False) -> GuardrailResult:
    """检测 Prompt Injection 模式。

    Args:
        text: 待检测文本
        hard_block: True 时高风险模式直接返回 BLOCK，否则返回 WARN

    Returns:
        GuardrailResult，action 为 ALLOW / WARN / BLOCK
    """
    detected: list[str] = []
    high_risk_count = 0

    for pattern, label in _INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            detected.append(label)
            # 「直接指令覆盖」「提示泄露」「分隔符攻击」为高风险
            if label in ("直接指令覆盖", "提示泄露", "可疑分隔符", "特殊令牌注入"):
                high_risk_count += 1

    if not detected:
        return GuardrailResult(
            name="prompt_injection",
            action=GuardrailAction.ALLOW,
            message="未检测到注入模式",
        )

    message = f"检测到注入模式: {', '.join(detected)}"
    action = (
        GuardrailAction.BLOCK
        if (hard_block and high_risk_count > 0)
        else GuardrailAction.WARN
    )
    return GuardrailResult(
        name="prompt_injection",
        action=action,
        message=message,
        details={"patterns": detected, "high_risk": high_risk_count},
    )


# ---------------------------------------------------------------------------
# 输入护栏：PII 检测与脱敏
# ---------------------------------------------------------------------------

# PII 正则模式
# 注意：中文环境下 text 可能包含中文字符，\b 对 Unicode 不总是可靠，
# 因此使用 (?<!\d) 和 (?!\d) 作为数字边界。
_PII_PATTERNS: list[tuple[str, str, str]] = [
    # 格式：(正则, 类型标签, 替换模板)
    # 中国身份证号（18 位）
    (r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)", "身份证号", "***身份证***"),
    # 中国手机号
    (r"(?<!\d)1[3-9]\d{9}(?!\d)", "手机号", "***手机号***"),
    # 邮箱地址
    (r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z0-9.])", "邮箱", "***邮箱***"),
    # 信用卡号（基本格式：13-19 位数字）
    (r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)", "疑似信用卡号", "***卡号***"),
    # IPv4 地址
    (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "IP 地址", "***IP***"),
]


def _detect_pii(text: str, hard_block: bool = False) -> GuardrailResult:
    """检测文本中的 PII。

    Args:
        text: 待检测文本
        hard_block: True 时检测到高风险 PII 直接返回 BLOCK

    Returns:
        GuardrailResult，含检测到的 PII 类型列表
    """
    found_types: list[str] = []
    high_risk = False

    for pattern, pii_type, _template in _PII_PATTERNS:
        if re.search(pattern, text):
            found_types.append(pii_type)
            if pii_type in ("身份证号", "信用卡号", "手机号"):
                high_risk = True

    if not found_types:
        return GuardrailResult(
            name="pii_detection",
            action=GuardrailAction.ALLOW,
            message="未检测到 PII",
        )

    # 只检查用户输入中的 PII（输出中 LLM 可能引用示例数据）
    message = f"检测到 PII 类型: {', '.join(found_types)}"
    action = (
        GuardrailAction.BLOCK
        if (hard_block and high_risk)
        else GuardrailAction.WARN
    )
    return GuardrailResult(
        name="pii_detection",
        action=action,
        message=message,
        details={"pii_types": found_types, "high_risk": high_risk},
    )


def _mask_pii(text: str) -> str:
    """对文本中的 PII 进行脱敏替换。"""
    masked = text
    for pattern, _pii_type, template in _PII_PATTERNS:
        masked = re.sub(pattern, template, masked)
    return masked


# ---------------------------------------------------------------------------
# 输出护栏：敏感内容/毒性审核
# ---------------------------------------------------------------------------

# 敏感词分类（中英文常用敏感/有害词）
_SENSITIVE_CATEGORIES: dict[str, list[str]] = {
    "暴力": [
        "杀死", "杀掉", "谋杀", "炸弹", "恐怖袭击", "屠杀",
        "kill", "murder", "bomb", "terrorist", "massacre",
    ],
    "色情": [
        "色情", "淫秽", "裸体", "性交",
        "porn", "explicit", "naked", "sexual",
    ],
    "仇恨言论": [
        "种族歧视", "种族灭绝", "纳粹",
        "racist", "nazi", "genocide", "hate speech",
    ],
    "自残/自杀": [
        "自杀", "自残", "割腕",
        "suicide", "self-harm", "kill myself",
    ],
    "非法活动": [
        "黑客攻击", "破解密码", "制作病毒", "贩卖毒品",
        "hack into", "crack password", "make virus", "sell drugs",
    ],
}


def _detect_toxicity(text: str, hard_block: bool = False) -> GuardrailResult:
    """检测输出内容中的有害/敏感内容。

    Args:
        text: 待检测的 LLM 输出
        hard_block: True 时检测到敏感词直接 BLOCK

    Returns:
        GuardrailResult，含检测到的类别
    """
    triggered_categories: set[str] = set()
    triggered_words: list[str] = []

    text_lower = text.lower()
    for category, words in _SENSITIVE_CATEGORIES.items():
        for word in words:
            if word.lower() in text_lower:
                triggered_categories.add(category)
                triggered_words.append(word)

    if not triggered_categories:
        return GuardrailResult(
            name="output_toxicity",
            action=GuardrailAction.ALLOW,
            message="未检测到有害内容",
        )

    message = f"检测到敏感内容类别: {', '.join(sorted(triggered_categories))}"
    action = (
        GuardrailAction.BLOCK if hard_block else GuardrailAction.WARN
    )
    return GuardrailResult(
        name="output_toxicity",
        action=action,
        message=message,
        details={
            "categories": sorted(triggered_categories),
            "triggered_words": triggered_words,
        },
    )


# ---------------------------------------------------------------------------
# 置信度门控：检索分数过低时拒绝回答
# ---------------------------------------------------------------------------

def _check_retrieval_confidence(
    contexts: list[str],
    scores: list[float] | None = None,
    threshold: float = 0.3,
) -> GuardrailResult:
    """检查检索结果的置信度。

    当所有检索片的分数均低于阈值时，说明知识库可能无法覆盖该问题，
    此时可主动拒绝回答（而非强行编造）。

    Args:
        contexts: 检索到的上下文文本列表
        scores: 对应的检索分数列表（如省略，仅检查是否为空）
        threshold: 置信度阈值，低于此值视为不可信
    """
    if not contexts:
        return GuardrailResult(
            name="retrieval_confidence",
            action=GuardrailAction.WARN,
            message="未检索到任何参考资料",
            details={"context_count": 0},
        )

    if scores is not None and len(scores) == len(contexts):
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        if max_score < threshold:
            return GuardrailResult(
                name="retrieval_confidence",
                action=GuardrailAction.WARN,
                message=f"检索置信度过低（最高分 {max_score:.3f} < 阈值 {threshold}）",
                details={
                    "max_score": max_score,
                    "avg_score": avg_score,
                    "threshold": threshold,
                    "context_count": len(contexts),
                },
            )

    return GuardrailResult(
        name="retrieval_confidence",
        action=GuardrailAction.ALLOW,
        message="检索置信度正常",
        details={"context_count": len(contexts)},
    )


# ---------------------------------------------------------------------------
# 统一护栏入口
# ---------------------------------------------------------------------------

@dataclass
class GuardrailsConfig:
    """护栏配置。"""

    enabled: bool = True
    # 输入护栏
    prompt_injection_enabled: bool = True
    prompt_injection_hard_block: bool = False  # True 时高风险模式直接拒绝
    pii_detection_enabled: bool = True
    pii_hard_block: bool = False
    # 输出护栏
    output_toxicity_enabled: bool = True
    output_toxicity_hard_block: bool = False
    # 置信度门控
    confidence_check_enabled: bool = True
    confidence_threshold: float = 0.3
    # 通用：硬拦截时是否抛出异常
    raise_on_block: bool = False


class Guardrails:
    """统一安全护栏入口。

    用法::

        gr = Guardrails(GuardrailsConfig(enabled=True))

        # 输入检查
        result = gr.check_input("用户输入内容")
        if result.blocked:
            return "抱歉，无法处理该请求。"

        # 输出检查
        result = gr.check_output("LLM 输出内容")

        # 检索置信度检查
        result = gr.check_confidence(contexts=[...], scores=[0.1, 0.2])

        # 脱敏
        safe = gr.mask("手机号 13800138000")
        # → "手机号 ***手机号***"
    """

    def __init__(self, config: GuardrailsConfig | None = None):
        self.config = config or GuardrailsConfig()

    # ---- 单一检查 ----

    def check_prompt_injection(self, text: str) -> GuardrailResult:
        """检查 Prompt Injection。"""
        return _detect_prompt_injection(
            text, hard_block=self.config.prompt_injection_hard_block
        )

    def check_pii(self, text: str) -> GuardrailResult:
        """检查 PII。"""
        return _detect_pii(text, hard_block=self.config.pii_hard_block)

    def check_output_toxicity(self, text: str) -> GuardrailResult:
        """检查输出毒性。"""
        return _detect_toxicity(
            text, hard_block=self.config.output_toxicity_hard_block
        )

    def check_confidence(
        self,
        contexts: list[str],
        scores: list[float] | None = None,
    ) -> GuardrailResult:
        """检查检索置信度。"""
        return _check_retrieval_confidence(
            contexts=contexts,
            scores=scores,
            threshold=self.config.confidence_threshold,
        )

    # ---- 组合检查 ----

    @dataclass
    class CheckResult:
        """组合检查结果汇总。"""
        blocked: bool = False
        blocked_by: str = ""
        results: list[GuardrailResult] = field(default_factory=list)

        @property
        def all_passed(self) -> bool:
            return not self.blocked

        def summary(self) -> str:
            """生成检查摘要。"""
            lines = []
            for r in self.results:
                icon = "✅" if r.action == GuardrailAction.ALLOW else "⚠️" if r.action == GuardrailAction.WARN else "❌"
                lines.append(f"{icon} [{r.name}] {r.message}")
            if self.blocked:
                lines.append(f"🔒 被 {self.blocked_by} 拦截")
            return "\n".join(lines)

    def check_input(self, text: str, skip_pii: bool = False) -> CheckResult:
        """执行输入护栏检查（Prompt Injection + PII）。

        Args:
            text: 用户输入文本
            skip_pii: True 时跳过 PII 检查
        """
        if not self.config.enabled:
            return self.CheckResult()

        results: list[GuardrailResult] = []
        blocked = False
        blocked_by = ""

        if self.config.prompt_injection_enabled:
            r = self.check_prompt_injection(text)
            results.append(r)
            if r.action == GuardrailAction.BLOCK:
                blocked = True
                blocked_by = "prompt_injection"
            _log_guardrail_result(r)

        if not skip_pii and self.config.pii_detection_enabled:
            r = self.check_pii(text)
            results.append(r)
            if r.action == GuardrailAction.BLOCK:
                blocked = True
                blocked_by = blocked_by or "pii_detection"
            _log_guardrail_result(r)

        return self.CheckResult(
            blocked=blocked, blocked_by=blocked_by, results=results
        )

    def check_output(self, text: str) -> CheckResult:
        """执行输出护栏检查（毒性审核）。"""
        if not self.config.enabled:
            return self.CheckResult()

        results: list[GuardrailResult] = []
        blocked = False
        blocked_by = ""

        if self.config.output_toxicity_enabled:
            r = self.check_output_toxicity(text)
            results.append(r)
            if r.action == GuardrailAction.BLOCK:
                blocked = True
                blocked_by = "output_toxicity"
            _log_guardrail_result(r)

        return self.CheckResult(
            blocked=blocked, blocked_by=blocked_by, results=results
        )

    def check_retrieval(
        self,
        contexts: list[str],
        scores: list[float] | None = None,
    ) -> GuardrailResult:
        """检查检索置信度（便捷方法）。"""
        if not self.config.enabled or not self.config.confidence_check_enabled:
            return GuardrailResult(
                name="retrieval_confidence",
                action=GuardrailAction.ALLOW,
                message="置信度检查已禁用",
            )
        return self.check_confidence(contexts, scores)

    # ---- 脱敏 ----

    @staticmethod
    def mask(text: str) -> str:
        """对文本中的 PII 进行替换脱敏。"""
        return _mask_pii(text)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _log_guardrail_result(result: GuardrailResult) -> None:
    """根据护栏结果等级输出日志。"""
    if result.action == GuardrailAction.BLOCK:
        logger.error("护栏拦截 [%s]: %s", result.name, result.message)
    elif result.action == GuardrailAction.WARN:
        logger.warning("护栏警告 [%s]: %s", result.name, result.message)
    else:
        logger.debug("护栏通过 [%s]: %s", result.name, result.message)
