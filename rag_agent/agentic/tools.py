"""Built-in tools for the agentic loop."""

from __future__ import annotations

import ast
import logging
import operator
from datetime import datetime, timezone

from rag_agent.agentic.base import BaseTool

logger = logging.getLogger(__name__)


class CalculatorTool(BaseTool):
    """Safely evaluate simple arithmetic expressions."""

    name = "calculator"

    # Supported AST node types for safe evaluation
    _ALLOWED_NODES: tuple[type, ...] = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Load,
    )

    _OPERATORS: dict[type[ast.operator], operator] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }

    def invoke(self, query: str) -> str:
        """Evaluate a numeric expression from the query.

        Falls back to extracting the first plausible expression if the whole
        query cannot be parsed.
        """
        import re

        # Try to isolate an expression: digits, operators, parentheses, decimals
        candidates = re.findall(r"[\d\+\-\*/\^\(\)\.\s]+", query)
        expr = ""
        for c in candidates:
            # Pick the longest candidate that contains at least one operator
            if len(c) > len(expr) and any(op in c for op in "+-*/^"):
                expr = c

        if not expr:
            return "无法从问题中提取可计算的表达式。"

        # Normalize ^ to ** for Python evaluation
        expr = expr.replace("^", "**")
        try:
            value = self._safe_eval(expr)
            return f"计算结果：{value}"
        except Exception as exc:
            logger.warning("Calculator failed for %r: %s", expr, exc)
            return f"计算失败：{exc}"

    def _safe_eval(self, expr: str) -> float:
        node = ast.parse(expr, mode="eval")
        return self._eval_node(node.body)

    def _eval_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("Only numeric constants are allowed")
        if isinstance(node, ast.BinOp):
            op = self._OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op(self._eval_node(node.left), self._eval_node(node.right))
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -self._eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return self._eval_node(node.operand)
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")


class DatetimeTool(BaseTool):
    """Return the current date and time."""

    name = "datetime"

    def invoke(self, query: str) -> str:
        now = datetime.now(timezone.utc).astimezone()
        return now.strftime("当前时间：%Y-%m-%d %H:%M:%S %Z")
