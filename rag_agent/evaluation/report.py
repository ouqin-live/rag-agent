"""Generate evaluation reports from persisted results."""

from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


_METRIC_NAMES = {
    "faithfulness": "忠实度 (Faithfulness)",
    "answer_relevance": "回答相关性 (Answer Relevance)",
    "context_precision": "上下文精确率 (Context Precision)",
}


class ReportGenerator:
    """Generate human-readable or machine-readable evaluation reports."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def get_failures(
        self,
        threshold: float = 0.6,
        since: datetime | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return recent failure records below ``threshold``."""
        since = since or (datetime.now(timezone.utc) - timedelta(days=7))
        since_str = since.isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, question, answer, contexts, scores, overall_score,
                       failed_rules, created_at
                FROM evaluations
                WHERE overall_score < ? AND created_at > ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (threshold, since_str, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def generate_text_report(
        self,
        threshold: float = 0.6,
        since: datetime | None = None,
        limit: int = 50,
    ) -> str:
        failures = self.get_failures(threshold, since, limit)
        lines = [
            "# RAG Agent 评估失败案例报告",
            f"生成时间: {datetime.now(timezone.utc).isoformat()}",
            f"阈值: {threshold}",
            f"低分样本数: {len(failures)}",
            "",
        ]
        for i, row in enumerate(failures, 1):
            lines.append(f"## 案例 {i}")
            lines.append(f"- 时间: {row['created_at']}")
            lines.append(f"- 综合分数: {row['overall_score']:.4f}")
            lines.append(f"- 问题: {row['question']}")
            lines.append(f"- 回答: {row['answer'][:200]}")
            try:
                scores = json.loads(row["scores"])
                for metric, score in scores.items():
                    display_name = _METRIC_NAMES.get(metric, metric)
                    lines.append(f"- {display_name}: {score:.4f}")
            except Exception:
                pass
            try:
                failed = json.loads(row["failed_rules"])
                if failed:
                    lines.append(f"- 未通过规则: {', '.join(failed)}")
            except Exception:
                pass
            lines.append("")
        return "\n".join(lines)

    def export_csv(
        self,
        output_path: str | Path,
        since: datetime | None = None,
    ) -> Path:
        """Export all evaluations since ``since`` to a CSV file."""
        since = since or (datetime.now(timezone.utc) - timedelta(days=30))
        since_str = since.isoformat()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM evaluations WHERE created_at > ? ORDER BY created_at DESC",
                (since_str,),
            ).fetchall()

        if not rows:
            output_path.write_text("", encoding="utf-8")
            return output_path

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows([dict(row) for row in rows])
        return output_path
