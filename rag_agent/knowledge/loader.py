"""Document loaders for local files and URLs."""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC
from pathlib import Path
from urllib.parse import urlparse

from rag_agent.knowledge.base import BaseLoader, Document

logger = logging.getLogger(__name__)


def _make_doc_id(source: str, content: str) -> str:
    """Generate a deterministic document id."""
    return hashlib.sha256(f"{source}:{content}".encode("utf-8")).hexdigest()[:32]


def _strip_markdown_frontmatter(text: str) -> str:
    """Remove YAML frontmatter from markdown if present."""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return text


class TextLoader(BaseLoader, ABC):
    """Load plain text files (.txt, .md)."""

    extensions: tuple[str, ...] = (".txt",)

    def load(self, source: str) -> list[Document]:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {source}")
        content = path.read_text(encoding="utf-8")
        doc_id = _make_doc_id(source, content)
        return [
            Document(
                id=doc_id,
                content=content,
                source=source,
                metadata={"type": "text", "filename": path.name},
            )
        ]


class MarkdownLoader(TextLoader):
    """Load markdown files and strip YAML frontmatter."""

    extensions = (".md", ".markdown")

    def load(self, source: str) -> list[Document]:
        docs = super().load(source)
        for doc in docs:
            doc.content = _strip_markdown_frontmatter(doc.content)
            doc.metadata["type"] = "markdown"
        return docs


class PdfLoader(BaseLoader, ABC):
    """Load PDF files using PyMuPDF (fitz)."""

    extensions = (".pdf",)

    def load(self, source: str) -> list[Document]:
        try:
            import fitz  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "PDF loading requires 'pymupdf'. Install it with: uv add pymupdf"
            ) from exc

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {source}")

        doc = fitz.open(source)
        parts = []
        for page in doc:
            text = page.get_text()
            if text:
                parts.append(text)
        content = "\n".join(parts)
        doc_id = _make_doc_id(source, content)
        return [
            Document(
                id=doc_id,
                content=content,
                source=source,
                metadata={"type": "pdf", "filename": path.name, "pages": len(doc)},
            )
        ]


class UrlLoader(BaseLoader, ABC):
    """Load a web page and extract readable text."""

    extensions = ()

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def load(self, source: str) -> list[Document]:
        try:
            import requests
        except ImportError as exc:
            raise ImportError(
                "URL loading requires 'requests'. Install it with: uv add requests"
            ) from exc

        parsed = urlparse(source)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL: {source}")

        response = requests.get(source, timeout=self.timeout)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")

        if "application/pdf" in content_type:
            raise NotImplementedError("PDF via URL is not supported yet")

        text = self._extract_text(response.text)
        doc_id = _make_doc_id(source, text)
        return [
            Document(
                id=doc_id,
                content=text,
                source=source,
                metadata={"type": "url", "content_type": content_type},
            )
        ]

    @staticmethod
    def _extract_text(html: str) -> str:
        """Very basic HTML-to-text extraction."""
        # Remove script/style blocks
        html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        # Replace common block tags with newlines
        html = re.sub(r"</(p|div|h[1-6]|li|tr|br)\s*>", "\n", html, flags=re.IGNORECASE)
        # Strip remaining tags
        text = re.sub(r"<[^>]+>", " ", html)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text


class AutoLoader(BaseLoader, ABC):
    """Pick an appropriate loader based on the source string."""

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self._loaders = [
            MarkdownLoader(),
            TextLoader(),
            PdfLoader(),
        ]
        self._url_loader = UrlLoader(timeout=timeout)

    def load(self, source: str) -> list[Document]:
        parsed = urlparse(source)
        if parsed.scheme in ("http", "https"):
            return self._url_loader.load(source)

        lower = source.lower()
        for loader in self._loaders:
            if any(lower.endswith(ext) for ext in loader.extensions):
                return loader.load(source)

        # Default to text loader for unknown local paths
        return TextLoader().load(source)
