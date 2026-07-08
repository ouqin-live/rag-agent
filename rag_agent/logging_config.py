"""Centralised logging configuration with noisy third-party loggers suppressed."""

from __future__ import annotations

import logging
import os

# Third-party loggers that are useful at WARNING but too verbose at INFO.
_NOISY_LOGGERS = [
    "httpx",
    "httpcore",
    "openai._base_client",
    "openai",
    "sentence_transformers",
    "chromadb",
    "urllib3",
    "requests",
    "filelock",
    "huggingface_hub",
    "PIL",
]


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure root logging and suppress overly verbose third-party loggers."""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Disable tqdm progress bars globally for cleaner CLI / server logs.
    os.environ.setdefault("TQDM_DISABLE", "1")

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
