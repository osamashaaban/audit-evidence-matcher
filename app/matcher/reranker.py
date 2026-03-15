"""
Reranker — passthrough implementation.

OpenAI does not offer a reranker endpoint. The pipeline still calls
reranker.rerank() but it returns candidates unchanged.
This maintains the 4-stage pipeline interface.
"""

from __future__ import annotations

import logging

from app.config import RerankerConfig

logger = logging.getLogger(__name__)


class NoOpReranker:
    """Passthrough reranker — returns candidates unchanged."""

    is_available = False

    def rerank(
        self,
        query_text: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        for c in candidates:
            c.setdefault("rerank_score", c.get("similarity", 0.0))
        return candidates[:top_k]


def get_reranker(config: RerankerConfig):
    logger.info("Reranker disabled — using passthrough")
    return NoOpReranker()