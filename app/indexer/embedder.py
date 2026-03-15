"""
Embedding layer using OpenAI text-embedding-3-small.

Batch support for cost efficiency (Req 5).
Cost logging integrated at this level.
"""

from __future__ import annotations

import logging

from app.config import EmbeddingConfig
from app.cost_logger import cost_logger

logger = logging.getLogger(__name__)


class Embedder:
    """
    Embedding via OpenAI API (text-embedding-3-small).

    Req 5: batch calls for efficiency, log tokens and cost per call.
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        from openai import OpenAI
        if not self.config.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY in your .env file."
            )
        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )
        return self._client

    def _embed_batch(self, texts: list[str], operation: str) -> list[list[float]]:
        """Core batch embedding with cost logging."""
        client = self._get_client()

        response = client.embeddings.create(
            model=self.config.model_name,
            input=texts,
        )

        embeddings = [item.embedding for item in response.data]

        # Cost calculation
        total_tokens = response.usage.total_tokens
        cost = (total_tokens / 1_000_000) * self.config.cost_per_1m_tokens

        cost_logger.log(
            stage="embedding",
            provider="openai",
            model=self.config.model_name,
            operation=operation,
            num_calls=1,
            input_tokens=total_tokens,
            estimated_cost_usd=cost,
        )

        logger.info(
            f"OpenAI embedding: {len(texts)} texts, "
            f"{total_tokens} tokens, ${cost:.6f}"
        )

        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents in a single batch call."""
        return self._embed_batch(texts, "embed_documents")

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        results = self._embed_batch([text], "embed_query")
        return results[0]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple queries in a single batch call."""
        return self._embed_batch(texts, "embed_queries")


def get_embedder(config: EmbeddingConfig) -> Embedder:
    """Create the embedder."""
    logger.info(f"Using OpenAI embedder: {config.model_name}")
    return Embedder(config)