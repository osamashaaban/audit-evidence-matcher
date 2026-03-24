"""
Matching Pipeline — 3-stage orchestrator.

Stage 1: Vector Search     — retrieve top-K candidates via embedding similarity
Stage 2: Deterministic     — rerank candidates by field-level evidence
Stage 3: LLM Explanation   — generate natural-language explanation (optional)

The deterministic score makes the MATCH/NO_MATCH decision.
The LLM provides the explanation only — it cannot override the decision.
"""

from __future__ import annotations

import logging

from app.config import AppConfig
from app.cost_logger import cost_logger
from app.models import Invoice, MatchCandidate, MatchResult
from app.indexer.vector_store import VectorStore
from app.matcher import reranker
from app.matcher import llm_judge

logger = logging.getLogger(__name__)


class MatchingPipeline:
    """Orchestrates the 3-stage matching pipeline."""

    def __init__(self, config: AppConfig, vector_store: VectorStore):
        self.config = config
        self.store = vector_store

    def match_invoice(self, invoice: Invoice) -> MatchResult:
        """
        Match one invoice against the journal store. Returns a MatchResult.

        Flow:
            1. Vector search → top-K candidates
            2. Deterministic reranking → scored + sorted by field evidence
            3. Decision → threshold check on deterministic score
            4. LLM explanation → one API call (optional)
        """

        # ── Stage 1: Vector Search (recall) ──
        raw_candidates = self.store.query_by_invoice(
            invoice,
            top_k=self.config.matching.top_k_retrieval,
        )

        if not raw_candidates:
            logger.warning(f"No candidates found for {invoice.invoice_no}")
            return MatchResult(
                invoice=invoice,
                outcome="NO_MATCH",
                confidence=0.0,
                explanation="No candidates found in the vector store.",
            )

        logger.info(
            f"Stage 1 (Vector): {len(raw_candidates)} candidates retrieved "
            f"for {invoice.invoice_no}"
        )

        # ── Stage 2: Deterministic Reranking ──
        ranked = reranker.rerank(
            invoice=invoice,
            raw_candidates=raw_candidates,
            config=self.config.matching,
        )

        if not ranked:
            return MatchResult(
                invoice=invoice,
                outcome="NO_MATCH",
                confidence=0.0,
                explanation="No candidates could be scored.",
            )

        logger.info(
            f"Stage 2 (Reranker): top = {ranked[0].journal_ref} "
            f"(score={ranked[0].combined_score:.3f})"
        )

        # ── Decision (deterministic threshold) ──
        best = ranked[0]
        is_match = best.combined_score >= self.config.matching.match_threshold

        decision = "MATCHED" if is_match else "NO_MATCH"
        best_match = best.journal_ref if is_match else None

        logger.info(
            f"Decision: {decision} "
            f"(score={best.combined_score:.3f}, "
            f"threshold={self.config.matching.match_threshold})"
        )

        # ── Stage 3: LLM Explanation (optional) ──
        if self.config.matching.use_llm_judge:
            llm_result = llm_judge.explain(
                invoice=invoice,
                candidates=ranked[:3],
                decision=decision,
                best_match=best_match,
                config=self.config.llm,
            )
            explanation = llm_result.get("explanation", "")
        else:
            # Build explanation from evidence (no API call)
            fallback = llm_judge._build_explanation_from_evidence(
                ranked, decision, best_match
            )
            explanation = fallback.get("explanation", "")

        # ── Build Result ──
        top_candidates = ranked[:max(3, self.config.matching.min_candidates_to_show)]

        return MatchResult(
            invoice=invoice,
            outcome=decision,
            best_match=best_match,
            confidence=best.combined_score,
            explanation=explanation,
            evidence=best.evidence if is_match else {},
            top_candidates=top_candidates,
        )

    def match_invoices(self, invoices: list[Invoice]) -> list[MatchResult]:
        """Match multiple invoices sequentially."""
        results = []
        for i, invoice in enumerate(invoices, 1):
            logger.info(
                f"Matching invoice {i}/{len(invoices)}: {invoice.invoice_no}"
            )
            result = self.match_invoice(invoice)
            results.append(result)
            logger.info(
                f"Result: {result.outcome} "
                f"(match={result.best_match}, confidence={result.confidence:.3f})"
            )
        return results