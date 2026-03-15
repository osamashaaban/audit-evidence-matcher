"""
Matching Pipeline — Orchestrates all 4 stages.

Stage 1: Vector Recall      — retrieve top-K candidates via embedding similarity
Stage 2: Cross-Encoder       — rerank candidates (optional)
Stage 3: Deterministic Score — field-level matching (invoice_no, vendor, amount, date)
Stage 4: LLM Judge           — explanation generation + advisory opinion

CRITICAL DESIGN: The deterministic combined score makes the final MATCH/NO_MATCH
decision. The LLM is an ADVISOR that provides explanations, NOT the decision maker.

Why: In testing, the LLM returned NO_MATCH with confidence 0.00
for INV-ACME-1001 even though deterministic scoring found an exact invoice number
match with combined score 0.807. The LLM cannot be trusted to override hard evidence.
In audit, field-level evidence (invoice_no, vendor, amount) is more reliable than
an LLM's opinion.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from app.config import AppConfig
from app.indexer.vector_store import VectorStore
from app.matcher.deterministic import (
    compute_deterministic_score,
    generate_candidate_rationale,
)
from app.matcher.llm_judge import get_llm_judge
from app.matcher.reranker import get_reranker
from app.models import Invoice, MatchCandidate, MatchResult

logger = logging.getLogger(__name__)


class MatchingPipeline:
    """
    End-to-end matching pipeline.

    Usage:
        pipeline = MatchingPipeline(config, vector_store)
        result = pipeline.match_invoice(invoice)
    """

    def __init__(self, config: AppConfig, vector_store: VectorStore):
        self.config = config
        self.store = vector_store
        self.reranker = get_reranker(config.reranker)
        self.judge = get_llm_judge(config.llm, enabled=config.matching.use_llm_judge)

    def match_invoice(self, invoice: Invoice) -> MatchResult:
        """
        Run the full 4-stage pipeline for a single invoice.

        Args:
            invoice: Parsed Invoice from PDF.

        Returns:
            MatchResult with outcome, confidence, explanation,
            evidence, and top candidates.
        """
        logger.info(f"Starting matching for invoice: {invoice.invoice_no}")

        # ── Stage 1: Vector Recall ──
        raw_candidates = self.store.query_by_invoice(
            invoice,
            top_k=self.config.matching.top_k_retrieval,
        )

        if not raw_candidates:
            logger.warning(f"No candidates found for {invoice.invoice_no}")
            return MatchResult(
                invoice=invoice,
                outcome="NO_MATCH",
                confidence=0.9,
                explanation="No journal groups found in the vector store.",
            )

        logger.info(
            f"Stage 1 (Vector): {len(raw_candidates)} candidates retrieved"
        )

        # ── Stage 2: Cross-Encoder Reranking ──
        invoice_text = invoice.to_embedding_text()

        reranked = self.reranker.rerank(
            invoice_text,
            raw_candidates,
            top_k=self.config.reranker.top_k_output
            if self.reranker.is_available
            else len(raw_candidates),
        )

        logger.info(
            f"Stage 2 (Reranker): {len(reranked)} candidates after reranking "
            f"(active: {self.reranker.is_available})"
        )

        # ── Stage 3: Deterministic Scoring ──
        match_candidates: list[MatchCandidate] = []

        for c in reranked:
            metadata = c.get("metadata", {})

            # Get all invoice numbers for this group (noisy dataset)
            candidate_inv_nos = None
            doc_text = c.get("document", "")
            if "All Invoice Numbers:" in doc_text:
                match = re.search(r"All Invoice Numbers:\s*(.+?)(?:\s*\||$)", doc_text)
                if match:
                    candidate_inv_nos = [
                        n.strip() for n in match.group(1).split(",")
                    ]

            det_score, evidence = compute_deterministic_score(
                invoice=invoice,
                candidate_metadata=metadata,
                config=self.config.matching,
                candidate_invoice_nos=candidate_inv_nos,
            )

            # Vector similarity is for RETRIEVAL (getting candidates into top-10).
            # Deterministic score is for RERANKING (ordering them by field evidence).
            # We don't mix them — deterministic alone decides the ranking.
            vector_score = c.get("similarity", 0.0)
            combined = det_score

            rationale = generate_candidate_rationale(evidence)

            mc = MatchCandidate(
                journal_ref=c.get("journal_ref", metadata.get("journal_ref", "")),
                vendor_name=metadata.get("vendor_name", ""),
                invoice_no=metadata.get("invoice_no", ""),
                total_amount=metadata.get("total_amount", 0.0),
                vector_score=vector_score,
                rerank_score=vector_score,  # passthrough (no separate reranker)
                deterministic_score=det_score,
                combined_score=combined,
                evidence=evidence,
                rationale=rationale,
            )
            match_candidates.append(mc)

        # Sort by combined score descending
        match_candidates.sort(key=lambda x: x.combined_score, reverse=True)

        best = match_candidates[0]
        logger.info(
            f"Stage 3 (Deterministic): Top candidate = "
            f"{best.journal_ref} (combined={best.combined_score:.3f})"
        )

        # ── Deterministic Decision (THE authority) ──
        det_says_match = best.combined_score >= self.config.matching.match_threshold

        # ── Stage 4: LLM Judge (ADVISOR for explanation only) ──
        top_for_judge = match_candidates[: max(self.config.matching.min_candidates_to_show, 3)]

        judge_result = self.judge.judge(invoice, top_for_judge)

        llm_decision = judge_result.get("decision", "NO_MATCH")
        llm_confidence = judge_result.get("confidence", 0.0)
        llm_explanation = judge_result.get("explanation", "")
        llm_match_ref = judge_result.get("best_match_journal_ref")

        logger.info(
            f"Stage 4 (LLM Judge advisory): {llm_decision} "
            f"(confidence={llm_confidence:.2f})"
        )

        # ── Reconcile: Deterministic decides, LLM explains ──
        #
        # The deterministic score is the FINAL AUTHORITY because:
        # 1. It's based on verifiable field evidence (invoice_no, vendor, amount)
        # 2. In testing, the LLM returned NO_MATCH (0.00 confidence) for an
        #    invoice with exact invoice_no match and 0.807 combined score
        # 3. For audit, hard evidence > LLM opinion
        #
        # The LLM ADDS VALUE by providing natural-language explanations
        # but CANNOT override the deterministic decision.

        if det_says_match:
            final_decision = "MATCHED"
            final_match = best.journal_ref
            final_confidence = best.combined_score

            if llm_decision == "MATCHED" and llm_explanation:
                # LLM agrees — use its explanation (richer language)
                final_explanation = llm_explanation
            else:
                # LLM disagrees or failed — build explanation from evidence
                final_explanation = self._build_match_explanation(invoice, best)
                if llm_decision != "MATCHED":
                    logger.warning(
                        f"LLM disagreed with deterministic match for "
                        f"{invoice.invoice_no} (LLM said {llm_decision}, "
                        f"det score={best.combined_score:.3f}). "
                        f"Deterministic decision overrides."
                    )
        else:
            final_decision = "NO_MATCH"
            final_match = None
            final_confidence = max(0.6, 1.0 - best.combined_score)

            if llm_decision == "NO_MATCH" and llm_explanation:
                # LLM agrees — use its explanation
                final_explanation = llm_explanation
            else:
                final_explanation = self._build_nomatch_explanation(invoice, best)

        # ── Merge LLM candidate rationales if available ──
        judge_analyses = {
            a.get("journal_ref", ""): a.get("rationale", "")
            for a in judge_result.get("candidates_analysis", [])
        }
        for mc in match_candidates:
            llm_rationale = judge_analyses.get(mc.journal_ref, "")
            if llm_rationale:
                mc.rationale = llm_rationale

        # ── Best match evidence ──
        best_evidence = {}
        if final_match:
            for mc in match_candidates:
                if mc.journal_ref == final_match:
                    best_evidence = mc.evidence
                    break

        result = MatchResult(
            invoice=invoice,
            outcome=final_decision,
            best_match=final_match,
            confidence=final_confidence,
            explanation=final_explanation,
            evidence=best_evidence,
            top_candidates=match_candidates[: max(
                self.config.matching.min_candidates_to_show, 3
            )],
        )

        logger.info(
            f"Final result for {invoice.invoice_no}: "
            f"{result.outcome} → {result.best_match} "
            f"(confidence={result.confidence:.2f})"
        )

        return result

    def _build_match_explanation(self, invoice: Invoice, best: MatchCandidate) -> str:
        """Build a clear explanation when deterministic scoring finds a match."""
        parts = [
            f"Invoice {invoice.invoice_no} matched to {best.journal_ref} "
            f"with a combined score of {best.combined_score:.2f}."
        ]

        ev = best.evidence
        matched_fields = []
        mismatched_fields = []

        for field_key, label in [
            ("invoice_no_match", "invoice number"),
            ("vendor_match", "vendor name"),
            ("amount_match", "amount"),
            ("date_match", "date"),
            ("vat_id_match", "VAT ID"),
        ]:
            field_ev = ev.get(field_key, {})
            if field_ev.get("matched"):
                matched_fields.append(label)
            elif field_ev.get("invoice_value") or field_ev.get("journal_value"):
                mismatched_fields.append(label)

        if matched_fields:
            parts.append(f"Matching fields: {', '.join(matched_fields)}.")
        if mismatched_fields:
            parts.append(f"Non-matching fields: {', '.join(mismatched_fields)}.")

        return " ".join(parts)

    def _build_nomatch_explanation(self, invoice: Invoice, best: MatchCandidate) -> str:
        """Build a clear explanation when no match is found."""
        return (
            f"No journal group matched invoice {invoice.invoice_no}. "
            f"The closest candidate was {best.journal_ref} with a combined "
            f"score of {best.combined_score:.2f}, which is below the match "
            f"threshold. {best.rationale}"
        )

    def match_invoices(self, invoices: list[Invoice]) -> list[MatchResult]:
        """Match multiple invoices sequentially."""
        results = []
        for invoice in invoices:
            result = self.match_invoice(invoice)
            results.append(result)
        return results