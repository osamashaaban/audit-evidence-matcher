"""
Reranker — reranks vector search candidates using deterministic field scoring.

Takes raw candidates from vector search and reranks them by comparing
5 fields: invoice_no, vendor_name, amount, date, VAT ID.

This is the reranking step described in the assignment:
  "Rerank and decide match using a hybrid approach:
   deterministic checks: invoice number similarity, vendor name
   similarity, amount tolerance, optional date proximity"
"""

from __future__ import annotations

import logging
import re

from app.config import MatchingConfig
from app.models import Invoice, MatchCandidate
from app.matcher.deterministic import (
    compute_deterministic_score,
    generate_candidate_rationale,
)

logger = logging.getLogger(__name__)


def rerank(
    invoice: Invoice,
    raw_candidates: list[dict],
    config: MatchingConfig,
) -> list[MatchCandidate]:
    """
    Rerank vector search candidates using deterministic field scoring.

    Args:
        invoice: The invoice being matched.
        raw_candidates: Raw candidates from vector_store.query_by_invoice().
            Each dict has: journal_ref, similarity, metadata, document.
        config: Matching configuration with weights and thresholds.

    Returns:
        List of MatchCandidate objects sorted by deterministic score (desc).
    """
    match_candidates: list[MatchCandidate] = []

    for c in raw_candidates:
        metadata = c.get("metadata", {})

        # Extract all invoice numbers for this group (noisy dataset
        # may have multiple formats on different rows)
        candidate_inv_nos = None
        doc_text = c.get("document", "")
        if "All Invoice Numbers:" in doc_text:
            match = re.search(
                r"All Invoice Numbers:\s*(.+?)(?:\s*\||$)", doc_text
            )
            if match:
                candidate_inv_nos = [
                    n.strip() for n in match.group(1).split(",")
                ]

        # Score this candidate on 5 fields
        det_score, evidence = compute_deterministic_score(
            invoice=invoice,
            candidate_metadata=metadata,
            config=config,
            candidate_invoice_nos=candidate_inv_nos,
        )

        vector_score = c.get("similarity", 0.0)
        rationale = generate_candidate_rationale(evidence)

        mc = MatchCandidate(
            journal_ref=c.get("journal_ref", metadata.get("journal_ref", "")),
            vendor_name=metadata.get("vendor_name", ""),
            invoice_no=metadata.get("invoice_no", ""),
            total_amount=metadata.get("total_amount", 0.0),
            vector_score=vector_score,
            rerank_score=det_score,
            deterministic_score=det_score,
            combined_score=det_score,
            evidence=evidence,
            rationale=rationale,
        )
        match_candidates.append(mc)

    # Sort by deterministic score — THIS IS THE RERANKING
    match_candidates.sort(key=lambda x: x.combined_score, reverse=True)

    if match_candidates:
        best = match_candidates[0]
        logger.info(
            f"Reranker: {len(match_candidates)} candidates scored. "
            f"Top = {best.journal_ref} ({best.combined_score:.3f})"
        )
    else:
        logger.warning("Reranker: no candidates to score")

    return match_candidates