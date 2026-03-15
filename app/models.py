"""
Core data models for the Evidence-to-Journal Matching pipeline.

All structured data flows through these dataclasses.
No business logic here — just data containers with clear types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ──────────────────────────────────────────────
# Invoice (parsed from PDF)
# ──────────────────────────────────────────────

@dataclass
class Invoice:
    """Structured representation of a parsed PDF invoice."""

    vendor_name: str
    invoice_no: str
    invoice_date: str  # kept as string for flexibility (YYYY-MM-DD)
    subtotal: Optional[float] = None
    vat_rate: Optional[float] = None
    vat_amount: Optional[float] = None
    total_amount: Optional[float] = None
    currency: str = "USD"
    vat_id: Optional[str] = None
    payment_terms: Optional[str] = None
    line_items: list[dict] = field(default_factory=list)
    raw_text: str = ""  # Req 4.3: "Keep a text copy of the OCR output"
    source_file: str = ""

    def to_embedding_text(self) -> str:
        """Build text representation for embedding (query side)."""
        parts = [
            f"Vendor: {self.vendor_name}",
            f"Invoice Number: {self.invoice_no}",
            f"Date: {self.invoice_date}",
        ]
        if self.subtotal is not None:
            parts.append(f"Subtotal: {self.subtotal} {self.currency}")
        if self.vat_amount is not None:
            parts.append(f"VAT: {self.vat_amount} {self.currency}")
        if self.total_amount is not None:
            parts.append(f"Total: {self.total_amount} {self.currency}")
        if self.vat_id:
            parts.append(f"VAT ID: {self.vat_id}")
        if self.payment_terms:
            parts.append(f"Payment Terms: {self.payment_terms}")
        if self.line_items:
            descs = [item.get("description", "") for item in self.line_items]
            parts.append(f"Items: {', '.join(descs)}")
        return " | ".join(parts)

    def to_dict(self) -> dict:
        """Serialize for display / JSON output."""
        return {
            "vendor_name": self.vendor_name,
            "invoice_no": self.invoice_no,
            "invoice_date": self.invoice_date,
            "subtotal": self.subtotal,
            "vat_rate": self.vat_rate,
            "vat_amount": self.vat_amount,
            "total_amount": self.total_amount,
            "currency": self.currency,
            "vat_id": self.vat_id,
            "payment_terms": self.payment_terms,
            "line_items": self.line_items,
            "source_file": self.source_file,
        }


# ──────────────────────────────────────────────
# Journal Group (parsed from XLSX, grouped by journal_ref)
# ──────────────────────────────────────────────

@dataclass
class JournalGroup:
    """
    One group of journal entries sharing the same journal_ref.

    Req 4.2: "create one embedding document per journal_ref group
    containing vendor name, invoice number(s), descriptions, totals,
    and line details."
    """

    journal_ref: str
    vendor_name: str
    vendor_vat_id: Optional[str] = None
    invoice_no: Optional[str] = None  # primary (first non-null in group)
    invoice_nos: list[str] = field(default_factory=list)  # all unique in group
    invoice_date: Optional[str] = None
    posting_date: Optional[str] = None
    total_amount: Optional[float] = None  # from total_amount column
    sum_of_amounts: Optional[float] = None  # sum of individual 'amount' values
    currency: str = "USD"
    payment_terms: Optional[str] = None
    descriptions: list[str] = field(default_factory=list)
    source_system: Optional[str] = None
    cost_center: Optional[str] = None
    project: Optional[str] = None
    line_count: int = 0
    lines: list[dict] = field(default_factory=list)  # raw line-level data

    def to_embedding_text(self) -> str:
        """
        Build text for embedding (document side).

        This is the single text stored in the vector store per group.
        Includes every field that could help matching.
        """
        parts = [
            f"Journal Reference: {self.journal_ref}",
            f"Vendor: {self.vendor_name}",
        ]
        if self.invoice_no:
            parts.append(f"Invoice Number: {self.invoice_no}")
        if len(self.invoice_nos) > 1:
            parts.append(f"All Invoice Numbers: {', '.join(self.invoice_nos)}")
        if self.vendor_vat_id:
            parts.append(f"VAT ID: {self.vendor_vat_id}")
        if self.invoice_date:
            parts.append(f"Invoice Date: {self.invoice_date}")
        if self.posting_date:
            parts.append(f"Posting Date: {self.posting_date}")
        if self.total_amount is not None:
            parts.append(f"Total Amount: {self.total_amount} {self.currency}")
        if self.sum_of_amounts is not None:
            parts.append(f"Sum of Line Amounts: {self.sum_of_amounts} {self.currency}")
        if self.payment_terms:
            parts.append(f"Payment Terms: {self.payment_terms}")
        if self.descriptions:
            parts.append(f"Descriptions: {' ; '.join(self.descriptions)}")
        return " | ".join(parts)

    def to_metadata(self) -> dict:
        """
        Build metadata dict for ChromaDB storage.

        Req 4.2: "Store metadata alongside embeddings
        (journal_ref, posting_date, totals, vendor, invoice_no)."

        ChromaDB metadata values must be str, int, float, or bool.
        """
        return {
            "journal_ref": self.journal_ref,
            "vendor_name": self.vendor_name or "",
            "vendor_vat_id": self.vendor_vat_id or "",
            "invoice_no": self.invoice_no or "",
            "invoice_date": self.invoice_date or "",
            "posting_date": self.posting_date or "",
            "total_amount": float(self.total_amount) if self.total_amount is not None else 0.0,
            "sum_of_amounts": float(self.sum_of_amounts) if self.sum_of_amounts is not None else 0.0,
            "currency": self.currency,
            "payment_terms": self.payment_terms or "",
            "source_system": self.source_system or "",
            "line_count": self.line_count,
        }

    def to_dict(self) -> dict:
        """Serialize for display."""
        return {
            "journal_ref": self.journal_ref,
            "vendor_name": self.vendor_name,
            "vendor_vat_id": self.vendor_vat_id,
            "invoice_no": self.invoice_no,
            "invoice_nos": self.invoice_nos,
            "invoice_date": self.invoice_date,
            "posting_date": self.posting_date,
            "total_amount": self.total_amount,
            "sum_of_amounts": self.sum_of_amounts,
            "currency": self.currency,
            "payment_terms": self.payment_terms,
            "descriptions": self.descriptions,
            "line_count": self.line_count,
        }


# ──────────────────────────────────────────────
# Match Candidate (one entry in the top-K list)
# ──────────────────────────────────────────────

@dataclass
class MatchCandidate:
    """A single candidate from the matching pipeline with all scores."""

    journal_ref: str
    vendor_name: str = ""
    invoice_no: str = ""
    total_amount: float = 0.0

    # Scores from each pipeline stage
    vector_score: float = 0.0        # Stage 1: cosine similarity
    rerank_score: float = 0.0        # Stage 2: cross-encoder (optional)
    deterministic_score: float = 0.0  # Stage 3: field-level matching
    combined_score: float = 0.0      # Final weighted combination

    # Field-level evidence for this candidate
    evidence: dict = field(default_factory=dict)
    rationale: str = ""  # concise rationale (Req 4.1)

    def to_dict(self) -> dict:
        return {
            "journal_ref": self.journal_ref,
            "vendor_name": self.vendor_name,
            "invoice_no": self.invoice_no,
            "total_amount": self.total_amount,
            "vector_score": round(self.vector_score, 4),
            "rerank_score": round(self.rerank_score, 4),
            "deterministic_score": round(self.deterministic_score, 4),
            "combined_score": round(self.combined_score, 4),
            "evidence": self.evidence,
            "rationale": self.rationale,
        }


# ──────────────────────────────────────────────
# Match Result (final output per invoice)
# ──────────────────────────────────────────────

@dataclass
class MatchResult:
    """
    Final matching result for one invoice.

    Req 4.4 Return:
      - best match (journal_ref) OR No Match
      - confidence score (0–1) and explanation
      - evidence used (which fields matched, which didn't)

    Req 4.1 Results View:
      - Extracted invoice fields
      - Matching outcome: Matched or No Match
      - Top candidates (≥ top-3): journal_ref, score, rationale
      - Explanation of why the match was chosen (or why No Match)
    """

    invoice: Invoice
    outcome: str  # "MATCHED" or "NO_MATCH"
    best_match: Optional[str] = None  # journal_ref or None
    confidence: float = 0.0
    explanation: str = ""
    evidence: dict = field(default_factory=dict)
    top_candidates: list[MatchCandidate] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "invoice": self.invoice.to_dict(),
            "outcome": self.outcome,
            "best_match": self.best_match,
            "confidence": round(self.confidence, 4),
            "explanation": self.explanation,
            "evidence": self.evidence,
            "top_candidates": [c.to_dict() for c in self.top_candidates],
        }


# ──────────────────────────────────────────────
# Cost Log Entry (Req 5: cost discipline tracking)
# ──────────────────────────────────────────────

@dataclass
class CostLogEntry:
    """
    Tracks a single API call for cost discipline.

    Req 5: "Log: number of OpenAI calls, approximate tokens,
    and estimated cost per stage."
    """

    stage: str       # "embedding", "reranking", "llm_judge"
    provider: str    # "openai" or "local"
    model: str       # "text-embedding-3-small", "gpt-4o-mini", etc.
    operation: str   # "embed_journals", "embed_invoices", "judge_invoice"
    num_calls: int = 1
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "provider": self.provider,
            "model": self.model,
            "operation": self.operation,
            "num_calls": self.num_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "timestamp": self.timestamp,
        }

