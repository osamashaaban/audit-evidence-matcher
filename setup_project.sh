#!/bin/bash

# ══════════════════════════════════════════════════════════════
# Evidence-to-Journal Matching — Full Project Setup Script
# ══════════════════════════════════════════════════════════════
# Usage:
#   1. Clone your repo:  git clone <repo-url> audit-evidence-matcher
#   2. cd audit-evidence-matcher
#   3. bash setup_project.sh
#   4. conda activate accord
#   5. streamlit run app/dashboard/streamlit_app.py
# ══════════════════════════════════════════════════════════════

set -e  # Exit on any error

echo "══════════════════════════════════════════════════════════"
echo "  Setting up Evidence-to-Journal Matching project"
echo "══════════════════════════════════════════════════════════"

# ── Create directory structure ──
echo "Creating directories..."
mkdir -p app
mkdir -p app/parsers
mkdir -p app/indexer
mkdir -p app/matcher
mkdir -p app/dashboard
mkdir -p data
mkdir -p vectordb
mkdir -p logs
mkdir -p tests
mkdir -p docs

echo "  Creating app/__init__.py..."
cat > 'app/__init__.py' << 'FILEEOF'

FILEEOF

echo "  Creating app/models.py..."
cat > 'app/models.py' << 'FILEEOF'
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

FILEEOF

echo "  Creating app/config.py..."
cat > 'app/config.py' << 'FILEEOF'
"""
Centralized configuration for the Evidence-to-Journal Matching pipeline.

Switch between local (Qwen3) and OpenAI by changing PROVIDER settings.
All model names, endpoints, and thresholds live here.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VECTORDB_DIR = PROJECT_ROOT / "vectordb"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
VECTORDB_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────
# Provider Toggle
# ──────────────────────────────────────────────

# Set via environment or sidebar toggle in Streamlit
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")  # "local" or "openai"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local")              # "local" or "openai"


# ──────────────────────────────────────────────
# Embedding Config
# ──────────────────────────────────────────────

@dataclass
class EmbeddingConfig:
    provider: str = "local"

    # Local (Qwen3-Embedding-0.6B via sentence-transformers)
    local_model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    local_task_instruction: str = (
        "Given an invoice from a vendor, retrieve the matching "
        "journal entry group that records this transaction"
    )

    # OpenAI
    openai_model_name: str = "text-embedding-3-small"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    # Shared
    embedding_dimension: int = 1024  # Qwen3-Embedding-0.6B default
    batch_size: int = 32

    # Cost tracking (OpenAI pricing per 1M tokens)
    openai_cost_per_1m_tokens: float = 0.02  # text-embedding-3-small

    def __post_init__(self):
        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY", "")


# ──────────────────────────────────────────────
# LLM Config
# ──────────────────────────────────────────────

@dataclass
class LLMConfig:
    provider: str = "local"

    # Local (Qwen3.5-35B via vLLM with OpenAI-compatible API)
    local_model_name: str = "Qwen/Qwen3.5-35B-A3B-FP8"
    local_base_url: str = "http://10.0.9.75:8010/v1"
    local_api_key: str = "na"
    local_extra_body: dict = field(default_factory=lambda: {
        "chat_template_kwargs": {"enable_thinking": False}
    })

    # OpenAI
    openai_model_name: str = "gpt-4o-mini"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    # Shared
    temperature: float = 0.0  # deterministic for audit tasks
    max_tokens: int = 1000

    # Cost tracking (OpenAI pricing per 1M tokens)
    openai_input_cost_per_1m: float = 0.15   # gpt-4o-mini input
    openai_output_cost_per_1m: float = 0.60  # gpt-4o-mini output

    def __post_init__(self):
        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY", "")


# ──────────────────────────────────────────────
# Reranker Config (Optional — differentiator)
# ──────────────────────────────────────────────

@dataclass
class RerankerConfig:
    enabled: bool = True
    model_name: str = "Qwen/Qwen3-Reranker-0.6B"
    top_k_input: int = 10   # how many candidates from vector search
    top_k_output: int = 5   # how many to pass to deterministic stage


# ──────────────────────────────────────────────
# Vector Store Config
# ──────────────────────────────────────────────

@dataclass
class VectorStoreConfig:
    persist_dir: str = str(VECTORDB_DIR)
    journal_collection_name: str = "journal_groups"
    invoice_collection_name: str = "invoices"  # Req 4.3: index invoices too
    distance_metric: str = "cosine"


# ──────────────────────────────────────────────
# Matching Config
# ──────────────────────────────────────────────

@dataclass
class MatchingConfig:
    # Vector search
    top_k_retrieval: int = 10  # K ≈ 5–10 as per assignment

    # Deterministic scoring weights
    weight_invoice_no: float = 0.35
    weight_vendor_name: float = 0.25
    weight_amount: float = 0.20
    weight_date: float = 0.10
    weight_vat_id: float = 0.10

    # Thresholds
    fuzzy_invoice_no_threshold: float = 0.92
    fuzzy_vendor_name_threshold: float = 0.75
    amount_tolerance_pct: float = 0.05  # 5% tolerance
    date_proximity_days: int = 7

    # Combined score weights
    weight_vector: float = 0.20
    weight_reranker: float = 0.30
    weight_deterministic: float = 0.50

    # Final decision
    match_threshold: float = 0.60
    min_candidates_to_show: int = 3  # Req 4.1: "at least top-3"

    # LLM judge
    use_llm_judge: bool = True  # "optional" per assignment but adds quality


# ──────────────────────────────────────────────
# Master Config
# ──────────────────────────────────────────────

@dataclass
class AppConfig:
    """Single config object passed throughout the application."""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)


def get_config(
    embedding_provider: str = "local",
    llm_provider: str = "local",
    reranker_enabled: bool = True,
    use_llm_judge: bool = True,
) -> AppConfig:
    """
    Build application config.

    Called once at startup or when user changes settings in the sidebar.
    """
    config = AppConfig()
    config.embedding.provider = embedding_provider
    config.llm.provider = llm_provider
    config.reranker.enabled = reranker_enabled
    config.matching.use_llm_judge = use_llm_judge
    return config

FILEEOF

echo "  Creating app/cost_logger.py..."
cat > 'app/cost_logger.py' << 'FILEEOF'
"""
Cost logging for API call tracking.

Req 5: "Log: number of OpenAI calls, approximate tokens,
and estimated cost per stage."

This module provides a singleton-style logger that any module can import
and log to. The Streamlit dashboard reads from it to display the sidebar
cost tracker.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from app.models import CostLogEntry
from app.config import LOGS_DIR


class CostLogger:
    """Accumulates cost log entries and provides per-stage summaries."""

    def __init__(self):
        self._entries: list[CostLogEntry] = []

    def log(
        self,
        stage: str,
        provider: str,
        model: str,
        operation: str,
        num_calls: int = 1,
        input_tokens: int = 0,
        output_tokens: int = 0,
        estimated_cost_usd: float = 0.0,
    ) -> CostLogEntry:
        """Record a single API call."""
        entry = CostLogEntry(
            stage=stage,
            provider=provider,
            model=model,
            operation=operation,
            num_calls=num_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )
        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> list[CostLogEntry]:
        return list(self._entries)

    def clear(self):
        """Reset all entries."""
        self._entries.clear()

    def summary_by_stage(self) -> dict[str, dict]:
        """
        Aggregate stats grouped by stage.

        Returns:
            {
                "embedding": {
                    "num_calls": 2,
                    "input_tokens": 500,
                    "output_tokens": 0,
                    "estimated_cost_usd": 0.001,
                    "provider": "local",
                    "model": "Qwen3-Embedding-0.6B"
                },
                "llm_judge": { ... },
                ...
            }
        """
        stages: dict[str, dict] = {}
        for entry in self._entries:
            if entry.stage not in stages:
                stages[entry.stage] = {
                    "num_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "estimated_cost_usd": 0.0,
                    "provider": entry.provider,
                    "model": entry.model,
                }
            s = stages[entry.stage]
            s["num_calls"] += entry.num_calls
            s["input_tokens"] += entry.input_tokens
            s["output_tokens"] += entry.output_tokens
            s["estimated_cost_usd"] += entry.estimated_cost_usd
        return stages

    def total_cost(self) -> float:
        """Total estimated cost across all entries."""
        return sum(e.estimated_cost_usd for e in self._entries)

    def total_calls(self) -> int:
        """Total number of API calls."""
        return sum(e.num_calls for e in self._entries)

    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return sum(e.input_tokens + e.output_tokens for e in self._entries)

    def save_to_file(self, filename: Optional[str] = None) -> Path:
        """Persist log to JSON file for reproducibility."""
        if filename is None:
            filename = "cost_log.json"
        filepath = LOGS_DIR / filename
        data = {
            "summary": {
                "total_calls": self.total_calls(),
                "total_tokens": self.total_tokens(),
                "total_cost_usd": round(self.total_cost(), 6),
            },
            "by_stage": {
                stage: {k: round(v, 6) if isinstance(v, float) else v for k, v in info.items()}
                for stage, info in self.summary_by_stage().items()
            },
            "entries": [e.to_dict() for e in self._entries],
        }
        filepath.write_text(json.dumps(data, indent=2))
        return filepath

    def format_for_display(self) -> str:
        """Human-readable summary for the Streamlit sidebar."""
        lines = []
        for stage, info in self.summary_by_stage().items():
            lines.append(
                f"**{stage}**: {info['num_calls']} calls, "
                f"~{info['input_tokens'] + info['output_tokens']} tokens, "
                f"${info['estimated_cost_usd']:.4f}"
            )
        lines.append(f"---")
        lines.append(f"**Total**: {self.total_calls()} calls, "
                      f"~{self.total_tokens()} tokens, "
                      f"**${self.total_cost():.4f}**")
        return "\n\n".join(lines)


# Module-level instance — import and use from anywhere
cost_logger = CostLogger()

FILEEOF

echo "  Creating app/parsers/__init__.py..."
cat > 'app/parsers/__init__.py' << 'FILEEOF'

FILEEOF

echo "  Creating app/parsers/xlsx_parser.py..."
cat > 'app/parsers/xlsx_parser.py' << 'FILEEOF'
"""
XLSX Parser — Journal Entries ingestion.

Req 4.2:
  - Parse the XLSX into journal lines, grouped by journal_ref.
  - Create one embedding document per journal_ref group containing
    vendor name, invoice number(s), descriptions, totals, and line details.
  - Store metadata alongside embeddings.

Req 8 (Noisy Dataset v2):
  - Vendor name variations ('LLC' vs 'L.L.C.')
  - Invoice number formatting changes (spaces, suffixes, missing)
  - Multi-line expense postings + separate VAT line
  - Distractors: similar vendor but different invoice/total
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import Union

import pandas as pd

from app.models import JournalGroup

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Normalization helpers (shared with deterministic matcher)
# ──────────────────────────────────────────────

def normalize_vendor_name(name: str) -> str:
    """
    Normalize vendor name for consistent comparison.

    Handles Req 8 variations:
      'ACME Office Supplies L.L.C.' → 'acme office supplies llc'
      'Nimbus Cloud Svcs, Inc.'     → 'nimbus cloud svcs inc'
    """
    if not name or name == "—":
        return ""
    result = name.lower().strip()
    # Remove dots from abbreviations: L.L.C. → LLC
    result = result.replace(".", "")
    # Remove commas
    result = result.replace(",", "")
    # Normalize common suffixes
    for suffix in ["llc", "l l c", "ltd", "limited", "inc", "incorporated", "co", "corp"]:
        # Collapse variations like "l l c" to "llc"
        result = result.replace("l l c", "llc")
    # Collapse multiple spaces
    result = " ".join(result.split())
    return result


def normalize_invoice_no(inv_no: str) -> str:
    """
    Normalize invoice number for consistent comparison.

    Handles Req 8 variations:
      'INV ACME 1001'  → 'invacme1001'
      'INV-ACME-1001'  → 'invacme1001'
      'NIMBUS-778-A'   → 'nimbus778a'
    """
    if not inv_no:
        return ""
    result = str(inv_no).lower().strip()
    # Remove dashes, spaces, dots
    for char in ["-", " ", ".", "_"]:
        result = result.replace(char, "")
    return result


# ──────────────────────────────────────────────
# Core Parser
# ──────────────────────────────────────────────

def parse_xlsx(file_input: Union[str, Path, BytesIO]) -> pd.DataFrame:
    """
    Read XLSX file into a clean DataFrame.

    Args:
        file_input: File path or BytesIO (from Streamlit uploader).

    Returns:
        DataFrame with all journal entry rows.

    Raises:
        ValueError: If file is empty or missing required columns.
    """
    df = pd.read_excel(file_input)

    if df.empty:
        raise ValueError("The uploaded XLSX file is empty.")

    # Validate required columns exist
    required_cols = {"journal_ref", "vendor_name", "amount", "description"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Clean up: convert date columns to strings for consistency
    for col in ["posting_date", "invoice_date"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: str(x)[:10] if pd.notna(x) else None
            )

    # Convert numeric columns safely
    for col in ["amount", "tax_amount", "total_amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(f"Parsed XLSX: {len(df)} rows, {df['journal_ref'].nunique()} unique groups")
    return df


def group_journal_entries(df: pd.DataFrame) -> list[JournalGroup]:
    """
    Group journal entry rows by journal_ref and build JournalGroup objects.

    Each group becomes one embedding document in the vector store.

    This function handles:
      - Multiple rows per group (expense + VAT lines)
      - Missing invoice_no on some lines (takes first non-null)
      - All unique invoice numbers collected
      - Sum of individual amounts computed
      - All descriptions collected
    """
    groups: list[JournalGroup] = []

    for ref, group_df in df.groupby("journal_ref", sort=False):
        # ── Vendor name: take first non-dash value ──
        vendor_names = group_df["vendor_name"].dropna().unique()
        vendor_name = next(
            (v for v in vendor_names if v != "—"), str(vendor_names[0]) if len(vendor_names) > 0 else ""
        )

        # ── Invoice numbers: collect all unique non-null values ──
        inv_nos = []
        if "invoice_no" in group_df.columns:
            inv_nos = [
                str(v) for v in group_df["invoice_no"].dropna().unique()
                if str(v).strip()
            ]
        primary_inv_no = inv_nos[0] if inv_nos else None

        # ── VAT ID ──
        vat_id = None
        if "vendor_vat_id" in group_df.columns:
            vat_ids = group_df["vendor_vat_id"].dropna()
            if not vat_ids.empty:
                vat_id = str(vat_ids.iloc[0])

        # ── Dates ──
        invoice_date = None
        if "invoice_date" in group_df.columns:
            dates = group_df["invoice_date"].dropna()
            if not dates.empty:
                invoice_date = str(dates.iloc[0])

        posting_date = None
        if "posting_date" in group_df.columns:
            dates = group_df["posting_date"].dropna()
            if not dates.empty:
                posting_date = str(dates.iloc[0])

        # ── Amounts ──
        total_amount = None
        if "total_amount" in group_df.columns:
            totals = group_df["total_amount"].dropna()
            if not totals.empty:
                total_amount = float(totals.iloc[0])

        sum_of_amounts = None
        if "amount" in group_df.columns:
            sum_of_amounts = float(group_df["amount"].sum())

        # ── Currency ──
        currency = "USD"
        if "currency" in group_df.columns:
            currencies = group_df["currency"].dropna()
            if not currencies.empty:
                currency = str(currencies.iloc[0])

        # ── Other fields ──
        payment_terms = None
        if "payment_terms" in group_df.columns:
            terms = group_df["payment_terms"].dropna()
            if not terms.empty:
                payment_terms = str(terms.iloc[0])

        source_system = None
        if "source_system" in group_df.columns:
            systems = group_df["source_system"].dropna()
            if not systems.empty:
                source_system = str(systems.iloc[0])

        cost_center = None
        if "cost_center" in group_df.columns:
            cc = group_df["cost_center"].dropna()
            if not cc.empty:
                cost_center = str(cc.iloc[0])

        project = None
        if "project" in group_df.columns:
            proj = group_df["project"].dropna()
            if not proj.empty:
                project = str(proj.iloc[0])

        # ── Descriptions: collect all unique ──
        descriptions = []
        if "description" in group_df.columns:
            descriptions = [
                str(d) for d in group_df["description"].dropna().unique()
                if str(d).strip()
            ]

        # ── Raw line data (for display / debugging) ──
        lines = group_df.to_dict(orient="records")

        group = JournalGroup(
            journal_ref=str(ref),
            vendor_name=vendor_name,
            vendor_vat_id=vat_id,
            invoice_no=primary_inv_no,
            invoice_nos=inv_nos,
            invoice_date=invoice_date,
            posting_date=posting_date,
            total_amount=total_amount,
            sum_of_amounts=sum_of_amounts,
            currency=currency,
            payment_terms=payment_terms,
            descriptions=descriptions,
            source_system=source_system,
            cost_center=cost_center,
            project=project,
            line_count=len(group_df),
            lines=lines,
        )

        groups.append(group)
        logger.debug(
            f"Group '{ref}': vendor={vendor_name}, inv={primary_inv_no}, "
            f"total={total_amount}, lines={len(group_df)}"
        )

    logger.info(f"Created {len(groups)} journal groups")
    return groups


def parse_and_group(file_input: Union[str, Path, BytesIO]) -> tuple[pd.DataFrame, list[JournalGroup]]:
    """
    Convenience function: parse XLSX and group in one call.

    Returns:
        (raw_dataframe, list_of_journal_groups)
    """
    df = parse_xlsx(file_input)
    groups = group_journal_entries(df)
    return df, groups

FILEEOF

echo "  Creating app/parsers/pdf_parser.py..."
cat > 'app/parsers/pdf_parser.py' << 'FILEEOF'
"""
PDF Parser — Invoice extraction.

Req 4.3:
  - Extract text from the PDFs (OCR not required for these samples).
  - Build a normalized invoice object containing at minimum:
    vendor_name, invoice_no, invoice_date, total_amount.
  - Index invoice text and/or normalized summary into the vector store.
  - Keep a text copy of the OCR output.
"""

from __future__ import annotations

import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

from app.models import Invoice

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Text Extraction
# ──────────────────────────────────────────────

def extract_text_from_pdf(file_input: Union[str, Path, BytesIO]) -> str:
    """
    Extract raw text from a PDF file.

    Uses PyMuPDF (fitz) — fast and reliable for text-based PDFs.
    The assignment states "OCR not required for these samples."

    Args:
        file_input: File path string/Path, or BytesIO from Streamlit uploader.

    Returns:
        Raw text content of the PDF.

    Raises:
        ValueError: If the PDF cannot be opened or contains no text.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is required. Install with: pip install PyMuPDF"
        )

    try:
        if isinstance(file_input, BytesIO):
            doc = fitz.open(stream=file_input.read(), filetype="pdf")
            # Reset stream position in case it's read again
            file_input.seek(0)
        else:
            doc = fitz.open(str(file_input))

        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()

        full_text = "\n".join(text_parts).strip()

        if not full_text:
            raise ValueError("PDF contains no extractable text.")

        logger.info(f"Extracted {len(full_text)} characters from PDF")
        return full_text

    except Exception as e:
        if "no extractable text" in str(e):
            raise
        raise ValueError(f"Failed to read PDF: {e}")


# ──────────────────────────────────────────────
# Structured Field Extraction
# ──────────────────────────────────────────────

def _extract_field(text: str, pattern: str, group: int = 1) -> Optional[str]:
    """Helper: extract a single field via regex, return None if not found."""
    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    if match:
        return match.group(group).strip()
    return None


def _parse_currency_value(raw: str) -> Optional[float]:
    """Parse a currency string like '1,725.00 USD' into a float."""
    if not raw:
        return None
    cleaned = re.sub(r"[,$€£]", "", raw).strip()
    cleaned = re.sub(r"\s*[A-Z]{3}$", "", cleaned).strip()
    try:
        return float(cleaned)
    except ValueError:
        logger.warning(f"Could not parse float from: '{raw}'")
        return None


def _extract_float(text: str, pattern: str, group: int = 1) -> Optional[float]:
    """Helper: extract a numeric value, handling commas."""
    raw = _extract_field(text, pattern, group)
    return _parse_currency_value(raw)


def _extract_amount_after_label(text: str, label: str) -> Optional[float]:
    """
    Extract a numeric value that appears on the NEXT line after a label.

    PyMuPDF often puts labels and values on separate lines:
        Subtotal
        1,500.00 USD

    This handles both same-line and next-line formats.
    """
    # Try same-line first: "Subtotal 1,500.00 USD"
    same_line = re.search(
        rf"{label}\s+([\d,]+\.?\d*\s*(?:USD|EUR|GBP)?)",
        text, re.IGNORECASE | re.MULTILINE
    )
    if same_line:
        val = _parse_currency_value(same_line.group(1))
        if val is not None and val > 1:  # sanity check: amounts should be > 1
            return val

    # Try next-line: "Subtotal\n1,500.00 USD"
    next_line = re.search(
        rf"{label}\s*\n\s*([\d,]+\.?\d*\s*(?:USD|EUR|GBP)?)",
        text, re.IGNORECASE | re.MULTILINE
    )
    if next_line:
        val = _parse_currency_value(next_line.group(1))
        if val is not None:
            return val

    return None


def _extract_line_items(text: str) -> list[dict]:
    """
    Extract line items from the invoice table.

    Looks for rows matching: # Description Qty UnitPrice LineTotal
    """
    items = []
    # Pattern: digit(s) then description then numbers
    pattern = r"(\d+)\s+(.+?)\s+(\d+)\s+([\d,]+\.?\d*)\s+(?:USD\s+)?([\d,]+\.?\d*)"
    for match in re.finditer(pattern, text):
        try:
            items.append({
                "line_no": int(match.group(1)),
                "description": match.group(2).strip(),
                "qty": int(match.group(3)),
                "unit_price": float(match.group(4).replace(",", "")),
                "line_total": float(match.group(5).replace(",", "")),
            })
        except (ValueError, IndexError):
            continue
    return items


def _extract_vendor_name(text: str) -> str:
    """
    Extract vendor name from invoice text.

    Strategy: The vendor name typically appears right after the
    "INVOICE" header line. We look for the first line that looks
    like a company name (contains LLC, Ltd, Inc, Co, etc. or is
    the first substantial line after "Invoice No/Date" block).
    """
    # Try to find a line with a company suffix
    company_pattern = r"^(.+?(?:LLC|L\.L\.C\.|Ltd\.?|Limited|Inc\.?|Co\.?|Corp\.?|Services|Supplies|Consulting).*)$"
    match = re.search(company_pattern, text, re.MULTILINE | re.IGNORECASE)
    if match:
        name = match.group(1).strip()
        # Don't return if it's the "Bill To" company
        if "accord partners" not in name.lower():
            return name

    # Fallback: look for the line right after "INVOICE"
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if "INVOICE" in line.upper() and i + 1 < len(lines):
            candidate = lines[i + 1].strip()
            if candidate and len(candidate) > 3 and "invoice" not in candidate.lower():
                return candidate

    return ""


def parse_invoice(
    text: str,
    source_file: str = "",
) -> Invoice:
    """
    Parse raw text into a structured Invoice object.

    Uses regex patterns tailored to the assignment's invoice format.
    Handles variations gracefully — missing fields become None.

    Args:
        text: Raw text extracted from the PDF.
        source_file: Original filename for reference.

    Returns:
        Invoice object with all extracted fields.
    """
    # ── Core required fields ──
    vendor_name = _extract_vendor_name(text)
    invoice_no = _extract_field(text, r"Invoice\s*No[:\s]+(.+?)(?:\n|$)")
    invoice_date = _extract_field(text, r"Invoice\s*Date[:\s]+(\d{4}-\d{2}-\d{2})")

    # ── Financial fields (handles values on same line or next line) ──
    subtotal = _extract_amount_after_label(text, "Subtotal")
    total_amount = _extract_amount_after_label(text, r"^Total")  # ^ avoids "Line Total"
    vat_amount = _extract_amount_after_label(text, r"VAT\s*\(\d+%?\)")
    vat_rate = _extract_float(text, r"VAT\s*\((\d+)%?\)")
    if vat_rate is not None:
        vat_rate = vat_rate / 100.0  # Convert 15 → 0.15

    # Fallback for total: if still None, try "Total" not at start of line
    if total_amount is None:
        total_amount = _extract_amount_after_label(text, "Total")

    # ── Optional fields ──
    vat_id = _extract_field(text, r"VAT\s*ID[:\s]+(\S+)")
    payment_terms = _extract_field(text, r"Payment\s*Terms[:\s]+(.+?)(?:\n|$)")

    # ── Currency (default USD) ──
    currency = "USD"
    if "EUR" in text:
        currency = "EUR"
    elif "GBP" in text or "£" in text:
        currency = "GBP"

    # ── Line items ──
    line_items = _extract_line_items(text)

    invoice = Invoice(
        vendor_name=vendor_name,
        invoice_no=invoice_no or "",
        invoice_date=invoice_date or "",
        subtotal=subtotal,
        vat_rate=vat_rate,
        vat_amount=vat_amount,
        total_amount=total_amount,
        currency=currency,
        vat_id=vat_id,
        payment_terms=payment_terms,
        line_items=line_items,
        raw_text=text,  # Req 4.3: "Keep a text copy"
        source_file=source_file,
    )

    logger.info(
        f"Parsed invoice: vendor={vendor_name}, no={invoice_no}, "
        f"date={invoice_date}, total={total_amount}"
    )

    return invoice


def extract_and_parse(
    file_input: Union[str, Path, BytesIO],
    source_file: str = "",
) -> Invoice:
    """
    Convenience function: extract text from PDF and parse in one call.

    Args:
        file_input: PDF file path or BytesIO.
        source_file: Original filename for reference.

    Returns:
        Parsed Invoice object.
    """
    text = extract_text_from_pdf(file_input)
    return parse_invoice(text, source_file=source_file)

FILEEOF

echo "  Creating app/indexer/__init__.py..."
cat > 'app/indexer/__init__.py' << 'FILEEOF'

FILEEOF

echo "  Creating app/indexer/embedder.py..."
cat > 'app/indexer/embedder.py' << 'FILEEOF'
"""
Embedding abstraction layer.

Provides a unified interface for generating text embeddings,
with two interchangeable backends:
  - LocalEmbedder:  Qwen3-Embedding-0.6B via sentence-transformers
  - OpenAIEmbedder: text-embedding-3-small via OpenAI API

Swap between them by changing one config flag — no other code changes needed.

Design decisions:
  - Instruction-aware embedding on query side (Qwen3 feature, improves retrieval 1-5%)
  - Batch support for cost efficiency (Req 5)
  - Cost logging integrated at this level
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

from app.config import EmbeddingConfig
from app.cost_logger import cost_logger

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Abstract Base
# ──────────────────────────────────────────────

class BaseEmbedder(ABC):
    """Interface that both embedding backends implement."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of documents (journal groups).
        No instruction prefix — documents are embedded as-is.
        """
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query (invoice).
        May include a task instruction prefix for instruction-aware models.
        """
        ...

    @abstractmethod
    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple queries in a batch."""
        ...


# ──────────────────────────────────────────────
# Local Embedder (Qwen3-Embedding-0.6B)
# ──────────────────────────────────────────────

class LocalEmbedder(BaseEmbedder):
    """
    Embedding via sentence-transformers running locally.

    Uses Qwen3-Embedding-0.6B by default.
    Supports instruction-aware queries for better retrieval.
    Zero API cost — all computation is local.
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = None  # lazy load
        self._model_name = config.local_model_name
        self._task_instruction = config.local_task_instruction

    def _load_model(self):
        """Lazy load to avoid importing torch at module level."""
        if self.model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading local embedding model: {self._model_name}")
            self.model = SentenceTransformer(
                self._model_name,
                trust_remote_code=True,
            )
            logger.info("Local embedding model loaded successfully")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install sentence-transformers torch"
            )

    def _add_instruction(self, text: str) -> str:
        """Prepend task instruction for query-side embedding."""
        if self._task_instruction:
            return f"Instruct: {self._task_instruction}\nQuery: {text}"
        return text

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents without instruction prefix."""
        self._load_model()

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
        )

        # Log cost (local = free, but track for comparison)
        cost_logger.log(
            stage="embedding",
            provider="local",
            model=self._model_name,
            operation="embed_documents",
            num_calls=1,
            input_tokens=sum(len(t.split()) for t in texts) * 2,  # rough estimate
            estimated_cost_usd=0.0,
        )

        logger.info(f"Embedded {len(texts)} documents locally")
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query with instruction prefix."""
        self._load_model()
        instructed = self._add_instruction(text)

        embedding = self.model.encode(
            instructed,
            normalize_embeddings=True,
        )

        cost_logger.log(
            stage="embedding",
            provider="local",
            model=self._model_name,
            operation="embed_query",
            num_calls=1,
            input_tokens=len(instructed.split()) * 2,
            estimated_cost_usd=0.0,
        )

        return embedding.tolist()

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple queries with instruction prefix."""
        self._load_model()
        instructed = [self._add_instruction(t) for t in texts]

        embeddings = self.model.encode(
            instructed,
            normalize_embeddings=True,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
        )

        cost_logger.log(
            stage="embedding",
            provider="local",
            model=self._model_name,
            operation="embed_queries",
            num_calls=1,
            input_tokens=sum(len(t.split()) for t in instructed) * 2,
            estimated_cost_usd=0.0,
        )

        logger.info(f"Embedded {len(texts)} queries locally")
        return embeddings.tolist()


# ──────────────────────────────────────────────
# OpenAI Embedder
# ──────────────────────────────────────────────

class OpenAIEmbedder(BaseEmbedder):
    """
    Embedding via OpenAI API (text-embedding-3-small).

    Req 5: batch calls for efficiency, log tokens and cost per call.
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._client = None  # lazy init

    def _get_client(self):
        """Lazy init OpenAI client."""
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.config.openai_api_key,
                base_url=self.config.openai_base_url,
            )
            return self._client
        except ImportError:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            )

    def _embed_batch(self, texts: list[str], operation: str) -> list[list[float]]:
        """Core batch embedding with cost logging."""
        client = self._get_client()

        response = client.embeddings.create(
            model=self.config.openai_model_name,
            input=texts,
        )

        embeddings = [item.embedding for item in response.data]

        # Cost calculation
        total_tokens = response.usage.total_tokens
        cost = (total_tokens / 1_000_000) * self.config.openai_cost_per_1m_tokens

        cost_logger.log(
            stage="embedding",
            provider="openai",
            model=self.config.openai_model_name,
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


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────

def get_embedder(config: EmbeddingConfig) -> BaseEmbedder:
    """
    Factory function: returns the appropriate embedder based on config.

    Usage:
        from app.config import get_config
        config = get_config(embedding_provider="local")
        embedder = get_embedder(config.embedding)
    """
    if config.provider == "openai":
        if not config.openai_api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY env variable "
                "or pass it in config."
            )
        logger.info(f"Using OpenAI embedder: {config.openai_model_name}")
        return OpenAIEmbedder(config)
    else:
        logger.info(f"Using local embedder: {config.local_model_name}")
        return LocalEmbedder(config)

FILEEOF

echo "  Creating app/indexer/vector_store.py..."
cat > 'app/indexer/vector_store.py' << 'FILEEOF'
"""
Vector Store operations using ChromaDB.

Two collections as required:
  1. journal_groups — one document per journal_ref group (Req 4.2)
  2. invoices       — indexed invoice text/summary (Req 4.3)

Req 4.2: "Store metadata alongside embeddings
          (journal_ref, posting_date, totals, vendor, invoice_no)."
Req 4.3: "Index invoice text and/or normalized summary into the vector store."

ChromaDB persists to disk automatically (Req: "persisting to disk is a plus").
"""

from __future__ import annotations

import logging
from typing import Optional

import chromadb
from chromadb.config import Settings

from app.config import VectorStoreConfig
from app.indexer.embedder import BaseEmbedder
from app.models import Invoice, JournalGroup

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages ChromaDB collections for journals and invoices.

    Usage:
        store = VectorStore(config, embedder)
        store.ingest_journal_groups(groups)
        results = store.query_by_invoice(invoice, top_k=10)
    """

    def __init__(self, config: VectorStoreConfig, embedder: BaseEmbedder):
        self.config = config
        self.embedder = embedder

        # Initialize ChromaDB with persistence
        self._client = chromadb.PersistentClient(
            path=config.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # Collections (created on first use)
        self._journal_collection = None
        self._invoice_collection = None

        logger.info(f"ChromaDB initialized at: {config.persist_dir}")

    # ──────────────────────────────────────────
    # Collection access
    # ──────────────────────────────────────────

    @property
    def journal_collection(self):
        """Get or create the journal groups collection."""
        if self._journal_collection is None:
            self._journal_collection = self._client.get_or_create_collection(
                name=self.config.journal_collection_name,
                metadata={"hnsw:space": self.config.distance_metric},
            )
        return self._journal_collection

    @property
    def invoice_collection(self):
        """Get or create the invoices collection."""
        if self._invoice_collection is None:
            self._invoice_collection = self._client.get_or_create_collection(
                name=self.config.invoice_collection_name,
                metadata={"hnsw:space": self.config.distance_metric},
            )
        return self._invoice_collection

    # ──────────────────────────────────────────
    # Journal Ingestion (Req 4.2)
    # ──────────────────────────────────────────

    def ingest_journal_groups(self, groups: list[JournalGroup]) -> int:
        """
        Embed and store journal groups in the vector store.

        Each group becomes one document with:
          - id: journal_ref
          - document: embedding text
          - embedding: vector from embedder
          - metadata: structured fields for filtering/display

        Args:
            groups: List of JournalGroup objects from xlsx_parser.

        Returns:
            Number of groups ingested.
        """
        if not groups:
            logger.warning("No journal groups to ingest")
            return 0

        # Build texts for embedding
        ids = [g.journal_ref for g in groups]
        documents = [g.to_embedding_text() for g in groups]
        metadatas = [g.to_metadata() for g in groups]

        # Generate embeddings (batched — Req 5: cost efficiency)
        logger.info(f"Generating embeddings for {len(groups)} journal groups...")
        embeddings = self.embedder.embed_documents(documents)

        # Upsert into ChromaDB (idempotent — safe to re-run)
        self.journal_collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(f"Ingested {len(groups)} journal groups into vector store")
        return len(groups)

    # ──────────────────────────────────────────
    # Invoice Ingestion (Req 4.3)
    # ──────────────────────────────────────────

    def ingest_invoices(self, invoices: list[Invoice]) -> int:
        """
        Embed and store invoices in the vector store.

        Req 4.3: "Index invoice text and/or normalized summary
                  into the vector store."

        Args:
            invoices: List of Invoice objects from pdf_parser.

        Returns:
            Number of invoices ingested.
        """
        if not invoices:
            logger.warning("No invoices to ingest")
            return 0

        ids = [inv.invoice_no or inv.source_file for inv in invoices]
        documents = [inv.to_embedding_text() for inv in invoices]
        metadatas = [
            {
                "vendor_name": inv.vendor_name or "",
                "invoice_no": inv.invoice_no or "",
                "invoice_date": inv.invoice_date or "",
                "total_amount": float(inv.total_amount) if inv.total_amount else 0.0,
                "currency": inv.currency,
                "vat_id": inv.vat_id or "",
                "source_file": inv.source_file or "",
            }
            for inv in invoices
        ]

        # Embed as documents (not queries — these are stored, not searched)
        logger.info(f"Generating embeddings for {len(invoices)} invoices...")
        embeddings = self.embedder.embed_documents(documents)

        self.invoice_collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(f"Ingested {len(invoices)} invoices into vector store")
        return len(invoices)

    # ──────────────────────────────────────────
    # Query (Matching — Req 4.4)
    # ──────────────────────────────────────────

    def query_by_invoice(
        self,
        invoice: Invoice,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Find the top-K matching journal groups for an invoice.

        Uses the invoice embedding text as a query against the
        journal_groups collection. Returns candidates with their
        similarity scores and metadata.

        Args:
            invoice: The Invoice to match.
            top_k: Number of candidates to retrieve (K ≈ 5–10 per assignment).

        Returns:
            List of dicts, each with:
              - journal_ref: str
              - document: str (the embedding text)
              - metadata: dict (structured fields)
              - distance: float (lower = more similar for cosine)
              - similarity: float (1 - distance, higher = better)
        """
        # Embed invoice as a query (with instruction prefix for local model)
        query_text = invoice.to_embedding_text()
        query_embedding = self.embedder.embed_query(query_text)

        # Query ChromaDB
        n_available = self.journal_collection.count()
        effective_k = min(top_k, n_available)

        if effective_k == 0:
            logger.warning("Journal collection is empty — no candidates to retrieve")
            return []

        results = self.journal_collection.query(
            query_embeddings=[query_embedding],
            n_results=effective_k,
            include=["documents", "metadatas", "distances"],
        )

        # Unpack ChromaDB results into clean dicts
        candidates = []
        if results["ids"] and results["ids"][0]:
            for i, ref in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                # ChromaDB cosine distance: 0 = identical, 2 = opposite
                # Convert to similarity: 1 = identical, 0 = orthogonal
                similarity = max(0.0, 1.0 - distance)

                candidates.append({
                    "journal_ref": ref,
                    "document": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": distance,
                    "similarity": similarity,
                })

        logger.info(
            f"Retrieved {len(candidates)} candidates for invoice "
            f"{invoice.invoice_no} (top similarity: "
            f"{candidates[0]['similarity']:.3f})" if candidates else "none"
        )

        return candidates

    # ──────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────

    def get_journal_count(self) -> int:
        """Number of journal groups currently stored."""
        return self.journal_collection.count()

    def get_invoice_count(self) -> int:
        """Number of invoices currently stored."""
        return self.invoice_collection.count()

    def get_all_journal_metadatas(self) -> list[dict]:
        """Retrieve all journal group metadata (for display/debugging)."""
        result = self.journal_collection.get(include=["metadatas"])
        return result["metadatas"] if result["metadatas"] else []

    def reset_journals(self):
        """Delete and recreate the journal collection."""
        try:
            self._client.delete_collection(self.config.journal_collection_name)
        except Exception:
            pass
        self._journal_collection = None
        logger.info("Journal collection reset")

    def reset_invoices(self):
        """Delete and recreate the invoice collection."""
        try:
            self._client.delete_collection(self.config.invoice_collection_name)
        except Exception:
            pass
        self._invoice_collection = None
        logger.info("Invoice collection reset")

    def reset_all(self):
        """Delete and recreate both collections."""
        self.reset_journals()
        self.reset_invoices()
        logger.info("All collections reset")

FILEEOF

echo "  Creating app/matcher/__init__.py..."
cat > 'app/matcher/__init__.py' << 'FILEEOF'

FILEEOF

echo "  Creating app/matcher/deterministic.py..."
cat > 'app/matcher/deterministic.py' << 'FILEEOF'
"""
Deterministic field-level matching & scoring.

Req 4.4: "deterministic checks: invoice number similarity,
          vendor name similarity, amount tolerance,
          optional date proximity"

Req 4.4 Return: "evidence used (which fields matched, which didn't)"

This module does NO semantic/AI work — pure string comparison, fuzzy
matching, and numeric tolerance checks. It provides the hard evidence
layer that catches distractors the vector search might miss.

Design decision: Deterministic scoring gets the highest weight (0.50)
in the combined score because for audit systems, exact field matches
(invoice_no, amount) are more trustworthy than semantic similarity.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Optional

from rapidfuzz import fuzz

from app.config import MatchingConfig
from app.models import Invoice, MatchCandidate
from app.parsers.xlsx_parser import normalize_invoice_no, normalize_vendor_name

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Individual Field Matchers
# ──────────────────────────────────────────────
# Each returns a score [0, 1] and an evidence dict
# explaining what was compared and the result.

def _common_prefix(a: str, b: str) -> str:
    """Return the longest common prefix of two strings."""
    prefix = []
    for ca, cb in zip(a, b):
        if ca == cb:
            prefix.append(ca)
        else:
            break
    return "".join(prefix)


def match_invoice_number(
    invoice_no: str,
    journal_no: str,
    journal_all_nos: Optional[list[str]] = None,
    threshold: float = 0.80,
) -> tuple[float, dict]:
    """
    Compare invoice numbers with normalization and fuzzy matching.

    Handles Req 8 variations:
      'INV-ACME-1001' vs 'INV ACME 1001' → match after normalization
      'NIMBUS-778' vs 'NIMBUS-778-A' → high fuzzy but not exact
      'ORION-404' vs 'ORION-403' → should NOT match

    If journal has multiple invoice numbers (noisy dataset),
    we check against all of them and take the best match.
    """
    if not invoice_no or (not journal_no and not journal_all_nos):
        return 0.0, {
            "matched": False,
            "invoice_value": invoice_no or "",
            "journal_value": journal_no or "",
            "method": "missing_field",
            "similarity": 0.0,
            "note": "One or both invoice numbers are missing",
        }

    # Collect all journal invoice numbers to check against
    candidates = []
    if journal_no:
        candidates.append(journal_no)
    if journal_all_nos:
        for n in journal_all_nos:
            if n and n not in candidates:
                candidates.append(n)

    best_score = 0.0
    best_match = ""
    best_method = ""

    inv_normalized = normalize_invoice_no(invoice_no)

    for jno in candidates:
        jno_normalized = normalize_invoice_no(jno)

        # Exact match after normalization
        if inv_normalized == jno_normalized:
            return 1.0, {
                "matched": True,
                "invoice_value": invoice_no,
                "journal_value": jno,
                "method": "exact_after_normalization",
                "similarity": 1.0,
                "note": f"'{inv_normalized}' == '{jno_normalized}'",
            }

        # Fuzzy match on normalized forms
        ratio = fuzz.ratio(inv_normalized, jno_normalized) / 100.0

        # Critical check: detect sequential invoice numbers.
        # "orion403" vs "orion404" has high fuzzy (~87%) but they
        # are different transactions. If the strings share a prefix
        # and differ only in trailing digits, it's NOT a match.
        if ratio > 0.75 and ratio < 1.0:
            common_prefix = _common_prefix(inv_normalized, jno_normalized)
            inv_suffix = inv_normalized[len(common_prefix):]
            jnl_suffix = jno_normalized[len(common_prefix):]
            if inv_suffix.isdigit() and jnl_suffix.isdigit() and inv_suffix != jnl_suffix:
                # Different trailing numbers → different transaction
                ratio = ratio * 0.3  # heavy penalty

        if ratio > best_score:
            best_score = ratio
            best_match = jno
            best_method = "fuzzy"

    matched = best_score >= threshold

    return (best_score if matched else best_score * 0.5), {
        "matched": matched,
        "invoice_value": invoice_no,
        "journal_value": best_match,
        "method": best_method,
        "similarity": round(best_score, 4),
        "note": (
            f"Fuzzy score {best_score:.2f} {'≥' if matched else '<'} "
            f"threshold {threshold}"
        ),
    }


def match_vendor_name(
    invoice_vendor: str,
    journal_vendor: str,
    threshold: float = 0.75,
) -> tuple[float, dict]:
    """
    Compare vendor names with normalization and fuzzy matching.

    Handles Req 8 variations:
      'ACME Office Supplies LLC' vs 'ACME Office Supplies L.L.C.' → match
      'Nimbus Cloud Services Inc.' vs 'Nimbus Cloud Svcs, Inc.' → partial
      'Orion Consulting Ltd.' vs 'Orion Consulting Limited' → partial
    """
    if not invoice_vendor or not journal_vendor or journal_vendor == "—":
        return 0.0, {
            "matched": False,
            "invoice_value": invoice_vendor or "",
            "journal_value": journal_vendor or "",
            "method": "missing_field",
            "similarity": 0.0,
        }

    inv_norm = normalize_vendor_name(invoice_vendor)
    jnl_norm = normalize_vendor_name(journal_vendor)

    # Exact after normalization
    if inv_norm == jnl_norm:
        return 1.0, {
            "matched": True,
            "invoice_value": invoice_vendor,
            "journal_value": journal_vendor,
            "method": "exact_after_normalization",
            "similarity": 1.0,
        }

    # Fuzzy match — use token_sort_ratio for word-order independence
    ratio = fuzz.token_sort_ratio(inv_norm, jnl_norm) / 100.0

    # Also check partial ratio for substring matches (e.g., "Svcs" vs "Services")
    partial = fuzz.partial_ratio(inv_norm, jnl_norm) / 100.0
    best = max(ratio, partial)

    matched = best >= threshold

    return (best if matched else best * 0.5), {
        "matched": matched,
        "invoice_value": invoice_vendor,
        "journal_value": journal_vendor,
        "method": "fuzzy",
        "similarity": round(best, 4),
        "note": f"token_sort={ratio:.2f}, partial={partial:.2f}",
    }


def match_amount(
    invoice_total: Optional[float],
    journal_total: Optional[float],
    journal_sum: Optional[float] = None,
    tolerance_pct: float = 0.05,
) -> tuple[float, dict]:
    """
    Compare amounts with tolerance.

    Complexity: Invoice total and journal total may differ because:
      - Journal records base + VAT as separate line amounts
      - Invoice total includes VAT
      - Example: Invoice total 1,725 vs Journal total_amount 2,300

    Strategy: Check against both total_amount and sum_of_amounts,
    and also check if invoice subtotal matches the non-VAT portion.
    """
    if invoice_total is None:
        return 0.0, {
            "matched": False,
            "invoice_value": None,
            "journal_value": journal_total,
            "method": "missing_field",
            "difference_pct": None,
        }

    # Collect amounts to compare against
    compare_values = []
    if journal_total is not None and journal_total > 0:
        compare_values.append(("total_amount", journal_total))
    if journal_sum is not None and journal_sum > 0:
        compare_values.append(("sum_of_amounts", journal_sum))

    if not compare_values:
        return 0.0, {
            "matched": False,
            "invoice_value": invoice_total,
            "journal_value": None,
            "method": "missing_field",
            "difference_pct": None,
        }

    best_score = 0.0
    best_info = {}

    for label, journal_val in compare_values:
        if journal_val == 0:
            continue

        diff_pct = abs(invoice_total - journal_val) / journal_val

        if diff_pct <= tolerance_pct:
            # Within tolerance — full match
            score = 1.0
        elif diff_pct <= tolerance_pct * 3:
            # Close but not exact — partial credit
            score = 0.7
        elif diff_pct <= 0.50:
            # Within 50% — small credit (could be base vs total mismatch)
            score = 0.3
        else:
            score = 0.0

        if score > best_score:
            best_score = score
            best_info = {
                "matched": score >= 0.7,
                "invoice_value": invoice_total,
                "journal_value": journal_val,
                "journal_field": label,
                "method": "tolerance",
                "difference_pct": round(diff_pct * 100, 2),
                "note": (
                    f"Diff {diff_pct*100:.1f}% "
                    f"{'≤' if diff_pct <= tolerance_pct else '>'} "
                    f"tolerance {tolerance_pct*100:.0f}%"
                ),
            }

    if not best_info:
        best_info = {
            "matched": False,
            "invoice_value": invoice_total,
            "journal_value": journal_total,
            "method": "no_comparison",
            "difference_pct": None,
        }

    return best_score, best_info


def match_date(
    invoice_date: str,
    journal_date: str,
    max_days: int = 7,
) -> tuple[float, dict]:
    """
    Compare dates by proximity.

    Dates within max_days get full credit, with linear decay beyond.
    """
    if not invoice_date or not journal_date:
        return 0.0, {
            "matched": False,
            "invoice_value": invoice_date or "",
            "journal_value": journal_date or "",
            "method": "missing_field",
            "days_diff": None,
        }

    try:
        inv_dt = datetime.strptime(invoice_date[:10], "%Y-%m-%d")
        jnl_dt = datetime.strptime(journal_date[:10], "%Y-%m-%d")
        days_diff = abs((inv_dt - jnl_dt).days)

        if days_diff == 0:
            score = 1.0
        elif days_diff <= max_days:
            score = 1.0 - (days_diff / (max_days * 2))  # gradual decay
        else:
            score = max(0.0, 1.0 - (days_diff / 30))  # steep decay beyond

        return score, {
            "matched": days_diff <= max_days,
            "invoice_value": invoice_date,
            "journal_value": journal_date,
            "method": "proximity",
            "days_diff": days_diff,
            "note": f"{days_diff} days apart",
        }

    except (ValueError, TypeError) as e:
        logger.warning(f"Date comparison failed: {e}")
        return 0.0, {
            "matched": False,
            "invoice_value": invoice_date,
            "journal_value": journal_date,
            "method": "parse_error",
            "days_diff": None,
        }


def match_vat_id(
    invoice_vat_id: Optional[str],
    journal_vat_id: Optional[str],
) -> tuple[float, dict]:
    """Exact match on VAT ID — no fuzzy needed."""
    if not invoice_vat_id or not journal_vat_id:
        return 0.0, {
            "matched": False,
            "invoice_value": invoice_vat_id or "",
            "journal_value": journal_vat_id or "",
            "method": "missing_field",
        }

    # Normalize: strip spaces, uppercase
    inv_clean = invoice_vat_id.strip().upper()
    jnl_clean = journal_vat_id.strip().upper()

    matched = inv_clean == jnl_clean

    return (1.0 if matched else 0.0), {
        "matched": matched,
        "invoice_value": invoice_vat_id,
        "journal_value": journal_vat_id,
        "method": "exact",
    }


# ──────────────────────────────────────────────
# Combined Deterministic Scorer
# ──────────────────────────────────────────────

def compute_deterministic_score(
    invoice: Invoice,
    candidate_metadata: dict,
    config: MatchingConfig,
    candidate_invoice_nos: Optional[list[str]] = None,
) -> tuple[float, dict]:
    """
    Compute weighted deterministic score across all fields.

    Args:
        invoice: The Invoice being matched.
        candidate_metadata: Metadata dict from ChromaDB for this candidate.
        config: Matching configuration with weights and thresholds.
        candidate_invoice_nos: All invoice numbers in this journal group.

    Returns:
        (score, evidence_dict) where:
          - score is [0, 1] weighted combination
          - evidence_dict has per-field details for explainability
    """
    # Extract candidate fields from metadata
    c_vendor = candidate_metadata.get("vendor_name", "")
    c_inv_no = candidate_metadata.get("invoice_no", "")
    c_total = candidate_metadata.get("total_amount", 0.0)
    c_sum = candidate_metadata.get("sum_of_amounts", 0.0)
    c_date = candidate_metadata.get("invoice_date", "") or candidate_metadata.get("posting_date", "")
    c_vat_id = candidate_metadata.get("vendor_vat_id", "")

    # Run each field matcher
    inv_no_score, inv_no_evidence = match_invoice_number(
        invoice.invoice_no, c_inv_no,
        journal_all_nos=candidate_invoice_nos,
        threshold=config.fuzzy_invoice_no_threshold,
    )

    vendor_score, vendor_evidence = match_vendor_name(
        invoice.vendor_name, c_vendor,
        threshold=config.fuzzy_vendor_name_threshold,
    )

    amount_score, amount_evidence = match_amount(
        invoice.total_amount, c_total,
        journal_sum=c_sum,
        tolerance_pct=config.amount_tolerance_pct,
    )

    date_score, date_evidence = match_date(
        invoice.invoice_date, c_date,
        max_days=config.date_proximity_days,
    )

    vat_score, vat_evidence = match_vat_id(
        invoice.vat_id, c_vat_id,
    )

    # Weighted combination
    weighted_score = (
        config.weight_invoice_no * inv_no_score
        + config.weight_vendor_name * vendor_score
        + config.weight_amount * amount_score
        + config.weight_date * date_score
        + config.weight_vat_id * vat_score
    )

    # ── Invoice Number Gating Rule ──
    # In audit: if a journal group HAS an invoice number and it does NOT
    # match the invoice being checked, then this is likely a different
    # transaction from the same vendor. Same vendor can have many invoices.
    # Without this rule, vendor+VAT+date alone can cause false matches
    # against distractors like GRP-ORION-403 vs invoice ORION-404.
    candidate_has_inv_no = bool(c_inv_no and c_inv_no.strip())
    invoice_has_inv_no = bool(invoice.invoice_no and invoice.invoice_no.strip())

    if candidate_has_inv_no and invoice_has_inv_no:
        if not inv_no_evidence.get("matched", False):
            # Cap the score — other fields matching is not enough
            weighted_score = min(weighted_score, 0.25)

    evidence = {
        "invoice_no_match": inv_no_evidence,
        "vendor_match": vendor_evidence,
        "amount_match": amount_evidence,
        "date_match": date_evidence,
        "vat_id_match": vat_evidence,
        "field_scores": {
            "invoice_no": round(inv_no_score, 4),
            "vendor_name": round(vendor_score, 4),
            "amount": round(amount_score, 4),
            "date": round(date_score, 4),
            "vat_id": round(vat_score, 4),
        },
        "weights": {
            "invoice_no": config.weight_invoice_no,
            "vendor_name": config.weight_vendor_name,
            "amount": config.weight_amount,
            "date": config.weight_date,
            "vat_id": config.weight_vat_id,
        },
    }

    logger.debug(
        f"Deterministic score for {candidate_metadata.get('journal_ref', '?')}: "
        f"{weighted_score:.3f} (inv={inv_no_score:.2f}, vendor={vendor_score:.2f}, "
        f"amt={amount_score:.2f}, date={date_score:.2f}, vat={vat_score:.2f})"
    )

    return weighted_score, evidence


def generate_candidate_rationale(evidence: dict) -> str:
    """
    Generate a concise rationale string from field evidence.

    Req 4.1: "Top candidates (at least top-3): journal_ref, score,
             and a concise rationale."
    """
    parts = []

    inv_ev = evidence.get("invoice_no_match", {})
    if inv_ev.get("matched"):
        parts.append(f"Invoice # matched ({inv_ev.get('method', '')})")
    elif inv_ev.get("similarity", 0) > 0:
        parts.append(f"Invoice # similar ({inv_ev.get('similarity', 0):.0%})")
    else:
        parts.append("Invoice # not matched")

    vendor_ev = evidence.get("vendor_match", {})
    if vendor_ev.get("matched"):
        parts.append("vendor matched")
    elif vendor_ev.get("similarity", 0) > 0.5:
        parts.append(f"vendor partial ({vendor_ev.get('similarity', 0):.0%})")
    else:
        parts.append("vendor not matched")

    amt_ev = evidence.get("amount_match", {})
    if amt_ev.get("matched"):
        parts.append("amount matched")
    elif amt_ev.get("difference_pct") is not None:
        parts.append(f"amount differs by {amt_ev['difference_pct']:.0f}%")

    return "; ".join(parts)

FILEEOF

echo "  Creating app/matcher/reranker.py..."
cat > 'app/matcher/reranker.py' << 'FILEEOF'
"""
Cross-Encoder Reranker (optional but quality differentiator).

While the vector search (Stage 1) uses a bi-encoder that embeds query
and document independently, the cross-encoder processes them TOGETHER,
capturing fine-grained interactions between the invoice and each candidate.

This is what catches tricky cases like:
  - ORION-404 vs GRP-ORION-403: vector search ranks them high (similar text),
    but the cross-encoder sees the subtle number difference more clearly.

Design: This stage sits between vector recall and deterministic scoring.
It narrows top-10 from vector search down to top-5 before deterministic
scoring, ensuring only the most semantically relevant candidates proceed.

If the reranker model is unavailable (e.g., no GPU), the pipeline
gracefully skips this stage — it's an enhancement, not a requirement.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.config import RerankerConfig
from app.cost_logger import cost_logger

logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-encoder reranker using Qwen3-Reranker-0.6B.

    Scores each (invoice_text, candidate_text) pair with deep
    cross-attention, producing more precise relevance scores
    than the bi-encoder similarity from vector search.
    """

    def __init__(self, config: RerankerConfig):
        self.config = config
        self._model = None
        self._available = None  # None = not yet checked

    @property
    def is_available(self) -> bool:
        """Check if the reranker model can be loaded."""
        if self._available is not None:
            return self._available

        if not self.config.enabled:
            self._available = False
            logger.info("Reranker disabled in config")
            return False

        try:
            self._load_model()
            self._available = True
            logger.info("Reranker model loaded successfully")
        except Exception as e:
            self._available = False
            logger.warning(f"Reranker not available (will skip): {e}")

        return self._available

    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is not None:
            return

        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(
            self.config.model_name,
            trust_remote_code=True,
        )

    def _build_query_text(self, invoice_text: str) -> str:
        """Add instruction prefix for the reranker."""
        instruction = (
            "Determine if this invoice matches the given journal entry. "
            "Consider invoice number, vendor name, amounts, and dates."
        )
        return f"Instruct: {instruction}\nQuery: {invoice_text}"

    def rerank(
        self,
        invoice_text: str,
        candidates: list[dict],
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Rerank candidates using the cross-encoder.

        Args:
            invoice_text: The invoice embedding text.
            candidates: List of candidate dicts from vector store query,
                        each must have a 'document' key with text.
            top_k: Number of top candidates to return (default from config).

        Returns:
            Reranked list of candidates, each with added 'rerank_score'.
            Sorted by rerank_score descending.
        """
        if not self.is_available:
            # Graceful fallback: return candidates unchanged
            for c in candidates:
                c["rerank_score"] = c.get("similarity", 0.0)
            return candidates

        if top_k is None:
            top_k = self.config.top_k_output

        # Build (query, document) pairs
        query = self._build_query_text(invoice_text)
        pairs = [(query, c["document"]) for c in candidates]

        # Score all pairs
        scores = self._model.predict(pairs)

        # Attach scores
        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = float(scores[i])

        # Sort by rerank_score descending
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Log cost (local = free)
        cost_logger.log(
            stage="reranking",
            provider="local",
            model=self.config.model_name,
            operation="rerank_candidates",
            num_calls=1,
            input_tokens=sum(len(p[0].split()) + len(p[1].split()) for p in pairs) * 2,
            estimated_cost_usd=0.0,
        )

        logger.info(
            f"Reranked {len(candidates)} candidates, "
            f"returning top {top_k}. "
            f"Best score: {candidates[0]['rerank_score']:.4f}"
        )

        return candidates[:top_k]


class NoOpReranker:
    """
    Passthrough reranker that does nothing.

    Used when reranking is disabled or unavailable.
    Keeps the pipeline interface consistent.
    """

    @property
    def is_available(self) -> bool:
        return False

    def rerank(self, invoice_text: str, candidates: list[dict], top_k: Optional[int] = None) -> list[dict]:
        for c in candidates:
            c["rerank_score"] = c.get("similarity", 0.0)
        if top_k:
            return candidates[:top_k]
        return candidates


def get_reranker(config: RerankerConfig):
    """Factory: returns Reranker if available, NoOpReranker otherwise."""
    if not config.enabled:
        logger.info("Reranker disabled — using passthrough")
        return NoOpReranker()
    return Reranker(config)

FILEEOF

echo "  Creating app/matcher/llm_judge.py..."
cat > 'app/matcher/llm_judge.py' << 'FILEEOF'
"""
LLM Judge — Final decision and explanation generation.

Req 4.4: "optional single LLM call per invoice to produce a
          structured decision + explanation from top candidates"

Req 5: "If using an LLM, prefer at most one call per invoice."

This module makes exactly ONE LLM call per invoice, sending it the
top candidates with their scores and evidence, and getting back a
structured JSON decision with confidence and explanation.

Supports both:
  - Local: Qwen3.5-35B-A3B via vLLM (OpenAI-compatible API)
  - OpenAI: gpt-4o-mini

Swap by changing config — the prompt and parsing are identical.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from app.config import LLMConfig
from app.cost_logger import cost_logger
from app.models import Invoice, MatchCandidate

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Prompt Template
# ──────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """You are a financial auditor matching invoices to journal entries.
You will receive an invoice and a ranked list of candidate journal entry groups.
Each candidate includes similarity scores and field-level evidence.

Your task:
1. Analyze each candidate against the invoice.
2. Decide the BEST match or determine NO_MATCH.
3. Provide a clear explanation for your decision.

Rules:
- Invoice number match is the STRONGEST signal.
- If no candidate has a matching invoice number AND matching vendor, return NO_MATCH.
- Amount differences may exist due to accounting splits (base + VAT as separate lines).
- A high vector similarity alone is NOT sufficient — verify with field evidence.
- Be skeptical of near-matches: ORION-403 is NOT the same as ORION-404.

Respond with ONLY valid JSON, no markdown, no backticks."""

JUDGE_USER_TEMPLATE = """## Invoice (Evidence Document)
- Vendor: {vendor_name}
- Invoice No: {invoice_no}
- Date: {invoice_date}
- Subtotal: {subtotal}
- VAT: {vat_amount}
- Total: {total_amount} {currency}
- VAT ID: {vat_id}
- Payment Terms: {payment_terms}

## Top Candidate Journal Groups (ranked by combined score)
{candidates_text}

## Required JSON Response Format
{{
  "decision": "MATCHED" or "NO_MATCH",
  "best_match_journal_ref": "journal_ref string or null",
  "confidence": 0.0 to 1.0,
  "explanation": "2-3 sentences explaining why this is the best match or why no match exists",
  "candidates_analysis": [
    {{
      "journal_ref": "...",
      "relevance": "high/medium/low",
      "rationale": "1 sentence explaining why this candidate does or does not match"
    }}
  ]
}}"""


def _format_candidates(candidates: list[MatchCandidate]) -> str:
    """Format candidates for the LLM prompt."""
    parts = []
    for i, c in enumerate(candidates, 1):
        evidence = c.evidence or {}

        inv_ev = evidence.get("invoice_no_match", {})
        vendor_ev = evidence.get("vendor_match", {})
        amt_ev = evidence.get("amount_match", {})
        date_ev = evidence.get("date_match", {})
        vat_ev = evidence.get("vat_id_match", {})

        parts.append(
            f"### Candidate {i}: {c.journal_ref}\n"
            f"- Vendor: {c.vendor_name}\n"
            f"- Invoice No: {c.invoice_no}\n"
            f"- Total Amount: {c.total_amount}\n"
            f"- Combined Score: {c.combined_score:.3f}\n"
            f"- Vector Similarity: {c.vector_score:.3f}\n"
            f"- Deterministic Score: {c.deterministic_score:.3f}\n"
            f"- Invoice # Match: {inv_ev.get('matched', 'N/A')} "
            f"(similarity: {inv_ev.get('similarity', 'N/A')})\n"
            f"- Vendor Match: {vendor_ev.get('matched', 'N/A')} "
            f"(similarity: {vendor_ev.get('similarity', 'N/A')})\n"
            f"- Amount Match: {amt_ev.get('matched', 'N/A')} "
            f"(diff: {amt_ev.get('difference_pct', 'N/A')}%)\n"
            f"- Date Match: {date_ev.get('matched', 'N/A')} "
            f"(days diff: {date_ev.get('days_diff', 'N/A')})\n"
            f"- VAT ID Match: {vat_ev.get('matched', 'N/A')}"
        )

    return "\n\n".join(parts)


# ──────────────────────────────────────────────
# LLM Judge
# ──────────────────────────────────────────────

class LLMJudge:
    """
    Makes final match decision with LLM-generated explanation.

    Exactly ONE call per invoice (Req 5: cost discipline).
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None

    def _get_client(self):
        """Lazy init OpenAI-compatible client (works for both local and OpenAI)."""
        if self._client is not None:
            return self._client

        from openai import OpenAI

        if self.config.provider == "openai":
            self._client = OpenAI(
                api_key=self.config.openai_api_key,
                base_url=self.config.openai_base_url,
            )
        else:
            # Local vLLM server with OpenAI-compatible API
            self._client = OpenAI(
                api_key=self.config.local_api_key,
                base_url=self.config.local_base_url,
            )

        return self._client

    def _get_model_name(self) -> str:
        if self.config.provider == "openai":
            return self.config.openai_model_name
        return self.config.local_model_name

    def judge(
        self,
        invoice: Invoice,
        candidates: list[MatchCandidate],
    ) -> dict:
        """
        Make final decision for one invoice.

        Args:
            invoice: The Invoice being matched.
            candidates: Top candidates with scores and evidence.

        Returns:
            Dict with keys:
              - decision: "MATCHED" or "NO_MATCH"
              - best_match_journal_ref: str or None
              - confidence: float [0, 1]
              - explanation: str
              - candidates_analysis: list of per-candidate analysis
        """
        client = self._get_client()
        model = self._get_model_name()

        # Build prompt
        user_prompt = JUDGE_USER_TEMPLATE.format(
            vendor_name=invoice.vendor_name,
            invoice_no=invoice.invoice_no,
            invoice_date=invoice.invoice_date,
            subtotal=invoice.subtotal,
            vat_amount=invoice.vat_amount,
            total_amount=invoice.total_amount,
            currency=invoice.currency,
            vat_id=invoice.vat_id or "N/A",
            payment_terms=invoice.payment_terms or "N/A",
            candidates_text=_format_candidates(candidates),
        )

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Build API call kwargs
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        # Add response_format for JSON mode (OpenAI supports this)
        if self.config.provider == "openai":
            kwargs["response_format"] = {"type": "json_object"}

        # Add extra_body for local models (e.g., disable thinking mode)
        if self.config.provider == "local" and self.config.local_extra_body:
            kwargs["extra_body"] = self.config.local_extra_body

        try:
            response = client.chat.completions.create(**kwargs)

            raw_content = response.choices[0].message.content

            # Parse JSON response
            result = self._parse_response(raw_content)

            # Log cost
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else len(user_prompt.split()) * 2
            output_tokens = usage.completion_tokens if usage else len(raw_content.split()) * 2

            if self.config.provider == "openai":
                cost = (
                    (input_tokens / 1_000_000) * self.config.openai_input_cost_per_1m
                    + (output_tokens / 1_000_000) * self.config.openai_output_cost_per_1m
                )
            else:
                cost = 0.0

            cost_logger.log(
                stage="llm_judge",
                provider=self.config.provider,
                model=model,
                operation=f"judge_{invoice.invoice_no}",
                num_calls=1,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=cost,
            )

            logger.info(
                f"LLM judge for {invoice.invoice_no}: "
                f"{result.get('decision', '?')} "
                f"(confidence: {result.get('confidence', 0):.2f})"
            )

            return result

        except Exception as e:
            logger.error(f"LLM judge failed for {invoice.invoice_no}: {e}")
            return self._fallback_result(invoice, candidates, str(e))

    def _parse_response(self, raw: str) -> dict:
        """Parse LLM response, handling markdown fences and malformed JSON."""
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Remove ```json and closing ```
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM JSON response: {cleaned[:200]}")
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{[\s\S]*\}', cleaned)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                except json.JSONDecodeError:
                    result = {}
            else:
                result = {}

        # Normalize fields
        result.setdefault("decision", "NO_MATCH")
        result.setdefault("best_match_journal_ref", None)
        result.setdefault("confidence", 0.0)
        result.setdefault("explanation", "LLM response could not be fully parsed.")
        result.setdefault("candidates_analysis", [])

        # Normalize decision string
        decision = result["decision"].upper().strip()
        if decision in ("MATCHED", "MATCH"):
            result["decision"] = "MATCHED"
        else:
            result["decision"] = "NO_MATCH"

        # Ensure confidence is float
        try:
            result["confidence"] = float(result["confidence"])
        except (ValueError, TypeError):
            result["confidence"] = 0.0

        return result

    def _fallback_result(
        self,
        invoice: Invoice,
        candidates: list[MatchCandidate],
        error: str,
    ) -> dict:
        """
        Generate a deterministic fallback if LLM call fails.

        This ensures the pipeline still produces a result even
        without the LLM — using deterministic scores alone.
        """
        if candidates and candidates[0].combined_score >= 0.60:
            best = candidates[0]
            return {
                "decision": "MATCHED",
                "best_match_journal_ref": best.journal_ref,
                "confidence": round(best.combined_score, 2),
                "explanation": (
                    f"LLM unavailable ({error}). Decision based on deterministic "
                    f"scoring: combined score {best.combined_score:.2f}."
                ),
                "candidates_analysis": [],
            }
        else:
            return {
                "decision": "NO_MATCH",
                "best_match_journal_ref": None,
                "confidence": 0.7,
                "explanation": (
                    f"LLM unavailable ({error}). No candidate scored above "
                    f"threshold based on deterministic matching."
                ),
                "candidates_analysis": [],
            }


class NoOpJudge:
    """
    Fallback judge that uses only deterministic scores.

    Used when LLM is disabled or unavailable.
    """

    def judge(self, invoice: Invoice, candidates: list[MatchCandidate]) -> dict:
        if candidates and candidates[0].combined_score >= 0.60:
            best = candidates[0]
            return {
                "decision": "MATCHED",
                "best_match_journal_ref": best.journal_ref,
                "confidence": round(best.combined_score, 2),
                "explanation": (
                    f"Matched based on deterministic scoring (no LLM). "
                    f"Best candidate {best.journal_ref} scored {best.combined_score:.2f}. "
                    f"{best.rationale}"
                ),
                "candidates_analysis": [
                    {
                        "journal_ref": c.journal_ref,
                        "relevance": "high" if c.combined_score > 0.5 else "low",
                        "rationale": c.rationale,
                    }
                    for c in candidates[:3]
                ],
            }
        else:
            return {
                "decision": "NO_MATCH",
                "best_match_journal_ref": None,
                "confidence": 0.8,
                "explanation": (
                    "No candidate scored above the match threshold. "
                    + (
                        f"Closest candidate {candidates[0].journal_ref} scored "
                        f"{candidates[0].combined_score:.2f}: {candidates[0].rationale}"
                        if candidates else "No candidates available."
                    )
                ),
                "candidates_analysis": [
                    {
                        "journal_ref": c.journal_ref,
                        "relevance": "low",
                        "rationale": c.rationale,
                    }
                    for c in candidates[:3]
                ],
            }


def get_llm_judge(config: LLMConfig, enabled: bool = True):
    """Factory: returns LLMJudge if enabled, NoOpJudge otherwise."""
    if not enabled:
        logger.info("LLM judge disabled — using deterministic-only fallback")
        return NoOpJudge()
    return LLMJudge(config)

FILEEOF

echo "  Creating app/matcher/pipeline.py..."
cat > 'app/matcher/pipeline.py' << 'FILEEOF'
"""
Matching Pipeline — Orchestrates all 4 stages.

Stage 1: Vector Recall      — retrieve top-K candidates via embedding similarity
Stage 2: Cross-Encoder       — rerank candidates with deeper semantic analysis
Stage 3: Deterministic Score — field-level matching (invoice_no, vendor, amount, date)
Stage 4: LLM Judge           — final decision with confidence and explanation

This is the single entry point for matching an invoice.
The Streamlit dashboard calls `pipeline.match_invoice()`.

Req 4.4: "Use vector search to retrieve top-K candidate journal groups.
          Rerank and decide match using a hybrid approach."
"""

from __future__ import annotations

import logging
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

            # Get all invoice numbers for this group (for noisy dataset handling)
            # ChromaDB metadata only stores the primary invoice_no,
            # but we stored it there. For noisy data, the group might
            # have multiple invoice_nos which we check via the embedding text.
            candidate_inv_nos = None
            doc_text = c.get("document", "")
            if "All Invoice Numbers:" in doc_text:
                import re
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

            # Combined score
            vector_score = c.get("similarity", 0.0)
            rerank_score = c.get("rerank_score", vector_score)

            combined = (
                self.config.matching.weight_vector * vector_score
                + self.config.matching.weight_reranker * rerank_score
                + self.config.matching.weight_deterministic * det_score
            )

            rationale = generate_candidate_rationale(evidence)

            mc = MatchCandidate(
                journal_ref=c.get("journal_ref", metadata.get("journal_ref", "")),
                vendor_name=metadata.get("vendor_name", ""),
                invoice_no=metadata.get("invoice_no", ""),
                total_amount=metadata.get("total_amount", 0.0),
                vector_score=vector_score,
                rerank_score=rerank_score,
                deterministic_score=det_score,
                combined_score=combined,
                evidence=evidence,
                rationale=rationale,
            )
            match_candidates.append(mc)

        # Sort by combined score descending
        match_candidates.sort(key=lambda x: x.combined_score, reverse=True)

        logger.info(
            f"Stage 3 (Deterministic): Top candidate = "
            f"{match_candidates[0].journal_ref} "
            f"(combined={match_candidates[0].combined_score:.3f})"
        )

        # ── Stage 4: LLM Judge ──
        top_for_judge = match_candidates[: self.config.matching.min_candidates_to_show]

        judge_result = self.judge.judge(invoice, top_for_judge)

        logger.info(
            f"Stage 4 (LLM Judge): {judge_result.get('decision', '?')} "
            f"(confidence={judge_result.get('confidence', 0):.2f})"
        )

        # ── Build Final Result ──
        # Merge LLM judge analysis into candidate rationales if available
        judge_analyses = {
            a.get("journal_ref", ""): a.get("rationale", "")
            for a in judge_result.get("candidates_analysis", [])
        }
        for mc in match_candidates:
            llm_rationale = judge_analyses.get(mc.journal_ref, "")
            if llm_rationale:
                mc.rationale = llm_rationale

        # Determine best match evidence
        best_match_ref = judge_result.get("best_match_journal_ref")
        best_evidence = {}
        if best_match_ref:
            for mc in match_candidates:
                if mc.journal_ref == best_match_ref:
                    best_evidence = mc.evidence
                    break

        result = MatchResult(
            invoice=invoice,
            outcome=judge_result.get("decision", "NO_MATCH"),
            best_match=best_match_ref,
            confidence=judge_result.get("confidence", 0.0),
            explanation=judge_result.get("explanation", ""),
            evidence=best_evidence,
            top_candidates=match_candidates[: max(
                self.config.matching.min_candidates_to_show,
                3
            )],
        )

        logger.info(
            f"Final result for {invoice.invoice_no}: "
            f"{result.outcome} → {result.best_match} "
            f"(confidence={result.confidence:.2f})"
        )

        return result

    def match_invoices(self, invoices: list[Invoice]) -> list[MatchResult]:
        """
        Match multiple invoices sequentially.

        Args:
            invoices: List of parsed Invoices.

        Returns:
            List of MatchResults in same order as input.
        """
        results = []
        for invoice in invoices:
            result = self.match_invoice(invoice)
            results.append(result)
        return results

FILEEOF

echo "  Creating app/dashboard/__init__.py..."
cat > 'app/dashboard/__init__.py' << 'FILEEOF'

FILEEOF

echo "  Creating app/dashboard/streamlit_app.py..."
cat > 'app/dashboard/streamlit_app.py' << 'FILEEOF'
"""
Streamlit Dashboard — Evidence-to-Journal Matching.

Req 4.1 Dashboard (UI):
  - Journal Upload: upload XLSX, preview rows, then ingest into vector store.
  - Evidence Upload: upload one or more PDF invoices, then run matching.
  - Results View per invoice:
    - Extracted invoice fields (vendor, invoice number, date, total).
    - Matching outcome: Matched or No Match.
    - Top candidates (at least top-3): journal_ref, score, and a concise rationale.
    - A short explanation of why the selected match was chosen (or why No Match).

Run with: streamlit run app/dashboard/streamlit_app.py
"""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
import streamlit as st
import pandas as pd
from io import BytesIO

from app.config import get_config, VectorStoreConfig, VECTORDB_DIR, LOGS_DIR
from app.cost_logger import cost_logger
from app.models import Invoice, JournalGroup, MatchResult
from app.parsers.xlsx_parser import parse_and_group
from app.parsers.pdf_parser import extract_and_parse
from app.indexer.embedder import get_embedder
from app.indexer.vector_store import VectorStore
from app.matcher.pipeline import MatchingPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Audit Evidence Matcher",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────

st.markdown("""
<style>
    /* Result cards */
    .match-card {
        padding: 1.2rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
        border-left: 5px solid;
    }
    .match-card.matched {
        background-color: rgba(52, 211, 153, 0.08);
        border-left-color: #34d399;
    }
    .match-card.no-match {
        background-color: rgba(248, 113, 113, 0.08);
        border-left-color: #f87171;
    }

    /* Badge styles */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 1rem;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-matched {
        background-color: rgba(52, 211, 153, 0.2);
        color: #059669;
    }
    .badge-nomatch {
        background-color: rgba(248, 113, 113, 0.2);
        color: #dc2626;
    }

    /* Evidence table */
    .evidence-row {
        display: flex;
        justify-content: space-between;
        padding: 0.3rem 0;
        border-bottom: 1px solid rgba(128,128,128,0.15);
        font-size: 0.9rem;
    }

    /* Step headers */
    .step-header {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    /* Confidence meter */
    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        background: rgba(128,128,128,0.2);
        margin-top: 0.3rem;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 4px;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "journal_df": None,
        "journal_groups": None,
        "journals_ingested": False,
        "invoices": [],
        "match_results": [],
        "matching_done": False,
        "vector_store": None,
        "pipeline": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

init_session_state()


# ──────────────────────────────────────────────
# Sidebar — Configuration & Cost Tracker
# ──────────────────────────────────────────────

def render_sidebar():
    """Render sidebar with configuration and cost tracking."""
    with st.sidebar:
        st.title("⚙️ Configuration")

        # Provider selection
        st.subheader("Model Providers")
        embedding_provider = st.radio(
            "Embedding Model",
            ["local", "openai"],
            index=0,
            help="Local: Qwen3-Embedding-0.6B | OpenAI: text-embedding-3-small",
        )

        llm_provider = st.radio(
            "LLM Judge",
            ["local", "openai"],
            index=0,
            help="Local: Qwen3.5-35B (vLLM) | OpenAI: gpt-4o-mini",
        )

        use_llm = st.checkbox("Enable LLM Judge", value=True,
                              help="If disabled, uses deterministic scoring only")

        use_reranker = st.checkbox("Enable Reranker", value=False,
                                   help="Qwen3-Reranker-0.6B cross-encoder")

        st.divider()

        # Dataset selection
        st.subheader("Dataset")
        dataset_version = st.radio(
            "Journal Entries Version",
            ["Clean (v1)", "Noisy (v2)"],
            index=0,
            help="v2 has vendor name variations, format changes, and distractors",
        )

        st.divider()

        # Cost tracker (Req 5)
        st.subheader("💰 Cost Tracker")
        if cost_logger.total_calls() > 0:
            st.markdown(cost_logger.format_for_display())

            if st.button("💾 Save Cost Log"):
                filepath = cost_logger.save_to_file()
                st.success(f"Saved to {filepath}")
        else:
            st.caption("No API calls logged yet.")

        st.divider()

        # Reset
        if st.button("🗑️ Reset Everything", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            cost_logger.clear()
            st.rerun()

    return {
        "embedding_provider": embedding_provider,
        "llm_provider": llm_provider,
        "use_llm": use_llm,
        "use_reranker": use_reranker,
        "dataset_version": dataset_version,
    }


# ──────────────────────────────────────────────
# Helper: Initialize Pipeline
# ──────────────────────────────────────────────

def get_or_create_pipeline(settings: dict):
    """Create or reuse the matching pipeline based on settings."""
    config = get_config(
        embedding_provider=settings["embedding_provider"],
        llm_provider=settings["llm_provider"],
        reranker_enabled=settings["use_reranker"],
        use_llm_judge=settings["use_llm"],
    )

    # Create embedder
    embedder = get_embedder(config.embedding)

    # Create vector store
    vs_config = VectorStoreConfig(persist_dir=str(VECTORDB_DIR))
    store = VectorStore(vs_config, embedder)

    # Create pipeline
    pipeline = MatchingPipeline(config, store)

    st.session_state.vector_store = store
    st.session_state.pipeline = pipeline

    return store, pipeline


# ──────────────────────────────────────────────
# Step 1: Journal Upload
# ──────────────────────────────────────────────

def render_step1_journal_upload(settings: dict):
    """Render journal entries upload and ingestion section."""
    st.markdown("### 📊 Step 1: Upload Journal Entries")

    uploaded_file = st.file_uploader(
        "Upload XLSX file",
        type=["xlsx"],
        key="xlsx_uploader",
        help="Upload sample_journal_entries.xlsx or sample_journal_entries_v2_noisy.xlsx",
    )

    if uploaded_file is not None:
        try:
            # Parse XLSX
            file_bytes = BytesIO(uploaded_file.read())
            df, groups = parse_and_group(file_bytes)

            st.session_state.journal_df = df
            st.session_state.journal_groups = groups

            # Preview
            st.success(f"✅ Parsed {len(df)} rows → {len(groups)} journal groups")

            # Show preview table
            with st.expander("📋 Preview Journal Entries", expanded=True):
                # Build summary table
                summary_data = []
                for g in groups:
                    summary_data.append({
                        "Journal Ref": g.journal_ref,
                        "Vendor": g.vendor_name,
                        "Invoice No": g.invoice_no or "—",
                        "Date": g.invoice_date or "—",
                        "Total": f"{g.total_amount:,.2f} {g.currency}" if g.total_amount else "—",
                        "Lines": g.line_count,
                        "Descriptions": "; ".join(g.descriptions[:2]),
                    })

                st.dataframe(
                    pd.DataFrame(summary_data),
                    use_container_width=True,
                    hide_index=True,
                )

            # Show raw data
            with st.expander("📄 Raw Data"):
                st.dataframe(df, use_container_width=True, hide_index=True)

            # Ingest button
            if st.button(
                "🔄 Ingest into Vector Store",
                type="primary",
                disabled=st.session_state.journals_ingested,
            ):
                with st.spinner("Generating embeddings and storing..."):
                    store, _ = get_or_create_pipeline(settings)
                    store.reset_journals()
                    n = store.ingest_journal_groups(groups)

                st.session_state.journals_ingested = True
                st.success(f"✅ Ingested {n} journal groups into vector store")
                st.rerun()

            if st.session_state.journals_ingested:
                st.info(f"✅ {len(groups)} journal groups are in the vector store. Proceed to Step 2.")

        except Exception as e:
            st.error(f"❌ Error parsing XLSX: {e}")
            logger.error(f"XLSX parse error: {e}", exc_info=True)


# ──────────────────────────────────────────────
# Step 2: Evidence Upload
# ──────────────────────────────────────────────

def render_step2_evidence_upload(settings: dict):
    """Render invoice PDF upload and field extraction section."""
    st.markdown("### 📄 Step 2: Upload Evidence Invoices")

    if not st.session_state.journals_ingested:
        st.warning("⚠️ Please upload and ingest journal entries first (Step 1).")
        return

    uploaded_pdfs = st.file_uploader(
        "Upload PDF invoices",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
        help="Upload one or more invoice PDFs",
    )

    if uploaded_pdfs:
        invoices = []
        for pdf_file in uploaded_pdfs:
            try:
                file_bytes = BytesIO(pdf_file.read())
                invoice = extract_and_parse(file_bytes, source_file=pdf_file.name)
                invoices.append(invoice)
            except Exception as e:
                st.error(f"❌ Error parsing {pdf_file.name}: {e}")

        if invoices:
            st.session_state.invoices = invoices

            # Show extracted fields
            st.success(f"✅ Extracted fields from {len(invoices)} invoice(s)")

            with st.expander("📋 Extracted Invoice Fields", expanded=True):
                inv_data = []
                for inv in invoices:
                    inv_data.append({
                        "File": inv.source_file,
                        "Vendor": inv.vendor_name,
                        "Invoice No": inv.invoice_no,
                        "Date": inv.invoice_date,
                        "Subtotal": f"{inv.subtotal:,.2f}" if inv.subtotal else "—",
                        "VAT": f"{inv.vat_amount:,.2f}" if inv.vat_amount else "—",
                        "Total": f"{inv.total_amount:,.2f} {inv.currency}" if inv.total_amount else "—",
                        "VAT ID": inv.vat_id or "—",
                    })

                st.dataframe(
                    pd.DataFrame(inv_data),
                    use_container_width=True,
                    hide_index=True,
                )

            # Show raw text (Req 4.3: "Keep a text copy of the OCR output")
            with st.expander("📝 Raw Extracted Text (OCR Output)"):
                for inv in invoices:
                    st.markdown(f"**{inv.source_file}**")
                    st.code(inv.raw_text, language=None)

            # Index invoices into vector store (Req 4.3)
            store = st.session_state.vector_store
            if store:
                store.ingest_invoices(invoices)

            # Run matching button
            if st.button("🎯 Run Matching", type="primary"):
                with st.spinner("Running 4-stage matching pipeline..."):
                    _, pipeline = get_or_create_pipeline(settings)
                    results = pipeline.match_invoices(invoices)

                st.session_state.match_results = results
                st.session_state.matching_done = True
                st.rerun()


# ──────────────────────────────────────────────
# Step 3: Results Display
# ──────────────────────────────────────────────

def render_confidence_bar(confidence: float, color: str) -> str:
    """Generate HTML for a confidence bar."""
    pct = int(confidence * 100)
    return f"""
    <div class="confidence-bar">
        <div class="confidence-fill" style="width:{pct}%; background:{color};"></div>
    </div>
    """


def render_evidence_table(evidence: dict):
    """Render field-level evidence as a clean table."""
    if not evidence:
        st.caption("No field-level evidence available.")
        return

    field_map = {
        "invoice_no_match": ("📝 Invoice Number", "invoice_value", "journal_value"),
        "vendor_match": ("🏢 Vendor Name", "invoice_value", "journal_value"),
        "amount_match": ("💰 Amount", "invoice_value", "journal_value"),
        "date_match": ("📅 Date", "invoice_value", "journal_value"),
        "vat_id_match": ("🆔 VAT ID", "invoice_value", "journal_value"),
    }

    rows = []
    for field_key, (label, inv_key, jnl_key) in field_map.items():
        ev = evidence.get(field_key, {})
        if not ev:
            continue

        matched = ev.get("matched", False)
        icon = "✅" if matched else "❌"
        inv_val = ev.get(inv_key, "—")
        jnl_val = ev.get(jnl_key, "—")

        # Extra detail
        detail = ""
        if "similarity" in ev and ev["similarity"] is not None:
            detail = f"({ev['similarity']:.0%} similarity)"
        elif "difference_pct" in ev and ev["difference_pct"] is not None:
            detail = f"({ev['difference_pct']:.1f}% difference)"
        elif "days_diff" in ev and ev["days_diff"] is not None:
            detail = f"({ev['days_diff']} days apart)"

        rows.append({
            "Field": f"{icon} {label}",
            "Invoice": str(inv_val),
            "Journal": str(jnl_val),
            "Detail": detail,
        })

    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )


def render_result_card(result: MatchResult, index: int):
    """Render a single invoice match result card."""
    inv = result.invoice
    is_matched = result.outcome == "MATCHED"
    card_class = "matched" if is_matched else "no-match"
    badge_class = "badge-matched" if is_matched else "badge-nomatch"
    badge_text = "✅ MATCHED" if is_matched else "❌ NO MATCH"
    color = "#34d399" if is_matched else "#f87171"

    # Card header
    st.markdown(f"""
    <div class="match-card {card_class}">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <strong style="font-size:1.1rem;">{inv.invoice_no}</strong>
                <span style="color:gray; margin-left:0.5rem;">{inv.vendor_name}</span>
            </div>
            <span class="badge {badge_class}">{badge_text}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        # Key info
        if is_matched:
            st.markdown(f"**Best Match:** `{result.best_match}`")
        st.markdown(f"**Confidence:** {result.confidence:.0%}")
        st.markdown(render_confidence_bar(result.confidence, color), unsafe_allow_html=True)

        # Extracted fields
        st.markdown("**Extracted Invoice Fields:**")
        field_data = {
            "Vendor": inv.vendor_name,
            "Invoice No": inv.invoice_no,
            "Date": inv.invoice_date,
            "Total": f"{inv.total_amount:,.2f} {inv.currency}" if inv.total_amount else "—",
            "VAT": f"{inv.vat_amount:,.2f}" if inv.vat_amount else "—",
            "VAT ID": inv.vat_id or "—",
        }
        for k, v in field_data.items():
            st.markdown(f"- **{k}:** {v}")

    with col2:
        # Explanation
        st.markdown("**Explanation:**")
        st.info(result.explanation)

        # Evidence table
        if result.evidence:
            st.markdown("**Field-Level Evidence:**")
            render_evidence_table(result.evidence)

    # Top candidates (expandable)
    with st.expander(f"🏆 Top {len(result.top_candidates)} Candidates"):
        for i, c in enumerate(result.top_candidates, 1):
            rank_emoji = ["🥇", "🥈", "🥉"][i - 1] if i <= 3 else f"#{i}"

            cols = st.columns([0.5, 2, 1, 1, 1, 3])
            cols[0].markdown(f"**{rank_emoji}**")
            cols[1].markdown(f"`{c.journal_ref}`")
            cols[2].metric("Combined", f"{c.combined_score:.3f}")
            cols[3].metric("Vector", f"{c.vector_score:.3f}")
            cols[4].metric("Determ.", f"{c.deterministic_score:.3f}")
            cols[5].caption(c.rationale)

            # Detailed evidence for this candidate
            if c.evidence:
                fs = c.evidence.get("field_scores", {})
                if fs:
                    score_cols = st.columns(5)
                    for j, (field, score) in enumerate(fs.items()):
                        score_cols[j].progress(
                            score,
                            text=f"{field}: {score:.2f}",
                        )

            if i < len(result.top_candidates):
                st.divider()


def render_step3_results():
    """Render matching results for all invoices."""
    st.markdown("### 🎯 Step 3: Matching Results")

    if not st.session_state.matching_done:
        st.info("⏳ Upload invoices and run matching in Step 2.")
        return

    results = st.session_state.match_results
    if not results:
        st.warning("No results to display.")
        return

    # Summary bar
    matched = sum(1 for r in results if r.outcome == "MATCHED")
    no_match = len(results) - matched

    cols = st.columns(3)
    cols[0].metric("Total Invoices", len(results))
    cols[1].metric("Matched", matched, delta=None)
    cols[2].metric("No Match", no_match, delta=None)

    st.divider()

    # Render each result card
    for i, result in enumerate(results):
        render_result_card(result, i)
        if i < len(results) - 1:
            st.markdown("---")

    # Export results
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Export Results as JSON"):
            export = [r.to_dict() for r in results]
            json_str = json.dumps(export, indent=2, default=str)
            st.download_button(
                "Download JSON",
                json_str,
                file_name="match_results.json",
                mime="application/json",
            )
    with col2:
        if st.button("💾 Save Cost Log"):
            filepath = cost_logger.save_to_file()
            st.success(f"Saved to {filepath}")


# ──────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────

def main():
    """Main application entry point."""
    # Title
    st.title("🔍 Evidence-to-Journal Matching")
    st.caption(
        "Upload journal entries (XLSX) and evidence invoices (PDF), "
        "then match invoices to the correct journal entry groups "
        "with explainable output."
    )

    # Sidebar
    settings = render_sidebar()

    # Main content — 3 steps
    render_step1_journal_upload(settings)

    st.divider()

    render_step2_evidence_upload(settings)

    st.divider()

    render_step3_results()


if __name__ == "__main__":
    main()

FILEEOF

echo "  Creating tests/__init__.py..."
cat > 'tests/__init__.py' << 'FILEEOF'

FILEEOF

echo "  Creating requirements.txt..."
cat > 'requirements.txt' << 'FILEEOF'
# PDF & XLSX Parsing
PyMuPDF>=1.24.0
pandas>=2.2.0
openpyxl>=3.1.0

# Embeddings (local — Qwen3-Embedding-0.6B)
sentence-transformers>=3.0.0
torch>=2.4.0

# Vector Store
chromadb>=0.5.0

# Fuzzy Matching
rapidfuzz>=3.9.0

# LLM Client (OpenAI-compatible — works with vLLM and OpenAI)
openai>=1.40.0

# LangChain (optional — for advanced LLM integration)
# langchain>=0.3.0
# langchain-openai>=0.2.0

# Backend
# fastapi>=0.115.0
# uvicorn>=0.30.0

# Frontend
streamlit>=1.38.0

# Utilities
python-dateutil>=2.9.0
python-dotenv>=1.0.0

FILEEOF

echo "  Creating .env.example..."
cat > '.env.example' << 'FILEEOF'
# ──────────────────────────────────────────────
# Evidence-to-Journal Matching — Environment Config
# ──────────────────────────────────────────────
# Copy this file to .env and fill in your values.

# ── Provider Toggle ──
# "local" = Qwen3 models via vLLM / sentence-transformers
# "openai" = OpenAI API
EMBEDDING_PROVIDER=local
LLM_PROVIDER=local

# ── OpenAI (when using openai provider) ──
OPENAI_API_KEY=sk-your-key-here

# ── Local vLLM Server (when using local provider) ──
LOCAL_LLM_BASE_URL=http://10.0.9.75:8010/v1
LOCAL_LLM_MODEL=Qwen/Qwen3.5-35B-A3B-FP8
LOCAL_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B

FILEEOF

echo "  Creating .gitignore..."
cat > '.gitignore' << 'FILEEOF'
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/

# Environment
.env
.venv/
venv/
env/

# Data & persistence (user-uploaded)
data/*
!data/.gitkeep
vectordb/*
!vectordb/.gitkeep
logs/*
!logs/.gitkeep

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Models (large files)
*.bin
*.pt
*.safetensors

FILEEOF

echo "  Creating Dockerfile..."
cat > 'Dockerfile' << 'FILEEOF'
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# System dependencies (for PyMuPDF and general build tools)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data vectordb logs

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "app/dashboard/streamlit_app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]

FILEEOF

echo "  Creating docker-compose.yml..."
cat > 'docker-compose.yml' << 'FILEEOF'
version: "3.8"

services:
  app:
    build: .
    ports:
      - "8501:8501"
    env_file:
      - .env
    volumes:
      # Persist vector store and logs across restarts
      - ./vectordb:/app/vectordb
      - ./logs:/app/logs
      - ./data:/app/data
    restart: unless-stopped

FILEEOF

echo "  Creating README.md..."
cat > 'README.md' << 'FILEEOF'
# 🔍 Evidence-to-Journal Matching (Audit Rollup Mini)

An end-to-end prototype that ingests journal entries (XLSX) and evidence invoices (PDF), stores them in a local vector store, and matches invoices to the correct journal entry group(s) with explainable output.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Streamlit Dashboard                       │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ Step 1:      │  │ Step 2:      │  │ Step 3:                │ │
│  │ Upload XLSX  │  │ Upload PDFs  │  │ Matching Results       │ │
│  │ Preview      │  │ Extract      │  │ • Outcome (Match/No)   │ │
│  │ Ingest       │  │ Index        │  │ • Confidence + Explain │ │
│  │              │  │ Run Match    │  │ • Evidence table        │ │
│  │              │  │              │  │ • Top-3 candidates     │ │
│  └──────┬───────┘  └──────┬───────┘  └────────────────────────┘ │
│         │                 │                                      │
│  [Sidebar: Config + Cost Tracker]                                │
└─────────┼─────────────────┼──────────────────────────────────────┘
          │                 │
          ▼                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Matching Pipeline                           │
│                                                                  │
│  Stage 1          Stage 2          Stage 3          Stage 4      │
│  Vector Recall    Reranker         Deterministic    LLM Judge    │
│  (Embedding →     (Cross-encoder   (invoice_no,     (Structured  │
│   ChromaDB        reranking,       vendor, amount,   decision +  │
│   top-K search)   optional)        date, VAT ID)    explanation) │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
          │                                            │
          ▼                                            ▼
┌──────────────────┐                    ┌──────────────────────────┐
│ ChromaDB         │                    │ LLM (swappable)          │
│ • journal_groups │                    │ • Local: Qwen3.5-35B     │
│ • invoices       │                    │ • OpenAI: gpt-4o-mini    │
│ (persisted)      │                    └──────────────────────────┘
└──────────────────┘
```

## Quick Start

### Option A: Local Setup

```bash
# 1. Clone and enter directory
git clone <repo-url>
cd audit-evidence-matcher

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your settings (API keys, model endpoints)

# 5. Run the dashboard
streamlit run app/dashboard/streamlit_app.py
```

### Option B: Docker

```bash
# Build and run
docker build -t audit-matcher .
docker run -p 8501:8501 --env-file .env audit-matcher

# Open http://localhost:8501
```

### Option C: Docker Compose

```bash
docker-compose up
# Open http://localhost:8501
```

## Usage

### Step 1 — Upload Journal Entries
1. Click "Upload XLSX file" and select `sample_journal_entries.xlsx` (or `_v2_noisy.xlsx`)
2. Preview the grouped journal entries in the table
3. Click **"Ingest into Vector Store"**

### Step 2 — Upload Evidence Invoices
1. Click "Upload PDF invoices" and select one or more invoice PDFs
2. Review the extracted fields (vendor, invoice number, date, total)
3. Click **"Run Matching"**

### Step 3 — View Results
For each invoice, the dashboard shows:
- **Outcome**: Matched or No Match
- **Confidence**: 0–100% with visual bar
- **Explanation**: Why this match was chosen (or why no match)
- **Evidence table**: Field-by-field comparison (which matched, which didn't)
- **Top-3 candidates**: journal_ref, score, and rationale

## Configuration

Use the sidebar to toggle:

| Setting | Options | Default |
|---------|---------|---------|
| Embedding Model | Local (Qwen3-Embedding-0.6B) / OpenAI (text-embedding-3-small) | Local |
| LLM Judge | Local (Qwen3.5-35B via vLLM) / OpenAI (gpt-4o-mini) | Local |
| Reranker | Enable/Disable (Qwen3-Reranker-0.6B) | Disabled |
| Dataset | Clean (v1) / Noisy (v2) | Clean |

### Switching to OpenAI

1. Set `OPENAI_API_KEY` in your `.env` file
2. In the sidebar, change Embedding and LLM to "openai"
3. Everything else works identically — no code changes needed

## Project Structure

```
audit-evidence-matcher/
├── app/
│   ├── config.py              # Centralized configuration
│   ├── cost_logger.py         # API cost tracking (Req 5)
│   ├── models.py              # Data classes
│   ├── parsers/
│   │   ├── xlsx_parser.py     # XLSX → grouped JournalGroups
│   │   └── pdf_parser.py      # PDF → structured Invoice
│   ├── indexer/
│   │   ├── embedder.py        # Embedding abstraction (local/OpenAI)
│   │   └── vector_store.py    # ChromaDB operations
│   ├── matcher/
│   │   ├── deterministic.py   # Field-level matching & scoring
│   │   ├── reranker.py        # Cross-encoder reranking
│   │   ├── llm_judge.py       # LLM decision & explanation
│   │   └── pipeline.py        # 4-stage orchestrator
│   └── dashboard/
│       └── streamlit_app.py   # Streamlit UI
├── data/                      # Uploaded files
├── vectordb/                  # ChromaDB persistence
├── logs/                      # Cost logs (JSON)
├── docs/
│   └── technical_writeup.md   # Architecture & design decisions
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Cost Discipline

Estimated costs when using OpenAI:

| Stage | Model | Calls | Est. Cost |
|-------|-------|-------|-----------|
| Embedding (journals) | text-embedding-3-small | 1 batch | ~$0.001 |
| Embedding (invoices) | text-embedding-3-small | 1 batch | ~$0.0005 |
| LLM Judge | gpt-4o-mini | 3 (1/invoice) | ~$0.01 |
| **Total** | | **5 calls** | **< $0.02** |

The cost tracker in the sidebar logs every API call with token counts and estimated cost per stage.

## Noisy Dataset Handling (v2)

The pipeline handles these variations through:

| Noise Type | Solution |
|------------|----------|
| Vendor name variations (`LLC` vs `L.L.C.`) | Normalization: strip dots, commas, collapse spaces |
| Invoice number formats (`INV-ACME-1001` vs `INV ACME 1001`) | Normalization: remove dashes, spaces, dots, lowercase |
| Missing `invoice_no` on some lines | Group-level aggregation: takes first non-null across all lines |
| Sequential distractors (`ORION-403` vs `ORION-404`) | Trailing-digit detection + invoice number gating rule |
| Same vendor, different invoice (`ACME-1001` vs `ACME-1002`) | Invoice number gating: if both have inv_no and they differ, score is capped |

## Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| PDF Parsing | PyMuPDF | Fast, lightweight, 100% accuracy for text-based PDFs |
| XLSX Parsing | pandas + openpyxl | Robust grouping and data manipulation |
| Vector Store | ChromaDB | Simple setup, persistent, Python-native |
| Embeddings | Qwen3-Embedding-0.6B / OpenAI | Instruction-aware, MTEB top-tier for size |
| Fuzzy Matching | rapidfuzz | 10x faster than fuzzywuzzy |
| LLM | Qwen3.5-35B / gpt-4o-mini | Structured JSON output for explainability |
| Dashboard | Streamlit | Fast prototyping with rich widgets |

FILEEOF

echo "  Creating docs/technical_writeup.md..."
cat > 'docs/technical_writeup.md' << 'FILEEOF'
# Technical Write-Up: Evidence-to-Journal Matching

## 1. Architecture Overview

The system follows a modular pipeline architecture with four distinct stages, each responsible for a specific aspect of the matching process:

**Parsers** handle data ingestion. The XLSX parser reads journal entries and groups them by `journal_ref`, producing one structured `JournalGroup` per group. The PDF parser extracts text via PyMuPDF and parses it into structured `Invoice` objects using regex patterns. Both parsers include normalization functions that handle the noisy dataset variations (vendor name abbreviations, invoice number format differences, missing fields).

**Indexer** manages embedding generation and vector storage. An abstract `BaseEmbedder` interface allows swapping between local (Qwen3-Embedding-0.6B via sentence-transformers) and OpenAI (text-embedding-3-small) by changing one config flag. ChromaDB stores two collections: `journal_groups` for the accounting records and `invoices` for the evidence documents. Both collections persist to disk.

**Matcher** implements the 4-stage matching pipeline. For each invoice, it runs: (1) vector search to retrieve top-K candidates, (2) optional cross-encoder reranking for precision, (3) deterministic field-level scoring with explainable evidence, and (4) an LLM judge that produces the final decision with a natural-language explanation.

**Dashboard** is a Streamlit application that presents the three-step workflow: upload XLSX and ingest, upload PDFs and run matching, then view results with full explainability. A sidebar provides model configuration toggles and a live cost tracker.

## 2. Chunking & Indexing Strategy

The key indexing decision is **one embedding document per `journal_ref` group**, not per row. A journal group like `GRP-INV-ACME-1001` may have two rows (one for the expense, one for VAT), but they represent a single transaction. Grouping them produces a richer document that includes all vendor names, invoice numbers, descriptions, totals, and line details in one embedding, improving retrieval accuracy because the invoice query text will share more semantic overlap with a complete transaction summary than with a single accounting line.

The embedding text for each group follows a structured format: `Journal Reference: ... | Vendor: ... | Invoice Number: ... | VAT ID: ... | Total Amount: ... | Descriptions: ...`. This format was chosen over free-text concatenation because embedding models perform better when field roles are explicitly labeled.

For the query side (invoices), the embedding text follows the same structured format. When using Qwen3-Embedding, queries are prefixed with a task instruction ("Given an invoice from a vendor, retrieve the matching journal entry group..."), which improves retrieval by 1-5% according to Qwen's benchmarks. Documents are embedded without this prefix, following the model's asymmetric design.

Metadata stored alongside each embedding includes `journal_ref`, `vendor_name`, `invoice_no`, `total_amount`, `posting_date`, and other fields. This allows the deterministic scoring stage to compare fields without re-parsing the original data.

## 3. Matching Strategy

The pipeline uses a **hybrid approach** combining semantic similarity with deterministic field matching:

**Stage 1 — Vector Recall:** The invoice embedding is queried against the journal_groups collection to retrieve the top-10 candidates by cosine similarity. This is a fast approximate recall step that narrows thousands of potential matches to a manageable set.

**Stage 2 — Cross-Encoder Reranking (optional):** A Qwen3-Reranker-0.6B cross-encoder processes each (invoice, candidate) pair jointly, producing more precise relevance scores than the bi-encoder similarity. This catches subtle mismatches that embedding similarity misses.

**Stage 3 — Deterministic Scoring:** Each candidate is scored across five fields: invoice number (weight 0.35), vendor name (0.25), amount (0.20), date (0.10), and VAT ID (0.10). Invoice number matching uses normalization (strip dashes, spaces, dots) followed by fuzzy comparison via rapidfuzz. A critical gating rule prevents false matches: if both the invoice and journal group have invoice numbers but they don't match, the score is capped at 0.25 regardless of other field scores. This rule specifically defeats distractors like `ORION-403` vs `ORION-404`, where vendor, VAT ID, and date might all match but the invoice numbers differ.

**Stage 4 — LLM Judge:** The top-3 candidates with their scores and field-level evidence are sent to the LLM in a single call per invoice. The LLM returns a structured JSON response with a match/no-match decision, confidence score, explanation, and per-candidate analysis. The prompt includes explicit instructions to be skeptical of near-matches and to verify field evidence before declaring a match.

The combined score weights favor deterministic scoring (50%) over vector similarity (20%) and reranking (30%). This reflects an audit domain principle: exact field matches are more reliable than semantic similarity for transaction matching.

## 4. Cost Controls

The system is designed to minimize API usage. Embeddings are generated in single batch calls (one for all journal groups, one for all invoices). The LLM is called at most once per invoice with all candidates included in a single prompt. With the provided dataset (5-7 journal groups, 3 invoices), the estimated OpenAI cost is under $0.02, well within the $5 budget.

Every API call is logged by the `CostLogger` with stage name, model, token counts, and estimated cost. The Streamlit sidebar displays a live per-stage cost summary, and the full log can be exported as JSON for audit purposes. During development, local models (Qwen3-Embedding-0.6B, Qwen3.5-35B) are used at zero cost, with a config-flag swap to OpenAI for production.

## 5. Key Design Decisions

**PyMuPDF over Docling/Marker for PDF parsing:** The provided invoices are text-based single-page documents. PyMuPDF extracts text in milliseconds with 100% accuracy. Heavier tools like Docling add seconds of latency and gigabytes of dependencies for no quality improvement on these samples. If the system needed to handle scanned multi-page documents, Docling would be the right choice.

**Invoice number gating rule:** The most impactful design decision for correctness. In the noisy dataset, `GRP-ORION-403` shares the same vendor, VAT ID, and similar date with invoice `ORION-404`. Without gating, vector similarity plus vendor/VAT matching produces a false positive. The gating rule recognizes that in accounting, the invoice number is the primary transaction identifier — if it doesn't match, other field similarities are coincidental (same vendor, different transaction).

**Trailing-digit penalty:** Sequential invoice numbers like `ORION-403` and `ORION-404` produce ~87% fuzzy similarity, which is dangerously close to typical match thresholds. A targeted check detects when strings share a prefix but differ only in trailing digits, applying a heavy penalty to prevent sequential-number false matches.

**Graceful degradation:** Every optional component (reranker, LLM judge) has a fallback. If the reranker model isn't available, candidates pass through unchanged. If the LLM call fails, the system falls back to deterministic-only scoring. The pipeline always produces a result.

FILEEOF

# ── Create .gitkeep files for empty dirs ──
touch data/.gitkeep
touch vectordb/.gitkeep
touch logs/.gitkeep

# ── Install dependencies ──
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Installing dependencies into conda env: accord"
echo "══════════════════════════════════════════════════════════"
echo ""

# Check if conda env is active
if [[ "$CONDA_DEFAULT_ENV" != "accord" ]]; then
    echo "⚠️  Conda environment 'accord' is not active."
    echo "   Run: conda activate accord"
    echo "   Then re-run this script, or install manually:"
    echo "   pip install -r requirements.txt"
    echo ""
    read -p "Try to install anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping dependency installation."
        echo ""
        SKIP_INSTALL=true
    fi
fi

if [[ "$SKIP_INSTALL" != "true" ]]; then
    pip install -r requirements.txt
fi

# ── Verify ──
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Verifying project structure"
echo "══════════════════════════════════════════════════════════"
echo ""

PASS=true
for f in \
    "app/models.py" \
    "app/config.py" \
    "app/cost_logger.py" \
    "app/parsers/xlsx_parser.py" \
    "app/parsers/pdf_parser.py" \
    "app/indexer/embedder.py" \
    "app/indexer/vector_store.py" \
    "app/matcher/deterministic.py" \
    "app/matcher/reranker.py" \
    "app/matcher/llm_judge.py" \
    "app/matcher/pipeline.py" \
    "app/dashboard/streamlit_app.py" \
    "requirements.txt" \
    ".env.example" \
    ".gitignore" \
    "Dockerfile" \
    "docker-compose.yml" \
    "README.md" \
    "docs/technical_writeup.md"; do
    if [[ -f "$f" ]]; then
        LINES=$(wc -l < "$f")
        echo "  ✅ $f ($LINES lines)"
    else
        echo "  ❌ MISSING: $f"
        PASS=false
    fi
done

echo ""
TOTAL=$(find . -name "*.py" -not -path "*__pycache__*" -exec cat {} \; | wc -l)
echo "Total Python lines: $TOTAL"
echo ""

if [[ "$PASS" == "true" ]]; then
    echo "══════════════════════════════════════════════════════════"
    echo "  ✅ PROJECT SETUP COMPLETE"
    echo "══════════════════════════════════════════════════════════"
    echo ""
    echo "Next steps:"
    echo "  1. conda activate accord"
    echo "  2. cp .env.example .env"
    echo "  3. Edit .env with your API keys / endpoints"
    echo "  4. streamlit run app/dashboard/streamlit_app.py"
    echo ""
else
    echo "❌ SETUP INCOMPLETE — some files are missing"
    exit 1
fi
