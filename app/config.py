"""
Centralized configuration for the Evidence-to-Journal Matching pipeline.
Uses OpenAI APIs for embeddings and LLM.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VECTORDB_DIR = PROJECT_ROOT / "vectordb"
LOGS_DIR = PROJECT_ROOT / "logs"

DATA_DIR.mkdir(exist_ok=True)
VECTORDB_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────
# Embedding Config (OpenAI)
# ──────────────────────────────────────────────

@dataclass
class EmbeddingConfig:
    model_name: str = "text-embedding-3-small"
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    batch_size: int = 32
    cost_per_1m_tokens: float = 0.02

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY", "")


# ──────────────────────────────────────────────
# LLM Config (OpenAI)
# ──────────────────────────────────────────────

@dataclass
class LLMConfig:
    model_name: str = "gpt-4o-mini"
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.0
    max_tokens: int = 1000
    input_cost_per_1m: float = 0.15
    output_cost_per_1m: float = 0.60

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY", "")


# ──────────────────────────────────────────────
# Vector Store Config
# ──────────────────────────────────────────────

@dataclass
class VectorStoreConfig:
    persist_dir: str = str(VECTORDB_DIR)
    journal_collection_name: str = "journal_groups"
    invoice_collection_name: str = "invoices"
    distance_metric: str = "cosine"


# ──────────────────────────────────────────────
# Matching Config
# ──────────────────────────────────────────────

@dataclass
class MatchingConfig:
    top_k_retrieval: int = 10

    # Deterministic scoring weights
    weight_invoice_no: float = 0.35
    weight_vendor_name: float = 0.25
    weight_amount: float = 0.20
    weight_date: float = 0.10
    weight_vat_id: float = 0.10

    # Thresholds
    fuzzy_invoice_no_threshold: float = 0.92
    fuzzy_vendor_name_threshold: float = 0.75
    amount_tolerance_pct: float = 0.05
    date_proximity_days: int = 7

    # Match threshold — deterministic score needed for MATCH
    match_threshold: float = 0.60
    min_candidates_to_show: int = 3

    # LLM explanation
    use_llm_judge: bool = True


# ──────────────────────────────────────────────
# Master Config
# ──────────────────────────────────────────────

@dataclass
class AppConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)


def get_config(use_llm_judge: bool = True) -> AppConfig:
    """Build application config."""
    config = AppConfig()
    config.matching.use_llm_judge = use_llm_judge
    return config