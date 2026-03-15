"""
Vector Store operations using ChromaDB.

Two collections:
  1. journal_groups — one document per journal_ref group (Req 4.2)
  2. invoices       — indexed invoice text/summary (Req 4.3)

Duplicate Detection (audit safety):
  - Journal groups: warns if a journal_ref already exists in the store
  - Invoices: warns if an invoice_no was already uploaded (double payment risk)

ChromaDB persists to disk automatically.
"""

from __future__ import annotations

import logging
from typing import Optional

import chromadb
from chromadb.config import Settings

from app.config import VectorStoreConfig
from app.indexer.embedder import Embedder
from app.models import Invoice, JournalGroup

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages ChromaDB collections for journals and invoices.

    Usage:
        store = VectorStore(config, embedder)
        count, warnings = store.ingest_journal_groups(groups)
        count, warnings = store.ingest_invoices(invoices)
        results = store.query_by_invoice(invoice, top_k=10)
    """

    def __init__(self, config: VectorStoreConfig, embedder: Embedder):
        self.config = config
        self.embedder = embedder

        self._client = chromadb.PersistentClient(
            path=config.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        self._journal_collection = None
        self._invoice_collection = None

        logger.info(f"ChromaDB initialized at: {config.persist_dir}")

    # ──────────────────────────────────────────
    # Collection access
    # ──────────────────────────────────────────

    @property
    def journal_collection(self):
        if self._journal_collection is None:
            self._journal_collection = self._client.get_or_create_collection(
                name=self.config.journal_collection_name,
                metadata={"hnsw:space": self.config.distance_metric},
            )
        return self._journal_collection

    @property
    def invoice_collection(self):
        if self._invoice_collection is None:
            self._invoice_collection = self._client.get_or_create_collection(
                name=self.config.invoice_collection_name,
                metadata={"hnsw:space": self.config.distance_metric},
            )
        return self._invoice_collection

    # ──────────────────────────────────────────
    # Journal Ingestion with duplicate detection
    # ──────────────────────────────────────────

    def ingest_journal_groups(self, groups: list[JournalGroup]) -> int:
        """
        Embed and store journal groups in the vector store.

        Duplicate detection is handled at the filename level in the
        Streamlit app, not here. This method simply upserts groups.

        Returns:
            Number of groups ingested.
        """
        if not groups:
            return 0

        # Deduplicate by journal_ref within the batch.
        # If two files have the same group (e.g., GRP-PAY-2026-01 in both
        # v1 and v2), the last one wins (later file overrides earlier).
        seen: dict[str, int] = {}
        for i, g in enumerate(groups):
            seen[g.journal_ref] = i  # last index wins
        unique_groups = [groups[i] for i in sorted(seen.values())]

        ids = [g.journal_ref for g in unique_groups]
        documents = [g.to_embedding_text() for g in unique_groups]
        metadatas = [g.to_metadata() for g in unique_groups]

        logger.info(f"Generating embeddings for {len(unique_groups)} journal groups "
                     f"({len(groups) - len(unique_groups)} duplicates merged)...")
        embeddings = self.embedder.embed_documents(documents)

        self.journal_collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(f"Ingested {len(unique_groups)} journal groups into vector store")
        return len(unique_groups)

    # ──────────────────────────────────────────
    # Invoice Ingestion with duplicate detection
    # ──────────────────────────────────────────

    def ingest_invoices(self, invoices: list[Invoice]) -> tuple[int, list[dict]]:
        """
        Embed and store invoices with duplicate detection.

        Duplicate detection is critical for audit:
          - Same invoice_no within one batch → duplicate submission
          - Same invoice_no already in store → previously uploaded
        Both are flagged as potential double payment risks.

        Returns:
            (ingested_count, list_of_duplicate_warnings)
        """
        if not invoices:
            return 0, []

        warnings: list[dict] = []
        unique_invoices: list[Invoice] = []

        # Track invoice_nos seen in this batch
        batch_seen: dict[str, str] = {}  # invoice_no → source_file

        # Get already-stored invoice_nos
        stored_inv_nos: dict[str, str] = {}  # invoice_no → source_file
        try:
            stored = self.invoice_collection.get(include=["metadatas"])
            if stored and stored["ids"]:
                for meta in stored["metadatas"]:
                    inv_no = meta.get("invoice_no", "")
                    src = meta.get("source_file", "unknown")
                    if inv_no:
                        stored_inv_nos[inv_no] = src
        except Exception:
            pass

        for inv in invoices:
            inv_no = (inv.invoice_no or "").strip()

            # Check 1: duplicate within this batch
            if inv_no and inv_no in batch_seen:
                first_file = batch_seen[inv_no]
                warnings.append({
                    "type": "DUPLICATE_INVOICE",
                    "severity": "HIGH",
                    "invoice_no": inv_no,
                    "vendor": inv.vendor_name,
                    "current_file": inv.source_file,
                    "original_file": first_file,
                    "message": (
                        f"DUPLICATE INVOICE DETECTED: Invoice '{inv_no}' from "
                        f"'{inv.vendor_name}' appears in both '{first_file}' and "
                        f"'{inv.source_file}'. This may indicate a duplicate "
                        f"submission (mistake) or fraud. An auditor should verify "
                        f"this is not a double payment risk."
                    ),
                })
                logger.warning(f"DUPLICATE INVOICE (batch): {inv_no} in {inv.source_file}")
                continue  # skip duplicate, don't store

            # Check 2: already stored from a previous upload
            if inv_no and inv_no in stored_inv_nos:
                existing_file = stored_inv_nos[inv_no]
                warnings.append({
                    "type": "DUPLICATE_INVOICE",
                    "severity": "HIGH",
                    "invoice_no": inv_no,
                    "vendor": inv.vendor_name,
                    "current_file": inv.source_file,
                    "original_file": existing_file,
                    "message": (
                        f"DUPLICATE INVOICE DETECTED: Invoice '{inv_no}' was "
                        f"previously uploaded in '{existing_file}'. Now re-submitted "
                        f"in '{inv.source_file}'. Duplicate invoices are a risk for "
                        f"double payments. An auditor should flag this for review."
                    ),
                })
                logger.warning(f"DUPLICATE INVOICE (stored): {inv_no} in {inv.source_file}")
                continue  # skip duplicate, don't store

            # New invoice — track and keep
            if inv_no:
                batch_seen[inv_no] = inv.source_file or ""
            unique_invoices.append(inv)

        # Store unique invoices
        if unique_invoices:
            ids = [inv.source_file or inv.invoice_no or f"inv_{i}"
                   for i, inv in enumerate(unique_invoices)]
            documents = [inv.to_embedding_text() for inv in unique_invoices]
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
                for inv in unique_invoices
            ]

            logger.info(f"Generating embeddings for {len(unique_invoices)} invoices...")
            embeddings = self.embedder.embed_documents(documents)

            self.invoice_collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            logger.info(f"Ingested {len(unique_invoices)} invoices into vector store")

        return len(unique_invoices), warnings

    # ──────────────────────────────────────────
    # Query
    # ──────────────────────────────────────────

    def query_by_invoice(
        self,
        invoice: Invoice,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Find the top-K matching journal groups for an invoice.

        Returns list of dicts with journal_ref, document, metadata,
        distance, and similarity.
        """
        query_text = invoice.to_embedding_text()
        query_embedding = self.embedder.embed_query(query_text)

        n_available = self.journal_collection.count()
        effective_k = min(top_k, n_available)

        if effective_k == 0:
            logger.warning("Journal collection is empty")
            return []

        results = self.journal_collection.query(
            query_embeddings=[query_embedding],
            n_results=effective_k,
            include=["documents", "metadatas", "distances"],
        )

        candidates = []
        if results["ids"] and results["ids"][0]:
            for i, ref in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
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
        return self.journal_collection.count()

    def get_invoice_count(self) -> int:
        return self.invoice_collection.count()

    def get_all_journal_metadatas(self) -> list[dict]:
        result = self.journal_collection.get(include=["metadatas"])
        return result["metadatas"] if result["metadatas"] else []

    def reset_journals(self):
        try:
            self._client.delete_collection(self.config.journal_collection_name)
        except Exception:
            pass
        self._journal_collection = None
        logger.info("Journal collection reset")

    def reset_invoices(self):
        try:
            self._client.delete_collection(self.config.invoice_collection_name)
        except Exception:
            pass
        self._invoice_collection = None
        logger.info("Invoice collection reset")

    def reset_all(self):
        self.reset_journals()
        self.reset_invoices()
        logger.info("All collections reset")