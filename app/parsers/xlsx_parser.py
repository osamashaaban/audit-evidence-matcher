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
import re
from io import BytesIO
from pathlib import Path
from typing import Union

import pandas as pd

from app.models import JournalGroup

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Normalization helpers (shared with deterministic matcher)
# ──────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """
    Generic text normalization — works on any text input.

    Steps (applied in order):
      1. Lowercase
      2. Strip leading/trailing whitespace
      3. Remove all punctuation (. , - ' " () {} [] / \\ : ; ! ? @ # $ % ^ & * ~)
      4. Collapse multiple spaces into one
      5. Strip again

    Example:
      'ACME Office Supplies L.L.C.'  → 'acme office supplies llc'
      'Nimbus Cloud Svcs, Inc.'      → 'nimbus cloud svcs inc'
      'Pharos Legal Advisors (LLP)'  → 'pharos legal advisors llp'
      'Atlas Construction & Eng.'    → 'atlas construction  eng'  → 'atlas construction eng'
    """
    if not text:
        return ""
    result = text.lower().strip()
    # Remove ALL punctuation — this handles any abbreviation style
    result = re.sub(r"[.\,\-\'\"()\[\]{}/\\:;!?@#$%^&*~]", "", result)
    # Collapse multiple spaces
    result = " ".join(result.split())
    return result


def normalize_vendor_name(name: str) -> str:
    """
    Normalize vendor name for comparison.

    Uses generic normalization, then optionally strips common
    legal suffixes so "Delta Logistics Co" and "Delta Logistics"
    can still be compared by the fuzzy matcher.

    Example:
      'ACME Office Supplies L.L.C.' → 'acme office supplies llc'
      'Zenith I.T. Solutions L.L.C.' → 'zenith it solutions llc'
      '— '                           → ''
    """
    if not name or name.strip() in ("—", "-", ""):
        return ""
    return normalize_text(name)


def normalize_invoice_no(inv_no: str) -> str:
    """
    Normalize invoice number for comparison.

    Removes ALL non-alphanumeric characters so any format
    variation produces the same result.

    Example:
      'INV-ACME-1001'  → 'invacme1001'
      'INV ACME 1001'  → 'invacme1001'
      'INV.ACME.1001'  → 'invacme1001'
      'NIMBUS-778-A'   → 'nimbus778a'
      'ZEN 880'        → 'zen880'
      'ZEN-880'        → 'zen880'
    """
    if not inv_no:
        return ""
    # Remove everything except letters and digits
    result = re.sub(r"[^a-zA-Z0-9]", "", str(inv_no))
    return result.lower()


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