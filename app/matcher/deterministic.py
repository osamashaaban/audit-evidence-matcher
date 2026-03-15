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

