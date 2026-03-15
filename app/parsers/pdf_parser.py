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

    Strategy: Position-based. In standard invoice layout, the vendor
    name is the first non-header, non-metadata line after the
    Invoice No / Invoice Date block. This works regardless of
    whether the company name contains LLC, Ltd, Inc, etc.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Skip lines that are clearly headers or metadata
    skip_patterns = [
        "invoice no", "invoice date", "invoice #", "invoice number",
        "bill to", "vat id", "tax id",
    ]

    found_header = False
    past_date = False

    for line in lines:
        lower = line.lower()

        # Mark that we've seen the INVOICE header
        if lower == "invoice" or lower.startswith("invoice"):
            found_header = True

        # Skip known metadata lines
        if any(p in lower for p in skip_patterns):
            if "date" in lower:
                past_date = True
            continue

        # Once we're past the Invoice Date line, the next
        # substantial line is the vendor name
        if found_header and past_date:
            # Sanity checks: not an address number, not too short
            if len(line) > 3 and not line[0].isdigit():
                return line

    return ""


def parse_invoice(
    text: str,
    source_file: str = "",
) -> Invoice:
    """
    Parse raw text into a structured Invoice object.

    Hybrid approach:
      1. Try regex extraction first (fast, free)
      2. If any key field is missing, fall back to gpt-4o-mini

    Args:
        text: Raw text extracted from the PDF.
        source_file: Original filename for reference.

    Returns:
        Invoice object with all extracted fields.
    """
    # ── Step 1: Regex extraction (fast, free) ──
    vendor_name = _extract_vendor_name(text)
    invoice_no = _extract_field(text, r"Invoice\s*No[:\s]+(.+?)(?:\n|$)")
    invoice_date = _extract_field(text, r"Invoice\s*Date[:\s]+(\d{4}-\d{2}-\d{2})")

    subtotal = _extract_amount_after_label(text, "Subtotal")
    total_amount = _extract_amount_after_label(text, r"^Total")
    vat_amount = _extract_amount_after_label(text, r"VAT\s*\(\d+%?\)")
    vat_rate = _extract_float(text, r"VAT\s*\((\d+)%?\)")
    if vat_rate is not None:
        vat_rate = vat_rate / 100.0

    if total_amount is None:
        total_amount = _extract_amount_after_label(text, "Total")

    vat_id = _extract_field(text, r"VAT\s*ID[:\s]+(\S+)")
    payment_terms = _extract_field(text, r"Payment\s*Terms[:\s]+(.+?)(?:\n|$)")

    # ── Step 2: LLM fallback if key fields are missing ──
    key_fields_missing = (
        not vendor_name
        or not invoice_no
        or total_amount is None
    )

    if key_fields_missing:
        logger.info(
            f"Regex missed key fields for {source_file} "
            f"(vendor={bool(vendor_name)}, inv_no={bool(invoice_no)}, "
            f"total={total_amount is not None}). Falling back to LLM."
        )
        llm_fields = _llm_extract_fields(text)

        if not vendor_name and llm_fields.get("vendor_name"):
            vendor_name = llm_fields["vendor_name"]
        if not invoice_no and llm_fields.get("invoice_no"):
            invoice_no = llm_fields["invoice_no"]
        if not invoice_date and llm_fields.get("invoice_date"):
            invoice_date = llm_fields["invoice_date"]
        if total_amount is None and llm_fields.get("total_amount") is not None:
            total_amount = llm_fields["total_amount"]
        if subtotal is None and llm_fields.get("subtotal") is not None:
            subtotal = llm_fields["subtotal"]
        if vat_amount is None and llm_fields.get("vat_amount") is not None:
            vat_amount = llm_fields["vat_amount"]
        if not vat_id and llm_fields.get("vat_id"):
            vat_id = llm_fields["vat_id"]

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
        raw_text=text,
        source_file=source_file,
    )

    logger.info(
        f"Parsed invoice: vendor={vendor_name}, no={invoice_no}, "
        f"date={invoice_date}, total={total_amount}"
    )

    return invoice


def _llm_extract_fields(text: str) -> dict:
    """
    Fallback: use gpt-4o-mini to extract invoice fields from raw text.
    Called only when regex misses key fields.
    Cost: ~$0.0003 per call.
    """
    try:
        import os
        from openai import OpenAI
        from app.cost_logger import cost_logger

        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("No OpenAI API key — cannot use LLM fallback for parsing")
            return {}

        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract invoice fields from the text. "
                        "Return ONLY valid JSON with these keys: "
                        "vendor_name, invoice_no, invoice_date (YYYY-MM-DD), "
                        "subtotal (number), vat_amount (number), total_amount (number), "
                        "vat_id. Use null for missing fields."
                    ),
                },
                {"role": "user", "content": text[:2000]},  # limit context
            ],
            temperature=0.0,
            max_tokens=300,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        cost = (input_tokens / 1_000_000) * 0.15 + (output_tokens / 1_000_000) * 0.60

        cost_logger.log(
            stage="pdf_parsing_fallback",
            provider="openai",
            model="gpt-4o-mini",
            operation="extract_fields",
            num_calls=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
        )

        import json
        result = json.loads(raw)

        # Convert numeric strings to floats
        for key in ["subtotal", "vat_amount", "total_amount"]:
            if key in result and result[key] is not None:
                try:
                    result[key] = float(result[key])
                except (ValueError, TypeError):
                    result[key] = None

        logger.info(f"LLM fallback extracted: {result}")
        return result

    except Exception as e:
        logger.error(f"LLM fallback failed: {e}")
        return {}


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