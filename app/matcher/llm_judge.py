"""
LLM Explainer — generates natural-language explanations using gpt-4o-mini.

Exactly ONE API call per invoice. The LLM does NOT make the match decision —
it only explains the result that the deterministic scoring already decided.

Req 4.4: "optional single LLM call per invoice to produce a structured
          decision + explanation from top candidates"
Req 5:   "If using an LLM, prefer at most one call per invoice."
"""

from __future__ import annotations

import json
import logging
import re

from app.config import LLMConfig
from app.cost_logger import cost_logger
from app.models import Invoice, MatchCandidate

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are a financial auditor explaining invoice-to-journal matching results.
You will receive an invoice, a ranked list of candidate journal groups with scores and
field-level evidence, and the system's decision (MATCHED or NO_MATCH).

Your task: explain WHY the decision is correct based on the evidence provided.

Rules:
- Invoice number match is the STRONGEST signal.
- If no candidate has a matching invoice number AND matching vendor, NO_MATCH is correct.
- Amount differences may exist due to VAT splits (base + VAT as separate journal lines).
- Be skeptical of near-matches: ORION-403 is NOT the same as ORION-404.

Respond with ONLY valid JSON:
{
  "explanation": "2-3 sentences explaining why the match was chosen or why no match exists",
  "candidates_analysis": [
    {"journal_ref": "...", "relevance": "high/medium/low", "rationale": "1 sentence"}
  ]
}"""

USER_TEMPLATE = """## Invoice
- Vendor: {vendor_name}
- Invoice No: {invoice_no}
- Date: {invoice_date}
- Total: {total_amount} {currency}
- VAT ID: {vat_id}

## System Decision: {decision}
{match_info}

## Top Candidates (ranked by deterministic score)
{candidates_text}"""


def _format_candidates(candidates: list[MatchCandidate]) -> str:
    """Format top candidates for the LLM prompt."""
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
            f"- Deterministic Score: {c.deterministic_score:.3f}\n"
            f"- Vector Similarity: {c.vector_score:.3f}\n"
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
# Explainer
# ──────────────────────────────────────────────

def explain(
    invoice: Invoice,
    candidates: list[MatchCandidate],
    decision: str,
    best_match: str | None,
    config: LLMConfig,
) -> dict:
    """
    Generate explanation for the matching result. One API call.

    Args:
        invoice: The invoice being matched.
        candidates: Ranked candidates (top 3 is enough).
        decision: "MATCHED" or "NO_MATCH" (already decided by deterministic).
        best_match: journal_ref of best match, or None.
        config: LLM configuration.

    Returns:
        {"explanation": "...", "candidates_analysis": [...]}
    """
    # Build prompt
    match_info = ""
    if decision == "MATCHED" and best_match:
        match_info = f"Best match: {best_match} (score: {candidates[0].combined_score:.3f})"
    else:
        top_score = candidates[0].combined_score if candidates else 0
        match_info = f"No match found. Closest candidate scored {top_score:.3f} (threshold: 0.60)"

    user_prompt = USER_TEMPLATE.format(
        vendor_name=invoice.vendor_name,
        invoice_no=invoice.invoice_no,
        invoice_date=invoice.invoice_date,
        total_amount=invoice.total_amount,
        currency=invoice.currency,
        vat_id=invoice.vat_id or "N/A",
        decision=decision,
        match_info=match_info,
        candidates_text=_format_candidates(candidates[:3]),
    )

    try:
        from openai import OpenAI

        if not config.api_key:
            return _build_explanation_from_evidence(candidates, decision, best_match)

        client = OpenAI(api_key=config.api_key, base_url=config.base_url)

        response = client.chat.completions.create(
            model=config.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content

        # Cost logging
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        cost = (
            (input_tokens / 1_000_000) * config.input_cost_per_1m
            + (output_tokens / 1_000_000) * config.output_cost_per_1m
        )

        cost_logger.log(
            stage="llm_judge",
            provider="openai",
            model=config.model_name,
            operation=f"explain_{invoice.invoice_no}",
            num_calls=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
        )

        result = _parse_response(raw)

        logger.info(
            f"LLM explanation for {invoice.invoice_no}: "
            f"{len(result.get('explanation', ''))} chars"
        )
        return result

    except Exception as e:
        logger.error(f"LLM explanation failed for {invoice.invoice_no}: {e}")
        return _build_explanation_from_evidence(candidates, decision, best_match)


def _parse_response(raw: str) -> dict:
    """Parse LLM JSON response with fallbacks."""
    cleaned = raw.strip()

    # Strip markdown backticks if present
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        json_match = re.search(r'\{[\s\S]*\}', cleaned)
        if json_match:
            try:
                result = json.loads(json_match.group())
            except json.JSONDecodeError:
                result = {}
        else:
            result = {}

    result.setdefault("explanation", "")
    result.setdefault("candidates_analysis", [])
    return result


def _build_explanation_from_evidence(
    candidates: list[MatchCandidate],
    decision: str,
    best_match: str | None,
) -> dict:
    """
    Build explanation from deterministic evidence when LLM is unavailable.
    """
    if decision == "MATCHED" and candidates:
        best = candidates[0]
        explanation = (
            f"Matched to {best.journal_ref} with deterministic score "
            f"{best.combined_score:.2f}. {best.rationale}"
        )
    elif candidates:
        best = candidates[0]
        explanation = (
            f"No match found. Closest candidate {best.journal_ref} scored "
            f"{best.combined_score:.2f} (below 0.60 threshold). {best.rationale}"
        )
    else:
        explanation = "No candidates found in the vector store."

    return {
        "explanation": explanation,
        "candidates_analysis": [
            {
                "journal_ref": c.journal_ref,
                "relevance": "high" if c.combined_score > 0.5 else "low",
                "rationale": c.rationale,
            }
            for c in candidates[:3]
        ],
    }