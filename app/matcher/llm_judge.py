"""
LLM Judge — Final decision and explanation generation using OpenAI gpt-4o-mini.

Req 4.4: "optional single LLM call per invoice to produce a
          structured decision + explanation from top candidates"

Req 5: "If using an LLM, prefer at most one call per invoice."

Exactly ONE LLM call per invoice.
"""

from __future__ import annotations

import json
import logging
import re
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
    """Makes final match decision with LLM-generated explanation. One call per invoice."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        from openai import OpenAI
        if not self.config.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env.")
        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )
        return self._client

    def judge(self, invoice: Invoice, candidates: list[MatchCandidate]) -> dict:
        """Make final decision for one invoice. Exactly 1 API call."""
        client = self._get_client()

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

        try:
            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"},
            )

            raw_content = response.choices[0].message.content
            result = self._parse_response(raw_content)

            # Cost logging
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            cost = (
                (input_tokens / 1_000_000) * self.config.input_cost_per_1m
                + (output_tokens / 1_000_000) * self.config.output_cost_per_1m
            )

            cost_logger.log(
                stage="llm_judge",
                provider="openai",
                model=self.config.model_name,
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
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            json_match = re.search(r'\{[\s\S]*\}', cleaned)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                except json.JSONDecodeError:
                    result = {}
            else:
                result = {}

        result.setdefault("decision", "NO_MATCH")
        result.setdefault("best_match_journal_ref", None)
        result.setdefault("confidence", 0.0)
        result.setdefault("explanation", "LLM response could not be fully parsed.")
        result.setdefault("candidates_analysis", [])

        decision = result["decision"].upper().strip()
        result["decision"] = "MATCHED" if decision in ("MATCHED", "MATCH") else "NO_MATCH"

        try:
            result["confidence"] = float(result["confidence"])
        except (ValueError, TypeError):
            result["confidence"] = 0.0

        return result

    def _fallback_result(self, invoice: Invoice, candidates: list[MatchCandidate], error: str) -> dict:
        if candidates and candidates[0].combined_score >= 0.60:
            best = candidates[0]
            return {
                "decision": "MATCHED",
                "best_match_journal_ref": best.journal_ref,
                "confidence": round(best.combined_score, 2),
                "explanation": f"LLM unavailable ({error}). Decision based on deterministic scoring.",
                "candidates_analysis": [],
            }
        return {
            "decision": "NO_MATCH",
            "best_match_journal_ref": None,
            "confidence": 0.7,
            "explanation": f"LLM unavailable ({error}). No candidate scored above threshold.",
            "candidates_analysis": [],
        }


class NoOpJudge:
    """Fallback judge using deterministic scores only (no LLM call)."""

    def judge(self, invoice: Invoice, candidates: list[MatchCandidate]) -> dict:
        if candidates and candidates[0].combined_score >= 0.60:
            best = candidates[0]
            return {
                "decision": "MATCHED",
                "best_match_journal_ref": best.journal_ref,
                "confidence": round(best.combined_score, 2),
                "explanation": (
                    f"Matched based on deterministic scoring. "
                    f"Best candidate {best.journal_ref} scored {best.combined_score:.2f}. "
                    f"{best.rationale}"
                ),
                "candidates_analysis": [
                    {"journal_ref": c.journal_ref,
                     "relevance": "high" if c.combined_score > 0.5 else "low",
                     "rationale": c.rationale}
                    for c in candidates[:3]
                ],
            }
        return {
            "decision": "NO_MATCH",
            "best_match_journal_ref": None,
            "confidence": 0.8,
            "explanation": (
                "No candidate scored above the match threshold. "
                + (f"Closest: {candidates[0].journal_ref} ({candidates[0].combined_score:.2f}): "
                   f"{candidates[0].rationale}" if candidates else "No candidates.")
            ),
            "candidates_analysis": [
                {"journal_ref": c.journal_ref, "relevance": "low", "rationale": c.rationale}
                for c in candidates[:3]
            ],
        }


def get_llm_judge(config: LLMConfig, enabled: bool = True):
    if not enabled:
        logger.info("LLM judge disabled — using deterministic-only fallback")
        return NoOpJudge()
    return LLMJudge(config)