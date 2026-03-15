"""
Cost logging for API call tracking.

Req 5: "Log: number of OpenAI calls, approximate tokens,
and estimated cost per stage."

Always estimates the OpenAI equivalent cost, even when running local models.
This allows accurate budget planning for OpenAI deployment.

OpenAI Pricing (as of March 2026):
  - text-embedding-3-small: $0.02 / 1M tokens
  - gpt-4o-mini: $0.15 / 1M input tokens, $0.60 / 1M output tokens
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from app.models import CostLogEntry
from app.config import LOGS_DIR

# OpenAI pricing constants
OPENAI_EMBED_COST_PER_1M = 0.02     # text-embedding-3-small
OPENAI_LLM_INPUT_PER_1M = 0.15      # gpt-4o-mini input
OPENAI_LLM_OUTPUT_PER_1M = 0.60     # gpt-4o-mini output


def estimate_openai_cost(stage: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate what this call would cost on OpenAI, regardless of actual provider."""
    if stage == "embedding":
        return (input_tokens / 1_000_000) * OPENAI_EMBED_COST_PER_1M
    elif stage in ("llm_judge", "pdf_parsing_fallback"):
        return (
            (input_tokens / 1_000_000) * OPENAI_LLM_INPUT_PER_1M
            + (output_tokens / 1_000_000) * OPENAI_LLM_OUTPUT_PER_1M
        )
    return 0.0


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
        # If provider is local (cost=0), still estimate OpenAI equivalent
        openai_estimate = estimate_openai_cost(stage, input_tokens, output_tokens)

        # Use actual cost if OpenAI, otherwise store the estimate
        actual_cost = estimated_cost_usd if provider == "openai" else 0.0

        entry = CostLogEntry(
            stage=stage,
            provider=provider,
            model=model,
            operation=operation,
            num_calls=num_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=actual_cost,
        )
        # Store OpenAI estimate as extra attribute
        entry._openai_estimate = openai_estimate
        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> list[CostLogEntry]:
        return list(self._entries)

    def clear(self):
        self._entries.clear()

    def summary_by_stage(self) -> dict[str, dict]:
        """Aggregate stats grouped by stage."""
        stages: dict[str, dict] = {}
        for entry in self._entries:
            if entry.stage not in stages:
                stages[entry.stage] = {
                    "num_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "estimated_cost_usd": 0.0,
                    "openai_equivalent_usd": 0.0,
                    "provider": entry.provider,
                    "model": entry.model,
                }
            s = stages[entry.stage]
            s["num_calls"] += entry.num_calls
            s["input_tokens"] += entry.input_tokens
            s["output_tokens"] += entry.output_tokens
            s["estimated_cost_usd"] += entry.estimated_cost_usd
            s["openai_equivalent_usd"] += getattr(entry, "_openai_estimate", 0.0)
        return stages

    def total_cost(self) -> float:
        return sum(e.estimated_cost_usd for e in self._entries)

    def total_openai_equivalent(self) -> float:
        return sum(getattr(e, "_openai_estimate", 0.0) for e in self._entries)

    def total_calls(self) -> int:
        return sum(e.num_calls for e in self._entries)

    def total_tokens(self) -> int:
        return sum(e.input_tokens + e.output_tokens for e in self._entries)

    def save_to_file(self, filename: Optional[str] = None) -> Path:
        """Persist log to JSON file."""
        if filename is None:
            filename = "cost_log.json"
        filepath = LOGS_DIR / filename

        openai_eq = self.total_openai_equivalent()
        data = {
            "summary": {
                "total_calls": self.total_calls(),
                "total_tokens": self.total_tokens(),
                "actual_cost_usd": round(self.total_cost(), 6),
                "openai_equivalent_usd": round(openai_eq, 6),
                "budget_remaining_usd": round(5.0 - openai_eq, 4),
                "budget_used_pct": round((openai_eq / 5.0) * 100, 4),
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
            provider_icon = "☁️" if info["provider"] == "openai" else "🏠"
            tokens = info["input_tokens"] + info["output_tokens"]
            openai_cost = info["openai_equivalent_usd"]
            lines.append(
                f"**{stage}** {provider_icon}  \n"
                f"`{info['num_calls']}` calls · `{tokens:,}` tokens · "
                f"OpenAI est: `${openai_cost:.4f}`"
            )

        lines.append("---")
        actual = self.total_cost()
        openai_eq = self.total_openai_equivalent()
        lines.append(
            f"**Actual cost:** `${actual:.4f}`  \n"
            f"**OpenAI equivalent:** `${openai_eq:.4f}`  \n"
            f"**Budget remaining:** `${5.0 - openai_eq:.4f}` of $5.00"
        )
        return "\n\n".join(lines)


# Module-level instance
cost_logger = CostLogger()