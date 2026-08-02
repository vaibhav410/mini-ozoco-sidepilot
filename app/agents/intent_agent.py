"""Agent 5 -- Intent Detection Agent.

Classifies every request into one of the SidePilot intents and
recommends the workflow that should fulfil it. Two-tier design:

1. Heuristic fast-path: high-precision keyword rules resolve the
   obvious cases ("summarize this", "draft an email ...") with zero
   LLM calls -- lower latency, no quota spent.
2. LLM classification: ambiguous requests go to the shared text model
   (Gemini -> Groq fallback) with document and screen context included
   in the prompt.

The agent is stateless: primitives in, :class:`IntentResult` out. It
plugs into the workflow's Understand stage through the intent-detector
injection point, and is also exposed directly via POST /intent/detect.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.agents.llm import build_llm, model_used
from app.rag.prompt import INTENT_PROMPT
from app.utils.json_utils import extract_json_object
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Every intent the SidePilot understands, mapped to the workflow that
# should fulfil it. Single source of truth -- the validator, the
# automation stage and the API docs all derive from this table.
INTENT_WORKFLOWS: dict[str, str] = {
    "question_answering": "rag_answer",
    "summarization": "rag_summarize",
    "screen_help": "screen_assist",
    "automation": "automation_generic",
    "email": "automation_email",
    "export": "automation_export",
    "search": "rag_answer",
    "navigation": "screen_assist",
    "classification": "document_classify",
}

# High-precision rules tried before any LLM call. Verb-anchored on
# purpose: "draft an email to HR" is an email intent, while "what is
# his email?" must fall through to Q&A classification.
_HEURISTICS: list[tuple[re.Pattern[str], str, float]] = [
    (
        re.compile(
            r"^\s*(summari[sz]e|tl;?dr\b|give me (a |an )?(summary|overview))",
            re.IGNORECASE,
        ),
        "summarization",
        0.95,
    ),
    (
        re.compile(
            r"\b(write|draft|compose|send|prepare)\b[^.?!]*\b(e-?mail|mail|reply|message)\b",
            re.IGNORECASE,
        ),
        "email",
        0.9,
    ),
    (
        re.compile(
            r"\b(export|download|save)\b[^.?!]*\b(pdf|markdown|md|file|report|notes?|summary)\b",
            re.IGNORECASE,
        ),
        "export",
        0.9,
    ),
    (
        re.compile(r"\b(classify|what (kind|type|category) of (doc|document|file))\b", re.IGNORECASE),
        "classification",
        0.85,
    ),
    (
        # Only consulted when screen context is attached (see detect()).
        re.compile(
            r"\b(on (my|the|this) screen|this (page|screen|window|tab)|what am i (doing|looking at))\b",
            re.IGNORECASE,
        ),
        "screen_help",
        0.85,
    ),
]

_MAX_DOC_SUMMARY = 100
_MAX_SCREEN_SUMMARY = 300


@dataclass(frozen=True)
class IntentResult:
    """Agent 5's verdict for one request."""

    intent: str
    confidence: float
    recommended_workflow: str
    method: str  # "heuristic" | "llm" | "fallback"


class IntentAgent:
    """Detects the user's intent from the request plus its context."""

    def __init__(self) -> None:
        # Temperature 0: classification must be deterministic.
        self._chain = INTENT_PROMPT | build_llm(temperature=0.0)

    def detect(
        self,
        question: str,
        registry: dict[str, dict[str, Any]],
        screen_context: dict[str, Any] | None = None,
    ) -> IntentResult:
        """Classify one request into an intent + recommended workflow.

        Args:
            question: The (standalone) user request.
            registry: ``doc_id -> metadata`` of the uploaded documents.
            screen_context: Optional Agent 4 screen analysis.

        Returns:
            IntentResult -- never raises; an unavailable LLM degrades to
            a question_answering fallback so the pipeline keeps moving.
        """
        cleaned = question.strip()

        for pattern, intent, confidence in _HEURISTICS:
            if intent == "screen_help" and screen_context is None:
                continue
            if pattern.search(cleaned):
                logger.info(
                    "AGENT 5 | intent=%s (%.2f) via heuristic", intent, confidence
                )
                return IntentResult(
                    intent, confidence, INTENT_WORKFLOWS[intent], "heuristic"
                )

        try:
            response = self._chain.invoke(
                {
                    "question": cleaned,
                    "documents": _documents_block(registry),
                    "screen": _screen_block(screen_context),
                }
            )
        except Exception as exc:
            logger.warning(
                "Agent 5 LLM call failed (%s); defaulting to question_answering",
                exc,
            )
            return IntentResult(
                "question_answering", 0.5, INTENT_WORKFLOWS["question_answering"],
                "fallback",
            )

        data = extract_json_object(str(response.content)) or {}
        intent = str(data.get("intent", "")).strip().lower()
        if intent not in INTENT_WORKFLOWS:
            intent = "question_answering"
        try:
            confidence = min(1.0, max(0.0, float(data.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5

        logger.info(
            "AGENT 5 | intent=%s (%.2f) via LLM [model: %s]",
            intent,
            confidence,
            model_used(response),
        )
        return IntentResult(intent, confidence, INTENT_WORKFLOWS[intent], "llm")


def _documents_block(registry: dict[str, dict[str, Any]]) -> str:
    """Prompt-ready one-line-per-document context."""
    if not registry:
        return "(no documents uploaded)"
    return "\n".join(
        f"- {meta['filename']} ({meta['category']}): "
        f"{meta['summary'][:_MAX_DOC_SUMMARY]}"
        for meta in registry.values()
    )


def _screen_block(screen_context: dict[str, Any] | None) -> str:
    """Prompt-ready screen context from Agent 4's analysis."""
    if not screen_context:
        return "(no screen context attached)"
    application = screen_context.get("application", "Unknown")
    summary = str(screen_context.get("summary", ""))[:_MAX_SCREEN_SUMMARY]
    return f"Application: {application}. {summary}"


@lru_cache(maxsize=1)
def get_intent_agent() -> IntentAgent:
    """Return the shared Agent 5 instance (created lazily)."""
    return IntentAgent()
