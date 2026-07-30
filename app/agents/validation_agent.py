"""Agent 3 -- Answer Validation Agent.

Runs after Agent 2 drafts an answer. Cross-checks every claim in the
draft against the retrieved context and returns a structured verdict.
Unsupported answers never reach the user -- the service layer replaces
them with an explicit "not found" response.

Design principle: the validator is a safety net, not a point of failure.
If the validation call itself fails or returns malformed output, the
draft passes through (with confidence "unknown") rather than breaking
the request.
"""

import json
import re
from functools import lru_cache

from langchain_core.documents import Document

from app.agents.llm import build_llm, model_used
from app.rag.prompt import VALIDATION_PROMPT
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ValidationAgent:
    """Verifies that Agent 2's draft answers are grounded in context."""

    def __init__(self) -> None:
        # Temperature 0: fact-checking must be deterministic.
        self._chain = VALIDATION_PROMPT | build_llm(temperature=0.0)

    def validate(
        self, question: str, answer: str, chunks: list[Document]
    ) -> dict:
        """Judge whether the draft answer is supported by the chunks.

        Args:
            question: The (standalone) user question.
            answer: Agent 2's draft answer.
            chunks: The retrieved context the answer must be grounded in.

        Returns:
            Dict with keys ``supported`` (bool), ``confidence`` (str),
            ``reason`` (str). Never raises.
        """
        context = "\n\n---\n\n".join(chunk.page_content for chunk in chunks)
        try:
            response = self._chain.invoke(
                {"context": context, "question": question, "answer": answer}
            )
        except Exception as exc:
            logger.warning("Agent 3 validation call failed (%s); passing draft", exc)
            return {"supported": True, "confidence": "unknown",
                    "reason": "validator unavailable"}

        verdict = self._parse(str(response.content))
        logger.info(
            "AGENT 3 | validation: supported=%s confidence=%s (%s) [model: %s]",
            verdict["supported"],
            verdict["confidence"],
            verdict["reason"],
            model_used(response),
        )
        return verdict

    @staticmethod
    def _parse(raw: str) -> dict:
        """Parse the JSON verdict; malformed output passes the draft."""
        fallback = {"supported": True, "confidence": "unknown",
                    "reason": "verdict unparseable"}
        cleaned = re.sub(r"```(?:json)?", "", raw).strip("` \n")
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return fallback
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return fallback
        confidence = str(data.get("confidence", "unknown")).lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "unknown"
        return {
            "supported": bool(data.get("supported", True)),
            "confidence": confidence,
            "reason": str(data.get("reason", "")).strip()[:200],
        }


@lru_cache(maxsize=1)
def get_validation_agent() -> ValidationAgent:
    """Return the shared Agent 3 instance (created lazily)."""
    return ValidationAgent()
