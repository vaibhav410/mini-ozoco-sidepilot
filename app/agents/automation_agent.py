"""Agent 6 -- Automation Agent.

Executes the action behind an automation intent through a registry of
modular handlers -- no hardcoded if/else chains. Each handler owns one
action end-to-end (LLM content generation + integration call) and
returns the same :class:`AutomationOutcome` shape:

    Intent -> handler lookup -> execute -> AutomationOutcome

New actions are added by registering a handler; the agent, the workflow
and the API never change. Handlers ground their output in the workflow
context (retrieved chunks + screen analysis) collected by the earlier
pipeline stages.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol

from app.agents.llm import build_llm
from app.integrations import export, gmail
from app.rag.prompt import (
    ACTION_PLAN_PROMPT,
    EMAIL_DRAFT_PROMPT,
    EXPORT_SUMMARY_PROMPT,
)
from app.utils.json_utils import extract_json_object
from app.utils.logger import get_logger
from app.workflow.context import WorkflowContext

logger = get_logger(__name__)

_MAX_CONTEXT_CHARS = 6000


@dataclass
class AutomationOutcome:
    """Result of one executed automation action."""

    action: str
    status: str  # "completed" | "failed"
    detail: str  # human-readable outcome shown as the answer
    file: str | None = None  # generated file in the exports directory
    download_url: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class ActionHandler(Protocol):
    """Contract every automation handler must satisfy."""

    action: str

    def execute(self, context: WorkflowContext) -> AutomationOutcome:
        """Perform the action and describe what happened."""
        ...


class EmailDraftHandler:
    """intent 'email' -> LLM-drafted email delivered via the Gmail
    integration (.eml download + mailto link, or a real API draft)."""

    action = "email_draft"

    def __init__(self) -> None:
        self._chain = EMAIL_DRAFT_PROMPT | build_llm(temperature=0.3)

    def execute(self, context: WorkflowContext) -> AutomationOutcome:
        response = self._chain.invoke(
            {
                "question": context.standalone_question,
                "context": _grounding_text(context),
            }
        )
        data = extract_json_object(str(response.content)) or {}
        subject = str(data.get("subject", "")).strip() or "Draft email"
        body = str(data.get("body", "")).strip() or str(response.content).strip()
        to = str(data.get("to", "")).strip()

        result = gmail.create_draft(to=to, subject=subject, body=body)
        file = result.get("file")
        detail = (
            f"Email draft ready: **{subject}**\n\n{body}\n\n"
            + (
                "Saved as a downloadable .eml draft."
                if result["backend"] == "eml_file"
                else "Created as a Gmail draft in your account."
            )
        )
        return AutomationOutcome(
            action=self.action,
            status="completed",
            detail=detail,
            file=file,
            download_url=f"/exports/{file}" if file else None,
            extra={
                "to": to,
                "subject": subject,
                "body": body,
                "mailto": result["mailto"],
                "backend": result["backend"],
            },
        )


class ExportSummaryHandler:
    """intent 'export' -> LLM-written Markdown summary exported as a
    Markdown or PDF file (format inferred from the request)."""

    action = "export_summary"

    def __init__(self) -> None:
        self._chain = EXPORT_SUMMARY_PROMPT | build_llm(temperature=0.2)

    def execute(self, context: WorkflowContext) -> AutomationOutcome:
        response = self._chain.invoke(
            {
                "question": context.standalone_question,
                "context": _grounding_text(context),
            }
        )
        content = str(response.content).strip().removeprefix("```markdown").strip("` \n")
        title = _export_title(context)

        wants_pdf = "pdf" in context.standalone_question.lower()
        path = (
            export.export_pdf(title, content)
            if wants_pdf
            else export.export_markdown(title, content)
        )
        detail = (
            f"Exported **{title}** as `{path.name}` "
            f"({'PDF' if wants_pdf else 'Markdown'}). "
            "Use the download link to save it."
        )
        return AutomationOutcome(
            action=self.action,
            status="completed",
            detail=detail,
            file=path.name,
            download_url=f"/exports/{path.name}",
            extra={"format": "pdf" if wants_pdf else "markdown"},
        )


class ActionPlanHandler:
    """intent 'automation' -> step-by-step action plan, also saved as
    a Markdown file for later reference."""

    action = "action_plan"

    def __init__(self) -> None:
        self._chain = ACTION_PLAN_PROMPT | build_llm(temperature=0.3)

    def execute(self, context: WorkflowContext) -> AutomationOutcome:
        response = self._chain.invoke(
            {
                "question": context.standalone_question,
                "context": _grounding_text(context),
            }
        )
        plan = str(response.content).strip().removeprefix("```markdown").strip("` \n")
        path = export.export_markdown("Action plan", plan)
        return AutomationOutcome(
            action=self.action,
            status="completed",
            detail=plan,
            file=path.name,
            download_url=f"/exports/{path.name}",
            extra={},
        )


class AutomationAgent:
    """Dispatches automation intents to their registered handlers."""

    def __init__(self, handlers: dict[str, ActionHandler] | None = None) -> None:
        # Dependency injection: pass a custom registry in tests; the
        # default covers the three shipped automation intents.
        self._handlers: dict[str, ActionHandler] = handlers or {
            "email": EmailDraftHandler(),
            "export": ExportSummaryHandler(),
            "automation": ActionPlanHandler(),
        }

    def register(self, intent: str, handler: ActionHandler) -> None:
        """Add or replace the handler for an intent (extensibility)."""
        self._handlers[intent] = handler

    @property
    def supported_intents(self) -> list[str]:
        """Intents this agent can currently execute."""
        return sorted(self._handlers)

    def execute(self, context: WorkflowContext) -> AutomationOutcome:
        """Run the handler for the context's intent.

        Returns:
            AutomationOutcome -- never raises; handler failures come
            back as a ``failed`` outcome so the pipeline stays alive.
        """
        handler = self._handlers.get(context.intent)
        if handler is None:
            return AutomationOutcome(
                action="none",
                status="failed",
                detail=f"No automation handler registered for intent "
                f"'{context.intent}'.",
            )
        try:
            outcome = handler.execute(context)
        except Exception as exc:
            logger.error("AGENT 6 | handler '%s' failed: %s", handler.action, exc)
            return AutomationOutcome(
                action=handler.action,
                status="failed",
                detail="The automation action failed (AI provider or file "
                "error). Please try again.",
            )
        logger.info(
            "AGENT 6 | action=%s status=%s file=%s",
            outcome.action,
            outcome.status,
            outcome.file or "-",
        )
        return outcome


def _grounding_text(context: WorkflowContext) -> str:
    """Concatenate retrieved chunks (incl. screen chunk) for prompts."""
    if not context.chunks:
        return "(no additional context)"
    parts = [
        f"[{chunk.metadata.get('filename', 'unknown')}]\n{chunk.page_content}"
        for chunk in context.chunks
    ]
    return "\n\n---\n\n".join(parts)[:_MAX_CONTEXT_CHARS]


def _export_title(context: WorkflowContext) -> str:
    """Human title for an exported file, from routing when available."""
    from app.rag.vector_store import vector_store_manager

    if context.routed_doc_id:
        meta = vector_store_manager.registry.get(context.routed_doc_id)
        if meta:
            return f"Summary of {meta['filename']}"
    return "Document summary"


@lru_cache(maxsize=1)
def get_automation_agent() -> AutomationAgent:
    """Return the shared Agent 6 instance (created lazily)."""
    return AutomationAgent()
