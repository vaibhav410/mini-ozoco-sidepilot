"""The workflow engine -- every request runs through the SidePilot
pipeline:

    Observe -> Understand -> Analyze -> Guide -> Automate

The engine is deliberately thin: it owns ordering, timing, tracing and
logging, while every decision lives inside the stages. Stages are
injected through the constructor, so any of them can be replaced (or new
ones inserted) without touching this file -- that is the extensibility
contract.

Status convention: a stage returns a short note for the trace; a note
starting with ``"skipped"`` records the stage as skipped, anything else
as completed. A raised exception records it as failed and propagates
(domain errors still become clean HTTP responses in the routes).
"""

from functools import lru_cache
from time import perf_counter
from typing import Protocol, Sequence, runtime_checkable

from app.utils.logger import get_logger
from app.workflow.context import StageTrace, WorkflowContext

logger = get_logger(__name__)

SKIPPED_PREFIX = "skipped"


@runtime_checkable
class WorkflowStage(Protocol):
    """Contract every pipeline stage must satisfy."""

    name: str

    def run(self, context: WorkflowContext) -> str | None:
        """Advance the workflow; return a short note for the trace."""
        ...


class WorkflowEngine:
    """Runs a WorkflowContext through an ordered list of stages."""

    def __init__(self, stages: Sequence[WorkflowStage] | None = None) -> None:
        """Create an engine over the given stages.

        Args:
            stages: Custom stage list; ``None`` uses the standard
                Observe -> Understand -> Analyze -> Guide -> Automate
                pipeline.
        """
        if stages is None:
            # Imported here so custom-stage users never pay for the
            # default stages' (heavier) dependency imports.
            from app.workflow.stages import build_default_stages

            stages = build_default_stages()
        self._stages: list[WorkflowStage] = list(stages)

    @property
    def stage_names(self) -> list[str]:
        """Ordered names of the configured stages (for docs/monitoring)."""
        return [stage.name for stage in self._stages]

    def run(self, context: WorkflowContext) -> WorkflowContext:
        """Execute all stages in order, recording a trace entry for each.

        Args:
            context: The request's workflow context (mutated in place).

        Returns:
            The same context, with ``trace`` filled and stage outputs set.

        Raises:
            AppError: Domain failures raised by stages (propagated).
            Exception: Unexpected stage crashes (propagated after being
                recorded in the trace).
        """
        pipeline_start = perf_counter()
        logger.info(
            "WORKFLOW | start (session=%s, question=%.60s)",
            context.session_id,
            context.question,
        )
        for stage in self._stages:
            stage_start = perf_counter()
            try:
                note = stage.run(context) or ""
            except Exception as exc:
                duration_ms = (perf_counter() - stage_start) * 1000
                context.trace.append(
                    StageTrace(
                        name=stage.name,
                        status="failed",
                        duration_ms=duration_ms,
                        note=str(exc)[:120],
                    )
                )
                logger.error(
                    "WORKFLOW | stage=%-10s FAILED after %4.0f ms | %s",
                    stage.name,
                    duration_ms,
                    exc,
                )
                raise

            duration_ms = (perf_counter() - stage_start) * 1000
            status = "skipped" if note.startswith(SKIPPED_PREFIX) else "completed"
            context.trace.append(
                StageTrace(
                    name=stage.name,
                    status=status,
                    duration_ms=duration_ms,
                    note=note,
                )
            )
            logger.info(
                "WORKFLOW | stage=%-10s %-9s %5.0f ms | %s",
                stage.name,
                status,
                duration_ms,
                note,
            )

        total_ms = (perf_counter() - pipeline_start) * 1000
        logger.info("WORKFLOW | done in %.0f ms", total_ms)
        return context


@lru_cache(maxsize=1)
def get_workflow_engine() -> WorkflowEngine:
    """Return the shared engine with the default pipeline (lazy)."""
    return WorkflowEngine()
