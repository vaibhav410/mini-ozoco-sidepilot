"""SidePilot workflow package.

Every request travels the pipeline:

    Observe -> Understand -> Analyze -> Guide -> Automate

Public surface: the shared :class:`WorkflowContext`, the engine, and the
stage protocol for anyone who wants to plug a custom stage in.
"""

from app.workflow.context import StageTrace, WorkflowContext
from app.workflow.workflow_engine import (
    WorkflowEngine,
    WorkflowStage,
    get_workflow_engine,
)

__all__ = [
    "StageTrace",
    "WorkflowContext",
    "WorkflowEngine",
    "WorkflowStage",
    "get_workflow_engine",
]
