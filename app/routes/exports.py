"""GET /exports -- generated file listing and downloads (HTTP layer only)."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.integrations.filesystem import list_exports, safe_export_path
from app.models.schemas import ExportsResponse
from app.utils.auth import require_user
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Exports"], dependencies=[Depends(require_user)])


@router.get(
    "/exports",
    response_model=ExportsResponse,
    summary="List generated files",
    description="Every file produced by the automation agent (email "
    "drafts, exported summaries, action plans), newest first.",
)
def get_exports() -> ExportsResponse:
    """List the exports directory."""
    return ExportsResponse(files=list_exports())


@router.get(
    "/exports/{filename}",
    summary="Download a generated file",
    response_class=FileResponse,
)
def download_export(filename: str) -> FileResponse:
    """Serve one generated file as an attachment download."""
    try:
        path = safe_export_path(filename)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path, filename=path.name)
