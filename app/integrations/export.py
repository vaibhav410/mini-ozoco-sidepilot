"""Document export integration: Markdown and PDF file generation.

PDF generation uses fpdf2 (pure Python, no system dependencies) so it
works identically on Windows, Docker and the Render free tier. The
built-in Helvetica font is Latin-1 only, so text is sanitised with
replacement characters rather than crashing on exotic glyphs.
"""

from pathlib import Path

from fpdf import FPDF

from app.integrations.filesystem import save_text, safe_export_path, timestamped_name
from app.utils.logger import get_logger

logger = get_logger(__name__)

_PAGE_WIDTH_MM = 190  # A4 width minus margins


def export_markdown(title: str, content: str) -> Path:
    """Write content as a Markdown file in the exports directory.

    Args:
        title: Document title (used for the heading and the filename).
        content: Markdown body.

    Returns:
        Path of the generated ``.md`` file.
    """
    body = content if content.lstrip().startswith("#") else f"# {title}\n\n{content}"
    return save_text(timestamped_name(title, "md"), body)


def export_pdf(title: str, content: str) -> Path:
    """Render content as a simple, clean PDF in the exports directory.

    Args:
        title: Document title (heading + filename).
        content: Plain or lightly-markdown text; rendered line by line.

    Returns:
        Path of the generated ``.pdf`` file.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(_PAGE_WIDTH_MM, 9, _latin1(title))
    pdf.ln(3)

    pdf.set_font("Helvetica", size=11)
    for raw_line in content.splitlines():
        line = raw_line.rstrip()
        if not line:
            pdf.ln(4)
            continue
        if line.startswith("#"):
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(_PAGE_WIDTH_MM, 7, _latin1(line.lstrip("# ")))
            pdf.set_font("Helvetica", size=11)
        elif line.lstrip().startswith(("- ", "* ")):
            pdf.multi_cell(_PAGE_WIDTH_MM, 6, _latin1("  - " + line.lstrip("-* ")))
        else:
            pdf.multi_cell(_PAGE_WIDTH_MM, 6, _latin1(line))

    path = safe_export_path(timestamped_name(title, "pdf"))
    pdf.output(str(path))
    logger.info("Saved export %s (%d bytes)", path.name, path.stat().st_size)
    return path


def _latin1(text: str) -> str:
    """Sanitise text for fpdf2's Latin-1 core fonts (bold ** stripped)."""
    return text.replace("**", "").encode("latin-1", "replace").decode("latin-1")
