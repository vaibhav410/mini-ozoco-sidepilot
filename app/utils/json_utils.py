"""Robust JSON extraction from LLM text output.

Models asked for "JSON only" still sometimes wrap the object in markdown
fences or surrounding prose. This helper strips that noise, parses the
outermost ``{...}`` block, and returns ``None`` instead of raising so
each caller can apply its own graceful fallback.
"""

import json
import re


def extract_json_object(raw: str) -> dict | None:
    """Parse the first JSON object found in an LLM response.

    Args:
        raw: The model's raw text output.

    Returns:
        The parsed dict, or ``None`` when no valid JSON object is present.
    """
    cleaned = re.sub(r"```(?:json)?", "", raw).strip("` \n")
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
