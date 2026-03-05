# parsing_utils.py
from __future__ import annotations

import json
import re
from typing import Any, Dict


def parse_json_strict(text: str) -> Dict[str, Any]:
    """
    Attempts strict JSON parse; if it fails, tries to extract the first {...} block.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract first JSON object block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("Could not find JSON object in critic output.")
    return json.loads(m.group(0))