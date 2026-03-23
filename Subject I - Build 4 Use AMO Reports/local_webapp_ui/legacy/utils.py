from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def safe_read_text(path: Path, limit: int = 80_000) -> str:
    if not path.exists():
        return ''
    txt = path.read_text(encoding='utf-8', errors='replace')
    if len(txt) > limit:
        return txt[:limit] + "\n\n... (tronqué)"
    return txt
