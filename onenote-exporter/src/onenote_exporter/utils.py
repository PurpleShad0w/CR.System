"""Utility helpers for file naming and slugs."""
from __future__ import annotations

import pathlib
import re
from urllib.parse import urlparse


def slugify(text: str, max_len: int = 80) -> str:
    """Create a filesystem-friendly slug.

    Args:
        text: Input string
        max_len: Maximum length of slug
    """
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().lower()
    text = re.sub(r"[-\s]+", "-", text)
    return text[:max_len] if text else "untitled"


def filename_from_url(url: str, fallback: str) -> str:
    """Try to infer a filename from a URL path."""
    try:
        path = urlparse(url).path
        name = pathlib.Path(path).name
        if name:
            return name
    except ValueError:
        # Leave fallback on parsing error
        return fallback
    return fallback
