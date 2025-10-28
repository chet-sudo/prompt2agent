"""Text utilities."""
from __future__ import annotations

import re


def slugify(value: str, *, default: str = "workflow") -> str:
    """Return a filesystem-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug or default
