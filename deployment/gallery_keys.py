from __future__ import annotations

import hashlib
import re
from pathlib import Path


def gallery_artifact_key(gallery_path: str | Path) -> str:
    resolved = Path(gallery_path).expanduser().resolve()
    slug = re.sub(r"[^a-z0-9]+", "-", resolved.name.lower()).strip("-")
    digest = hashlib.sha256(resolved.as_posix().encode("utf-8")).hexdigest()[:12]
    if slug:
        return f"{slug}-{digest}"
    return digest
