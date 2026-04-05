from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


def is_searchable_query(query_text: str) -> bool:
    query = query_text.strip()
    if not query:
        return False
    if len(query) >= 2:
        return True
    return bool(re.search(r"[\u4e00-\u9fff]", query))


def load_metadata_texts(config: dict, image_paths) -> list[str] | None:
    manifest_path = config.get("metadata_manifest")
    if not manifest_path:
        return None

    manifest_path = Path(manifest_path).expanduser().resolve()
    if not manifest_path.exists():
        return None

    path_to_text = {}
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            image_path = Path(row["image_path"]).expanduser()
            if not image_path.is_absolute():
                image_path = (manifest_path.parent / image_path).resolve()
            captions = row.get("captions") or ([row["caption"]] if row.get("caption") else [])
            text = " ".join(caption for caption in captions if isinstance(caption, str)).lower()
            path_to_text[image_path.as_posix()] = text

    return [path_to_text.get(str(path), "") for path in image_paths]


def query_terms(query_text: str) -> list[str]:
    query = query_text.strip().lower()
    if not query:
        return []

    terms = set()
    for chunk in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", query):
        if len(chunk) <= 1 and not re.fullmatch(r"[\u4e00-\u9fff]", chunk):
            continue
        terms.add(chunk)
        if re.fullmatch(r"[\u4e00-\u9fff]+", chunk) and len(chunk) > 2:
            for n in range(2, min(len(chunk), 4) + 1):
                for start in range(0, len(chunk) - n + 1):
                    terms.add(chunk[start : start + n])

    return sorted(terms, key=len, reverse=True)


def apply_metadata_boost(scores: np.ndarray, metadata_texts, query_text: str) -> np.ndarray:
    if not metadata_texts:
        return scores

    query = query_text.strip().lower()
    if not query:
        return scores

    boosted = scores.copy()
    tokens = query_terms(query)
    for index, metadata_text in enumerate(metadata_texts):
        if not metadata_text:
            continue
        if query in metadata_text:
            boosted[index] += 0.40
            continue

        token_hits = [token for token in tokens if token in metadata_text]
        if tokens and len(token_hits) >= min(3, len(tokens)):
            boosted[index] += 0.24
        elif tokens and len(token_hits) >= 2:
            boosted[index] += 0.14
        elif token_hits:
            boosted[index] += 0.08 * len(token_hits)

        if any(len(token) >= 4 for token in token_hits):
            boosted[index] += 0.06
    return boosted
