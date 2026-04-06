from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import mlx.core as mx
import numpy as np
from PIL import Image

import sys

sys.path.append(Path(__file__).resolve().parents[1].as_posix())

from deployment.search_utils import apply_metadata_boost, is_searchable_query, load_metadata_texts
from mlx_pipeline import l2_normalize, load_mlx_siglip_model, open_rgb_image

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".heic", ".heif"}


def load_search_config(config_path: str) -> dict:
    return json.loads(Path(config_path).expanduser().resolve().read_text(encoding="utf-8"))


def load_indexed_paths(config: dict) -> list[str]:
    indexed_paths_file = config.get("indexed_paths_file")
    if indexed_paths_file:
        return Path(indexed_paths_file).expanduser().resolve().read_text(encoding="utf-8").splitlines()

    gallery_path = Path(config["gallery_path"]).expanduser().resolve()
    return [
        str(path)
        for path in sorted(
            candidate
            for candidate in gallery_path.rglob("*")
            if candidate.is_file()
            and not any(part.startswith(".") for part in candidate.relative_to(gallery_path).parts)
            and candidate.suffix.lower() in SUPPORTED_SUFFIXES
        )
    ]


class BaseSearchEngine:
    model_label: str
    config: dict
    gallery_path: Path
    image_paths: list[str]
    embeddings: np.ndarray
    metadata_texts: list[str] | None

    def search(self, query_text: str, k: int = 20) -> List[Tuple[str, Optional[str]]]:
        raise NotImplementedError

    def search_by_image(self, image: Image.Image, k: int = 20) -> List[Tuple[str, Optional[str]]]:
        raise NotImplementedError

    def search_similar(self, image_path: str | Path, k: int = 20) -> List[Tuple[str, Optional[str]]]:
        raise NotImplementedError

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            handle.write(content)
            temp_path = Path(handle.name)
        temp_path.replace(path)

    @staticmethod
    def _atomic_write_json(path: Path, payload) -> None:
        BaseSearchEngine._atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))

    @staticmethod
    def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
            np.save(handle, np.asarray(array, dtype="float32"))
            temp_path = Path(handle.name)
        temp_path.replace(path)

    @staticmethod
    def _resolve_manifest_image_path(manifest_path: Path, value: str) -> Path:
        image_path = Path(value).expanduser()
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        return image_path.resolve()

    def _remove_from_metadata_manifest(self, image_path: Path) -> None:
        self._remove_from_metadata_manifest_many({image_path})

    def _remove_from_metadata_manifest_many(self, image_paths: set[Path]) -> None:
        manifest_value = self.config.get("metadata_manifest")
        if not manifest_value:
            return

        manifest_path = Path(manifest_value).expanduser().resolve()
        if not manifest_path.exists():
            return

        kept_lines: list[str] = []
        changed = False
        with open(manifest_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    kept_lines.append(line)
                    continue
                row_path = row.get("image_path")
                if not row_path:
                    kept_lines.append(line)
                    continue
                resolved = self._resolve_manifest_image_path(manifest_path, row_path)
                if resolved in image_paths:
                    changed = True
                    continue
                kept_lines.append(line)

        if changed:
            content = "\n".join(kept_lines)
            if content:
                content += "\n"
            self._atomic_write_text(manifest_path, content)

    def _remove_from_skipped_images(self, image_path: Path) -> None:
        self._remove_from_skipped_images_many({image_path})

    def _remove_from_skipped_images_many(self, image_paths: set[Path]) -> None:
        skipped_value = self.config.get("skipped_images_file")
        if not skipped_value:
            return

        skipped_path = Path(skipped_value).expanduser().resolve()
        if not skipped_path.exists():
            return

        try:
            payload = json.loads(skipped_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        if not isinstance(payload, list):
            return

        kept_rows = []
        changed = False
        for row in payload:
            row_path = row.get("path") if isinstance(row, dict) else None
            if not row_path:
                kept_rows.append(row)
                continue
            resolved = self._resolve_manifest_image_path(skipped_path, row_path)
            if resolved in image_paths:
                changed = True
                continue
            kept_rows.append(row)

        if changed:
            self._atomic_write_json(skipped_path, kept_rows)

    def delete_image(self, image_path: str | Path) -> bool:
        removed = self.delete_images([image_path])
        return Path(image_path).expanduser().resolve().as_posix() in removed

    def delete_images(self, image_paths: list[str | Path]) -> set[str]:
        target_paths = []
        seen_paths = set()
        for image_path in image_paths:
            target = Path(image_path).expanduser().resolve()
            if target in seen_paths:
                continue
            seen_paths.add(target)
            target_paths.append(target)

        if not target_paths:
            return set()

        target_set = set(target_paths)
        keep_indices = []
        removed_index_paths = set()
        for idx, candidate in enumerate(self.image_paths):
            candidate_path = Path(candidate).expanduser().resolve()
            if candidate_path in target_set:
                removed_index_paths.add(candidate_path.as_posix())
                continue
            keep_indices.append(idx)

        if len(keep_indices) != len(self.image_paths):
            new_image_paths = [self.image_paths[idx] for idx in keep_indices]
            new_embeddings = np.asarray(self.embeddings)[keep_indices].astype("float32", copy=False)
            new_metadata_texts = None
            if self.metadata_texts is not None:
                new_metadata_texts = [self.metadata_texts[idx] for idx in keep_indices]

            indexed_paths_value = self.config.get("indexed_paths_file")
            embeddings_value = self.config.get("embeddings_file")
            if indexed_paths_value:
                indexed_paths_path = Path(indexed_paths_value).expanduser().resolve()
                indexed_payload = "\n".join(new_image_paths)
                if indexed_payload:
                    indexed_payload += "\n"
                self._atomic_write_text(indexed_paths_path, indexed_payload)
            if embeddings_value:
                embeddings_path = Path(embeddings_value).expanduser().resolve()
                self._atomic_save_npy(embeddings_path, new_embeddings)

            self.image_paths = new_image_paths
            self.embeddings = new_embeddings
            self.metadata_texts = new_metadata_texts

        self._remove_from_metadata_manifest_many(target_set)
        self._remove_from_skipped_images_many(target_set)
        return removed_index_paths


class MLXSigLIPSearchEngine(BaseSearchEngine):
    def __init__(self, config: dict, device: str = "auto"):
        del device

        self.config = config
        self.gallery_path = Path(config["gallery_path"]).expanduser().resolve()
        self.image_paths = load_indexed_paths(config)
        self.embeddings = np.load(Path(config["embeddings_file"]).expanduser().resolve(), mmap_mode="r")
        self.metadata_texts = load_metadata_texts(config, self.image_paths)
        self.max_length = int(config.get("text_max_length", 64))
        self.model_label = Path(config["model_name"]).name
        self.model, self.processor = load_mlx_siglip_model(
            config["model_name"],
            weights_file=config.get("weights_file"),
            precision=config.get("model_precision", "bfloat16"),
            lazy=False,
        )

    def _rank_scores(
        self,
        scores: np.ndarray,
        *,
        k: int,
        exclude_index: int | None = None,
    ) -> List[Tuple[str, Optional[str]]]:
        if not self.image_paths or k <= 0:
            return []

        ranked_scores = np.asarray(scores, dtype=np.float32).copy()
        if exclude_index is not None and 0 <= exclude_index < ranked_scores.shape[0]:
            ranked_scores[exclude_index] = -np.inf

        finite_count = int(np.isfinite(ranked_scores).sum())
        if finite_count <= 0:
            return []

        k = min(k, finite_count)
        top_indices = np.argpartition(-ranked_scores, k - 1)[:k]
        top_indices = top_indices[np.argsort(-ranked_scores[top_indices])]
        return [(self.image_paths[int(index)], None) for index in top_indices]

    def _find_image_index(self, image_path: str | Path) -> int:
        target_path = Path(image_path).expanduser().resolve()
        return next(
            (
                idx
                for idx, candidate in enumerate(self.image_paths)
                if Path(candidate).expanduser().resolve() == target_path
            ),
            -1,
        )

    def encode_query(self, query_text: str) -> np.ndarray:
        inputs = self.processor(
            text=[query_text],
            return_tensors="mlx",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        embedding = self.model.get_text_features(**inputs)
        embedding = l2_normalize(embedding)
        mx.eval(embedding)
        return np.asarray(embedding, dtype=np.float32)[0]

    def encode_query_image(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=[image], return_tensors="mlx")
        image_inputs = {"pixel_values": inputs["pixel_values"]}
        if "pixel_attention_mask" in inputs:
            image_inputs["pixel_attention_mask"] = inputs["pixel_attention_mask"]
        embedding = self.model.get_image_features(**image_inputs)
        embedding = l2_normalize(embedding)
        mx.eval(embedding)
        return np.asarray(embedding, dtype=np.float32)[0]

    def search(self, query_text: str, k: int = 20) -> List[Tuple[str, Optional[str]]]:
        if not is_searchable_query(query_text) or not self.image_paths or k <= 0:
            return []

        query_vector = self.encode_query(query_text)
        scores = self.embeddings @ query_vector
        scores = apply_metadata_boost(scores=scores, metadata_texts=self.metadata_texts, query_text=query_text)
        return self._rank_scores(scores, k=k)

    def search_by_image(self, image: Image.Image, k: int = 20) -> List[Tuple[str, Optional[str]]]:
        if not self.image_paths or k <= 0:
            return []
        query_vector = self.encode_query_image(image)
        scores = self.embeddings @ query_vector
        return self._rank_scores(scores, k=k)

    def search_similar(self, image_path: str | Path, k: int = 20) -> List[Tuple[str, Optional[str]]]:
        if not self.image_paths or k <= 0:
            return []

        index = self._find_image_index(image_path)
        if index >= 0:
            query_vector = np.asarray(self.embeddings[index], dtype=np.float32)
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm
            scores = self.embeddings @ query_vector
            return self._rank_scores(scores, k=k, exclude_index=index)

        image = open_rgb_image(image_path)
        query_vector = self.encode_query_image(image)
        scores = self.embeddings @ query_vector
        return self._rank_scores(scores, k=k)


def build_search_engine(config_path: str, device: str):
    config = load_search_config(config_path)
    backend = config.get("backend", "mlx_siglip2")
    if backend != "mlx_siglip2":
        raise ValueError(f"Unsupported backend: {backend}")
    return MLXSigLIPSearchEngine(config=config, device=device)
