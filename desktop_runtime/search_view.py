from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ActiveSearchView:
    paths: list[str]
    matrix: np.ndarray
    _index_by_path: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._index_by_path = {path: index for index, path in enumerate(self.paths)}

    @classmethod
    def from_store(cls, store, folder_path: str, encoder_signature: str) -> "ActiveSearchView":
        rows = store.get_folder_rows(folder_path, encoder_signature)
        if not rows:
            return cls(paths=[], matrix=np.zeros((0, 0), dtype=np.float32))

        paths = [row["absolute_path"] for row in rows]
        matrix = np.stack(
            [
                np.frombuffer(row["embedding_blob"], dtype=np.float32, count=row["embedding_dim"]).copy()
                for row in rows
            ],
            axis=0,
        )
        return cls(paths=paths, matrix=matrix)

    def search(self, query_vector: np.ndarray, limit: int, *, exclude_path: str | None = None) -> list[str]:
        if not self.paths:
            return []

        query = np.asarray(query_vector, dtype=np.float32)
        matrix = np.asarray(self.matrix, dtype=np.float32)
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0.0:
            return []

        row_norms = np.linalg.norm(matrix, axis=1)
        safe_norms = np.where(row_norms == 0.0, 1.0, row_norms)
        scores = (matrix @ query) / (safe_norms * query_norm)
        order = np.argsort(scores)[::-1]
        results: list[str] = []
        for index in order:
            path = self.paths[index]
            if exclude_path is not None and path == exclude_path:
                continue
            results.append(path)
            if len(results) >= limit:
                break
        return results

    def search_similar(self, image_path: str, limit: int) -> list[str]:
        index = self._index_by_path.get(image_path)
        if index is None:
            return []

        query_vector = np.asarray(self.matrix[index], dtype=np.float32)
        return self.search(query_vector, limit, exclude_path=image_path)

    def without_paths(self, removed_paths: list[str]) -> "ActiveSearchView":
        if not self.paths or not removed_paths:
            return self

        removed = set(removed_paths)
        keep_indices = [index for index, path in enumerate(self.paths) if path not in removed]
        if len(keep_indices) == len(self.paths):
            return self
        if not keep_indices:
            width = self.matrix.shape[1] if self.matrix.ndim == 2 else 0
            return ActiveSearchView(paths=[], matrix=np.zeros((0, width), dtype=np.float32))

        next_paths = [self.paths[index] for index in keep_indices]
        next_matrix = np.asarray(self.matrix, dtype=np.float32)[keep_indices].copy()
        return ActiveSearchView(paths=next_paths, matrix=next_matrix)
