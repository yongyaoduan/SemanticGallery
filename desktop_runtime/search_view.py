from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ActiveSearchView:
    paths: list[str]
    matrix: np.ndarray

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

    def search(self, query_vector: np.ndarray, limit: int) -> list[str]:
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
        order = np.argsort(scores)[::-1][:limit]
        return [self.paths[index] for index in order]
