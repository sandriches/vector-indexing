from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Index(Protocol):
    """Every index in this repo implements this so the harness can treat them alike.

    Vectors are float32, L2-normalised rows. Distances are cosine distance
    (1 - dot), so smaller is closer.
    """

    name: str

    def build(self, vectors: np.ndarray) -> None: ...

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (ids, distances), each shape (k,), sorted closest first."""
        ...

    def memory_bytes(self) -> int:
        """Estimated bytes held by the index structure itself."""
        ...
