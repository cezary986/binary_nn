
from typing import List, Tuple
import numpy as np


class Batcher:

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X: np.ndarray = X
        self.y: np.ndarray = y

    def unison_shuffled_copies(self, a: np.ndarray, b: np.ndarray):
        assert a.shape[0] == b.shape[0]
        permutation = np.random.permutation(len(a))
        return a[permutation], b[permutation]

    def generate_batches(self, n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        shuffled_X, shuffled_y = self.unison_shuffled_copies(self.X, self.y)
        X_batches: List[np.ndarray] = np.array_split(shuffled_X, n)
        y_batches: List[np.ndarray] = np.array_split(shuffled_y, n)
        return [
            (X_batches[i], y_batches[i]) for i in range(0, n)
        ]
