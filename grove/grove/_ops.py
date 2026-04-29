"""Reduce operations and buffer pool."""

import numpy as np
from ._types import ReduceOp

REDUCE_FN = {
    ReduceOp.SUM: np.add,
    ReduceOp.PROD: np.multiply,
    ReduceOp.MIN: np.minimum,
    ReduceOp.MAX: np.maximum,
}


class BufferPool:
    def __init__(self) -> None:
        self._bufs: dict[str, np.ndarray] = {}

    def get(self, name: str, size: int, dtype: np.dtype) -> np.ndarray:
        buf = self._bufs.get(name)
        if buf is None or buf.size < size or buf.dtype != dtype:
            buf = np.empty(size, dtype=dtype)
            self._bufs[name] = buf
        return buf[:size]
