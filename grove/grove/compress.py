"""Chunk-wise top-k compression with optional DCT transform."""

import math
import numpy as np
import mlx.core as mx


def _make_dct_basis(n: int) -> np.ndarray:
    k = np.arange(n, dtype=np.float32)
    basis = np.cos(np.pi * k[:, None] * (2 * k[None, :] + 1) / (2 * n))
    basis[0] *= 1.0 / math.sqrt(n)
    basis[1:] *= math.sqrt(2.0 / n)
    return basis


def _get_divisors(n: int) -> list[int]:
    divs = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return sorted(divs)


def _best_chunk(dim: int, target: int) -> int:
    for d in reversed(_get_divisors(dim)):
        if d <= target:
            return d
    return dim


class TopKCompressor:
    def __init__(
        self,
        total_elems: int,
        chunk_target: int,
        topk: int,
        use_dct: bool = False,
    ):
        self.chunk_size = _best_chunk(total_elems, chunk_target)
        self.n_chunks = total_elems // self.chunk_size
        self.topk = min(topk, self.chunk_size)
        self.use_dct = use_dct

        if use_dct:
            basis = _make_dct_basis(self.chunk_size)
            self.fwd_mx = mx.array(basis)
            self.inv_mx = mx.array(basis.T)
            self.fwd = basis
            self.inv = basis.T.copy()

    def compress(self, flat: "mx.array | np.ndarray") -> tuple[np.ndarray, np.ndarray, mx.array]:
        if isinstance(flat, np.ndarray):
            flat = mx.array(flat)
        chunks = flat.reshape(self.n_chunks, self.chunk_size)

        if self.use_dct:
            encoded = chunks @ self.fwd_mx.T
        else:
            encoded = chunks

        if self.topk >= self.chunk_size:
            idx = mx.broadcast_to(mx.arange(self.chunk_size), encoded.shape)
            val = encoded
        else:
            idx = mx.argpartition(-mx.abs(encoded), kth=self.topk, axis=-1)[..., :self.topk]
            val = mx.take_along_axis(encoded, idx, axis=-1)

        sparse = mx.put_along_axis(mx.zeros_like(encoded), idx, val, axis=-1)

        if self.use_dct:
            transmitted = (sparse @ self.inv_mx.T).reshape(-1)
        else:
            transmitted = sparse.reshape(-1)

        mx.eval(idx, val, transmitted)

        idx_np = np.array(idx).ravel().astype(np.int32)
        val_np = np.array(val).ravel().astype(np.float32)

        return idx_np, val_np, transmitted

    def decompress(self, idx_flat: np.ndarray, val_flat: np.ndarray) -> np.ndarray:
        idx = idx_flat.astype(np.intp).reshape(self.n_chunks, self.topk)
        val = val_flat.astype(np.float32).reshape(self.n_chunks, self.topk)
        sparse = np.zeros((self.n_chunks, self.chunk_size), dtype=np.float32)
        np.put_along_axis(sparse, idx, val, axis=-1)

        if self.use_dct:
            return (sparse @ self.inv).ravel()
        return sparse.ravel()
