"""Gradient packing: flatten to float32, reduce, cast back."""

from typing import Any
import numpy as np
import mlx.core as mx


class _GradientPacker:
    def __init__(self):
        self._metadata: list | None = None
        self._buf: np.ndarray | None = None
        self._total: int = 0

    def flatten(self, grads: dict) -> np.ndarray:
        if self._metadata is None:
            self._metadata = []
            self._total = 0
            self._scan(grads, "")
            self._buf = np.empty(self._total, dtype=np.float32)

        buf = self._buf
        offset = 0
        for path, shape, mlx_dtype in self._metadata:
            size = int(np.prod(shape))
            buf[offset : offset + size] = np.array(
                self._resolve(grads, path).astype(mx.float32), copy=False
            ).ravel()
            offset += size
        return buf

    def unflatten(self, buf: np.ndarray, scale: float, grads: dict) -> dict:
        result = {}
        offset = 0
        for path, shape, mlx_dtype in self._metadata:
            size = int(np.prod(shape))
            chunk = buf[offset : offset + size].reshape(shape)
            mx_arr = mx.array(chunk)
            if scale != 1.0:
                mx_arr = mx_arr * scale
            self._set_nested(result, path, mx_arr.astype(mlx_dtype))
            offset += size
        self._merge_non_tensors(grads, result)
        return result

    def _scan(self, d: dict, prefix: str) -> None:
        for key, val in d.items():
            path = f"{prefix}/{key}" if prefix else key
            if isinstance(val, dict):
                self._scan(val, path)
            elif hasattr(val, "shape"):
                self._metadata.append((path, val.shape, val.dtype))
                self._total += int(np.prod(val.shape))

    @staticmethod
    def _resolve(d: dict, path: str) -> Any:
        for part in path.split("/")[:-1]:
            d = d[part]
        return d[path.split("/")[-1]]

    @staticmethod
    def _set_nested(d: dict, path: str, value) -> None:
        parts = path.split("/")
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value

    @staticmethod
    def _merge_non_tensors(src: dict, dst: dict) -> None:
        for key, val in src.items():
            if isinstance(val, dict):
                if key not in dst:
                    dst[key] = {}
                _GradientPacker._merge_non_tensors(val, dst[key])
            elif key not in dst:
                dst[key] = val
