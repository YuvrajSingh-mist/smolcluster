"""DiLoCo (Douillard et al., 2023) with Streaming DiLoCo extensions."""

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from concurrent.futures import ThreadPoolExecutor, Future
from mlx.utils import tree_map
from .mlx_comm import _GradientPacker
from ._types import ReduceOp


def _deep_copy(params: dict) -> dict:
    return tree_map(lambda x: mx.array(x) if isinstance(x, mx.array) else x, params)


def _quantize_e3m0(buf: np.ndarray) -> tuple[np.ndarray, float]:
    absvals = np.abs(buf)
    nonzero = absvals[absvals > 0]
    scale = float(np.median(nonzero)) if len(nonzero) > 0 else 1.0
    if scale == 0:
        scale = 1.0

    scaled = buf / scale
    sign = np.sign(scaled)
    mag = np.abs(scaled)

    log_mag = np.log2(np.maximum(mag, 0.25))
    exp = np.clip(np.round(log_mag).astype(np.int8), -2, 3)
    quantized_mag = np.where(mag < 0.125, 0.0, 2.0 ** exp)

    sign_bit = (sign < 0).astype(np.uint8)
    exp_code = np.where(quantized_mag == 0, 0, exp + 3).astype(np.uint8)
    code = (sign_bit << 3) | exp_code

    return code, scale


def _dequantize_e3m0(code: np.ndarray, scale: float) -> np.ndarray:
    sign = np.where((code >> 3) & 1, -1.0, 1.0)
    exp_code = (code & 0x07).astype(np.float32)
    mag = np.where(exp_code == 0, 0.0, 2.0 ** (exp_code - 3.0))
    return (sign * mag * scale).astype(np.float32)


class DiLoCo:
    def __init__(
        self,
        model: nn.Module,
        outer_lr: float = 0.7,
        outer_momentum: float = 0.9,
        H: int = 500,
        overlap: bool = False,
        mix_alpha: float = 0.5,
        quantize: bool = False,
    ) -> None:
        self._H = H
        self._overlap = overlap
        self._mix_alpha = mix_alpha
        self._quantize = quantize
        self._inner_step = 0
        self._outer_step = 0
        self._initial_params = _deep_copy(dict(model.trainable_parameters()))
        self._outer_optim = optim.SGD(
            learning_rate=outer_lr,
            momentum=outer_momentum,
            nesterov=True,
        )
        self._outer_optim.init(tree_map(lambda x: x, self._initial_params))
        self._packer = _GradientPacker()

        if overlap:
            self._executor = ThreadPoolExecutor(max_workers=1)
            self._pending: Future | None = None
            self._pending_initial: dict | None = None

    def step(self, model: nn.Module) -> bool:
        self._inner_step += 1
        if self._inner_step < self._H:
            return False

        self._inner_step = 0
        self._outer_step += 1

        import grove

        if self._overlap and grove.world_size > 1:
            return self._step_async(model, grove)
        if grove.world_size > 1:
            return self._step_blocking(model, grove)
        return self._step_single(model)

    def _step_single(self, model) -> bool:
        current = dict(model.trainable_parameters())
        pseudo_grads = tree_map(lambda i, c: i - c, self._initial_params, current)
        mx.eval(tree_map(lambda x: x, pseudo_grads))
        model.update(self._initial_params)
        self._outer_optim.apply_gradients(pseudo_grads, model)
        mx.eval(model.parameters())
        self._initial_params = _deep_copy(dict(model.trainable_parameters()))
        return True

    def _step_blocking(self, model, grove) -> bool:
        current = dict(model.trainable_parameters())
        pseudo_grads = tree_map(lambda i, c: i - c, self._initial_params, current)
        mx.eval(tree_map(lambda x: x, pseudo_grads))

        buf = self._packer.flatten(pseudo_grads)
        averaged = self._allreduce_buf(buf, grove)
        pseudo_grads = self._packer.unflatten(averaged, 1.0 / grove.world_size, pseudo_grads)

        model.update(self._initial_params)
        self._outer_optim.apply_gradients(pseudo_grads, model)
        mx.eval(model.parameters())
        self._initial_params = _deep_copy(dict(model.trainable_parameters()))
        return True

    def _step_async(self, model, grove) -> bool:
        if self._pending is not None:
            averaged_buf = self._pending.result()
            self._apply_async_result(model, averaged_buf, grove.world_size)

        current = dict(model.trainable_parameters())
        pseudo_grads = tree_map(lambda i, c: i - c, self._initial_params, current)
        mx.eval(tree_map(lambda x: x, pseudo_grads))

        buf = self._packer.flatten(pseudo_grads)
        self._pending_initial = _deep_copy(self._initial_params)
        self._initial_params = _deep_copy(current)

        self._pending = self._executor.submit(self._allreduce_buf, buf, grove)
        return True

    def _apply_async_result(self, model, averaged_buf, world_size) -> None:
        dummy = tree_map(lambda x: x, self._pending_initial)
        pseudo_grads = self._packer.unflatten(averaged_buf, 1.0 / world_size, dummy)

        saved_params = _deep_copy(dict(model.trainable_parameters()))
        model.update(self._pending_initial)
        self._outer_optim.apply_gradients(pseudo_grads, model)
        mx.eval(model.parameters())

        synced = dict(model.trainable_parameters())
        mixed = tree_map(
            lambda local, s: self._mix_alpha * local + (1 - self._mix_alpha) * s,
            saved_params,
            synced,
        )
        model.update(mixed)
        mx.eval(model.parameters())
        self._initial_params = _deep_copy(dict(model.trainable_parameters()))
        self._pending_initial = None

    def _allreduce_buf(self, buf: np.ndarray, grove) -> np.ndarray:
        if self._quantize:
            code, scale = _quantize_e3m0(buf)
            gathered_codes = np.array(grove._comm.all_gather(code))
            scale_arr = np.array([scale], dtype=np.float32)
            gathered_scales = np.array(grove._comm.all_gather(scale_arr))
            ws = grove.world_size
            per_worker = len(code)
            accumulated = np.zeros_like(buf)
            for w in range(ws):
                w_code = gathered_codes[w * per_worker:(w + 1) * per_worker]
                w_scale = gathered_scales[w]
                accumulated += _dequantize_e3m0(w_code, w_scale)
            return accumulated
        else:
            grove._comm.all_reduce(buf, ReduceOp.SUM)
            return buf

    @property
    def inner_step(self) -> int:
        return self._inner_step

    @property
    def outer_step(self) -> int:
        return self._outer_step
