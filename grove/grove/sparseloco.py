"""SparseLoCo (Sarfi et al., 2025): DiLoCo with top-k error feedback compression."""

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from concurrent.futures import ThreadPoolExecutor, Future
from mlx.utils import tree_map, tree_unflatten
from mlx.nn.utils import tree_flatten
from .compress import TopKCompressor


def _deep_copy(params: dict) -> dict:
    return tree_map(lambda x: mx.array(x) if isinstance(x, mx.array) else x, params)


class SparseLoCo:
    def __init__(
        self,
        model: nn.Module,
        outer_lr: float = 1.0,
        H: int = 30,
        error_decay: float = 0.95,
        topk: int = 64,
        chunk: int = 4096,
        overlap: bool = False,
        mix_alpha: float = 0.5,
    ) -> None:
        self._outer_lr = outer_lr
        self._H = H
        self._error_decay = error_decay
        self._overlap = overlap
        self._mix_alpha = mix_alpha
        self._inner_step = 0
        self._outer_step = 0

        self._initial_params = _deep_copy(dict(model.trainable_parameters()))

        self._param_info: list[tuple[str, tuple]] = []
        self._error_buffers: list[mx.array] = []
        self._compressors: list[TopKCompressor] = []

        for name, param in tree_flatten(dict(model.trainable_parameters())):
            total = int(np.prod(param.shape))
            self._param_info.append((name, param.shape))
            self._error_buffers.append(mx.zeros((total,)))
            self._compressors.append(TopKCompressor(total, chunk, topk, use_dct=False))

        if overlap:
            self._executor = ThreadPoolExecutor(max_workers=1)
            self._pending: Future | None = None

    def step(self, model: nn.Module) -> bool:
        self._inner_step += 1
        if self._inner_step < self._H:
            return False

        self._inner_step = 0
        self._outer_step += 1

        import grove

        if self._overlap:
            return self._step_async(model, grove)
        return self._step_blocking(model, grove)

    def _step_blocking(self, model, grove) -> bool:
        current = dict(model.trainable_parameters())
        pseudo_grads = tree_map(lambda i, c: i - c, self._initial_params, current)
        mx.eval(tree_map(lambda x: x, pseudo_grads))

        flat_grads = [g.reshape(-1) for _, g in tree_flatten(pseudo_grads)]
        all_idx, all_val = self._compress(flat_grads)

        if grove.world_size > 1:
            updates = self._gather_decompress(all_idx, all_val, grove)
        else:
            updates = [
                comp.decompress(idx, val)
                for comp, idx, val in zip(self._compressors, all_idx, all_val)
            ]

        self._apply_updates(model, updates)
        self._initial_params = _deep_copy(dict(model.trainable_parameters()))
        return True

    def _step_async(self, model, grove) -> bool:
        if self._pending is not None:
            synced_flat = self._pending.result()
            current_flat = tree_flatten(dict(model.trainable_parameters()))
            mixed_flat = {}
            for (name, local), (_, synced) in zip(current_flat, synced_flat):
                mixed_flat[name] = self._mix_alpha * local + (1 - self._mix_alpha) * synced
            model.update(tree_unflatten(list(mixed_flat.items())))
            mx.eval(model.parameters())

        current = dict(model.trainable_parameters())
        pseudo_grads = tree_map(lambda i, c: i - c, self._initial_params, current)
        mx.eval(tree_map(lambda x: x, pseudo_grads))

        flat_grads = [g.reshape(-1) for _, g in tree_flatten(pseudo_grads)]
        all_idx, all_val = self._compress(flat_grads)

        self._initial_params = _deep_copy(current)
        initial_snapshot = [
            (name, np.array(param.astype(mx.float32), copy=False))
            for name, param in tree_flatten(self._initial_params)
        ]

        self._pending = self._executor.submit(
            self._sync_worker, all_idx, all_val, initial_snapshot, grove,
        )
        return True

    def _sync_worker(self, all_idx, all_val, initial_snapshot, grove) -> list[tuple[str, mx.array]]:
        if grove.world_size > 1:
            updates = self._gather_decompress(all_idx, all_val, grove)
        else:
            updates = [
                comp.decompress(idx, val)
                for comp, idx, val in zip(self._compressors, all_idx, all_val)
            ]

        result = []
        for (name, init_np), update, (_, shape) in zip(initial_snapshot, updates, self._param_info):
            updated = init_np.reshape(shape) - self._outer_lr * update.reshape(shape)
            result.append((name, mx.array(updated.astype(np.float32))))
        return result

    def _compress(self, flat_grads: list[mx.array]) -> tuple[list, list]:
        all_idx = []
        all_val = []
        for i, (grad, comp) in enumerate(zip(flat_grads, self._compressors)):
            ef = self._error_buffers[i] * self._error_decay + grad
            idx, val, transmitted = comp.compress(ef)
            self._error_buffers[i] = ef - transmitted
            all_idx.append(idx)
            all_val.append(val)
        mx.eval(*self._error_buffers)
        return all_idx, all_val

    def _gather_decompress(self, all_idx, all_val, grove) -> list[np.ndarray]:
        ws = grove.world_size
        updates = []
        for i, ((_, shape), comp) in enumerate(zip(self._param_info, self._compressors)):
            idx_gathered = np.array(grove._comm.all_gather(all_idx[i]))
            val_gathered = np.array(grove._comm.all_gather(all_val[i]))
            per_worker = len(all_idx[i])
            accumulated = np.zeros(int(np.prod(shape)), dtype=np.float32)
            for w in range(ws):
                start = w * per_worker
                accumulated += comp.decompress(
                    idx_gathered[start:start + per_worker],
                    val_gathered[start:start + per_worker],
                )
            accumulated /= ws
            updates.append(accumulated)
        return updates

    def _apply_updates(self, model, updates) -> None:
        new_flat = {}
        flat_initial = tree_flatten(self._initial_params)
        for (name, init_param), update, (_, shape) in zip(flat_initial, updates, self._param_info):
            new_flat[name] = init_param - self._outer_lr * mx.array(update.reshape(shape))
        model.update(tree_unflatten(list(new_flat.items())))
        mx.eval(model.parameters())

    @property
    def inner_step(self) -> int:
        return self._inner_step

    @property
    def outer_step(self) -> int:
        return self._outer_step
