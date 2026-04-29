"""DeMo (Peng et al., 2024): Decoupled Momentum with DCT compression."""

import numpy as np
import mlx.core as mx
from mlx.utils import tree_unflatten
from mlx.nn.utils import tree_flatten
from .compress import TopKCompressor


class DeMo:
    def __init__(
        self,
        model: "nn.Module",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        decay: float = 0.999,
        topk: int = 32,
        chunk: int = 64,
    ) -> None:
        self._lr = lr
        self._weight_decay = weight_decay
        self._decay = decay
        self._step_count = 0

        self._params: list[tuple[str, tuple]] = []
        self._deltas: list[mx.array] = []
        self._compressors: list[TopKCompressor] = []

        for name, param in tree_flatten(dict(model.trainable_parameters())):
            total = int(np.prod(param.shape))
            self._params.append((name, param.shape))
            self._deltas.append(mx.zeros((total,)))
            self._compressors.append(TopKCompressor(total, chunk, topk, use_dct=True))

    def step(self, model: "nn.Module", grads: dict) -> None:
        import grove

        self._step_count += 1

        flat_grads = [g.reshape(-1) for _, g in tree_flatten(grads)]
        flat_params = tree_flatten(dict(model.trainable_parameters()))

        all_idx = []
        all_val = []

        for i, (grad, comp) in enumerate(zip(flat_grads, self._compressors)):
            if self._weight_decay != 0:
                _, param = flat_params[i]
                grad = grad + self._weight_decay * param.reshape(-1)

            delta = self._deltas[i] * self._decay + self._lr * grad
            idx, val, transmitted = comp.compress(delta)
            self._deltas[i] = delta - transmitted
            all_idx.append(idx)
            all_val.append(val)

        mx.eval(*self._deltas)

        if grove.world_size > 1:
            updates = self._all_gather_decompress(all_idx, all_val, grove)
        else:
            updates = [
                comp.decompress(idx, val)
                for comp, idx, val in zip(self._compressors, all_idx, all_val)
            ]

        new_flat = {}
        for (name, param), update, (_, shape) in zip(flat_params, updates, self._params):
            new_flat[name] = param - self._lr * mx.sign(mx.array(update.reshape(shape)))

        model.update(tree_unflatten(list(new_flat.items())))
        mx.eval(model.parameters())

    def _all_gather_decompress(self, all_idx, all_val, grove) -> list[np.ndarray]:
        ws = grove.world_size
        updates = []
        for i, ((_, shape), comp) in enumerate(zip(self._params, self._compressors)):
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

    @property
    def step_count(self) -> int:
        return self._step_count
