"""AMP-style mixed precision utilities for MLX GRPO training.

Implements the key stability techniques from mixed precision training:
    1. Dynamic loss scaling.
    2. Overflow detection with skipped updates.
    3. FP32 master weights and FP32 optimizer state updates.
"""

import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Tuple

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from ..utils.training_utils import _scale_grads

logger = logging.getLogger(__name__)

_FLOAT_DTYPES = frozenset({mx.float16, mx.bfloat16, mx.float32})


class GradScaler:
    """Scale loss before backward so reduced-precision gradients don't underflow.

    Algorithm (mirrors PyTorch GradScaler):
      1. Multiply loss by ``scale`` before backward pass.
      2. After accumulation, divide all grads by ``scale`` (unscale).
      3. Check for inf/nan in unscaled grads:
           - found  → halve scale, skip optimizer step
           - clean  → apply step; double scale every ``growth_interval`` clean steps
    """

    def __init__(
        self,
        init_scale: float = 2 ** 15,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ) -> None:
        self._scale = float(init_scale)
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._growth_tracker = 0
        self.enabled = enabled

    def get_scale(self) -> float:
        return self._scale if self.enabled else 1.0

    def scale(self, value: mx.array) -> mx.array:
        """PyTorch-like API: scaler.scale(loss_or_tensor)."""
        return value * self._scale if self.enabled else value

    def scale_loss(self, loss: mx.array) -> mx.array:
        """Multiply loss by scale before the backward pass."""
        return self.scale(loss)

    def scale_tensor(self, value: mx.array) -> mx.array:
        """PyTorch-like alias for scaling a tensor/loss value."""
        return self.scale(value)

    def unscale(self, grads: Any) -> Any:
        """Divide every gradient tensor by the current scale.

        Must be called after gradient accumulation and before any gradient
        clipping or the optimizer step.
        """
        return _scale_grads(grads, 1.0 / self._scale) if self.enabled else grads

    def unscale_(self, grads: Any) -> Any:
        """PyTorch-like alias for in-place-style unscale (returns new tree)."""
        return self.unscale(grads)

    def has_inf_nan(self, grads: Any) -> bool:
        """Return True if any gradient tensor contains inf or nan."""
        for _, g in tree_flatten(grads):
            mx.eval(g)
            if not bool(mx.all(mx.isfinite(g)).item()):
                return True
        return False

    def update(self, found_inf: bool) -> None:
        """Adjust the loss scale after each optimizer step attempt.

        Args:
            found_inf: True if inf/nan was detected in the unscaled gradients.
        """
        if not self.enabled:
            return
        if found_inf:
            self._scale = max(self._scale * self._backoff_factor, 1.0)
            self._growth_tracker = 0
            logger.warning("[AMP] overflow detected — scale backed off to %.0f", self._scale)
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                self._scale = min(self._scale * self._growth_factor, float(2 ** 24))
                self._growth_tracker = 0
                logger.info("[AMP] scale grown to %.0f", self._scale)

    def state_dict(self) -> Dict[str, Any]:
        return {"scale": self._scale, "growth_tracker": self._growth_tracker}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._scale = float(state["scale"])
        self._growth_tracker = int(state["growth_tracker"])


@contextmanager
def autocast(enabled: bool = True) -> Iterator[None]:
    """Compatibility context manager akin to torch.cuda.amp.autocast.

    MLX handles dtype behavior at tensor/module level, so this is intentionally
    a no-op context used for API symmetry and call-site clarity.
    """
    _ = enabled
    yield


class MasterWeightAdamW:
    """AdamW wrapper that always updates FP32 master weights.

    The model keeps train-time compute dtype weights (for example bfloat16).
    This optimizer maintains:
      - FP32 master parameters.
      - FP32 Adam first/second moments.
    After each update, master weights are cast back to each parameter's original
    model dtype and written into the model.
    """

    def __init__(
        self,
        model: Any,
        learning_rate: float,
        betas: Tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 1e-1,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self._fallback = optim.AdamW(
            learning_rate=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self._lr = float(learning_rate)
        self._beta1 = float(betas[0])
        self._beta2 = float(betas[1])
        self._eps = float(eps)
        self._weight_decay = float(weight_decay)
        self._step = 0

        self._param_dtypes: Dict[Any, Any] = {}
        self._master_params: Any = None
        self._m: Any = None
        self._v: Any = None

        if self.enabled:
            self._init_state(model)

    @property
    def state(self) -> Dict[str, Any]:
        if not self.enabled:
            return self._fallback.state
        return {
            "step": self._step,
            "m": self._m,
            "v": self._v,
            "lr": self._lr,
        }

    def _init_state(self, model: Any) -> None:
        flat = tree_flatten(model.parameters())
        self._param_dtypes = {k: v.dtype for k, v in flat}
        # Only cast floating-point params to float32 master weights.
        # Integer params (uint32 packed int4) must NOT go through the lossy
        # uint32→float32→uint32 round-trip — 99.8% of packed words are corrupted
        # because float32 has only 24 bits of mantissa.
        self._master_params = tree_unflatten([
            (k, v.astype(mx.float32) if v.dtype in _FLOAT_DTYPES else v)
            for k, v in flat
        ])
        float_flat = [(k, v) for k, v in flat if v.dtype in _FLOAT_DTYPES]
        self._m = tree_unflatten([(k, mx.zeros(v.shape, dtype=mx.float32)) for k, v in float_flat])
        self._v = tree_unflatten([(k, mx.zeros(v.shape, dtype=mx.float32)) for k, v in float_flat])
        mx.eval(self._master_params, self._m, self._v)

    def update(self, model: Any, grads: Any) -> None:
        if not self.enabled:
            self._fallback.update(model, grads)
            return

        self._step += 1
        step = float(self._step)
        b1_correction = 1.0 - (self._beta1**step)
        b2_correction = 1.0 - (self._beta2**step)

        flat_master = dict(tree_flatten(self._master_params))
        flat_m = dict(tree_flatten(self._m))
        flat_v = dict(tree_flatten(self._v))
        flat_grads = dict(tree_flatten(grads))

        new_master = []
        new_m = []
        new_v = []
        new_model_params = []

        for key, param in flat_master.items():
            # Integer (quantized) params: non-differentiable, no optimizer state.
            # Pass through unchanged — no cast, no update.
            if self._param_dtypes[key] not in _FLOAT_DTYPES:
                new_master.append((key, param))
                new_model_params.append((key, param))
                continue

            grad_tensor = flat_grads.get(key)
            if grad_tensor is None:
                # Some parameters may be absent from the grad tree (for example
                # disconnected/frozen params in a step). In that case, keep
                # parameter and optimizer moments unchanged.
                new_master.append((key, param))
                new_m.append((key, flat_m[key]))
                new_v.append((key, flat_v[key]))
                new_model_params.append((key, param.astype(self._param_dtypes[key])))
                continue

            grad = grad_tensor.astype(mx.float32)
            m = self._beta1 * flat_m[key] + (1.0 - self._beta1) * grad
            v = self._beta2 * flat_v[key] + (1.0 - self._beta2) * mx.square(grad)

            m_hat = m / b1_correction
            v_hat = v / b2_correction
            update = m_hat / (mx.sqrt(v_hat) + self._eps)
            if self._weight_decay != 0.0:
                update = update + self._weight_decay * param

            next_param = param - self._lr * update
            new_master.append((key, next_param))
            new_m.append((key, m))
            new_v.append((key, v))
            new_model_params.append((key, next_param.astype(self._param_dtypes[key])))

        self._master_params = tree_unflatten(new_master)
        self._m = tree_unflatten(new_m)
        self._v = tree_unflatten(new_v)
        model.update(tree_unflatten(new_model_params))
        mx.eval(model.parameters(), self._master_params, self._m, self._v)
