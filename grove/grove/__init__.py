"""Grove: Zero-config distributed ML for Apple Silicon.

    import grove
    world = grove.init()

    for batch in data[world.rank()::world.size()]:
        loss, grads = loss_and_grad(model, batch)
        grads = grove.average_gradients(grads)
        optimizer.update(model, grads)
        mx.eval(model.state, optimizer.state)
"""

__version__ = "0.1.0"

rank: int = 0
world_size: int = 1

_comm = None
_packer = None
_coordinator = None
_worker_client = None
_received_script: tuple[str, str] | None = None
_sync_every: int = 1


class World:
    def rank(self) -> int:
        import grove
        return grove.rank

    def size(self) -> int:
        import grove
        return grove.world_size


def init(
    cluster: str | None = None,
    world_size: int | None = None,
    sync_every: int = 1,
    timeout: float = 120.0,
    transport: str = "p2p",
) -> World:
    globals()["_sync_every"] = sync_every

    if cluster is not None and world_size is not None and world_size > 1:
        if transport == "p2p":
            _init_p2p(cluster, world_size, timeout)
        else:
            _init_cluster(cluster, world_size, timeout)
        return World()

    return World()


_step_counter: int = 0


def average_gradients(grads: dict) -> dict:
    if world_size <= 1:
        return grads

    import time
    import mlx.core as mx
    from ._types import ReduceOp

    arrays = []
    def _collect(d):
        for val in d.values():
            if isinstance(val, dict):
                _collect(val)
            elif hasattr(val, "shape"):
                arrays.append(val)
    _collect(grads)
    mx.eval(*arrays)

    t0 = time.monotonic()
    buf = _packer.flatten(grads)
    _comm.all_reduce(buf, ReduceOp.SUM)
    result = _packer.unflatten(buf, 1.0 / world_size, grads)
    sync_ms = (time.monotonic() - t0) * 1000

    globals()["_step_counter"] = _step_counter + 1
    if _worker_client is not None:
        _worker_client.set_stats(step=_step_counter, sync_ms=sync_ms)
    if _coordinator is not None:
        with _coordinator._lock:
            _coordinator._step_counts[rank] = _step_counter
            _coordinator._sync_ms[rank] = sync_ms

    return result


def report(loss: float) -> None:
    if _worker_client is not None:
        _worker_client._loss = loss
    if _coordinator is not None:
        with _coordinator._lock:
            _coordinator._loss[rank] = loss


def demo(model, lr: float = 1e-3, decay: float = 0.999, topk: int = 32, chunk: int = 64) -> "DeMo":
    from .demo import DeMo
    return DeMo(model, lr=lr, decay=decay, topk=topk, chunk=chunk)


def sparseloco(
    model,
    outer_lr: float = 1.0,
    H: int = 30,
    error_decay: float = 0.95,
    topk: int = 64,
    chunk: int = 4096,
    overlap: bool = True,
) -> "SparseLoCo":
    from .sparseloco import SparseLoCo
    return SparseLoCo(model, outer_lr, H, error_decay, topk, chunk, overlap)


def diloco(
    model,
    outer_lr: float = 0.7,
    outer_momentum: float = 0.9,
    H: int = 500,
    overlap: bool = False,
    quantize: bool = False,
) -> "DiLoCo":
    from .diloco import DiLoCo
    return DiLoCo(model, outer_lr, outer_momentum, H, overlap, quantize=quantize)


def is_available() -> bool:
    return world_size > 1


def all_sum(x):
    import mlx.core as mx
    import numpy as np
    if world_size <= 1:
        return x
    mx.eval(x)
    buf = np.array(x.astype(mx.float32), copy=False).copy()
    _comm.all_reduce(buf)
    return mx.array(buf).astype(x.dtype)


def all_gather(x):
    import mlx.core as mx
    import numpy as np
    if world_size <= 1:
        return x
    mx.eval(x)
    buf = np.array(x.astype(mx.float32), copy=False).copy()
    result = _comm.all_gather(buf)
    return mx.array(result).astype(x.dtype)


def send(x, dst: int):
    import mlx.core as mx
    import numpy as np
    mx.eval(x)
    _comm.send(np.array(x.astype(mx.float32), copy=False), dst)


def recv(shape, dtype, src: int):
    import mlx.core as mx
    import numpy as np
    buf = np.empty(shape, dtype=np.float32)
    _comm.recv(buf, src)
    return mx.array(buf).astype(dtype)


def recv_like(x, src: int):
    return recv(x.shape, x.dtype, src)


def barrier() -> None:
    if _comm is not None:
        _comm.barrier()


from ._init import _init_cluster, _init_p2p, _init_packer  # noqa: E402
