"""Ring all-reduce with sub-chunking to fit kernel socket buffers."""

import threading
import numpy as np
from ..transport.base import Transport
from .._types import ReduceOp
from .._ops import REDUCE_FN, BufferPool
from .._utils import get_logger

log = get_logger("ring")

_SLICE_ELEMS = 1024 * 1024


class _AsyncSend:
    def __init__(self, conn, data: np.ndarray, pool: BufferPool) -> None:
        self._error: Exception | None = None
        buf = pool.get("send", data.size, data.dtype)
        np.copyto(buf, data)
        self._thread = threading.Thread(target=self._run, args=(conn, buf), daemon=True)
        self._thread.start()

    def _run(self, conn, buf) -> None:
        try:
            conn.send(buf)
        except Exception as e:
            self._error = e

    def wait(self) -> None:
        self._thread.join()
        if self._error is not None:
            raise self._error


class RingAllReduce:
    def __init__(
        self,
        rank: int,
        world_size: int,
        transport: Transport,
        live_ranks: list[int] | None = None,
    ):
        self._rank = rank
        self._transport = transport
        self._send_conn = None
        self._recv_conn = None
        self._blocking = True
        self._world_size = world_size
        if world_size > 1:
            if live_ranks is None:
                live_ranks = list(range(world_size))
            self._setup_ring(live_ranks)
        self._pool = BufferPool()

    def _setup_ring(self, live_ranks: list[int]) -> None:
        idx = live_ranks.index(self._rank)
        send_to = live_ranks[(idx + 1) % len(live_ranks)]
        recv_from = live_ranks[(idx - 1) % len(live_ranks)]
        self._send_conn = self._transport.connect(send_to)
        self._recv_conn = self._transport.connect(recv_from)
        self._blocking = getattr(self._send_conn, "is_blocking", True)
        self._world_size = len(live_ranks)

    def reform(self, live_ranks: list[int]) -> None:
        self._setup_ring(live_ranks)

    def _send(self, data: np.ndarray) -> _AsyncSend | None:
        if not self._blocking:
            self._send_conn.send(data)
            return None
        return _AsyncSend(self._send_conn, data, self._pool)

    def _ring_reduce_slice(self, flat: np.ndarray, reduce_fn: "Callable") -> None:
        N = self._world_size
        r = self._rank
        chunk_size = (flat.size + N - 1) // N
        chunks = [flat[i * chunk_size : min((i + 1) * chunk_size, flat.size)] for i in range(N)]
        tmp = self._pool.get("tmp", chunk_size, flat.dtype)

        for step in range(N - 1):
            send_idx = (r - step) % N
            recv_idx = (r - step - 1) % N
            pending = self._send(chunks[send_idx])
            recv_buf = tmp[: chunks[recv_idx].size]
            self._recv_conn.recv(recv_buf)
            if pending:
                pending.wait()
            reduce_fn(chunks[recv_idx], recv_buf, out=chunks[recv_idx])

        for step in range(N - 1):
            send_idx = (r - step + 1) % N
            recv_idx = (r - step) % N
            pending = self._send(chunks[send_idx])
            recv_buf = tmp[: chunks[recv_idx].size]
            self._recv_conn.recv(recv_buf)
            if pending:
                pending.wait()
            np.copyto(chunks[recv_idx], recv_buf)

    def all_reduce(self, buf: np.ndarray, op: ReduceOp = ReduceOp.SUM) -> None:
        N = self._world_size
        if N == 1:
            return

        if not buf.flags["C_CONTIGUOUS"]:
            raise ValueError("all_reduce requires C-contiguous array")

        reduce_fn = REDUCE_FN[op]
        flat = buf.ravel()

        slice_size = max(_SLICE_ELEMS * N, N)
        for start in range(0, flat.size, slice_size):
            self._ring_reduce_slice(flat[start : start + slice_size], reduce_fn)

    def all_gather(self, buf: np.ndarray) -> np.ndarray:
        N = self._world_size
        if N == 1:
            return buf.copy()

        r = self._rank
        piece_size = buf.size
        result = np.empty(N * piece_size, dtype=buf.dtype)
        result[r * piece_size : (r + 1) * piece_size] = buf.ravel()

        for step in range(N - 1):
            send_idx = (r - step) % N
            recv_idx = (r - step - 1) % N
            pending = self._send(result[send_idx * piece_size : (send_idx + 1) * piece_size])
            self._recv_conn.recv(result[recv_idx * piece_size : (recv_idx + 1) * piece_size])
            if pending:
                pending.wait()

        return result
