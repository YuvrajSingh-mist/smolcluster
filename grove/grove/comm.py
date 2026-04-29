"""Collective operations over a process group."""

import numpy as np
from .group import Group
from ._types import ReduceOp
from .algorithms.ring import RingAllReduce
from ._utils import get_logger

log = get_logger("comm")


class Communicator:
    def __init__(self, group: Group, worker_client: "WorkerClient | None" = None):
        self._group = group
        self._worker_client = worker_client
        self._membership_epoch = 0
        self._algo = RingAllReduce(group.rank, group.world_size, group.transport)

    @property
    def rank(self) -> int:
        return self._group.rank

    @property
    def world_size(self) -> int:
        return self._group.world_size

    @property
    def group(self) -> Group:
        return self._group

    def _maybe_reform(self) -> None:
        if self._worker_client is None:
            return
        membership = self._worker_client.get_pending_membership()
        if membership is None or membership.epoch <= self._membership_epoch:
            return
        self._algo.reform(list(membership.live_ranks))
        self._group._world_size = membership.size
        self._membership_epoch = membership.epoch
        self._worker_client.ack_reform(membership.epoch)
        self._worker_client.wait_reform_complete(membership.epoch)
        import grove
        grove.world_size = membership.size
        log.info("Reformed ring: epoch=%d, nodes=%s", membership.epoch, membership.live_ranks)

    def _wait_for_reform(self, timeout: float = 30.0) -> None:
        self._worker_client.report_failure()
        self._worker_client.wait_for_membership(timeout)
        self._maybe_reform()

    def all_reduce(self, buf: np.ndarray, op: ReduceOp = ReduceOp.SUM) -> None:
        self._maybe_reform()
        if self._worker_client is not None:
            buf_backup = buf.copy()
            try:
                self._algo.all_reduce(buf, op)
            except (ConnectionError, OSError, TimeoutError) as e:
                log.warning("All-reduce failed: %s. Waiting for reform...", e)
                np.copyto(buf, buf_backup)
                self._wait_for_reform()
                self._algo.all_reduce(buf, op)
        else:
            self._algo.all_reduce(buf, op)

    def all_gather(self, buf: np.ndarray) -> np.ndarray:
        self._maybe_reform()
        if self._worker_client is not None:
            try:
                return self._algo.all_gather(buf)
            except (ConnectionError, OSError, TimeoutError) as e:
                log.warning("All-gather failed: %s. Waiting for reform...", e)
                self._wait_for_reform()
                return self._algo.all_gather(buf)
        return self._algo.all_gather(buf)

    def broadcast(self, buf: np.ndarray, root: int = 0) -> None:
        N = self._group.world_size
        r = self._group.rank
        if N == 1:
            return
        transport = self._group.transport
        if r == root:
            for peer in range(N):
                if peer != root:
                    transport.connect(peer).send(buf)
        else:
            transport.connect(root).recv(buf)

    def send(self, buf: np.ndarray, dst: int) -> None:
        self._group.transport.connect(dst).send(buf)

    def recv(self, buf: np.ndarray, src: int) -> None:
        self._group.transport.connect(src).recv(buf)

    def barrier(self) -> None:
        if self._group.store is not None:
            self._group.barrier()
            return

        N = self._group.world_size
        if N <= 1:
            return

        buf = np.ones(1, dtype=np.float32)
        self._algo.all_reduce(buf)
