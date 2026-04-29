"""Test infrastructure: mock transport and distributed test runner."""

import queue
import threading
import numpy as np
from grove.transport.base import Transport, Connection
from grove.comm import Communicator
from grove.group import Group


class MockConnection(Connection):
    def __init__(self, send_q: queue.Queue, recv_q: queue.Queue):
        self._send_q = send_q
        self._recv_q = recv_q

    def send(self, buf: np.ndarray) -> None:
        self._send_q.put(buf.ravel().copy())

    def recv(self, buf: np.ndarray) -> None:
        data = self._recv_q.get(timeout=30)
        np.copyto(buf.ravel(), data[:buf.size])

    def close(self) -> None:
        pass


class MockTransport(Transport):
    def __init__(self, rank: int, world_size: int, queues: dict):
        self._rank = rank
        self._world_size = world_size
        self._queues = queues
        self._connections: dict[int, MockConnection] = {}

    def connect(self, peer_rank: int) -> MockConnection:
        if peer_rank not in self._connections:
            send_q = self._queues[(self._rank, peer_rank)]
            recv_q = self._queues[(peer_rank, self._rank)]
            self._connections[peer_rank] = MockConnection(send_q, recv_q)
        return self._connections[peer_rank]

    def close(self) -> None:
        self._connections.clear()

    @property
    def transport_type(self) -> str:
        return "mock"


def create_mock_transports(world_size: int) -> list[MockTransport]:
    queues = {}
    for src in range(world_size):
        for dst in range(world_size):
            if src != dst:
                queues[(src, dst)] = queue.Queue()
    return [MockTransport(r, world_size, queues) for r in range(world_size)]


def run_ranks(fn, world_size, **kwargs):
    """Run fn(rank, world_size, comm, **kwargs) in parallel threads."""
    transports = create_mock_transports(world_size)
    errors = []

    def worker(rank):
        try:
            group = Group(rank, world_size, store=None, transport=transports[rank])
            comm = Communicator(group)
            fn(rank, world_size, comm, **kwargs)
        except Exception as e:
            errors.append((rank, e))

    threads = [threading.Thread(target=worker, args=(r,)) for r in range(world_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    if errors:
        msgs = [f"rank {r}: {e}" for r, e in errors]
        raise RuntimeError(f"Failed: {'; '.join(msgs)}")
