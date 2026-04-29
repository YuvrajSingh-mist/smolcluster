"""Core collective operations."""

import numpy as np
from conftest import run_ranks


def _allreduce(rank, world_size, comm):
    data = np.full(1000, rank + 1, dtype=np.float32)
    comm.all_reduce(data)
    expected = world_size * (world_size + 1) / 2
    assert np.allclose(data, expected)


def _allgather(rank, world_size, comm):
    data = np.full(100, rank, dtype=np.float32)
    result = comm.all_gather(data)
    assert result.shape == (world_size * 100,)
    for i in range(world_size):
        assert np.allclose(result[i * 100 : (i + 1) * 100], i)


def _broadcast(rank, world_size, comm):
    if rank == 0:
        data = np.arange(500, dtype=np.float64)
    else:
        data = np.zeros(500, dtype=np.float64)
    comm.broadcast(data, root=0)
    assert np.allclose(data, np.arange(500, dtype=np.float64))


def test_allreduce_2():
    run_ranks(_allreduce, world_size=2)

def test_allreduce_4():
    run_ranks(_allreduce, world_size=4)

def test_allgather_4():
    run_ranks(_allgather, world_size=4)

def test_broadcast_4():
    run_ranks(_broadcast, world_size=4)
