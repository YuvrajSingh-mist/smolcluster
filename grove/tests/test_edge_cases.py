"""Edge cases: dtypes, sizes, reduce ops, point-to-point."""

import math
import numpy as np
from grove._types import ReduceOp
from conftest import run_ranks


def _allreduce_dtype(rank, world_size, comm, dtype_str=None):
    data = np.full(256, rank + 1, dtype=np.dtype(dtype_str))
    comm.all_reduce(data)
    expected = world_size * (world_size + 1) / 2
    assert np.allclose(data, expected)


def _allreduce_size(rank, world_size, comm, size=None):
    data = np.full(size, rank + 1, dtype=np.float32)
    comm.all_reduce(data)
    expected = world_size * (world_size + 1) / 2
    assert np.allclose(data, expected)


def _large(rank, world_size, comm):
    data = np.full(8_000_000, rank + 1, dtype=np.float32)
    comm.all_reduce(data)
    assert np.allclose(data, world_size * (world_size + 1) / 2)


def _reduce_ops(rank, world_size, comm):
    data = np.full(100, rank + 1, dtype=np.float32)
    comm.all_reduce(data, ReduceOp.SUM)
    assert np.allclose(data, world_size * (world_size + 1) / 2)

    comm.barrier()
    data = np.full(100, rank + 1, dtype=np.float32)
    comm.all_reduce(data, ReduceOp.MIN)
    assert np.allclose(data, 1.0)

    comm.barrier()
    data = np.full(100, rank + 1, dtype=np.float32)
    comm.all_reduce(data, ReduceOp.MAX)
    assert np.allclose(data, world_size)

    comm.barrier()
    data = np.full(100, rank + 1, dtype=np.float64)
    comm.all_reduce(data, ReduceOp.PROD)
    assert np.allclose(data, math.factorial(world_size))


def _p2p(rank, world_size, comm):
    if rank == 0:
        data = np.arange(100, dtype=np.float32)
        comm.send(data, dst=1)
        buf = np.zeros(100, dtype=np.float32)
        comm.recv(buf, src=1)
        assert np.allclose(buf, np.arange(100, dtype=np.float32) * 2)
    elif rank == 1:
        buf = np.zeros(100, dtype=np.float32)
        comm.recv(buf, src=0)
        assert np.allclose(buf, np.arange(100, dtype=np.float32))
        comm.send(buf * 2, dst=0)


def test_float64():
    run_ranks(_allreduce_dtype, world_size=2, dtype_str="float64")

def test_int32():
    run_ranks(_allreduce_dtype, world_size=2, dtype_str="int32")

def test_int64():
    run_ranks(_allreduce_dtype, world_size=2, dtype_str="int64")

def test_size_7():
    run_ranks(_allreduce_size, world_size=3, size=7)

def test_size_1001():
    run_ranks(_allreduce_size, world_size=4, size=1001)

def test_size_1():
    run_ranks(_allreduce_size, world_size=2, size=1)

def test_large_32mb():
    run_ranks(_large, world_size=2)

def test_reduce_ops():
    run_ranks(_reduce_ops, world_size=4)

def test_p2p():
    run_ranks(_p2p, world_size=2)
