"""Classic Data Parallelism with Ring-AllReduce algorithm."""

from .worker import run_classicdp_worker

__all__ = ["run_classicdp_worker"]
