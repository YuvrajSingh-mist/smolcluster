"""Centralized logging configuration for smolcluster."""
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from smolcluster.utils.logging_startup import ensure_logging_infrastructure


class RankFilter(logging.Filter):
    """Add rank information to log records."""
    
    def __init__(self, rank: Optional[int] = None, component: str = "server"):
        super().__init__()
        self.rank = rank if rank is not None else -1
        self.component = component
    
    def filter(self, record):
        record.rank = self.rank
        record.component = self.component
        return True


def setup_cluster_logging(
    logger: logging.Logger,
    component: str,
    rank: Optional[int] = None,
    hostname: Optional[str] = None,
    log_dir: str = "/tmp/smolcluster-logs",
    level: int = logging.INFO,
) -> None:
    """
    Add file logging to existing logger with structured format for Loki.
    Does NOT start infrastructure - assumes Promtail is running to ship logs.
    
    Args:
        logger: Existing logger to configure
        component: "server" or "worker"
        rank: Worker rank (None for server)
        hostname: Machine hostname
        log_dir: Directory for log files
        level: Logging level
    """
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Determine log file name
    if component == "server":
        log_file = Path(log_dir) / f"server-{hostname or 'unknown'}.log"
    else:
        log_file = Path(log_dir) / f"worker-rank{rank}-{hostname or 'unknown'}.log"
    
    # Check if file handler already exists for this logger
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_file):
            logger.info(f"📝 Logging already configured for: {log_file}")
            return
    
    # Add rank filter
    rank_filter = RankFilter(rank=rank, component=component)
    logger.addFilter(rank_filter)
    
    # File handler (structured for Loki)
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(level)
    
    # Structured format: timestamp | level | rank:X | step:Y | message
    # This makes it easy for Loki to parse and add labels
    if component == "server":
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | rank:server | %(message)s'
        )
    else:
        file_formatter = logging.Formatter(
            f'%(asctime)s | %(levelname)s | rank:{rank} | %(message)s'
        )
    
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"📝 Logging initialized: {log_file}")
    logger.info(f"Component: {component}, Rank: {rank}, Hostname: {hostname}")


def log_step(logger: logging.Logger, step: int, message: str, level: int = logging.INFO):
    """
    Log a message with step information for better Loki filtering.
    
    Args:
        logger: Logger instance
        step: Training step number
        message: Log message
        level: Logging level
    """
    logger.log(level, f"step:{step} | {message}")


def log_metric(
    logger: logging.Logger,
    step: int,
    metric_name: str,
    value: float,
    extra_info: Optional[str] = None
):
    """
    Log a metric in a structured way.
    
    Args:
        logger: Logger instance
        step: Training step
        metric_name: Name of the metric (e.g., "loss", "accuracy")
        value: Metric value
        extra_info: Optional additional context
    """
    msg = f"step:{step} | metric:{metric_name} | value:{value:.6f}"
    if extra_info:
        msg += f" | {extra_info}"
    logger.info(msg)


# Example Grafana LogQL queries (add to docstring or README)
EXAMPLE_QUERIES = """
# Grafana Loki Query Examples for smolcluster

## View all server logs
{component="server"}

## View specific worker
{component="worker", rank="0"}

## Find errors across all workers
{component="worker"} |= "ERROR"

## Filter by training step
{component="worker"} | regexp "step:(?P<step>\\d+)" | step > 1000

## View gradient updates
{job="smolcluster-worker"} |= "gradient"

## Find timeouts or connection issues
{job=~"smolcluster.*"} |~ "timeout|connection|error"

## Compare loss across workers
{component="worker"} | regexp "metric:loss.*value:(?P<loss>[\\d.]+)"

## View recent logs (last 5 minutes)
{component="worker"}[5m]

## Count errors per worker
sum by (rank) (count_over_time({component="worker"} |= "ERROR" [1h]))
"""
