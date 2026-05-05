"""Centralized logging configuration for smolcluster."""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# ANSI colour palette
# ---------------------------------------------------------------------------

_RESET = "\033[0m"

_LEVEL_COLOURS = {
    logging.DEBUG:    "\033[36m",    # cyan
    logging.INFO:     "\033[32m",    # green
    logging.WARNING:  "\033[33m",    # yellow
    logging.ERROR:    "\033[31m",    # red
    logging.CRITICAL: "\033[1;31m",  # bold red
}

_TAG_COLOUR = "\033[35m"   # magenta — for bracketed tags like [MODEL], [LORA]
_DIM = "\033[2m"
_CTX_COLOUR = "\033[1;35m"  # bold magenta — for the context prefix

# ---------------------------------------------------------------------------
# Global log context — set once at process startup via set_log_context()
# ---------------------------------------------------------------------------

_CTX: dict[str, str] = {}
_CTX_ORDER = ("arch", "algorithm", "role", "hardware")


def _infer_hardware(hostname: str) -> str:
    """Derive human-readable hardware label from hostname."""
    if not hostname:
        return ""
    h = hostname.lower()
    if re.match(r"(macmini|mini)\d*$", h):
        return "Mac Mini"
    if re.match(r"jetson\w*$", h):
        return "Jetson"
    if re.match(r"(rpi|raspi|pi)\d*$", h):
        return "RPi"
    if re.match(r"macbook\w*$", h):
        return "MacBook"
    return hostname


def set_log_context(
    *,
    algorithm: str = "",
    arch: str = "",
    role: str = "",
    hardware: str = "",
) -> None:
    """Set global context fields that appear on every log line.

    Call once from the main entry-point after setup_logging().
    Fields are shown as  [arch | algorithm | role | hardware]  in each line.
    """
    for key, val in (("algorithm", algorithm), ("arch", arch), ("role", role), ("hardware", hardware)):
        if val:
            _CTX[key] = val


class ColourFormatter(logging.Formatter):
    """Single-line coloured formatter.

    Format:  YYYY-MM-DD HH:MM:SS  LEVEL     [ctx]  logger.name  message
    Bracketed tags like [MODEL], [checkpoint], [vllm worker 0] are highlighted.
    Context prefix (arch | algorithm | role | hardware) is shown when set via set_log_context().
    """

    _FMT = "{dim}{asctime}{reset}  {level_col}{levelname:<8}{reset}  {ctx}{dim}{name}{reset}  {msg}"

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        record.message = record.getMessage()
        record.asctime = self.formatTime(record, "%Y-%m-%d %H:%M:%S")

        msg = re.sub(
            r"(\[[^\]]{1,40}\])",
            rf"{_TAG_COLOUR}\1{_RESET}",
            record.message,
        )

        ctx_parts = [_CTX[k] for k in _CTX_ORDER if _CTX.get(k)]
        ctx = f"{_CTX_COLOUR}[{' | '.join(ctx_parts)}]{_RESET}  " if ctx_parts else ""

        line = self._FMT.format(
            dim=_DIM,
            asctime=record.asctime,
            reset=_RESET,
            level_col=_LEVEL_COLOURS.get(record.levelno, ""),
            levelname=record.levelname,
            ctx=ctx,
            name=record.name,
            msg=msg,
        )

        if record.exc_info:
            line = f"{line}\n{_LEVEL_COLOURS.get(logging.ERROR, '')}{self.formatException(record.exc_info)}{_RESET}"

        return line


def setup_logging(
    level: int = logging.INFO,
    *,
    force: bool = False,
) -> None:
    """Configure the root logger with a coloured console handler.

    Call once from the main entry-point of each script.
    Subsequent calls are no-ops unless ``force=True``.
    """
    root = logging.getLogger()

    if root.handlers and not force:
        return

    if force:
        root.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(ColourFormatter())
    root.addHandler(handler)
    root.setLevel(level)

    # Quieten noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "datasets", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Cluster logging (file-based, structured)
# ---------------------------------------------------------------------------

class RankFilter(logging.Filter):
    """Attach rank and component fields to every log record."""

    def __init__(self, rank: Optional[int] = None, component: str = "server"):
        super().__init__()
        self.rank = rank if rank is not None else -1
        self.component = component

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self.rank
        record.component = self.component
        return True


def setup_cluster_logging(
    logger: logging.Logger,
    component: str,
    rank: Optional[int] = None,
    hostname: Optional[str] = None,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    algorithm: str = "",
    arch: str = "",
) -> None:
    """Add structured file logging to an existing logger."""

    def _project_log_dir() -> Path:
        return Path(__file__).resolve().parents[3] / "logging" / "cluster-logs"

    def _pick_writable(preferred: Optional[str]) -> Path:
        default = _project_log_dir()
        for candidate in [Path(preferred) if preferred else default, default, Path.cwd() / "smolcluster-logs"]:
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                probe = candidate / ".write_probe"
                probe.open("a").close()
                probe.unlink(missing_ok=True)
                return candidate
            except OSError:
                continue
        raise OSError("No writable directory found for cluster logs")

    hardware = _infer_hardware(hostname or "")

    # Set global context so ColourFormatter picks it up on every line
    set_log_context(
        algorithm=algorithm,
        arch=arch,
        role=("server" if component == "server" else f"worker-{rank}" if rank is not None else "worker"),
        hardware=hardware,
    )

    algo_prefix = f"{algorithm}-" if algorithm else ""
    log_file = _pick_writable(log_dir) / (
        f"{algo_prefix}server-{hostname or 'unknown'}.log" if component == "server"
        else f"{algo_prefix}worker-rank{rank}-{hostname or 'unknown'}.log"
    )

    # Avoid duplicate handlers
    if any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file) for h in logger.handlers):
        return

    logger.addFilter(RankFilter(rank=rank, component=component))

    try:
        fh = logging.FileHandler(log_file, mode="a")
    except PermissionError:
        fallback = _pick_writable(str(_project_log_dir().parent / "cluster-logs-fallback"))
        fh = logging.FileHandler(fallback / log_file.name, mode="a")

    # Build a clean, human-readable format for the file (no ANSI colours)
    ctx_parts = [p for p in (arch, algorithm, ("server" if component == "server" else f"worker-{rank}"), hardware) if p]
    ctx_str = " | ".join(ctx_parts)
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(
        f"%(asctime)s  %(levelname)-8s  [{ctx_str}]  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)
    logger.info(
        "Logging initialised: %s  [algorithm=%s arch=%s role=%s hardware=%s]",
        log_file, algorithm or "?", arch or "?", component, hardware or "?",
    )


def log_step(logger: logging.Logger, step: int, message: str, level: int = logging.INFO) -> None:
    logger.log(level, "step:%d | %s", step, message)


def log_metric(logger: logging.Logger, step: int, metric_name: str, value: float, extra_info: Optional[str] = None) -> None:
    msg = f"step:{step} | metric:{metric_name} | value:{value:.6f}"
    if extra_info:
        msg += f" | {extra_info}"
    logger.info(msg)


def emit_transport_event(phase: str, **fields) -> None:
    """Emit machine-readable transport events for dashboard particle animation.

    The dashboard listens for lines in the form:
            [TRANSPORT_EVENT] {"phase":"request"|"response", ...}
    """
    payload = {"phase": str(phase or "").strip().lower()}
    for k, v in fields.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            payload[k] = v
        else:
            payload[k] = str(v)
    print(f"[TRANSPORT_EVENT] {json.dumps(payload)}", flush=True)


def emit_smol_event(event_type: str, direction: str, arch: str) -> None:
    """Emit a dashboard particle-animation event.

    This is the canonical way to trigger topology animations in the dashboard.
    It prints a ``[SMOL_EVENT]`` line that the log stream picks up and routes to
    the frontend's ``_handleSmolEvent`` → ``_smolEventQueue`` pipeline.

    Args:
        event_type: One of ``"gradients"``, ``"weights"``, ``"rollout"``, ``"weight_sync"``.
        direction:  ``"out"`` when the local node is *sending* data to peers;
                    ``"in"``  when the local node has *received* data from peers.
        arch:       Algorithm identifier: ``"syncps"``, ``"classicdp"``, ``"fsdp"``, ``"grpo"``.

    Each call produces exactly **one** ``[SMOL_EVENT]`` line, so calling this
    once per peer / per prompt gives one animation burst per exchange — rather
    than a single burst for the whole batch.
    """
    print(f'[SMOL_EVENT] {json.dumps({"type": event_type, "dir": direction, "arch": arch})}', flush=True)
