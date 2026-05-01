"""Training dashboard launchers."""

import logging
import os
import sys
import threading
from .tui import DashboardApp, LogCapture


def _disable_tqdm_locks() -> None:
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    try:
        import tqdm.std
        class _DummyLock:
            def __init__(self, *a, **kw): pass
            def acquire(self, *a, **kw): return True
            def release(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
        tqdm.std.TqdmDefaultWriteLock = _DummyLock
    except ImportError:
        pass


def _setup_log_capture() -> LogCapture:
    _disable_tqdm_locks()
    capture = LogCapture()

    # Redirect both stdout and stderr so nothing leaks to the raw terminal
    sys.stdout = capture
    sys.stderr = capture

    # Replace all existing logging handlers with one that writes to capture
    class _CaptureHandler(logging.Handler):
        def emit(self, record):
            try:
                capture.write(self.format(record))
            except Exception:
                pass

    class _AnsiLevelFormatter(logging.Formatter):
        _RESET = "\x1b[0m"
        _LEVEL_COLORS = {
            logging.DEBUG: "\x1b[2m",      # dim
            logging.WARNING: "\x1b[33m",   # yellow
            logging.ERROR: "\x1b[31m",     # red
            logging.CRITICAL: "\x1b[1;31m",# bold red
        }

        def format(self, record):
            msg = super().format(record)
            color = self._LEVEL_COLORS.get(record.levelno)
            if not color:
                return msg
            return f"{color}{msg}{self._RESET}"

    handler = _CaptureHandler()
    handler.setFormatter(_AnsiLevelFormatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # Suppress noisy grove internals
    for name in list(logging.Logger.manager.loggerDict):
        if name.startswith("grove"):
            logging.getLogger(name).setLevel(logging.WARNING)
    return capture


class Dashboard:
    def __init__(self, coordinator: "CoordinatorServer", cluster_name: str, uid: str) -> None:
        self._coord = coordinator
        self._cluster = cluster_name
        self._uid = uid

    def run_with_training(self, train_fn) -> None:
        capture = _setup_log_capture()
        done = threading.Event()
        error = threading.Event()

        def _train():
            try:
                train_fn()
                capture.write("\x1b[1;32mTraining complete.\x1b[0m Press q to exit.")
            except SystemExit as e:
                if e.code not in (None, 0):
                    error.set()
                    capture.write(f"\x1b[1;31mExited with code {e.code}\x1b[0m")
            except Exception as e:
                import traceback
                error.set()
                capture.write(f"\x1b[1;31mERROR: {e}\x1b[0m")
                capture.write(traceback.format_exc())
            finally:
                done.set()

        train_thread = threading.Thread(target=_train, daemon=True)
        train_thread.start()

        app = DashboardApp(
            self._coord._build_stats,
            self._cluster,
            self._uid,
            my_rank=0,
            log_capture=capture,
            done_event=done,
            error_event=error,
        )
        app.run()


class WorkerDashboard:
    def __init__(self, worker_client: "WorkerClient", cluster_name: str, uid: str, rank: int) -> None:
        self._client = worker_client
        self._cluster = cluster_name
        self._uid = uid
        self._rank = rank

    def run_with_training(self, train_fn) -> None:
        capture = _setup_log_capture()
        done = threading.Event()
        error = threading.Event()

        def _train():
            try:
                train_fn()
                capture.write("\x1b[1;32mTraining complete.\x1b[0m Press q to exit.")
            except SystemExit as e:
                if e.code not in (None, 0):
                    error.set()
                    capture.write(f"\x1b[1;31mExited with code {e.code}\x1b[0m")
            except Exception as e:
                import traceback
                error.set()
                capture.write(f"\x1b[1;31mERROR: {e}\x1b[0m")
                capture.write(traceback.format_exc())
            finally:
                done.set()

        train_thread = threading.Thread(target=_train, daemon=True)
        train_thread.start()

        app = DashboardApp(
            self._client.get_cluster_stats,
            self._cluster,
            self._uid,
            my_rank=self._rank,
            log_capture=capture,
            done_event=done,
            error_event=error,
        )
        app.run()
