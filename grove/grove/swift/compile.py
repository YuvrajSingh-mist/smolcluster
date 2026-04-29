"""Compile the Swift P2P helper binary."""

import shutil
import subprocess
from pathlib import Path
from .._utils import get_logger

log = get_logger("swift")

_BIN_DIR = Path.home() / ".grove" / "bin"
_BIN_NAME = "grove-p2p-helper"
_SWIFT_SRC = Path(__file__).parent / "p2p_helper.swift"


def binary_path() -> Path:
    return _BIN_DIR / _BIN_NAME


def is_available() -> bool:
    return shutil.which("swiftc") is not None


def ensure_compiled() -> Path:
    bin_path = binary_path()

    if bin_path.exists() and bin_path.stat().st_mtime >= _SWIFT_SRC.stat().st_mtime:
        return bin_path

    if not is_available():
        raise RuntimeError(
            "Swift compiler (swiftc) not found. "
            "Install Xcode command-line tools: xcode-select --install"
        )

    _BIN_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Compiling P2P helper: {_SWIFT_SRC} -> {bin_path}")

    result = subprocess.run(
        ["swiftc", "-O", "-o", str(bin_path), str(_SWIFT_SRC)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Swift compilation failed:\n{result.stderr}")

    log.info("P2P helper compiled successfully")
    return bin_path
