"""P2P Swift helper compilation tests."""

import os
import pytest
from grove.swift.compile import ensure_compiled, is_available, binary_path


@pytest.mark.skipif(not is_available(), reason="swiftc not installed")
class TestP2PCompilation:
    def test_swiftc_available(self):
        assert is_available()

    def test_compile(self):
        path = binary_path()
        if path.exists():
            path.unlink()

        result = ensure_compiled()
        assert result.exists()
        assert os.access(result, os.X_OK)

    def test_compile_cached(self):
        import time
        ensure_compiled()
        start = time.perf_counter()
        ensure_compiled()
        elapsed = time.perf_counter() - start
        assert elapsed < 0.01

    def test_binary_runs(self):
        import subprocess
        path = ensure_compiled()
        result = subprocess.run(
            [str(path)], capture_output=True, text=True, timeout=5,
        )
        assert result.returncode != 0
        assert "Usage" in result.stderr
