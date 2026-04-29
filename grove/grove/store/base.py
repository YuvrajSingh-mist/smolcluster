"""Abstract base for key-value stores."""

from abc import ABC, abstractmethod


class Store(ABC):
    @abstractmethod
    def set(self, key: str, value: bytes) -> None: ...

    @abstractmethod
    def get(self, key: str) -> bytes: ...

    @abstractmethod
    def wait(self, keys: list[str], timeout: float | None = None) -> None: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    def check(self, keys: list[str]) -> bool:
        for k in keys:
            try:
                self.get_nowait(k)
            except KeyError:
                return False
        return True

    def get_nowait(self, key: str) -> bytes:
        raise KeyError(key)

    def barrier(self, rank: int, world_size: int, tag: str = "") -> None:
        prefix = f"barrier/{tag}" if tag else "barrier"
        self.set(f"{prefix}/{rank}", b"1")
        keys = [f"{prefix}/{i}" for i in range(world_size)]
        self.wait(keys)
