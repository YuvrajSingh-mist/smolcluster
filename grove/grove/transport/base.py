"""Abstract base classes for transports and connections."""

from abc import ABC, abstractmethod
import numpy as np


class Connection(ABC):
    @abstractmethod
    def send(self, buf: np.ndarray) -> None: ...

    @abstractmethod
    def recv(self, buf: np.ndarray) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class Transport(ABC):
    @abstractmethod
    def connect(self, peer_rank: int) -> Connection: ...

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def transport_type(self) -> str: ...
