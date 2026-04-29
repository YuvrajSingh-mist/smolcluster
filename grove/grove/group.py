"""Process group: binds rank, world size, store, and transport."""

from .store.base import Store
from .transport.base import Transport
from ._types import TransportType
from ._utils import get_logger

log = get_logger("group")


class Group:
    def __init__(
        self,
        rank: int,
        world_size: int,
        store: Store | None,
        transport_type: TransportType = TransportType.TCP,
        transport: "Transport | None" = None,
    ):
        self._rank = rank
        self._world_size = world_size
        self._store = store

        if transport is not None:
            self._transport = transport
        else:
            match transport_type:
                case TransportType.TCP:
                    from .transport.tcp import TCPTransport
                    self._transport = TCPTransport(rank, world_size, store)
                case TransportType.P2P:
                    from .transport.p2p import P2PTransport
                    self._transport = P2PTransport(rank, world_size, store)

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def store(self) -> Store | None:
        return self._store

    @property
    def transport(self) -> Transport:
        return self._transport

    def barrier(self, tag: str = "default") -> None:
        self._store.barrier(self._rank, self._world_size, tag)

    def destroy(self) -> None:
        self._transport.close()
