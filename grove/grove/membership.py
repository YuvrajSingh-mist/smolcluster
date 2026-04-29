"""Membership tracking for elastic training."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Membership:
    epoch: int
    live_ranks: tuple[int, ...]
    addr_map: dict[int, str]

    @property
    def size(self) -> int:
        return len(self.live_ranks)

    def ring_neighbors(self, rank: int) -> tuple[int, int]:
        idx = self.live_ranks.index(rank)
        send_to = self.live_ranks[(idx + 1) % self.size]
        recv_from = self.live_ranks[(idx - 1) % self.size]
        return send_to, recv_from
