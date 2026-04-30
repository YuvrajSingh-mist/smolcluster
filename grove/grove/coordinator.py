"""Coordinator server and worker clients for fault tolerance."""

import os
import socket
import threading
import time
from .control import MsgType, send_msg, recv_msg, encode_ctrl, decode_ctrl
from .membership import Membership
from ._utils import get_logger, configure_socket, get_local_ip

log = get_logger("coordinator")


class CoordinatorServer:
    def __init__(
        self,
        host: str,
        port: int,
        world_size: int,
        addr_map: dict[int, str],
    ) -> None:
        self._host = host
        self._port = port
        self._world_size = world_size
        self._addr_map = dict(addr_map)
        self._live_ranks = list(range(world_size))
        self._epoch = 0
        self._lock = threading.Lock()
        self._running = True

        self._last_heartbeat: dict[int, float] = {}
        self._step_counts: dict[int, int] = {}
        self._loss: dict[int, float] = {}
        self._sync_ms: dict[int, float] = {}
        self._hostnames: dict[int, str] = {}
        self._status_map: dict[int, str] = {}
        self._start_time = time.monotonic()
        self._dead_ranks: set[int] = set()
        self._script_content: str = ""
        self._script_name: str = ""
        self._worker_socks: dict[int, socket.socket] = {}
        self._reform_acks: dict[int, threading.Event] = {}

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        configure_socket(self._sock, tcp=True)
        self._sock.bind((host, port))
        self._sock.listen(world_size + 4)
        self._sock.settimeout(1.0)

        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

        self._checker_thread = threading.Thread(target=self._heartbeat_checker, daemon=True)
        self._checker_thread.start()

        log.info(f"Coordinator listening on {host}:{port}")

    def set_script(self, path: str) -> None:
        self._script_name = os.path.basename(path)
        with open(path) as f:
            self._script_content = f.read()

    def _accept_loop(self) -> None:
        while self._running:
            try:
                conn, addr = self._sock.accept()
                configure_socket(conn, tcp=True)
                conn.settimeout(5.0)
                t = threading.Thread(target=self._handle_worker, args=(conn,), daemon=True)
                t.start()
            except socket.timeout:
                continue
            except OSError:
                break

    def _handle_worker(self, conn: socket.socket) -> None:
        rank = -1
        try:
            msg_type, payload = recv_msg(conn)
            if msg_type != MsgType.HEARTBEAT:
                conn.close()
                return
            rank = payload["rank"]
            with self._lock:
                self._worker_socks[rank] = conn
                self._last_heartbeat[rank] = time.monotonic()
                self._step_counts[rank] = payload.get("step", 0)
                self._hostnames[rank] = payload.get("hostname", f"rank-{rank}")
            send_msg(conn, MsgType.HEARTBEAT_ACK, {"epoch": self._epoch})
            log.info(f"Worker {rank} connected ({self._hostnames[rank]})")

            conn.settimeout(10.0)
            while self._running:
                msg_type, payload = recv_msg(conn)
                if msg_type == MsgType.HEARTBEAT:
                    with self._lock:
                        self._last_heartbeat[rank] = time.monotonic()
                        self._step_counts[rank] = payload.get("step", 0)
                        if "loss" in payload:
                            self._loss[rank] = payload["loss"]
                        if "sync_ms" in payload:
                            self._sync_ms[rank] = payload["sync_ms"]
                        if "status" in payload:
                            self._status_map[rank] = payload["status"]
                    send_msg(conn, MsgType.HEARTBEAT_ACK, {"epoch": self._epoch})
                elif msg_type == MsgType.SCRIPT_REQUEST:
                    send_msg(conn, MsgType.SCRIPT_RESPONSE, {
                        "name": self._script_name,
                        "content": self._script_content,
                    })
                elif msg_type == MsgType.REFORM_ACK:
                    epoch = payload["epoch"]
                    with self._lock:
                        event = self._reform_acks.get(epoch)
                        if event:
                            self._reform_acks[f"{epoch}_{rank}"] = True
                            remaining = [r for r in self._live_ranks
                                         if not self._reform_acks.get(f"{epoch}_{r}")]
                            if not remaining:
                                event.set()
        except (ConnectionError, OSError, socket.timeout):
            pass
        finally:
            with self._lock:
                if rank >= 0 and self._worker_socks.get(rank) is conn:
                    self._worker_socks.pop(rank, None)
            conn.close()

    def _build_stats(self) -> dict:
        with self._lock:
            return {
                "epoch": self._epoch,
                "live_ranks": list(self._live_ranks),
                "dead_ranks": list(self._dead_ranks),
                "steps": dict(self._step_counts),
                "loss": dict(self._loss),
                "sync_ms": dict(self._sync_ms),
                "hostnames": dict(self._hostnames),
                "status": dict(self._status_map),
            }

    def _broadcast_stats(self) -> None:
        stats = self._build_stats()
        with self._lock:
            socks = {r: s for r, s in self._worker_socks.items()
                     if r in self._live_ranks}
        for rank, sock in socks.items():
            try:
                send_msg(sock, MsgType.STATS_UPDATE, stats)
            except (ConnectionError, OSError):
                pass

    def _heartbeat_checker(self) -> None:
        broadcast_counter = 0
        while self._running:
            time.sleep(1.0)
            now = time.monotonic()
            with self._lock:
                dead = [r for r in self._live_ranks
                        if r in self._last_heartbeat
                        and now - self._last_heartbeat[r] > 60.0]
                stragglers = self._detect_stragglers()
            for rank in dead:
                log.warning(f"Node {rank} missed heartbeat — removing")
                self._remove_node(rank)
            for rank in stragglers:
                if rank not in dead:
                    log.warning(f"Node {rank} is a straggler — removing")
                    self._remove_node(rank)

            broadcast_counter += 1
            if broadcast_counter >= 2:
                self._broadcast_stats()
                broadcast_counter = 0

    def _detect_stragglers(self) -> list[int]:
        if len(self._step_counts) < 3:
            return []
        steps = {r: s for r, s in self._step_counts.items() if r in self._live_ranks}
        if len(steps) < 3:
            return []
        sorted_steps = sorted(steps.values())
        median = sorted_steps[len(sorted_steps) // 2]
        if median < 5:
            return []
        return [r for r, s in steps.items() if s < median - 2]

    def _remove_node(self, rank: int) -> None:
        with self._lock:
            if rank not in self._live_ranks:
                return
            self._live_ranks.remove(rank)
            self._dead_ranks.add(rank)
            self._epoch += 1
            epoch = self._epoch
            live = tuple(self._live_ranks)
            addr_map = {r: self._addr_map[r] for r in live if r in self._addr_map}
            self._last_heartbeat.pop(rank, None)
            self._worker_socks.pop(rank, None)

        membership = Membership(epoch=epoch, live_ranks=live, addr_map=addr_map)
        log.info(f"Epoch {epoch}: live_ranks={live}")
        self._broadcast_membership(membership)

    def _broadcast_membership(self, membership: Membership) -> None:
        payload = {
            "epoch": membership.epoch,
            "live_ranks": list(membership.live_ranks),
            "addr_map": membership.addr_map,
        }
        event = threading.Event()
        with self._lock:
            self._reform_acks[membership.epoch] = event
            for r in membership.live_ranks:
                for r2 in membership.live_ranks:
                    self._reform_acks.pop(f"{membership.epoch}_{r2}", None)

        with self._lock:
            socks = {r: s for r, s in self._worker_socks.items()
                     if r in membership.live_ranks}

        for rank, sock in socks.items():
            try:
                send_msg(sock, MsgType.MEMBERSHIP_UPDATE, payload)
            except (ConnectionError, OSError):
                log.warning(f"Failed to send membership update to rank {rank}")

        if event.wait(timeout=15.0):
            for rank, sock in socks.items():
                try:
                    send_msg(sock, MsgType.REFORM_COMPLETE, {"epoch": membership.epoch})
                except (ConnectionError, OSError):
                    pass
            log.info(f"Epoch {membership.epoch}: reform complete")
        else:
            log.warning(f"Epoch {membership.epoch}: reform timed out waiting for ACKs")

    def close(self) -> None:
        self._running = False
        self._sock.close()


class WorkerClient:
    def __init__(self, coordinator_host: str, coordinator_port: int, rank: int) -> None:
        self._rank = rank
        self._host = coordinator_host
        self._port = coordinator_port
        self._step = 0
        self._loss = 0.0
        self._sync_ms = 0.0
        self._status = ""
        _h = os.environ.get("GROVE_HOSTNAME", socket.gethostname())
        self._hostname = _h if "." in _h else _h + ".local"
        self._running = True
        self._lock = threading.Lock()
        self._pending_membership: Membership | None = None
        self._reform_complete: dict[int, threading.Event] = {}
        self._cluster_stats: dict | None = None
        self._script_response = None
        self._membership_event = threading.Event()
        self._script_event: threading.Event | None = None

        self._sock = self._connect(coordinator_host, coordinator_port)

        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

    def _connect(self, host: str, port: int, timeout: float = 30.0) -> socket.socket:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                configure_socket(sock, tcp=True)
                sock.settimeout(10.0)
                sock.connect((host, port))
                send_msg(sock, MsgType.HEARTBEAT, {
                    "rank": self._rank, "step": 0, "ts": time.time(),
                    "loss": 0.0, "sync_ms": 0.0, "hostname": self._hostname,
                })
                recv_msg(sock)
                return sock
            except (ConnectionRefusedError, OSError):
                time.sleep(0.1)
        raise TimeoutError(f"Could not connect to coordinator at {host}:{port}")

    def _heartbeat_loop(self) -> None:
        while self._running:
            time.sleep(2.0)
            try:
                with self._lock:
                    send_msg(self._sock, MsgType.HEARTBEAT, {
                        "rank": self._rank, "step": self._step, "ts": time.time(),
                        "loss": self._loss, "sync_ms": self._sync_ms,
                        "hostname": self._hostname, "status": self._status,
                    })
            except (ConnectionError, OSError):
                log.warning("Lost connection to coordinator")
                break

    def _recv_loop(self) -> None:
        try:
            while self._running:
                msg_type, payload = recv_msg(self._sock)
                if msg_type == MsgType.HEARTBEAT_ACK:
                    pass
                elif msg_type == MsgType.MEMBERSHIP_UPDATE:
                    membership = Membership(
                        epoch=payload["epoch"],
                        live_ranks=tuple(payload["live_ranks"]),
                        addr_map=payload.get("addr_map", {}),
                    )
                    with self._lock:
                        self._pending_membership = membership
                        self._reform_complete[membership.epoch] = threading.Event()
                    self._membership_event.set()
                elif msg_type == MsgType.REFORM_COMPLETE:
                    epoch = payload["epoch"]
                    with self._lock:
                        event = self._reform_complete.get(epoch)
                        if event:
                            event.set()
                elif msg_type == MsgType.STATS_UPDATE:
                    with self._lock:
                        self._cluster_stats = payload
                elif msg_type == MsgType.SCRIPT_RESPONSE:
                    with self._lock:
                        self._script_response = payload
                    if self._script_event:
                        self._script_event.set()
        except (ConnectionError, OSError, socket.timeout):
            log.warning("Control connection lost")
            self._membership_event.set()

    def get_pending_membership(self) -> Membership | None:
        with self._lock:
            m = self._pending_membership
            return m

    def ack_reform(self, epoch: int) -> None:
        try:
            with self._lock:
                send_msg(self._sock, MsgType.REFORM_ACK,
                         {"rank": self._rank, "epoch": epoch})
        except (ConnectionError, OSError):
            pass

    def wait_reform_complete(self, epoch: int, timeout: float = 15.0) -> bool:
        with self._lock:
            event = self._reform_complete.get(epoch)
        if event is None:
            return True
        result = event.wait(timeout)
        with self._lock:
            self._pending_membership = None
            self._reform_complete.pop(epoch, None)
        return result

    def set_stats(self, step: int, loss: float = 0.0, sync_ms: float = 0.0) -> None:
        with self._lock:
            self._step = step
            self._loss = loss
            self._sync_ms = sync_ms

    def set_status(self, msg: str) -> None:
        with self._lock:
            self._status = msg

    def fetch_script(self) -> tuple[str, str]:
        self._script_event = threading.Event()
        with self._lock:
            send_msg(self._sock, MsgType.SCRIPT_REQUEST, {})
        if not self._script_event.wait(timeout=10.0):
            raise TimeoutError("No script response from coordinator")
        with self._lock:
            resp = self._script_response
            self._script_response = None
        return resp["name"], resp["content"]

    def get_cluster_stats(self) -> dict | None:
        with self._lock:
            return self._cluster_stats

    def wait_for_membership(self, timeout: float = 30.0) -> None:
        self._membership_event.wait(timeout)

    def report_failure(self) -> None:
        pass

    def close(self) -> None:
        self._running = False
        try:
            self._sock.close()
        except OSError:
            pass


class P2PCoordinatorServer:
    def __init__(self, world_size: int, transport: "P2PTransport") -> None:
        self._world_size = world_size
        self._transport = transport
        self._lock = threading.Lock()
        self._running = True

        self._live_ranks = list(range(world_size))
        self._epoch = 0
        self._step_counts: dict[int, int] = {}
        self._loss: dict[int, float] = {}
        self._sync_ms: dict[int, float] = {}
        _h0 = socket.gethostname()
        self._hostnames: dict[int, str] = {0: _h0 if "." in _h0 else _h0 + ".local"}
        self._peer_ips: dict[int, str] = {0: get_local_ip()}
        self._status_map: dict[int, str] = {}
        self._dead_ranks: set[int] = set()
        self._start_time = time.monotonic()
        self._script_content: str = ""
        self._script_name: str = ""

        self._last_heartbeat: dict[int, float] = {}
        self._reform_acks: dict = {}
        self._local_worker: "P2PWorkerClient | None" = None

        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

        self._stats_thread = threading.Thread(target=self._stats_loop, daemon=True)
        self._stats_thread.start()

        self._checker_thread = threading.Thread(target=self._heartbeat_checker, daemon=True)
        self._checker_thread.start()

    def set_script(self, path: str) -> None:
        self._script_name = os.path.basename(path)
        with open(path) as f:
            self._script_content = f.read()

    def set_local_worker(self, worker: "P2PWorkerClient") -> None:
        self._local_worker = worker

    def _recv_loop(self) -> None:
        while self._running:
            with self._lock:
                ranks_to_poll = [r for r in self._live_ranks if r != 0]
            for rank in ranks_to_poll:
                while True:
                    try:
                        raw = self._transport.recv_ctrl(rank, timeout=0.1)
                        if raw == b"PING":
                            continue
                        msg_type, payload = decode_ctrl(raw)
                        self._handle_msg(rank, msg_type, payload)
                    except TimeoutError:
                        break
                    except (ConnectionError, OSError, ValueError) as e:
                        log.warning(f"P2P coord recv error from rank {rank}: {e}")
                        break

    def _handle_msg(self, rank: int, msg_type, payload: dict) -> None:
        if msg_type == MsgType.HEARTBEAT:
            if payload.get("data_plane_failure"):
                log.warning(f"P2P node {rank} reported data plane failure")
            with self._lock:
                self._last_heartbeat[rank] = time.monotonic()
                self._step_counts[rank] = payload.get("step", 0)
                if "loss" in payload:
                    self._loss[rank] = payload["loss"]
                if "sync_ms" in payload:
                    self._sync_ms[rank] = payload["sync_ms"]
                if "hostname" in payload:
                    self._hostnames[rank] = payload["hostname"]
                if "status" in payload:
                    self._status_map[rank] = payload["status"]
                if "ip" in payload:
                    self._peer_ips[rank] = payload["ip"]
        elif msg_type == MsgType.SCRIPT_REQUEST:
            resp = encode_ctrl(MsgType.SCRIPT_RESPONSE, {
                "name": self._script_name,
                "content": self._script_content,
            })
            self._transport.send_raw(rank, resp)
        elif msg_type == MsgType.REFORM_ACK:
            epoch = payload["epoch"]
            with self._lock:
                event = self._reform_acks.get(epoch)
                if event:
                    self._reform_acks[f"{epoch}_{rank}"] = True
                    remaining = [
                        r for r in self._live_ranks
                        if r != 0 and not self._reform_acks.get(f"{epoch}_{r}")
                    ]
                    if not remaining:
                        event.set()

    def _heartbeat_checker(self) -> None:
        while self._running:
            time.sleep(1.0)
            now = time.monotonic()

            transport_dead = set()
            if hasattr(self._transport, 'get_disconnected_peers'):
                transport_dead = self._transport.get_disconnected_peers()

            with self._lock:
                dead = [
                    r for r in self._live_ranks
                    if r != 0 and (
                        r in transport_dead
                        or (r in self._last_heartbeat and now - self._last_heartbeat[r] > 60.0)
                    )
                ]
                stragglers = self._detect_stragglers()
            for rank in dead:
                if rank in transport_dead:
                    log.warning(f"P2P node {rank} disconnected — removing")
                else:
                    log.warning(f"P2P node {rank} heartbeat timeout — removing")
                self._remove_node(rank)
            for rank in stragglers:
                if rank not in dead:
                    log.warning(f"P2P node {rank} is a straggler — removing")
                    self._remove_node(rank)

    def _detect_stragglers(self) -> list[int]:
        if len(self._step_counts) < 3:
            return []
        steps = {r: s for r, s in self._step_counts.items() if r in self._live_ranks}
        if len(steps) < 3:
            return []
        sorted_steps = sorted(steps.values())
        median = sorted_steps[len(sorted_steps) // 2]
        if median < 5:
            return []
        return [r for r, s in steps.items() if s < median - 2]

    def _remove_node(self, rank: int) -> None:
        with self._lock:
            if rank not in self._live_ranks:
                return
            self._live_ranks.remove(rank)
            self._dead_ranks.add(rank)
            self._epoch += 1
            epoch = self._epoch
            live = tuple(self._live_ranks)
            self._last_heartbeat.pop(rank, None)

        membership = Membership(epoch=epoch, live_ranks=live, addr_map={})
        log.info(f"P2P Epoch {epoch}: live_ranks={live}")
        self._broadcast_membership(membership)

    def _broadcast_membership(self, membership: Membership) -> None:
        payload = {
            "epoch": membership.epoch,
            "live_ranks": list(membership.live_ranks),
            "addr_map": membership.addr_map,
        }
        msg = encode_ctrl(MsgType.MEMBERSHIP_UPDATE, payload)

        event = threading.Event()
        with self._lock:
            self._reform_acks[membership.epoch] = event
            for r in membership.live_ranks:
                self._reform_acks.pop(f"{membership.epoch}_{r}", None)

        for rank in membership.live_ranks:
            if rank == 0:
                continue
            try:
                self._transport.send_raw(rank, msg)
            except (ConnectionError, OSError):
                log.warning(f"Failed to send membership update to rank {rank}")

        if self._local_worker is not None:
            with self._local_worker._lock:
                self._local_worker._pending_membership = membership
                self._local_worker._reform_complete[membership.epoch] = threading.Event()

        if event.wait(timeout=15.0):
            complete_msg = encode_ctrl(MsgType.REFORM_COMPLETE, {"epoch": membership.epoch})
            for rank in membership.live_ranks:
                if rank == 0:
                    continue
                try:
                    self._transport.send_raw(rank, complete_msg)
                except (ConnectionError, OSError):
                    pass
            if self._local_worker is not None:
                with self._local_worker._lock:
                    evt = self._local_worker._reform_complete.get(membership.epoch)
                    if evt:
                        evt.set()
            log.info(f"P2P Epoch {membership.epoch}: reform complete")
        else:
            log.warning(f"P2P Epoch {membership.epoch}: reform timed out")

    def _stats_loop(self) -> None:
        while self._running:
            time.sleep(2.0)
            stats = self._build_stats()
            msg = encode_ctrl(MsgType.STATS_UPDATE, stats)
            with self._lock:
                live = [r for r in self._live_ranks if r != 0]
            for rank in live:
                try:
                    self._transport.send_raw(rank, msg)
                except (ConnectionError, OSError):
                    pass

    def _build_stats(self) -> dict:
        with self._lock:
            return {
                "epoch": self._epoch,
                "live_ranks": list(self._live_ranks),
                "dead_ranks": list(self._dead_ranks),
                "steps": dict(self._step_counts),
                "loss": dict(self._loss),
                "sync_ms": dict(self._sync_ms),
                "hostnames": dict(self._hostnames),
                "ips": dict(self._peer_ips),
                "status": dict(self._status_map),
            }

    def close(self) -> None:
        self._running = False


class P2PWorkerClient:
    def __init__(self, transport: "P2PTransport", rank: int) -> None:
        self._transport = transport
        self._rank = rank
        self._step = 0
        self._loss = 0.0
        self._sync_ms = 0.0
        self._status = ""
        _h = os.environ.get("GROVE_HOSTNAME", socket.gethostname())
        self._hostname = _h if "." in _h else _h + ".local"
        self._running = True
        self._lock = threading.Lock()
        self._cluster_stats: dict | None = None
        self._script_response = None
        self._script_event: threading.Event | None = None
        self._pending_membership: Membership | None = None
        self._membership_event = threading.Event()
        self._reform_complete: dict[int, threading.Event] = {}
        self._last_coordinator_contact = time.monotonic()
        self.coordinator_alive = True

        if rank != 0:
            self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._heartbeat_thread.start()

            self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
            self._recv_thread.start()

    def _heartbeat_loop(self) -> None:
        while self._running:
            time.sleep(2.0)
            try:
                with self._lock:
                    msg = encode_ctrl(MsgType.HEARTBEAT, {
                        "rank": self._rank, "step": self._step, "ts": time.time(),
                        "loss": self._loss, "sync_ms": self._sync_ms,
                        "hostname": self._hostname, "status": self._status,
                        "ip": get_local_ip(),
                    })
                self._transport.send_raw(0, msg)
            except (ConnectionError, OSError):
                break

    def _recv_loop(self) -> None:
        while self._running:
            if time.monotonic() - self._last_coordinator_contact > 10.0:
                self.coordinator_alive = False
                log.warning("Coordinator disconnected")
                break
            try:
                raw = self._transport.recv_ctrl(0, timeout=1.0)
                self._last_coordinator_contact = time.monotonic()
                if raw == b"PING":
                    continue
                msg_type, payload = decode_ctrl(raw)
                if msg_type == MsgType.STATS_UPDATE:
                    with self._lock:
                        self._cluster_stats = payload
                elif msg_type == MsgType.SCRIPT_RESPONSE:
                    with self._lock:
                        self._script_response = payload
                    if self._script_event:
                        self._script_event.set()
                elif msg_type == MsgType.MEMBERSHIP_UPDATE:
                    membership = Membership(
                        epoch=payload["epoch"],
                        live_ranks=tuple(payload["live_ranks"]),
                        addr_map=payload.get("addr_map", {}),
                    )
                    with self._lock:
                        self._pending_membership = membership
                        self._reform_complete[membership.epoch] = threading.Event()
                    self._membership_event.set()
                elif msg_type == MsgType.REFORM_COMPLETE:
                    epoch = payload["epoch"]
                    with self._lock:
                        event = self._reform_complete.get(epoch)
                        if event:
                            event.set()
            except TimeoutError:
                continue
            except (ConnectionError, OSError, ValueError):
                continue

    def get_cluster_stats(self) -> dict | None:
        with self._lock:
            return self._cluster_stats

    def get_pending_membership(self) -> Membership | None:
        with self._lock:
            return self._pending_membership

    def ack_reform(self, epoch: int) -> None:
        if self._rank == 0:
            return
        try:
            msg = encode_ctrl(MsgType.REFORM_ACK, {"rank": self._rank, "epoch": epoch})
            self._transport.send_raw(0, msg)
        except (ConnectionError, OSError):
            pass

    def wait_reform_complete(self, epoch: int, timeout: float = 15.0) -> bool:
        with self._lock:
            event = self._reform_complete.get(epoch)
        if event is None:
            return True
        result = event.wait(timeout)
        with self._lock:
            self._pending_membership = None
            self._reform_complete.pop(epoch, None)
        return result

    def set_stats(self, step: int, loss: float = 0.0, sync_ms: float = 0.0) -> None:
        with self._lock:
            self._step = step
            self._loss = loss
            self._sync_ms = sync_ms

    def set_status(self, msg: str) -> None:
        with self._lock:
            self._status = msg

    def fetch_script(self) -> tuple[str, str]:
        self._script_event = threading.Event()
        self._transport.send_raw(0, encode_ctrl(MsgType.SCRIPT_REQUEST, {}))
        if not self._script_event.wait(timeout=10.0):
            raise TimeoutError("No script response from coordinator")
        with self._lock:
            resp = self._script_response
            self._script_response = None
        return resp["name"], resp["content"]

    def wait_for_membership(self, timeout: float = 30.0) -> None:
        self._membership_event.wait(timeout)

    def report_failure(self) -> None:
        try:
            msg = encode_ctrl(MsgType.HEARTBEAT, {
                "rank": self._rank, "step": self._step, "ts": time.time(),
                "hostname": self._hostname, "data_plane_failure": True,
            })
            self._transport.send_raw(0, msg)
        except (ConnectionError, OSError):
            pass

    def close(self) -> None:
        self._running = False
