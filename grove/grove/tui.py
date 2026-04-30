"""Terminal UI components."""

import io
import time
import threading
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Static, RichLog
from textual.binding import Binding
from rich.text import Text


class LogCapture(io.TextIOBase):
    def __init__(self, max_lines: int = 200) -> None:
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._max = max_lines
        self._new_lines: list[str] = []

    def write(self, s: str) -> int:
        if not s or s == "\n":
            return len(s) if s else 0
        with self._lock:
            for line in s.splitlines():
                if line.strip():
                    self._lines.append(line)
                    self._new_lines.append(line)
                    if len(self._lines) > self._max:
                        self._lines.pop(0)
        return len(s)

    def flush(self) -> None:
        pass

    def drain_new(self) -> list[str]:
        with self._lock:
            lines = list(self._new_lines)
            self._new_lines.clear()
            return lines


class ClusterRow:
    def __init__(self, data: dict):
        self.name = data.get("name", "?")
        self.uid = data.get("uid", "?")
        self.current = int(data.get("current", 0))
        self.expected = int(data.get("expected", 0))
        self.hostname = data.get("hostname", "?")
        self.started = float(data.get("started", time.time()))
        self.data = data

    @property
    def age(self) -> str:
        ago = time.time() - self.started
        if ago < 60:
            return f"{int(ago)}s"
        elif ago < 3600:
            return f"{int(ago // 60)}m"
        return f"{int(ago // 3600)}h"

    @property
    def full(self) -> bool:
        return self.current >= self.expected


class JoinApp(App):
    COMMANDS = set()
    CSS = """
    Screen { background: $surface; }
    #title { padding: 1 2; }
    #hint { color: $text-muted; padding: 0 2 1 2; }
    DataTable { height: 1fr; }
    DataTable > .datatable--cursor { background: $accent; color: $text; }
    """

    BINDINGS = [Binding("q", "quit", "Quit", show=False)]

    def __init__(self, browser: "LiveBrowser"):
        super().__init__()
        self._browser = browser
        self.selected_cluster: dict | None = None
        self._clusters: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Static("[b]grove[/]  select a cluster", id="title")
        table = DataTable(id="clusters")
        table.cursor_type = "row"
        table.add_columns("Name", "ID", "Nodes", "Host", "Age")
        yield table
        yield Static("  [dim]↑↓ select   enter join   q quit[/]", id="hint")

    def on_mount(self) -> None:
        self.set_interval(1.0, self._refresh_table)
        self._refresh_table()
        self.query_one("#clusters", DataTable).focus()

    def _refresh_table(self) -> None:
        self._clusters = self._browser.get_clusters()
        table = self.query_one("#clusters", DataTable)
        cursor = table.cursor_row
        table.clear()
        for c in self._clusters:
            row = ClusterRow(c)
            style = "dim" if row.full else ""
            nodes_style = "dim" if row.full else "green"
            table.add_row(
                Text(row.name, style=f"bold {style}" if not row.full else style),
                Text(row.uid, style="cyan dim"),
                Text(f"{row.current}/{row.expected}", style=nodes_style),
                Text(row.hostname, style=style),
                Text(row.age, style="dim"),
            )
        if cursor is not None and self._clusters:
            table.move_cursor(row=min(cursor, len(self._clusters) - 1))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        idx = event.cursor_row
        if idx < len(self._clusters):
            self.selected_cluster = self._clusters[idx]
        self.exit()

    def action_quit(self) -> None:
        self.selected_cluster = None
        self.exit()



def _format_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


class DashboardApp(App):
    COMMANDS = set()
    CSS = """
    Screen { background: $surface; layout: vertical; }
    #header { padding: 1 2 0 2; }
    #nodes { height: auto; max-height: 50%; margin: 0 1; }
    #stats { color: $text-muted; padding: 0 2; }
    #logs { min-height: 3; height: 1fr; margin: 0 1; }
    #footer { dock: bottom; padding: 0 2; color: $text-muted; }
    """

    BINDINGS = [Binding("q", "quit", "Quit")]

    def __init__(
        self,
        get_state: "Callable",
        cluster_name: str,
        uid: str,
        my_rank: int | None = None,
        log_capture: "LogCapture | None" = None,
        done_event: "threading.Event | None" = None,
    ):
        super().__init__()
        self._get_state = get_state
        self._cluster = cluster_name
        self._uid = uid
        self._my_rank = my_rank
        self._log_capture = log_capture
        self._done_event = done_event
        self._training_done = False
        self._start_time = time.monotonic()

    def compose(self) -> ComposeResult:
        role = f"rank {self._my_rank}" if self._my_rank is not None else "coordinator"
        yield Static(
            f"[b]grove[/]  {self._cluster}  [dim cyan]{self._uid}[/]  {role}",
            id="header",
        )
        table = DataTable(id="nodes")
        table.cursor_type = "none"
        table.add_columns("Rank", "Host", "Status", "Step", "Loss", "Sync")
        yield table
        yield Static("", id="stats")
        log = RichLog(id="logs", wrap=True, markup=True)
        log.show_vertical_scrollbar = False
        yield log
        yield Static("  [dim]q quit[/]", id="footer")

    def on_mount(self) -> None:
        self.set_interval(1.0, self._refresh)
        self._refresh()

    def _refresh(self) -> None:
        self._refresh_header()
        self._refresh_table()
        self._refresh_logs()

    def _refresh_header(self) -> None:
        elapsed = _format_elapsed(time.monotonic() - self._start_time)
        role = f"rank {self._my_rank}" if self._my_rank is not None else "coordinator"
        if self._training_done:
            self.query_one("#header", Static).update(
                f"[bold green]training complete[/]  {self._cluster}  [dim]{elapsed}[/]  press q to exit"
            )
        else:
            self.query_one("#header", Static).update(
                f"[b]grove[/]  {self._cluster}  [dim cyan]{self._uid}[/]  {role}  [dim]{elapsed}[/]"
            )
        if not self._training_done and self._done_event and self._done_event.is_set():
            self._training_done = True

    def _refresh_table(self) -> None:
        state = self._get_state()
        if not state:
            return
        if callable(state):
            state = state()

        table = self.query_one("#nodes", DataTable)
        table.clear()

        live_ranks = state.get("live_ranks", [])
        dead_ranks = state.get("dead_ranks", [])
        all_ranks = sorted(set(live_ranks) | set(dead_ranks))
        steps = state.get("steps", {})
        losses = state.get("loss", {})
        syncs = state.get("sync_ms", {})
        hostnames = state.get("hostnames", {})
        epoch = state.get("epoch", 0)

        def _get(d, rank):
            return d.get(str(rank), d.get(rank, d.get(int(rank), None)))

        for rank in all_ranks:
            hostname = _get(hostnames, rank) or f"node-{rank}"
            is_me = self._my_rank is not None and int(rank) == int(self._my_rank)
            is_dead = int(rank) in [int(d) for d in dead_ranks]

            if is_dead:
                step_val = _get(steps, rank) or "—"
                table.add_row(
                    Text(str(rank), style="dim"),
                    Text(str(hostname), style="dim"),
                    Text("dead", style="red"),
                    Text(str(step_val), style="dim"),
                    Text("—", style="dim"),
                    Text("—", style="dim"),
                )
            else:
                step_val = _get(steps, rank) or 0
                loss_val = _get(losses, rank)
                sync_val = _get(syncs, rank)
                loss_str = f"{float(loss_val):.4f}" if loss_val else "—"
                sync_str = f"{float(sync_val):.0f}ms" if sync_val else "—"
                style = "bold cyan" if is_me else ""
                marker = " ◀" if is_me else ""

                table.add_row(
                    Text(str(rank), style=style),
                    Text(str(hostname) + marker, style=style),
                    Text("ok", style="green"),
                    Text(str(step_val), style=style),
                    Text(loss_str, style=style),
                    Text(sync_str, style=style),
                )

        n_live = len(live_ranks)
        n_dead = len(dead_ranks)

        parts = [f"  {n_live} node{'s' if n_live != 1 else ''}"]
        if n_dead:
            parts.append(f"  {n_dead} dead")
        if len(live_ranks) > 1:
            mesh = " ─ ".join(str(r) for r in live_ranks)
            parts.append(f"  {mesh}  (all-to-all)")
        if epoch:
            parts.append(f"  epoch {epoch}")

        self.query_one("#stats", Static).update(Text("".join(parts), style="dim"))

    def _refresh_logs(self) -> None:
        if self._log_capture is None:
            return
        log_widget = self.query_one("#logs", RichLog)
        for line in self._log_capture.drain_new():
            log_widget.write(line)

    def action_quit(self) -> None:
        self.exit()
