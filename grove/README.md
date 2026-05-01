# grove

Terminal dashboard for smolcluster. Watch every node in your training run from a single pane — no SSH, no IP addresses, no config files.

Part of [smolcluster](https://github.com/YuvrajSingh-mist/smolcluster). See the [quickstart guide](https://www.smolcluster.com/quickstart.html) for full cluster setup.

## Install

From the smolcluster repo root:

```bash
source .venv/bin/activate
uv pip install -e grove/
```

## Usage

**Mac (AirDrop/mDNS — no IPs needed):**

```bash
# Coordinator (mini1):
grove start scripts/training/launch_dp_train_gpt.sh -n 3

# Each worker (mini2, mini3) — open a terminal and run:
grove join
```

**Jetson / Linux (TCP + mDNS):**

```bash
# Coordinator:
GROVE_TRANSPORT=tcp grove start train.py -n 3

# Each worker:
GROVE_TRANSPORT=tcp grove join

# Pin to a named cluster:
GROVE_TRANSPORT=tcp grove join --cluster swift-oak
```

**Single machine:**

```bash
grove run examples/distributed_train.py
```

Workers auto-discover the coordinator — on Mac over AirDrop/WiFi, on Linux/Jetson via mDNS on the shared subnet.

## Dashboard

One row per rank, updated every second:

| Rank | Host | Status | Step | Loss | Grad | tok/s | Sync | Net Mb |
|------|------|--------|------|------|------|-------|------|--------|

**Keybindings:** `l` toggles the inline log panel, `q` quits.

All smolcluster training algorithms (FSDP, ClassicDP, EDP, SyncPS, EP, GRPO) report into grove automatically — no changes to your training scripts required.

## Reporting from a custom script

```python
try:
    import grove as _grove
except ImportError:
    _grove = None

# inside your training loop:
if _grove is not None:
    _grove.report(loss, step=step, grad_norm=grad_norm, tok_per_sec=tok_per_sec)
    _grove.status("training")
```

`grove.report()` signature:

```python
grove.report(
    loss,
    step=None,
    *,
    grad_norm=None,
    tok_per_sec=None,
    tx_mbps=None,
    rx_mbps=None,
)
```

## CLI reference

```
grove run <script>              Run on a single device
grove start <script> -n N       Start coordinator, expect N total nodes
grove start <script> --name X   Use a specific cluster name
grove join [name]               Join a cluster (interactive picker if no name)
grove status                    Show nearby clusters and system info
```

## Environment variables

| Variable | Effect |
|----------|--------|
| `GROVE_TRANSPORT` | Set to `tcp` to force TCP transport (required on Linux/Jetson) |

## Requirements

- Python 3.10+
- Mac: macOS 12+ with Apple Silicon for AirDrop transport; TCP transport works on any OS
- Linux/Jetson: TCP transport (`GROVE_TRANSPORT=tcp`)

## License

MIT
