
# Distributed Logging

SmolCluster writes structured logs from every training node into a shared directory and exposes them through the dashboard.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Log Files](#log-files)
  - [Location](#location)
  - [Naming Convention](#naming-convention)
  - [Line Format](#line-format)
- [Dashboard Monitoring](#dashboard-monitoring)
- [Configuration Files](#configuration-files)

---

## Overview

Every launch script calls `start_logging_stack` which does two things:

1. Creates `logging/cluster-logs/` in the repository root (if it doesn't exist).
2. Starts Redis (required for dashboard state persistence).

Each training process — server and workers — writes a structured log file into that directory via `setup_cluster_logging()` in `src/smolcluster/utils/logging_utils.py`. No external log shipper is required for basic monitoring.

## Architecture

```
  Controller machine                  Worker nodes
  ─────────────────                  ─────────────
  ┌──────────────────┐               ┌─────────────────────┐
  │ Dashboard :8765  │◄──── Redis ───│ training metrics    │
  │ (primary UI)     │               └─────────────────────┘
  └──────────────────┘
         │
         ▼
  logging/cluster-logs/
  ├── server-mini1.log
  ├── worker-rank0-mini2.log
  └── worker-rank1-mini3.log
         ▲               ▲
  Written by server   Written by each worker
  on its own machine  on its own machine
```

> If your nodes share the repository via a network mount (NFS / Samba), all log files land in the same `logging/cluster-logs/` automatically. If nodes have independent checkouts, copy or tail logs back to the controller as needed.

## Log Files

### Location

```
<repo-root>/logging/cluster-logs/
```

`logging_utils.py` tries this path first. If it is not writable it falls back to `logging/cluster-logs-fallback/` inside the repo, then `<cwd>/smolcluster-logs/`.

### Naming Convention

| Process | File name |
|---------|-----------|
| Parameter server / leader | `server-{hostname}.log` |
| Worker rank N | `worker-rank{N}-{hostname}.log` |

Example for a three-node cluster:

```
server-mini1.log
worker-rank0-mini2.log
worker-rank1-mini3.log
```

### Line Format

Every log line follows the same structured format so it can be parsed by both humans and Promtail:

```
2026-04-06 12:34:56,789 | INFO | rank:0 | [Step 42] Loss: 2.314
```

| Field | Description |
|-------|-------------|
| timestamp | `YYYY-MM-DD HH:MM:SS,ms` |
| level | `INFO`, `WARNING`, `ERROR`, `DEBUG` |
| rank | `server` for the parameter server, integer for workers |
| message | Free-form log message |

## Dashboard Monitoring

The primary monitoring interface is the SmolCluster dashboard:

```bash
bash scripts/inference/launch_api.sh
```

Open **http://localhost:8765** in your browser. The dashboard shows:

- Live training metrics (loss, grad norm, step count) streamed via Server-Sent Events
- Node connectivity and health
- Inference token throughput when running a chat server

Redis (started automatically by `start_logging_stack`) persists the metrics state so the dashboard survives page refreshes.

---

## Configuration Files

| File | Purpose |
|------|---------|
| `scripts/lib/logging_helpers.sh` | `start_logging_stack` / `ensure_redis_running` shell helpers |
| `src/smolcluster/utils/logging_utils.py` | `setup_cluster_logging()` — Python logging initialisation |
