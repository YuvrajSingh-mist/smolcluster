
# Distributed Logging System

SmolCluster includes a production-grade distributed logging system that aggregates logs from all training nodes in real-time, making it easy to monitor and debug multi-node training jobs.

## Table of Contents

- [Overview](#overview)
  - [Architecture](#architecture)
  - [Key Features](#key-features)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Automatic Setup (Recommended)](#automatic-setup-recommended)
  - [Accessing Grafana](#accessing-grafana)
- [Log Structure](#log-structure)
- [Manual Setup](#manual-setup)
  - [1. Start Loki + Grafana (Controller)](#1-start-loki--grafana-controller)
  - [2. Start Promtail on Server Node](#2-start-promtail-on-server-node)
  - [3. Start Promtail on Worker Nodes](#3-start-promtail-on-worker-nodes)
  - [4. Verify Promtail is Running](#4-verify-promtail-is-running)
- [Grafana Query Examples](#grafana-query-examples)
- [Configuration Files](#configuration-files)

---

## Overview

The logging system is built on the Grafana observability stack:

- **Grafana** (http://localhost:3000): Web-based UI for log visualization and exploration
- **Loki**: High-performance log aggregation backend that receives and indexes logs
- **Promtail**: Lightweight log shipper that runs on each training node and forwards logs to Loki

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Controller Machine (10.10.0.4)                             │
│  ┌──────────┐         ┌──────────┐                         │
│  │ Grafana  │◄────────┤   Loki   │                         │
│  │  :3000   │         │  :3100   │                         │
│  └──────────┘         └─────▲────┘                         │
│                             │                               │
└─────────────────────────────┼───────────────────────────────┘
                              │
                              │ HTTP Push
                              │
              ┌───────────────┼───────────────┐
              │               │               │
        ┌─────▼─────┐   ┌─────▼─────┐  ┌─────▼─────┐
        │ Promtail  │   │ Promtail  │  │ Promtail  │
        │  mini1    │   │  mini2    │  │  mini3    │
        └─────▲─────┘   └─────▲─────┘  └─────▲─────┘
              │               │              │
        ┌─────┴─────┐   ┌─────┴─────┐  ┌─────┴─────┐
        │Server logs│   │Worker logs│  │Worker logs│
        │/tmp/...   │   │/tmp/...   │  │/tmp/...   │
        └───────────┘   └───────────┘  └───────────┘
```

### Key Features

- **Real-time Log Streaming**: See logs from all nodes as they happen
- **Structured Logging**: Automatic label extraction (job, host, rank, level)
- **Powerful Querying**: Filter, search, and correlate logs across nodes using LogQL
- **Timezone Handling**: Automatic conversion from local timezone (IST) to UTC
- **Auto-start Integration**: Launch scripts automatically start logging infrastructure
- **No Code Changes**: Works with existing Python logging without modifications

## Quick Start

### Prerequisites

1. **Docker** (controller machine only):
   ```bash
   # macOS
   brew install --cask docker
   
   # Linux
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   
   # Start and verify
   docker --version
   ```

2. **Promtail** (all training nodes):
   ```bash
   # macOS (via Homebrew)
   brew install promtail
   
   # Linux (manual installation)
   curl -O -L "https://github.com/grafana/loki/releases/download/v2.9.0/promtail-linux-amd64.zip"
   unzip promtail-linux-amd64.zip
   chmod +x promtail-linux-amd64
   sudo mv promtail-linux-amd64 /usr/local/bin/promtail
   
   # Verify installation
   promtail --version
   ```

### Automatic Setup (Recommended)

The easiest way is to use the launch scripts - they handle everything:

```bash
bash launch_edp_gpt.sh
```

The launch script automatically:
1. Starts Grafana + Loki in Docker containers on your controller machine
2. Detects Promtail installation on each remote node
3. Starts Promtail with the correct configuration
4. Configures timezone handling (IST → UTC conversion)
5. Cleans up old position files for fresh log ingestion

### Accessing Grafana

1. Open your browser to http://localhost:3000
2. Login with default credentials:
   - Username: `admin`
   - Password: `admin`
   - (You'll be prompted to change the password on first login)
3. Navigate to **Explore** (compass icon in sidebar)
4. Select **Loki** as the data source


## Log Structure

Logs are automatically parsed and labeled with the following metadata:

| Label | Description | Example Values |
|-------|-------------|----------------|
| `job` | Type of process | `smolcluster-server`, `smolcluster-worker` |
| `component` | Component type | `server`, `worker` |
| `host` | Hostname/identifier | `server-mini1`, `worker-rank0-mini2` |
| `rank` | Worker rank | `0`, `1`, `2` |
| `level` | Log level | `INFO`, `ERROR`, `WARNING`, `DEBUG` |
| `filename` | Source log file | `/tmp/smolcluster-logs/worker-rank0-mini2.log` |

## Manual Setup 

If you need more control over the logging stack:

### 1. Start Loki + Grafana (Controller)

```bash
cd logging
docker-compose up -d

# Verify containers are running
docker ps | grep -E "loki|grafana"

# Check Loki is ready
curl http://localhost:3100/ready
```

### 2. Start Promtail on Server Node

```bash
ssh mini1 "cd ~/Desktop/smolcluster && \
  export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:\$HOME/bin:\$PATH && \
  nohup promtail -config.file=logging/promtail-server-remote.yaml > /tmp/promtail.log 2>&1 &"
```

### 3. Start Promtail on Worker Nodes

```bash
# Worker on mini2
ssh mini2 "cd ~/Desktop/smolcluster && \
  export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:\$HOME/bin:\$PATH && \
  nohup promtail -config.file=logging/promtail-worker-remote.yaml > /tmp/promtail.log 2>&1 &"

# Worker on mini3
ssh mini3 "cd ~/Desktop/smolcluster && \
  export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:\$HOME/bin:\$PATH && \
  nohup promtail -config.file=logging/promtail-worker-remote.yaml > /tmp/promtail.log 2>&1 &"
```

### 4. Verify Promtail is Running

```bash
# Check process
ssh mini2 "ps aux | grep promtail | grep -v grep"

# Check logs for errors
ssh mini2 "tail -20 /tmp/promtail.log"

# Check metrics endpoint
ssh mini2 "curl -s localhost:9080/metrics | grep promtail_sent_entries"
```
