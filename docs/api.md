# Inference API Reference (MP + DP)

Complete API reference for the chat/inference backend used by Model Parallelism (MP) and Data Parallelism (DP) modes.

## Table of Contents

- [Overview](#overview)
- [Base URL](#base-url)
- [Backend Modes](#backend-modes)
- [Endpoints](#endpoints)
  - [GET /](#get-)
  - [GET /health](#get-health)
  - [GET /config](#get-config)
  - [GET /memory/status](#get-memorystatus)
  - [GET /memory/history](#get-memoryhistory)
  - [POST /memory/clear](#post-memoryclear)
  - [POST /chat](#post-chat)
  - [POST /query](#post-query)
  - [POST /reconnect](#post-reconnect)
- [Streaming Format](#streaming-format)
- [curl Cookbook by Algorithm](#curl-cookbook-by-algorithm)
  - [MP — Model Parallelism](#mp--model-parallelism)
    - [Health and config](#mp-health-and-config)
    - [Stream generation (plain prompt)](#mp-stream-generation-plain-prompt)
    - [Stream generation (instruction format)](#mp-stream-generation-instruction-format)
    - [Non-streaming JSON query](#mp-non-streaming-json-query)
    - [Session history](#mp-session-history)
  - [SyncPS — Synchronous Parameter Server](#syncps--synchronous-parameter-server)
    - [Health and config](#syncps-health-and-config)
    - [Stream generation with worker rank](#syncps-stream-generation-with-worker-rank)
    - [Stream generation (instruction format)](#syncps-stream-generation-instruction-format)
    - [Non-streaming JSON query](#syncps-non-streaming-json-query)
    - [Session history and clear](#syncps-session-history-and-clear)
  - [ClassicDP — Classic Data Parallelism](#classicdp--classic-data-parallelism)
    - [Health and config](#classicdp-health-and-config)
    - [Stream generation with worker rank](#classicdp-stream-generation-with-worker-rank)
    - [Stream generation (instruction format)](#classicdp-stream-generation-instruction-format)
    - [Non-streaming JSON query](#classicdp-non-streaming-json-query)
    - [Session history and clear](#classicdp-session-history-and-clear)
- [Errors and Notes](#errors-and-notes)

---

## Overview

The SmolCluster chat backend exposes one unified HTTP API for distributed inference:

- **MP (Model Parallelism)** — request is routed through model-sharded ranks.
- **SyncPS (Synchronous Parameter Server)** — request targets a selected worker rank; server holds a single parameter copy.
- **ClassicDP (Classic Data Parallelism)** — each worker holds a full model copy; request targets a specific worker rank.

The API streams generation tokens using **Server-Sent Events (SSE)** on `POST /chat`.

## Base URL

```
http://localhost:8080
```

The port is read from `cluster_config_inference.yaml → web_interface.api_port` (default `8080`).
The HTML frontend runs on port `5050` by default.

## Backend Modes

Set via environment variables injected by the launch scripts:

| Algorithm | `INFERENCE_BACKEND` | `INFERENCE_ALGORITHM` |
|-----------|--------------------|-----------------------|
| MP        | `model_parallelism` | `mp` |
| SyncPS    | `data_parallelism`  | `syncps` |
| ClassicDP | `classicdp`         | `classicdp` |

Inspect the active mode at runtime with `GET /config`.

## Endpoints

### GET /

Simple service check.

```bash
curl http://localhost:8080/
```

**Response**
```json
{
  "status": "ok",
  "service": "SmolCluster Chat API"
}
```

---

### GET /health

Checks API connection to the inference server socket.

```bash
curl http://localhost:8080/health
```

**Response (healthy)**
```json
{
  "status": "connected",
  "healthy": true
}
```

---

### GET /config

Returns full runtime configuration including ports, backend mode, model metadata, available worker ranks, memory backend, and decoding defaults.

```bash
curl http://localhost:8080/config
```

**Response (example — SyncPS)**
```json
{
  "api_port": 8080,
  "frontend_port": 5050,
  "server_host": "10.10.1.1",
  "server_port": 65432,
  "inference_backend": "data_parallelism",
  "inference_algorithm": "syncps",
  "inference_architecture": "data_parallelism",
  "model_name": "meta-llama/Llama-3.2-1B-Instruct",
  "available_worker_ranks": [1, 2],
  "memory_enabled": true,
  "memory_backend": "redis_vector",
  "redis_url": "redis://localhost:6379/0",
  "max_new_tokens": 256,
  "decoding_strategy": "top_p",
  "temperature": 0.6,
  "top_p": { "temperature": 0.6, "p": 0.9 }
}
```

---

### GET /memory/status

Returns Redis memory availability.

```bash
curl http://localhost:8080/memory/status
```

**Response**
```json
{
  "enabled": true,
  "backend": "redis_vector",
  "redis_url": "redis://localhost:6379/0",
  "error": null
}
```

---

### GET /memory/history

Fetches stored turns for a session (for UI refresh/session restore).

**Query parameters**

| Parameter  | Required | Default | Description |
|------------|----------|---------|-------------|
| `session_id` | yes    | —       | Memory session key |
| `limit`    | no       | `100`   | Max messages returned |

```bash
curl "http://localhost:8080/memory/history?session_id=my-session&limit=50"
```

**Response**
```json
{
  "enabled": true,
  "session_id": "my-session",
  "messages": [
    {"role": "user",      "content": "Hi"},
    {"role": "assistant", "content": "Hello!"}
  ]
}
```

---

### POST /memory/clear

Deletes stored history for one or more sessions.

```bash
curl -X POST http://localhost:8080/memory/clear \
  -H "Content-Type: application/json" \
  -d '{"session_ids": ["my-session"]}'
```

---

### POST /chat

Main inference endpoint. Returns an **SSE stream** of tokens.

**Request body fields**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | one of `text`/`messages` | Plain prompt (base models) |
| `messages` | array | one of `text`/`messages` | Chat messages `[{"role":"user","content":"..."}]` |
| `max_tokens` | int | no | Override max generation tokens |
| `temperature` | float | no | Sampling temperature |
| `top_p` | float | no | Top-p nucleus sampling |
| `top_k` | int | no | Top-k sampling |
| `decoding_strategy` | string | no | `greedy`, `sampling`, `top_p`, `top_k` |
| `worker_rank` | int | no | DP worker to query (SyncPS/ClassicDP) |
| `session_id` | string | no | Memory session key (default: `"default"`) |
| `use_memory` | bool | no | Store/retrieve conversation memory (default: `true`) |

---

### POST /query

Plain non-streaming inference. Same request body as `POST /chat`. Returns a single JSON object — no SSE, no memory read/write.

**Response**
```json
{
  "generated_text": "...",
  "success": true,
  "error": null,
  "worker_rank": 1,
  "total_time_ms": 2100.4,
  "time_to_first_token_ms": 132.0,
  "tokens_per_second": 43.8,
  "num_tokens": 92
}
```

---

### POST /reconnect

Forces API socket reconnect to the inference server.

```bash
curl -X POST http://localhost:8080/reconnect
```

**Response**
```json
{ "status": "reconnected" }
```

---

## Streaming Format

`POST /chat` uses SSE with `data: <json>` lines.

**Token event**
```json
{"token": "hello", "done": false}
```

**Final event**
```json
{
  "done": true,
  "full_text": "final generated text",
  "worker_rank": 1,
  "total_time_ms": 2100.4,
  "time_to_first_token_ms": 132.0,
  "tokens_per_second": 43.8,
  "num_tokens": 92
}
```

**Error event**
```json
{"error": "message", "done": true}
```

---

## curl Cookbook by Algorithm

All examples assume the API is running locally on port `8080`.
Use `curl -N` (no-buffer) to receive SSE tokens as they stream.

---

### MP — Model Parallelism

Launched with `INFERENCE_BACKEND=model_parallelism INFERENCE_ALGORITHM=mp`.
The model is sharded across ranks; no `worker_rank` is needed.

#### MP — Health and config

```bash
curl http://localhost:8080/health
curl http://localhost:8080/config
```

#### MP — Stream generation (plain prompt)

```bash
curl -N -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Explain model parallelism in simple terms",
    "max_tokens": 120,
    "decoding_strategy": "top_k",
    "top_k": 40,
    "session_id": "mp-session-1",
    "use_memory": true
  }'
```

#### MP — Stream generation (instruction format)

```bash
curl -N -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a concise ML tutor."},
      {"role": "user",   "content": "What are the trade-offs of pipeline vs tensor parallelism?"}
    ],
    "max_tokens": 160,
    "decoding_strategy": "top_p",
    "top_p": 0.9,
    "temperature": 0.7,
    "session_id": "mp-session-1",
    "use_memory": true
  }'
```

#### MP — Non-streaming JSON query

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "text": "List 3 advantages of model parallelism",
    "max_tokens": 96,
    "decoding_strategy": "greedy"
  }'
```

#### MP — Session history

```bash
# Fetch last 50 turns
curl "http://localhost:8080/memory/history?session_id=mp-session-1&limit=50"

# Clear session
curl -X POST http://localhost:8080/memory/clear \
  -H "Content-Type: application/json" \
  -d '{"session_ids": ["mp-session-1"]}'
```

---

### SyncPS — Synchronous Parameter Server

Launched with `INFERENCE_BACKEND=data_parallelism INFERENCE_ALGORITHM=syncps`.
A dedicated parameter server (rank 0) holds weights; workers (rank 1, 2, …) run inference.
Specify `worker_rank` to target a specific worker.

#### SyncPS — Health and config

```bash
curl http://localhost:8080/health
curl http://localhost:8080/config
# Check available_worker_ranks in the response to see which ranks are live
```

#### SyncPS — Stream generation with worker rank

```bash
curl -N -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Explain parameter server architecture",
    "worker_rank": 1,
    "max_tokens": 128,
    "decoding_strategy": "top_p",
    "top_p": 0.9,
    "temperature": 0.7,
    "session_id": "syncps-worker-1",
    "use_memory": true
  }'
```

```bash
curl -N -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "text": "What is gradient synchronization?",
    "worker_rank": 2,
    "max_tokens": 128,
    "session_id": "syncps-worker-2",
    "use_memory": true
  }'
```

#### SyncPS — Stream generation (instruction format)

```bash
curl -N -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a distributed-systems expert."},
      {"role": "user",   "content": "Compare SyncPS and ClassicDP inference."}
    ],
    "worker_rank": 1,
    "max_tokens": 200,
    "decoding_strategy": "top_p",
    "top_p": 0.9,
    "session_id": "syncps-worker-1",
    "use_memory": true
  }'
```

#### SyncPS — Non-streaming JSON query

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Give 3 bullet points on synchronous gradient updates",
    "worker_rank": 1,
    "max_tokens": 96,
    "decoding_strategy": "top_p",
    "top_p": 0.9
  }'
```

#### SyncPS — Session history and clear

```bash
# Fetch stored turns for a worker session
curl "http://localhost:8080/memory/history?session_id=syncps-worker-1&limit=50"

# Clear one session
curl -X POST http://localhost:8080/memory/clear \
  -H "Content-Type: application/json" \
  -d '{"session_ids": ["syncps-worker-1"]}'

# Clear all worker sessions at once
curl -X POST http://localhost:8080/memory/clear \
  -H "Content-Type: application/json" \
  -d '{"session_ids": ["syncps-worker-1", "syncps-worker-2"]}'
```

---

### ClassicDP — Classic Data Parallelism

Launched with `INFERENCE_BACKEND=classicdp INFERENCE_ALGORITHM=classicdp`.
Every worker hosts a full model replica (rank 0, 1, 2, …). Use `worker_rank` to
target a specific replica. Requests load-balance across replicas when no rank is specified.

#### ClassicDP — Health and config

```bash
curl http://localhost:8080/health
curl http://localhost:8080/config
# available_worker_ranks lists all live ranks (e.g. [0, 1, 2])
```

#### ClassicDP — Stream generation with worker rank

```bash
curl -N -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Explain data parallelism in one paragraph",
    "worker_rank": 1,
    "max_tokens": 128,
    "decoding_strategy": "top_p",
    "top_p": 0.9,
    "temperature": 0.7,
    "session_id": "classicdp-worker-1",
    "use_memory": true
  }'
```

```bash
curl -N -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "text": "What is ring-allreduce?",
    "worker_rank": 2,
    "max_tokens": 128,
    "session_id": "classicdp-worker-2",
    "use_memory": true
  }'
```

#### ClassicDP — Stream generation (instruction format)

```bash
curl -N -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a concise ML tutor."},
      {"role": "user",   "content": "Compare SyncPS and ClassicDP briefly."}
    ],
    "worker_rank": 2,
    "max_tokens": 160,
    "decoding_strategy": "top_p",
    "top_p": 0.9,
    "session_id": "classicdp-worker-2",
    "use_memory": true
  }'
```

#### ClassicDP — Non-streaming JSON query

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Give 3 points on gradient synchronization",
    "worker_rank": 1,
    "max_tokens": 96,
    "decoding_strategy": "top_p",
    "top_p": 0.9
  }'
```

#### ClassicDP — Session history and clear

```bash
# Fetch stored turns for a worker session
curl "http://localhost:8080/memory/history?session_id=classicdp-worker-1&limit=50"

# Clear one session
curl -X POST http://localhost:8080/memory/clear \
  -H "Content-Type: application/json" \
  -d '{"session_ids": ["classicdp-worker-1"]}'

# Clear all ClassicDP worker sessions at once
curl -X POST http://localhost:8080/memory/clear \
  -H "Content-Type: application/json" \
  -d '{"session_ids": ["classicdp-worker-1", "classicdp-worker-2"]}'
```

---

## Errors and Notes

- If a selected worker rank is unavailable in DP mode, `POST /chat` emits `{"error": "...", "done": true}`.
- If Redis is unavailable, chat still works; memory endpoints return a disabled/empty state.
- If the client disconnects during streaming, generation may continue server-side until completion.
- For long-running sessions use distinct `session_id` values per conversation scope.
- `POST /query` bypasses Redis memory entirely — use it for stateless one-shot lookups.
- Available worker ranks for the active run are always listed in `GET /config → available_worker_ranks`.
