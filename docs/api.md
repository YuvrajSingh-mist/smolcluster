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
  - [POST /query](#post-query)
  - [POST /chat](#post-chat)
  - [POST /reconnect](#post-reconnect)
- [Streaming Format](#streaming-format)
- [Examples](#examples)
  - [MP Request (plain prompt)](#mp-request-plain-prompt)
  - [DP Request (instruction messages + worker rank)](#dp-request-instruction-messages--worker-rank)
  - [Restore Session History](#restore-session-history)
- [Errors and Notes](#errors-and-notes)

---

## Overview

The SmolCluster chat backend exposes one unified HTTP API for distributed inference:

- **MP (Model Parallelism)**: request is routed through model-sharded ranks.
- **DP (Data Parallelism, SyncPS/ClassicDP)**: request targets a selected worker rank.

The API streams generation tokens using **Server-Sent Events (SSE)**.

## Base URL

Default local endpoints:

- API: `http://localhost:8080`
- Health: `http://localhost:8080/health`

Ports are read from `cluster_config_inference.yaml`.

## Backend Modes

Backend mode is selected by launch configuration:

- `model_parallelism`
- `data_parallelism`

Inspect active mode with `GET /config`.

## Endpoints

### GET /

Simple service check.

**Response**
```json
{
  "status": "ok",
  "service": "SmolCluster Chat API"
}
```

### GET /health

Checks API connection to inference server.

**Response (healthy)**
```json
{
  "status": "connected",
  "healthy": true
}
```

### GET /config

Returns runtime configuration for frontend and clients.

Includes:

- API and frontend ports
- backend/algorithm/architecture metadata
- model metadata
- available worker ranks (DP)
- memory backend status
- active decoding defaults

**Response (example)**
```json
{
  "api_port": 8080,
  "frontend_port": 5050,
  "server_host": "10.10.0.1",
  "server_port": 65432,
  "inference_backend": "data_parallelism",
  "inference_algorithm": "syncps",
  "inference_architecture": "data_parallelism",
  "model_name": "meta-llama/Llama-3.2-1B-Instruct",
  "model_architecture": "meta-llama/Llama-3.2-1B-Instruct",
  "available_worker_ranks": [1, 2],
  "memory_enabled": true,
  "memory_backend": "redis_vector",
  "redis_url": "redis://localhost:6379/0",
  "memory_error": null,
  "is_instruction_based": true,
  "max_new_tokens": 256,
  "decoding_strategy": "top_p",
  "temperature": 0.6,
  "top_p": {
    "temperature": 0.6,
    "p": 0.9
  }
}
```

### GET /memory/status

Returns Redis memory availability.

**Response**
```json
{
  "enabled": true,
  "backend": "redis_vector",
  "redis_url": "redis://localhost:6379/0",
  "error": null
}
```

### GET /memory/history

Fetches stored turns for a session (for UI refresh/session restore).

**Query params**

- `session_id` (required): memory session key
- `limit` (optional, default 100): max messages returned

**Response**
```json
{
  "enabled": true,
  "session_id": "browser-session-1",
  "messages": [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"}
  ]
}
```

### POST /chat

Main inference endpoint. Returns SSE stream.

**Request body**

- `text` (optional): plain prompt (base models)
- `messages` (optional): chat messages for instruction models
- `max_tokens` (optional)
- `temperature` (optional)
- `top_p` (optional)
- `top_k` (optional)
- `decoding_strategy` (optional): `greedy`, `sampling`, `top_p`, `top_k`
- `worker_rank` (optional): DP worker to query
- `session_id` (optional, default `default`): memory session key
- `use_memory` (optional, default `true`): retrieve/store memory

At least one of `text` or `messages` must be provided.

### POST /query

Plain (non-SSE) inference endpoint for direct JSON-style querying.

- No streaming response.
- No Redis memory retrieval or storage.

Uses the same request body fields as `POST /chat` (`text`/`messages`, generation params, optional `worker_rank`).

**Response**
```json
{
  "generated_text": "final generated text",
  "success": true,
  "error": null,
  "worker_rank": 1,
  "total_time_ms": 2100.4,
  "time_to_first_token_ms": 132.0,
  "tokens_per_second": 43.8,
  "num_tokens": 92
}
```

### POST /reconnect

Forces API socket reconnect to inference server.

**Response**
```json
{
  "status": "reconnected"
}
```

## Streaming Format

`POST /chat` uses SSE with `data: <json>` lines.

Token event:
```json
{"token": "hello", "done": false}
```

Final event:
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

Error event:
```json
{"error": "message", "done": true}
```

## Examples

### MP Request (plain prompt)

```bash
curl -N -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Explain model parallelism in simple terms",
    "max_tokens": 120,
    "decoding_strategy": "top_k",
    "top_k": 40,
    "session_id": "mp-demo"
  }'
```

### DP Request (instruction messages + worker rank)

```bash
curl -N -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Summarize attention mechanism"}
    ],
    "worker_rank": 2,
    "max_tokens": 160,
    "decoding_strategy": "top_p",
    "top_p": 0.9,
    "session_id": "dp-worker-2",
    "use_memory": true
  }'
```

### Plain JSON Query (no SSE)

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Give me 3 bullet points about data parallelism",
    "worker_rank": 1,
    "max_tokens": 120,
    "decoding_strategy": "top_p",
    "top_p": 0.9
  }'
```

### Restore Session History

```bash
curl "http://localhost:8080/memory/history?session_id=dp-worker-2&limit=50"
```

## Errors and Notes

- If a selected worker is unavailable in DP mode, `POST /chat` emits an error event.
- If Redis is unavailable, chat still works; memory endpoints return disabled/empty state.
- If client disconnects during streaming, generation may continue server-side and complete.
- For long-running sessions, use distinct `session_id` values per conversation scope.
