"""
FastAPI backend for chat application.
Handles user input and communicates with the distributed inference server.
"""

import json
import logging
import os
import socket
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator
from typing import Optional

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from smolcluster.applications.chat.backend.memory_store import RedisVectorMemory
from smolcluster.utils.common_utils import (
    get_effective_decoding_strategies,
    get_inference_metrics,
    receive_message,
    send_message,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load inference configuration
CONFIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "configs"
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "model_parallelism")
INFERENCE_ALGORITHM = os.getenv("INFERENCE_ALGORITHM", INFERENCE_BACKEND)

backend_model_cfg = (
    CONFIG_DIR / "inference" / INFERENCE_BACKEND / "model_config_inference.yaml"
)
flat_model_cfg = CONFIG_DIR / "inference" / "model_config_inference.yaml"

model_cfg_path = backend_model_cfg if backend_model_cfg.exists() else flat_model_cfg
with open(model_cfg_path) as f:
    model_configs = yaml.safe_load(f)

if INFERENCE_BACKEND == "data_parallelism":
    model_configs = model_configs.get("dp", model_configs)
    if "hf_model_name" in model_configs:
        model_config = model_configs
    else:
        model_name = model_configs.get("active_model", "causal_gpt2")
        model_config = model_configs[model_name]
else:
    model_configs = model_configs.get("mp", model_configs)
    model_name = model_configs.get("active_model", "causal_gpt2")
    model_config = model_configs[model_name]

# Load cluster config for web interface ports and server connection
with open(CONFIG_DIR / "inference" / "cluster_config_inference.yaml") as f:
    cluster_config = yaml.safe_load(f)

# Get active model config name for metadata endpoints.
MODEL_NAME = model_configs.get("active_model", "hf_model")
MODEL_DISPLAY_NAME = model_config.get("hf_model_name", MODEL_NAME)
MODEL_IDENTIFIER = model_config.get("hf_model_name") or model_config.get("weights_model_name") or MODEL_NAME
HF_TOKEN = os.environ.get("HF_TOKEN") or None

EFFECTIVE_STRATEGIES = get_effective_decoding_strategies(
    model_config,
    hf_token=HF_TOKEN,
    logger=logger,
)

configured_workers = cluster_config.get("workers", {}).get("regular", [])
worker_ranks: set[int] = set()
for worker in configured_workers:
    if not isinstance(worker, dict):
        continue
    rank = worker.get("rank")
    if rank is None:
        continue
    worker_ranks.add(int(rank))

# Include rank 0 only for algorithms that have a coordinating server at rank 0
# (SyncPS parameter server, MP model-shard coordinator). ClassicDP has no server.
if INFERENCE_ALGORITHM in ("syncps", "mp"):
    worker_ranks.add(0)
AVAILABLE_WORKER_RANKS = sorted(worker_ranks) or [0]


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Manage server connection lifecycle for FastAPI startup/shutdown."""
    logger.info("Starting FastAPI chat backend...")
    try:
        connect_to_server()
    except Exception as e:
        logger.warning(f"Could not connect to server on startup: {e}")

    yield

    logger.info("Shutting down FastAPI chat backend...")
    disconnect_from_server()


app = FastAPI(title="SmolCluster Chat API", lifespan=lifespan)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global socket connection to server
server_socket: Optional[socket.socket] = None

# Get server connection details from cluster config
server_hostname = cluster_config["server"]
SERVER_HOST = os.getenv("INFERENCE_SERVER_HOST", cluster_config["host_ip"][server_hostname])
port_config = cluster_config["port"]
if isinstance(port_config, dict):
    default_port = port_config.get(server_hostname, port_config.get("default", 65432))
else:
    default_port = port_config

SERVER_PORT = int(os.getenv("INFERENCE_SERVER_PORT", default_port))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Get web interface port from cluster config
API_PORT = cluster_config["web_interface"]["api_port"]

MAX_CONNECTION_RETRIES = 10
RETRY_DELAY = 3  # seconds

memory_store: Optional[RedisVectorMemory] = None
memory_store_error: Optional[str] = None
try:
    memory_store = RedisVectorMemory(redis_url=REDIS_URL)
    logger.info("Redis vector memory connected at %s", REDIS_URL)
except Exception as exc:
    memory_store_error = str(exc)
    logger.warning("Redis vector memory unavailable: %s", exc)


class ChatRequest(BaseModel):
    """Chat request for inference.
    
    For base models (causal_gpt2): use 'text' with a plain text prompt.
    For instruction-based models (Llama-Instruct): use 'messages' with chat format.
    
    Example for instruction-based model:
        messages = [
            {"role": "system", "content": "You are a helpful programming assistant."},
            {"role": "user", "content": "Is Rust better than Python?"},
        ]
        request = ChatRequest(messages=messages)
    
    Example for base model:
        request = ChatRequest(text="The quick brown fox")
    
    Example with specific worker or server:
        request = ChatRequest(text="Hello", worker_rank=1)  # Query worker 1
        request = ChatRequest(text="Hello", worker_rank=0)  # Query server (rank 0)
    """
    text: Optional[str] = None  # Plain text prompt for base models
    messages: Optional[list[dict[str, str]]] = None  # Messages for instruction-based models
    max_tokens: Optional[int] = None  # Will use model config default
    temperature: Optional[float] = None  # Will use model config default
    top_p: Optional[float] = None  # Will use model config default
    top_k: Optional[int] = None  # Will use model config default
    decoding_strategy: Optional[str] = None  # Will use model config default
    worker_rank: Optional[int] = None  # Specific worker/server to query (None = rank 0 server)
    session_id: Optional[str] = "default"  # Memory session identifier
    use_memory: bool = True  # Enable Redis vector memory retrieval


class ChatResponse(BaseModel):
    generated_text: str
    success: bool
    error: Optional[str] = None
    # Inference metrics
    total_time_ms: Optional[float] = None
    time_to_first_token_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None
    num_tokens: Optional[int] = None


class QueryResponse(ChatResponse):
    worker_rank: Optional[int] = None


class MemoryClearRequest(BaseModel):
    session_ids: list[str]


def connect_to_server():
    """Establish connection to Model Parallelism server with retry logic."""
    global server_socket

    for attempt in range(1, MAX_CONNECTION_RETRIES + 1):
        try:
            if server_socket is None:
                logger.info(
                    f"Attempt {attempt}/{MAX_CONNECTION_RETRIES}: Connecting to {SERVER_HOST}:{SERVER_PORT}..."
                )
                server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server_socket.settimeout(10)  # 10 second timeout
                server_socket.connect((SERVER_HOST, SERVER_PORT))
                server_socket.settimeout(None)  # Remove timeout after connection
                logger.info(f"Connected to server at {SERVER_HOST}:{SERVER_PORT}")

                # Register as client
                send_message(server_socket, ("register_client", 0))
                response = receive_message(server_socket)
                if response and response[0] == "client_registered":
                    logger.info("Successfully registered with server")
                    return server_socket
                else:
                    raise Exception(f"Failed to register with server: {response}")

            return server_socket
        except Exception as e:
            logger.warning(f"Attempt {attempt}/{MAX_CONNECTION_RETRIES} failed: {e}")
            server_socket = None

            if attempt < MAX_CONNECTION_RETRIES:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error(
                    f"Failed to connect after {MAX_CONNECTION_RETRIES} attempts"
                )
                raise HTTPException(
                    status_code=503,
                    detail=f"Server unavailable after {MAX_CONNECTION_RETRIES} attempts: {str(e)}",
                ) from e


def disconnect_from_server():
    """Close connection to server."""
    global server_socket
    if server_socket:
        try:
            send_message(server_socket, ("disconnect", None))
            server_socket.close()
        except Exception:
            pass
        server_socket = None
        logger.info("Disconnected from server")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "SmolCluster Chat API"}


@app.get("/config")
async def get_config():
    """Get API and model configuration."""
    active_strategy = model_config.get("active_decoding_strategy")
    if not isinstance(active_strategy, str) or not active_strategy:
        raise HTTPException(
            status_code=500,
            detail=(
                f"'active_decoding_strategy' in model_config_inference.yaml is missing or not a string "
                f"(got {active_strategy!r}). Set it to one of: greedy, sampling, top_p, top_k."
            ),
        )

    strategies = EFFECTIVE_STRATEGIES
    strategy_params = strategies.get(active_strategy)
    if not isinstance(strategy_params, dict):
        raise HTTPException(
            status_code=500,
            detail=(
                f"No decoding_strategies entry for '{active_strategy}' in model_config_inference.yaml. "
                f"Add a '{active_strategy}:' block under decoding_strategies."
            ),
        )

    if "temperature" not in strategy_params:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Decoding strategy '{active_strategy}' has no 'temperature' value. "
                f"Either add 'temperature:' under decoding_strategies.{active_strategy} in "
                f"model_config_inference.yaml, or set tokenizer.use_hf_defaults: true so it is "
                f"loaded from the HuggingFace generation_config (requires network + HF_TOKEN for gated models)."
            ),
        )

    return {
        "api_port": API_PORT,
        "frontend_port": cluster_config["web_interface"]["frontend_port"],
        "server_host": SERVER_HOST,
        "server_port": SERVER_PORT,
        "inference_backend": INFERENCE_BACKEND,
        "inference_algorithm": INFERENCE_BACKEND,
        "inference_architecture": INFERENCE_ALGORITHM,
        "model_name": MODEL_DISPLAY_NAME,
        "model_architecture": MODEL_IDENTIFIER,
        "available_worker_ranks": AVAILABLE_WORKER_RANKS,
        "memory_enabled": memory_store is not None,
        "memory_backend": "redis_vector",
        "redis_url": REDIS_URL,
        "memory_error": memory_store_error,
        "is_instruction_based": model_config.get("is_instruction_based", False),
        "max_new_tokens": model_config["max_new_tokens"],
        "decoding_strategy": active_strategy,
        "temperature": strategy_params["temperature"],
        f"{active_strategy}": strategy_params,
    }


@app.get("/health")
async def health():
    """Check if server connection is healthy."""
    try:
        if server_socket is None:
            return {"status": "disconnected", "healthy": False}
        return {"status": "connected", "healthy": True}
    except Exception as e:
        return {"status": "error", "healthy": False, "error": str(e)}


@app.get("/memory/status")
async def memory_status():
    """Expose Redis memory availability and connection details."""
    return {
        "enabled": memory_store is not None,
        "backend": "redis_vector",
        "redis_url": REDIS_URL,
        "error": memory_store_error,
    }


@app.get("/memory/history")
async def memory_history(session_id: str, limit: int = 100):
    """Return stored chat turns for a session for UI rehydration."""
    if memory_store is None:
        return {"enabled": False, "session_id": session_id, "messages": []}

    try:
        history = memory_store.get_session_history(session_id=session_id, limit=limit)
    except Exception as exc:
        logger.warning("Failed to fetch memory history: %s", exc)
        return {"enabled": True, "session_id": session_id, "messages": []}

    return {
        "enabled": True,
        "session_id": session_id,
        "messages": [
            {"role": item.role, "content": item.content}
            for item in history
        ],
    }


@app.post("/memory/clear")
async def memory_clear(request: MemoryClearRequest):
    """Delete stored chat history for one or more sessions."""
    if memory_store is None:
        raise HTTPException(
            status_code=503,
            detail=f"Memory backend unavailable: {memory_store_error or 'redis not connected'}",
        )

    if not request.session_ids:
        raise HTTPException(status_code=400, detail="session_ids must be a non-empty list")

    total_deleted = 0
    errors: list[str] = []
    for session_id in request.session_ids:
        if not isinstance(session_id, str) or not session_id.strip():
            errors.append("Encountered empty or non-string session_id")
            continue
        try:
            total_deleted += memory_store.clear_session_history(session_id)
        except Exception as exc:
            errors.append(f"{session_id}: {exc}")

    if errors:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to clear one or more sessions",
                "errors": errors,
                "deleted": total_deleted,
            },
        )

    return {
        "ok": True,
        "deleted": total_deleted,
        "sessions_requested": len(request.session_ids),
    }


@app.post("/chat")
async def chat(chat_request: ChatRequest, http_request: Request):
    """
    Stream tokens from model parallelism server with accurate TTFT measurement.

    Args:
        request: ChatRequest with text and generation parameters

    Returns:
        StreamingResponse with Server-Sent Events
    """

    async def generate():
        # Get inference metrics tracker
        metrics_tracker = get_inference_metrics()
        metrics_tracker.reset()

        try:
            # Validate input
            if not chat_request.text and not chat_request.messages:
                yield f"data: {json.dumps({'error': 'Either text or messages must be provided', 'done': True})}\n\n"
                return

            session_id = chat_request.session_id or "default"
            user_query = chat_request.text or ""
            if chat_request.messages and isinstance(chat_request.messages, list):
                last_user = next(
                    (
                        msg.get("content", "")
                        for msg in reversed(chat_request.messages)
                        if isinstance(msg, dict) and msg.get("role") == "user"
                    ),
                    "",
                )
                user_query = last_user or user_query

            memories = []
            if chat_request.use_memory and memory_store and user_query:
                try:
                    memories = memory_store.search(session_id, user_query, k=4)
                except Exception as e:
                    logger.warning(f"Memory search failed, continuing without context: {e}")
                    memories = []
                
                if memories:
                    memory_lines = [
                        f"- {item.role}: {item.content}"
                        for item in memories
                    ]
                    memory_block = "Relevant prior context from memory:\n" + "\n".join(memory_lines)
                    if chat_request.messages:
                        chat_request.messages = [
                            {
                                "role": "system",
                                "content": memory_block,
                            }
                        ] + chat_request.messages
                    else:
                        chat_request.text = f"{memory_block}\n\nCurrent user query:\n{chat_request.text or ''}"

            # Ensure connection
            sock = connect_to_server()
            if sock is None:
                yield f"data: {json.dumps({'error': 'Could not connect to server', 'done': True})}\n\n"
                return

            # Send inference request to server
            inference_request = {
                "command": "inference",
                "max_tokens": chat_request.max_tokens,
                "temperature": chat_request.temperature,
                "top_p": chat_request.top_p,
                "top_k": chat_request.top_k,
                "decoding_strategy": chat_request.decoding_strategy,
                "worker_rank": chat_request.worker_rank if chat_request.worker_rank is not None else 0,
            }
            selected_worker_rank = inference_request["worker_rank"]
            
            # Add prompt or messages
            if chat_request.messages:
                inference_request["messages"] = chat_request.messages
            else:
                inference_request["prompt"] = chat_request.text

            log_str = f"messages (count={len(chat_request.messages)})" if chat_request.messages else f"{(chat_request.text or '')[:50]}..."
            worker_info = f" [Worker {selected_worker_rank}]"
            logger.info(f"Sending streaming inference request{worker_info}: {log_str}")

            # Start timing
            metrics_tracker.start_inference()
            send_message(sock, ("inference", inference_request))

            # Stream tokens as they arrive
            full_text = ""
            client_disconnected = False
            
            while True:
                response = receive_message(sock)

                if response is None:
                    error_data = {"error": "Connection lost", "done": True}
                    yield f"data: {json.dumps(error_data)}\n\n"
                    break

                command, result = response

                if command == "token":
                    # Received token from worker
                    token_text = result.get("text", "")
                    token_idx = result.get("token_idx", 0)
                    
                    full_text += token_text
                    
                    # Record token for metrics
                    metrics_tracker.record_token()

                    if not client_disconnected and await http_request.is_disconnected():
                        client_disconnected = True
                        logger.info(
                            "SSE client disconnected while Worker %s is generating; continuing inference in background",
                            selected_worker_rank,
                        )

                    # Send token only while SSE client is connected
                    if not client_disconnected:
                        token_data = {
                            "token": token_text,
                            "done": False,
                        }
                        yield f"data: {json.dumps(token_data)}\n\n"

                    logger.info(f"[Worker {selected_worker_rank}] Token {token_idx}: {repr(token_text)}")

                elif command == "inference_complete":
                    # Generation complete
                    metrics_tracker.end_inference()
                    perf_metrics = metrics_tracker.get_metrics()

                    logger.info(f"Streaming complete. Metrics: {perf_metrics}")

                    worker_rank = result.get("worker_rank", 0)
                    
                    # Send final metrics
                    final_data = {
                        "done": True,
                        "full_text": full_text,
                        "worker_rank": worker_rank,
                        "total_time_ms": perf_metrics.get("total_time_ms"),
                        "time_to_first_token_ms": perf_metrics.get(
                            "time_to_first_token_ms"
                        ),
                        "tokens_per_second": perf_metrics.get("tokens_per_second"),
                        "num_tokens": perf_metrics.get("num_tokens"),
                    }
                    if not client_disconnected:
                        yield f"data: {json.dumps(final_data)}\n\n"

                    if chat_request.use_memory and memory_store:
                        try:
                            if user_query:
                                memory_store.add_turn(session_id, "user", user_query)
                            if full_text.strip():
                                memory_store.add_turn(session_id, "assistant", full_text)
                        except Exception as exc:
                            logger.warning("Failed storing chat memory: %s", exc)
                    break

                elif command == "error":
                    error_msg = result.get("message", "Unknown error")
                    logger.error(f"Server error: {error_msg}")
                    error_data = {"error": error_msg, "done": True}
                    yield f"data: {json.dumps(error_data)}\n\n"
                    break

                else:
                    logger.warning(f"Unexpected command: {command}")
                    error_data = {
                        "error": f"Unexpected response: {command}",
                        "done": True,
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    break

        except Exception as e:
            logger.error(f"Error during streaming inference: {e}")
            disconnect_from_server()
            error_data = {"error": str(e), "done": True}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/query", response_model=QueryResponse)
async def query(query_request: ChatRequest):
    """Run inference and return a plain JSON response (no SSE, no memory operations)."""
    if not query_request.text and not query_request.messages:
        raise HTTPException(
            status_code=400,
            detail="Either text or messages must be provided",
        )

    sock = connect_to_server()
    if sock is None:
        raise HTTPException(status_code=503, detail="Could not connect to server")

    inference_request = {
        "command": "inference",
        "max_tokens": query_request.max_tokens,
        "temperature": query_request.temperature,
        "top_p": query_request.top_p,
        "top_k": query_request.top_k,
        "decoding_strategy": query_request.decoding_strategy,
        "worker_rank": query_request.worker_rank if query_request.worker_rank is not None else 0,
    }
    selected_worker_rank = inference_request["worker_rank"]

    if query_request.messages:
        inference_request["messages"] = query_request.messages
    else:
        inference_request["prompt"] = query_request.text

    metrics_tracker = get_inference_metrics()
    metrics_tracker.reset()
    metrics_tracker.start_inference()

    try:
        send_message(sock, ("inference", inference_request))
        full_text = ""
        final_worker_rank = selected_worker_rank

        while True:
            response = receive_message(sock)
            if response is None:
                raise HTTPException(status_code=503, detail="Connection lost")

            command, result = response

            if command == "token":
                token_text = result.get("text", "")
                full_text += token_text
                metrics_tracker.record_token()
                continue

            if command == "inference_complete":
                metrics_tracker.end_inference()
                perf_metrics = metrics_tracker.get_metrics()
                final_worker_rank = result.get("worker_rank", selected_worker_rank)
                return QueryResponse(
                    generated_text=full_text,
                    success=True,
                    worker_rank=final_worker_rank,
                    total_time_ms=perf_metrics.get("total_time_ms"),
                    time_to_first_token_ms=perf_metrics.get("time_to_first_token_ms"),
                    tokens_per_second=perf_metrics.get("tokens_per_second"),
                    num_tokens=perf_metrics.get("num_tokens"),
                )

            if command == "error":
                error_msg = result.get("message", "Unknown error")
                raise HTTPException(status_code=500, detail=error_msg)

            raise HTTPException(status_code=500, detail=f"Unexpected response: {command}")

    except HTTPException:
        raise
    except Exception as exc:
        disconnect_from_server()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/reconnect")
async def reconnect():
    """Manually reconnect to server."""
    try:
        disconnect_from_server()
        connect_to_server()
        return {"status": "reconnected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
