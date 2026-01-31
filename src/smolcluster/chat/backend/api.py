"""  
FastAPI backend for chat application with Model Parallelism server.
Handles user input and communicates with the distributed inference server.
"""
import asyncio
import json
import logging
import socket
import time
from pathlib import Path
from typing import Optional
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from smolcluster.utils.common_utils import send_message, receive_message, get_inference_metrics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model config
CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"
with open(CONFIG_DIR / "model_parallelism" / "model_config_inference.yaml") as f:
    model_configs = yaml.safe_load(f)

# Load cluster config for web interface ports and server connection
with open(CONFIG_DIR / "model_parallelism" / "cluster_config_inference.yaml") as f:
    cluster_config = yaml.safe_load(f)

# Get active model config (default to causal_gpt2)
MODEL_NAME = 'causal_gpt2'
model_config = model_configs[MODEL_NAME]

app = FastAPI(title="SmolCluster Chat API")

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
SERVER_HOST = cluster_config["host_ip"][server_hostname]
port_config = cluster_config["port"]
if isinstance(port_config, dict):
    SERVER_PORT = port_config.get(server_hostname, port_config.get("default", 65432))
else:
    SERVER_PORT = port_config

# Get web interface port from cluster config
API_PORT = cluster_config["web_interface"]["api_port"]

MAX_CONNECTION_RETRIES = 10
RETRY_DELAY = 3  # seconds


class ChatRequest(BaseModel):
    text: str
    max_tokens: Optional[int] = None  # Will use model config default
    temperature: Optional[float] = None  # Will use model config default
    top_p: Optional[float] = None  # Will use model config default
    top_k: Optional[int] = None  # Will use model config default
    decoding_strategy: Optional[str] = None  # Will use model config default


class ChatResponse(BaseModel):
    generated_text: str
    success: bool
    error: Optional[str] = None
    # Inference metrics
    total_time_ms: Optional[float] = None
    time_to_first_token_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None
    num_tokens: Optional[int] = None


def connect_to_server():
    """Establish connection to Model Parallelism server with retry logic."""
    global server_socket
    
    for attempt in range(1, MAX_CONNECTION_RETRIES + 1):
        try:
            if server_socket is None:
                logger.info(f"Attempt {attempt}/{MAX_CONNECTION_RETRIES}: Connecting to {SERVER_HOST}:{SERVER_PORT}...")
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
                logger.error(f"Failed to connect after {MAX_CONNECTION_RETRIES} attempts")
                raise HTTPException(status_code=503, detail=f"Server unavailable after {MAX_CONNECTION_RETRIES} attempts: {str(e)}")


def disconnect_from_server():
    """Close connection to server."""
    global server_socket
    if server_socket:
        try:
            send_message(server_socket, ("disconnect", None))
            server_socket.close()
        except:
            pass
        server_socket = None
        logger.info("Disconnected from server")


@app.on_event("startup")
async def startup_event():
    """Connect to server on startup."""
    logger.info("Starting FastAPI chat backend...")
    try:
        connect_to_server()
    except Exception as e:
        logger.warning(f"Could not connect to server on startup: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Disconnect from server on shutdown."""
    logger.info("Shutting down FastAPI chat backend...")
    disconnect_from_server()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "SmolCluster Chat API"}


@app.get("/config")
async def get_config():
    """Get API and model configuration."""
    active_strategy = model_config["active_decoding_strategy"]
    strategies = model_config.get("decoding_strategies", {})
    strategy_params = strategies.get(active_strategy, {})
    
    return {
        "api_port": API_PORT,
        "frontend_port": cluster_config["web_interface"]["frontend_port"],
        "server_host": SERVER_HOST,
        "server_port": SERVER_PORT,
        "model_name": MODEL_NAME,
        "max_new_tokens": model_config["max_new_tokens"],
        "decoding_strategy": active_strategy,
        "temperature": strategy_params.get("temperature", 1.0),
        f'{active_strategy}' : strategy_params
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


@app.post("/chat")
async def chat(request: ChatRequest):
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
            # Ensure connection
            sock = connect_to_server()
            if sock is None:
                yield f"data: {json.dumps({'error': 'Could not connect to server', 'done': True})}\n\n"
                return
            
            # Send inference request to server
            inference_request = {
                "command": "inference",
                "prompt": request.text,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "decoding_strategy": request.decoding_strategy
            }
            
            logger.info(f"Sending streaming inference request: {request.text[:50]}...")
            
            # Start timing
            metrics_tracker.start_inference()
            send_message(sock, ("inference", inference_request))
            
            # Stream tokens as they arrive
            full_text = ""
            while True:
                response = receive_message(sock)
                
                if response is None:
                    error_data = {'error': 'Connection lost', 'done': True}
                    yield f"data: {json.dumps(error_data)}\n\n"
                    break
                
                command, result = response
                
                if command == "token":
                    # Received a new token
                    token_text = result.get("text", "")
                    token_idx = result.get("token_idx", 0)
                    
                    # Record token for metrics
                    metrics_tracker.record_token()
                    full_text += token_text
                    
                    # Send token to frontend (json.dumps handles escaping)
                    token_data = {'token': token_text, 'done': False}
                    yield f"data: {json.dumps(token_data)}\n\n"
                    logger.info(f"Streamed token {token_idx}: {repr(token_text)}")
                    
                elif command == "inference_complete":
                    # Generation complete
                    metrics_tracker.end_inference()
                    perf_metrics = metrics_tracker.get_metrics()
                    
                    logger.info(f"Streaming complete. Metrics: {perf_metrics}")
                    
                    # Send final metrics
                    final_data = {
                        'done': True,
                        'full_text': full_text,
                        'total_time_ms': perf_metrics.get('total_time_ms'),
                        'time_to_first_token_ms': perf_metrics.get('time_to_first_token_ms'),
                        'tokens_per_second': perf_metrics.get('tokens_per_second'),
                        'num_tokens': perf_metrics.get('num_tokens')
                    }
                    yield f"data: {json.dumps(final_data)}\n\n"
                    break
                    
                elif command == "error":
                    error_msg = result.get("message", "Unknown error")
                    logger.error(f"Server error: {error_msg}")
                    error_data = {'error': error_msg, 'done': True}
                    yield f"data: {json.dumps(error_data)}\n\n"
                    break
                    
                else:
                    logger.warning(f"Unexpected command: {command}")
                    error_data = {'error': f'Unexpected response: {command}', 'done': True}
                    yield f"data: {json.dumps(error_data)}\n\n"
                    break
                    
        except Exception as e:
            logger.error(f"Error during streaming inference: {e}")
            disconnect_from_server()
            error_data = {'error': str(e), 'done': True}
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/reconnect")
async def reconnect():
    """Manually reconnect to server."""
    try:
        disconnect_from_server()
        connect_to_server()
        return {"status": "reconnected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)