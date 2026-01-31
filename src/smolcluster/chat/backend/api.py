"""
FastAPI backend for chat application with Model Parallelism server.
Handles user input and communicates with the distributed inference server.
"""
import logging
import socket
import time
from pathlib import Path
from typing import Optional
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
    active_strategy = model_config.get("active_decoding_strategy", "top_p")
    strategies = model_config.get("decoding_strategies", {})
    strategy_params = strategies.get(active_strategy, {})
    
    return {
        "api_port": API_PORT,
        "frontend_port": cluster_config["web_interface"]["frontend_port"],
        "server_host": SERVER_HOST,
        "server_port": SERVER_PORT,
        "model_name": MODEL_NAME,
        "max_new_tokens": model_config.get("max_new_tokens", 50),
        "decoding_strategy": active_strategy,
        "temperature": strategy_params.get("temperature", 1.0),
        "top_p": strategy_params.get("p", 0.9),
        "top_k": strategy_params.get("k", 50)
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


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send user text to model parallelism server and get generated response.
    
    Args:
        request: ChatRequest with text and generation parameters
        
    Returns:
        ChatResponse with generated text and performance metrics
    """
    # Get inference metrics tracker
    metrics_tracker = get_inference_metrics()
    metrics_tracker.reset()
    
    try:
        # Ensure connection
        sock = connect_to_server()
        if sock is None:
            raise HTTPException(status_code=503, detail="Could not connect to server")
        
        # Send inference request to server
        inference_request = {
            "command": "inference",
            "prompt": request.text,
            "max_tokens": request.max_tokens or model_config.get("max_new_tokens", 50),
            "temperature": request.temperature or model_config.get("temperature", 0.7),
            "top_p": request.top_p or model_config.get("top_p", 0.9),
            "top_k": request.top_k or model_config.get("top_k", 50)
        }
        
        logger.info(f"Sending inference request: {request.text[:50]}...")
        
        # Start timing
        metrics_tracker.start_inference()
        send_message(sock, ("inference", inference_request))
        
        # Receive generated response
        response = receive_message(sock)
        metrics_tracker.end_inference()
        
        if response is None:
            raise HTTPException(status_code=500, detail="No response from server")
        
        command, result = response
        
        if command == "inference_result":
            generated_text = result.get("text", "")
            logger.info(f"Received response: {generated_text[:50]}...")
            
            # Count tokens (simple approximation: split by whitespace)
            # For more accurate counting, use a tokenizer
            num_tokens = len(generated_text.split())
            for _ in range(num_tokens):
                metrics_tracker.record_token()
            
            # Get performance metrics
            perf_metrics = metrics_tracker.get_metrics()
            logger.info(f"Inference metrics: {perf_metrics}")
            
            return ChatResponse(
                generated_text=generated_text,
                success=True,
                total_time_ms=perf_metrics.get('total_time_ms'),
                time_to_first_token_ms=perf_metrics.get('time_to_first_token_ms'),
                tokens_per_second=perf_metrics.get('tokens_per_second'),
                num_tokens=perf_metrics.get('num_tokens')
            )
        elif command == "error":
            error_msg = result.get("message", "Unknown error")
            logger.error(f"Server error: {error_msg}")
            perf_metrics = metrics_tracker.get_metrics()
            return ChatResponse(
                generated_text="",
                success=False,
                error=error_msg,
                total_time_ms=perf_metrics.get('total_time_ms'),
                time_to_first_token_ms=perf_metrics.get('time_to_first_token_ms'),
                tokens_per_second=perf_metrics.get('tokens_per_second'),
                num_tokens=perf_metrics.get('num_tokens')
            )
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Unexpected response from server: {command}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        # Try to reconnect on next request
        disconnect_from_server()
        perf_metrics = metrics_tracker.get_metrics()
        return ChatResponse(
            generated_text="",
            success=False,
            error=str(e),
            total_time_ms=perf_metrics.get('total_time_ms'),
            time_to_first_token_ms=perf_metrics.get('time_to_first_token_ms'),
            tokens_per_second=perf_metrics.get('tokens_per_second'),
            num_tokens=perf_metrics.get('num_tokens')
        )


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