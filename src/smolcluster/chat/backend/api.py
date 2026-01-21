"""
FastAPI backend for chat application with Model Parallelism server.
Handles user input and communicates with the distributed inference server.
"""
import logging
import socket
from pathlib import Path
from typing import Optional
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from smolcluster.utils.common_utils import send_message, receive_message

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model config
CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"
with open(CONFIG_DIR / "model_parallelism" / "model_config.yaml") as f:
    model_configs = yaml.safe_load(f)

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
SERVER_HOST = "10.10.0.1"  # Update with your server host
SERVER_PORT = 65432  # Update with your server port


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


def connect_to_server():
    """Establish connection to Model Parallelism server."""
    global server_socket
    try:
        if server_socket is None:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.connect((SERVER_HOST, SERVER_PORT))
            logger.info(f"Connected to server at {SERVER_HOST}:{SERVER_PORT}")
            
            # Register as client
            send_message(server_socket, ("register_client", 0))
            response = receive_message(server_socket)
            if response and response[0] == "client_registered":
                logger.info("Successfully registered with server")
            else:
                raise Exception(f"Failed to register with server: {response}")
                
        return server_socket
    except Exception as e:
        logger.error(f"Failed to connect to server: {e}")
        server_socket = None
        raise HTTPException(status_code=503, detail=f"Server unavailable: {str(e)}")


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


@app.get("/health")
async def health():
    """Check if server connection is healthy."""
    try:
        if server_socket is None:
            return {"status": "disconnected", "healthy": False}
        return {"status": "connected", "healthy": True}
    except Exception as e:
        return {"status": "error", "healthy": False, "error": str(e)}


@app.get("/config")
async def get_config():
    """Get model configuration values for frontend."""
    active_strategy = model_config.get("active_decoding_strategy", "top_p")
    strategies = model_config.get("decoding_strategies", {})
    strategy_params = strategies.get(active_strategy, {})
    
    return {
        "model_name": MODEL_NAME,
        "max_new_tokens": model_config.get("max_new_tokens", 50),
        "decoding_strategy": active_strategy,
        "temperature": strategy_params.get("temperature", 1.0),
        "top_p": strategy_params.get("top_p", 0.9),
        "top_k": strategy_params.get("top_k", 50)
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send user text to model parallelism server and get generated response.
    
    Args:
        request: ChatRequest with text and generation parameters
        
    Returns:
        ChatResponse with generated text
    """
    try:
        # Ensure connection
        sock = connect_to_server()
        
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
        send_message(sock, ("inference", inference_request))
        
        # Receive generated response
        response = receive_message(sock)
        
        if response is None:
            raise HTTPException(status_code=500, detail="No response from server")
        
        command, result = response
        
        if command == "inference_result":
            generated_text = result.get("text", "")
            logger.info(f"Received response: {generated_text[:50]}...")
            
            return ChatResponse(
                generated_text=generated_text,
                success=True
            )
        elif command == "error":
            error_msg = result.get("message", "Unknown error")
            logger.error(f"Server error: {error_msg}")
            return ChatResponse(
                generated_text="",
                success=False,
                error=error_msg
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
        return ChatResponse(
            generated_text="",
            success=False,
            error=str(e)
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
    
    uvicorn.run(app, host="0.0.0.0", port=8000)