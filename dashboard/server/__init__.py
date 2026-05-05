"""FastAPI server package — re-exports the `app` instance for uvicorn."""
from .app import app

__all__ = ["app"]
