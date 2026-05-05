"""FastAPI application factory, lifespan, static mounts, and root route."""
import asyncio
import logging
import socket
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import ctx
from . import node_meta, redis as _redis_mod
from .broadcasters import events_broadcaster, log_broadcaster
from .paths import FRONTEND_DIR
from .ssh_config import _build_static_nodes_inventory, _lookup_ssh_entry, local_node_metadata
from .routes import inference as _routes_inference, misc as _routes_misc, nodes as _routes_nodes, training as _routes_training
from . import sse as _sse
from dashboard.node_manager import NodeManager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Redis ──────────────────────────────────────────────────────────────────
    redis_boot = _redis_mod.ensure_redis_running()
    ctx.redis = aioredis.from_url(_redis_mod.REDIS_URL, decode_responses=True)
    _redis_mod._redis_diag["enabled"] = True
    _redis_mod._redis_diag["status"]  = "connected"
    _redis_mod.redis_mark(f"connected ({redis_boot})")
    logger.info(f"[redis] connected ({redis_boot}) url={_redis_mod.REDIS_URL}")

    # ── Node manager & host identity ───────────────────────────────────────────
    ctx.node_manager     = NodeManager()
    ctx.server_hostname  = socket.gethostname().removesuffix(".local")
    local_ips             = node_meta.collect_local_ips()
    ctx.static_nodes     = _build_static_nodes_inventory(ctx.server_hostname, local_ips)
    ctx.ssh_aliases.clear()
    ctx.ssh_aliases.update({h: h for h in ctx.static_nodes})
    ctx.node_os[ctx.server_hostname] = local_node_metadata()

    # ── Restore persisted state from Redis ────────────────────────────────────
    await _redis_mod.restore_state_from_redis()

    # Flush stale log stream (broadcaster re-tails from EOF on start).
    await ctx.redis.delete("smolcluster:logs")

    # Seed SSH usernames from config for nodes not yet probed.
    seed_pipe = ctx.redis.pipeline()
    for h in ctx.static_nodes:
        if not ctx.probed.get(h):
            user = _lookup_ssh_entry(h, ctx.static_nodes[h].get("ip", "")).get("user", "")
            if user:
                ctx.probed[h] = user
                seed_pipe.hset("smolcluster:probed", h, user)
    await seed_pipe.execute()

    logger.info(f"[dashboard] http://{ctx.server_hostname}.local:9090")

    broadcast_task = asyncio.create_task(events_broadcaster())
    log_task       = asyncio.create_task(log_broadcaster())
    metadata_task  = asyncio.create_task(node_meta.prime_node_metadata())

    yield

    broadcast_task.cancel()
    log_task.cancel()
    metadata_task.cancel()
    await asyncio.gather(broadcast_task, log_task, metadata_task, return_exceptions=True)
    await ctx.node_manager.stop_all()
    await ctx.redis.aclose()


# ── Application ────────────────────────────────────────────────────────────────
app = FastAPI(title="smolcluster Dashboard", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Static assets — mounts before routers (ASGI prefix routing handles ordering correctly).
app.mount("/css", StaticFiles(directory=FRONTEND_DIR / "css"), name="css")
app.mount("/js",  StaticFiles(directory=FRONTEND_DIR / "js"),  name="js")

# Include all route groups.
app.include_router(_routes_nodes.router)
app.include_router(_routes_training.router)
app.include_router(_routes_inference.router)
app.include_router(_routes_misc.router)
app.include_router(_sse.router)


@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")
