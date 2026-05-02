"""SSE endpoints: /api/events and /api/logs."""
import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from . import _ctx
from ._redis import REDIS_EVENTS_KEY

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/events")
async def sse_events(request: Request):
    """Server-Sent Events stream: full cluster state at ~6–7 Hz."""
    async def gen():
        while True:
            if await request.is_disconnected():
                break
            try:
                raw = await _ctx.redis.get(REDIS_EVENTS_KEY)
                if raw:
                    yield f"data: {raw}\n\n"
            except Exception:
                pass
            await __import__("asyncio").sleep(0.15)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/api/logs")
async def sse_logs(request: Request):
    """Server-Sent Events stream: live log lines via Redis Stream."""
    async def gen():
        # Send recent history first so the UI doesn't miss logs from before connect.
        try:
            history = await _ctx.redis.xrevrange("smolcluster:logs", "+", "-", count=500)
            if history:
                history.reverse()
                lines = [
                    {"hostname": e["hostname"], "line": e["line"],
                     "session": e.get("session", ""), "ts": float(e.get("ts") or 0)}
                    for _, e in history
                ]
                if lines:
                    yield f"data: {json.dumps(lines)}\n\n"
                last_id = history[-1][0]
            else:
                last_id = "$"
        except Exception:
            last_id = "$"

        while True:
            if await request.is_disconnected():
                break
            try:
                results = await _ctx.redis.xread(
                    {"smolcluster:logs": last_id}, count=200, block=400,
                )
                if results:
                    _, entries = results[0]
                    if entries:
                        last_id = entries[-1][0]
                        lines = [
                            {"hostname": e["hostname"], "line": e["line"],
                             "session": e.get("session", ""), "ts": float(e.get("ts") or 0)}
                            for _, e in entries
                        ]
                        yield f"data: {json.dumps(lines)}\n\n"
            except Exception:
                await __import__("asyncio").sleep(0.35)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
