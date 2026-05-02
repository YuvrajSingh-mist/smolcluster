"""Misc routes: connectivity check, UI state persistence, chat proxy."""
import asyncio
import json
import logging
import time

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from . import _ctx
from ._helpers import _get_inference_api_url
from ._paths import INFERENCE_FILE
from ._redis import REDIS_UI_KEY, redis_mark
from ._ssh_config import _lookup_ssh_entry

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Connectivity ───────────────────────────────────────────────────────────────

def _resolve_connectivity_target(hostname: str, snap: dict) -> str:
    node_ip   = snap.get(hostname, {}).get("ip", "")
    ssh_entry = _lookup_ssh_entry(hostname, node_ip)
    if ssh_entry.get("hostname"):
        return ssh_entry["hostname"]
    if node_ip:
        return node_ip
    return f"{hostname}.local"


async def _run_tcp_checks(selected: dict, snap: dict) -> None:
    total   = len(selected)
    results = []
    INFERENCE_FILE.write_text(json.dumps({
        "mode": "connectivity", "status": "checking",
        "results": [], "total": total, "message": f"Checking {total} node(s)…",
    }))
    for hostname in selected:
        t0     = time.monotonic()
        target = _resolve_connectivity_target(hostname, snap)
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(target, 22), timeout=5.0,
            )
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            ms = round((time.monotonic() - t0) * 1000, 1)
            results.append({"hostname": hostname, "status": "ok", "ms": ms, "target": target})
        except asyncio.TimeoutError:
            results.append({"hostname": hostname, "status": "timeout", "ms": None, "target": target})
        except Exception as e:
            results.append({"hostname": hostname, "status": "error",
                            "error": str(e)[:60], "ms": None, "target": target})
        INFERENCE_FILE.write_text(json.dumps({
            "mode": "connectivity", "status": "checking",
            "results": results, "total": total, "message": f"Checked {len(results)}/{total}…",
        }))
    ok = sum(1 for r in results if r["status"] == "ok")
    INFERENCE_FILE.write_text(json.dumps({
        "mode": "connectivity", "status": "done",
        "results": results, "total": total,
        "message": f"All {total} reachable ✓" if ok == total else f"{ok}/{total} reachable",
    }))


@router.post("/api/connectivity/check")
async def connectivity_check():
    selected = _ctx.node_manager.snapshot_selected()
    if not selected:
        raise HTTPException(400, "No nodes selected")
    asyncio.create_task(_run_tcp_checks(selected, dict(_ctx.static_nodes)))
    return {"status": "checking"}


# ── UI state ───────────────────────────────────────────────────────────────────

@router.get("/api/ui-state")
async def get_ui_state():
    if _ctx.redis:
        try:
            raw = await _ctx.redis.get(REDIS_UI_KEY)
            redis_mark("ui-state get", op_key="ui_get")
            if raw:
                return json.loads(raw)
        except Exception:
            pass
    return {}


@router.post("/api/ui-state")
async def post_ui_state(request: Request):
    if _ctx.redis:
        try:
            patch = await request.json()
            raw   = await _ctx.redis.get(REDIS_UI_KEY)
            cur   = json.loads(raw) if raw else {}
            cur.update(patch)
            await _ctx.redis.set(REDIS_UI_KEY, json.dumps(cur))
            keys = ",".join(sorted(str(k) for k in patch.keys())) if isinstance(patch, dict) else "unknown"
            redis_mark(f"ui-state set keys={keys}", op_key="ui_set")
        except Exception:
            pass
    return {"ok": True}


# ── Chat proxy ─────────────────────────────────────────────────────────────────

@router.post("/chat")
async def chat_proxy(request: Request):
    """Proxy chat requests to the local inference API as a transparent SSE stream."""
    api_url = _get_inference_api_url()
    if not api_url:
        raise HTTPException(503, "Inference API server not configured")

    body         = await request.body()
    content_type = request.headers.get("content-type", "application/json")

    async def stream_from_api():
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST", f"{api_url}/chat",
                    content=body, headers={"content-type": content_type},
                ) as response:
                    if response.status_code != 200:
                        resp_text = (await response.aread()).decode(errors="ignore").strip()
                        err = f"Inference API error: {response.status_code}"
                        if resp_text:
                            err = f"{err} - {resp_text[:180]}"
                        yield f"data: {json.dumps({'error': err, 'done': True})}\n\n"
                        return
                    async for chunk in response.aiter_bytes():
                        if chunk:
                            yield chunk
        except httpx.TimeoutException:
            yield f"data: {json.dumps({'error': 'Inference API timeout', 'done': True})}\n\n"
        except Exception as e:
            logger.error(f"[chat proxy] {e}")
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

    return StreamingResponse(stream_from_api(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
