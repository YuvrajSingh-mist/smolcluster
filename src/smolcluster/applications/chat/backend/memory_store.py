"""Redis-backed lightweight vector memory for chat turns."""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass

import numpy as np

import redis


logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    role: str
    content: str
    score: float


class RedisVectorMemory:
    """Store and retrieve chat turns using Redis in-memory vector index."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        index_name: str = "smolcluster_chat_memory_idx",
        key_prefix: str = "chatmem:",
        embedding_dim: int = 256,
    ) -> None:
        if redis is None:
            raise RuntimeError("redis package not installed")

        self.redis_url = redis_url
        self.index_name = index_name
        self.key_prefix = key_prefix
        self.embedding_dim = embedding_dim
        self.client = redis.Redis.from_url(redis_url, decode_responses=False)
        self.client.ping()
        self._ensure_index()

    def _ensure_index(self) -> None:
        try:
            self.client.execute_command("FT.INFO", self.index_name)
            return
        except Exception:
            pass

        self.client.execute_command(
            "FT.CREATE",
            self.index_name,
            "ON",
            "HASH",
            "PREFIX",
            "1",
            self.key_prefix,
            "SCHEMA",
            "session",
            "TAG",
            "SEPARATOR",
            "|",
            "role",
            "TAG",
            "SEPARATOR",
            "|",
            "content",
            "TEXT",
            "vector",
            "VECTOR",
            "HNSW",
            "6",
            "TYPE",
            "FLOAT32",
            "DIM",
            str(self.embedding_dim),
            "DISTANCE_METRIC",
            "COSINE",
        )

    def _session_tag(self, session_id: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        if not text:
            return vec

        for token in re.findall(r"\w+", text.lower()):
            idx = hash(token) % self.embedding_dim
            vec[idx] += 1.0

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        if not content.strip():
            return

        tag = self._session_tag(session_id)
        vector = self._embed(content)
        key = f"{self.key_prefix}{tag}:{int(time.time() * 1000)}:{uuid.uuid4().hex[:8]}"
        self.client.hset(
            key,
            mapping={
                "session": tag.encode("utf-8"),
                "role": role.encode("utf-8"),
                "content": content.encode("utf-8"),
                "vector": vector.tobytes(),
            },
        )

    def get_session_history(self, session_id: str, limit: int = 100) -> list[MemoryItem]:
        """Return chat turns for a session ordered by insertion timestamp."""
        tag = self._session_tag(session_id)
        pattern = f"{self.key_prefix}{tag}:*"

        records: list[tuple[int, MemoryItem]] = []
        try:
            keys = list(self.client.scan_iter(match=pattern, count=1000))
        except Exception as exc:
            logger.warning("Redis memory history scan failed: %s", exc)
            return []

        for raw_key in keys:
            try:
                key = raw_key.decode("utf-8", errors="ignore") if isinstance(raw_key, (bytes, bytearray)) else str(raw_key)
                parts = key.split(":")
                ts = int(parts[-2]) if len(parts) >= 3 and parts[-2].isdigit() else 0

                row = self.client.hgetall(raw_key)
                role = row.get(b"role", b"").decode("utf-8", errors="ignore")
                content = row.get(b"content", b"").decode("utf-8", errors="ignore")
                if not content:
                    continue
                records.append((ts, MemoryItem(role=role or "assistant", content=content, score=0.0)))
            except Exception:
                continue

        records.sort(key=lambda item: item[0])
        if limit > 0:
            records = records[-limit:]
        return [item for _, item in records]

    def clear_session_history(self, session_id: str) -> int:
        """Delete all stored turns for a session and return deleted key count."""
        tag = self._session_tag(session_id)
        pattern = f"{self.key_prefix}{tag}:*"

        deleted = 0
        try:
            keys = list(self.client.scan_iter(match=pattern, count=1000))
        except Exception as exc:
            logger.warning("Redis memory clear scan failed: %s", exc)
            return 0

        if not keys:
            return 0

        try:
            deleted = int(self.client.delete(*keys))
        except Exception as exc:
            logger.warning("Redis memory clear delete failed: %s", exc)
            return 0

        return deleted

    def search(self, session_id: str, query: str, k: int = 4) -> list[MemoryItem]:
        if not query.strip():
            return []

        tag = self._session_tag(session_id)
        query_vec = self._embed(query).astype(np.float32).tobytes()
        redis_query = (
            f"(@session:{{{tag}}})=>[KNN {k} @vector $vec AS score]"
        )

        try:
            result = self.client.execute_command(
                "FT.SEARCH",
                self.index_name,
                redis_query,
                "PARAMS",
                "2",
                "vec",
                query_vec,
                "SORTBY",
                "score",
                "ASC",
                "RETURN",
                "3",
                "role",
                "content",
                "score",
                "LIMIT",
                "0",
                str(k),
                "DIALECT",
                "2",
            )
        except Exception as exc:
            logger.warning("Redis memory search failed: %s", exc)
            return []

        items: list[MemoryItem] = []
        if not isinstance(result, list) or len(result) < 2:
            return items

        for i in range(1, len(result), 2):
            payload = result[i + 1] if i + 1 < len(result) else []
            fields = {
                payload[j].decode("utf-8", errors="ignore"): payload[j + 1]
                for j in range(0, len(payload), 2)
            }
            role = fields.get("role", b"").decode("utf-8", errors="ignore")
            content = fields.get("content", b"").decode("utf-8", errors="ignore")
            score_raw = fields.get("score", b"1")
            try:
                score = float(
                    score_raw.decode("utf-8", errors="ignore")
                    if isinstance(score_raw, (bytes, bytearray))
                    else score_raw
                )
            except Exception:
                score = 1.0
            if content:
                items.append(MemoryItem(role=role or "assistant", content=content, score=score))

        return items
