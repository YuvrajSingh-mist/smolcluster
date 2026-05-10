"""Pydantic request-body models for the dashboard API."""

from pydantic import BaseModel


class SelectRequest(BaseModel):
    ssh_user: str = ""
    rank: int | None = None


class StartRequest(BaseModel):
    algorithm: str = "syncps"


class InferenceLaunchRequest(BaseModel):
    algorithm: str = "syncps"
    server_hostname: str = ""  # which selected node is the server/rank-0
