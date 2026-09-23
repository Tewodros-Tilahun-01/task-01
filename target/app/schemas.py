"""
schemas.py — Pydantic models for request and response shapes.
"""

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    detail: str
    type: str | None = None
