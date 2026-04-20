from typing import Any

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    intent: str
    message: str
    data: Any = None
    error: str | None = None
