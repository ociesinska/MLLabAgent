from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    intent: str
    message: str
    data: Any = None
    error: str | None = None


class ParsedUserRequest(BaseModel):
    intent: Literal[
        "show",
        "compare",
        "summarize_compare",
        "show_best_run",
        "show_latest_run",
        "agent_analyze",
        "unknown",
    ]
    run_identifiers: list[str] = Field(default_factory=list)
    metric: str | None = None
