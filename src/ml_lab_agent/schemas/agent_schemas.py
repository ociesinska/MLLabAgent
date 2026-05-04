from typing import Literal
from pydantic import BaseModel, Field


class AgentToolCall(BaseModel):
    tool: Literal[
        "get_latest_run",
        "get_best_run_by_metric",
        "compare_runs",
        "generate_summary"
    ]
    args: dict = Field(default_factory=dict)


class AgentPlan(BaseModel):
    goal: str
    steps: list[AgentToolCall]