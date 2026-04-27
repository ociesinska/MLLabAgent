from typing_extensions import TypedDict

from ml_lab_agent.schemas.chat_schemas import ChatResponse


class State(TypedDict):
    message: str
    intent: str | None
    metric: str | None
    run_ids: list[str]
    compare_results: dict | None
    llm_error: str | None
    final_response: ChatResponse | None
