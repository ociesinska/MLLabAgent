from fastapi import APIRouter

from ml_lab_agent.api.agents.chat_graph.graph import graph
from ml_lab_agent.schemas.chat_schemas import ChatRequest, ChatResponse

chat_router = APIRouter()


@chat_router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = graph.invoke({"message": request.message})
    return result["final_response"]
