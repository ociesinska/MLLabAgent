from fastapi import APIRouter

from ml_lab_agent.schemas.chat_schemas import ChatRequest, ChatResponse
from ml_lab_agent.services.chat_service import process_request

chat_router = APIRouter()


@chat_router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    return process_request(request.message)
