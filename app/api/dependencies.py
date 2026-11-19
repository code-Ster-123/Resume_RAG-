from fastapi import Request
from app.services.chat_service import ChatService

def get_chat_service(request: Request) -> ChatService:
    rag_engine = request.app.state.rag_engine
    semantic_cache = getattr(request.app.state, "semantic_cache", None)
    return ChatService(rag_engine=rag_engine, cache=semantic_cache)
