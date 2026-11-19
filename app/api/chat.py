from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from app.api.dependencies import get_chat_service
from app.services.chat_service import ChatService
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/v1", tags=["Chat"])

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    response: Response,
    chat_service: ChatService = Depends(get_chat_service)
):
    try:
        answer, is_cached, similarity = await chat_service.get_answer(request.query)
        response.headers["X-Cache"] = "HIT" if is_cached else "MISS"
        if similarity is not None:
            response.headers["X-Cache-Similarity"] = f"{similarity:.4f}"
        return ChatResponse(response=answer, is_cached=is_cached, similarity=similarity)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/stream")
async def chat_stream_endpoint(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
):
    try:
        match = await chat_service.check_cache(request.query)
        is_hit = match is not None

        headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Cache": "HIT" if is_hit else "MISS",
        }
        if is_hit and match is not None:
            headers["X-Cache-Similarity"] = f"{match.similarity:.4f}"

        async def event_generator():
            try:
                if is_hit and match is not None:
                    async for chunk in chat_service.stream_cached_response(match.response):
                        yield chunk
                else:
                    async for chunk in chat_service.stream_live_and_cache(request.query):
                        yield chunk
            except Exception as e:
                yield f"\n[Error: {str(e)}]"

        return StreamingResponse(
            event_generator(),
            media_type="text/plain",
            headers=headers
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
