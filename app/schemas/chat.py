from typing import Optional
from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    is_cached: bool = False
    similarity: Optional[float] = None
