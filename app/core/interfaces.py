from dataclasses import dataclass
from typing import Protocol, AsyncGenerator, Optional

@dataclass
class CacheMatch:
    query: str
    response: str
    similarity: float

class RAGEngine(Protocol):
    async def query(self, text: str) -> str:
        """Process a text query and return an AI response."""
        ...

    async def stream(self, text: str) -> AsyncGenerator[str, None]:
        """Process a text query and yield AI response chunks as they arrive."""
        ...

class SemanticCache(Protocol):
    async def get(self, query: str) -> Optional[CacheMatch]:
        """Look up semantically equivalent cached response. Returns None on cache miss."""
        ...

    async def set(self, query: str, response: str) -> None:
        """Store query vector and its generated answer in the cache."""
        ...

