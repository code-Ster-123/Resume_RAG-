import asyncio
from typing import AsyncGenerator, Optional, Tuple
from app.core.interfaces import RAGEngine, SemanticCache, CacheMatch

class ChatService:
    def __init__(self, rag_engine: RAGEngine, cache: Optional[SemanticCache] = None):
        self.rag_engine = rag_engine
        self.cache = cache

    async def check_cache(self, user_query: str) -> Optional[CacheMatch]:
        """Check if query has a semantic cache match."""
        if self.cache is not None:
            return await self.cache.get(user_query)
        return None

    async def get_answer(self, user_query: str) -> Tuple[str, bool, Optional[float]]:
        """
        Returns (answer_text, is_cached, similarity_score).
        """
        match = await self.check_cache(user_query)
        if match:
            return match.response, True, match.similarity

        # Cache miss: run live RAG
        answer = await self.rag_engine.query(user_query)
        if self.cache is not None and answer:
            await self.cache.set(user_query, answer)

        return answer, False, None

    async def stream_cached_response(self, text: str) -> AsyncGenerator[str, None]:
        """
        Stream cached text in natural word bursts so UI streams smoothly.
        """
        words = text.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield chunk
            await asyncio.sleep(0.01)

    async def stream_live_and_cache(self, user_query: str) -> AsyncGenerator[str, None]:
        """
        Stream chunks live from RAG engine and save full response to cache on completion.
        """
        accumulated_chunks = []
        async for chunk in self.rag_engine.stream(user_query):
            accumulated_chunks.append(chunk)
            yield chunk

        full_answer = "".join(accumulated_chunks).strip()
        if self.cache is not None and full_answer:
            await self.cache.set(user_query, full_answer)

    async def get_answer_stream(self, user_query: str) -> AsyncGenerator[str, None]:
        """
        Unified streaming method that checks cache or streams live.
        """
        match = await self.check_cache(user_query)
        if match:
            async for chunk in self.stream_cached_response(match.response):
                yield chunk
        else:
            async for chunk in self.stream_live_and_cache(user_query):
                yield chunk
