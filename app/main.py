import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from app.core.config import settings
from app.infrastructure.llamaindex_adapter import LlamaIndexRAGAdapter
from app.infrastructure.llm_factory import LLMAdapter
from app.infrastructure.semantic_cache import InMemorySemanticCache
from app.api.chat import router as chat_router

static_dir = os.path.join(os.path.dirname(__file__), "static")
index_html_path = os.path.join(static_dir, "index.html")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading embedding model, LLM, and indexing resume documents... This might take a moment.")
    
    # 1. Initialize shared embedding model once in memory
    embed_model = HuggingFaceEmbedding(model_name=settings.embed_model)

    # 2. Initialize Semantic Cache using shared embedder
    if settings.semantic_cache_enabled:
        semantic_cache = InMemorySemanticCache(
            embed_model=embed_model,
            threshold=settings.semantic_cache_threshold,
            persist_path=settings.semantic_cache_file,
            max_entries=settings.semantic_cache_max_entries,
        )
        app.state.semantic_cache = semantic_cache
        print(f"Semantic Cache initialized (threshold={settings.semantic_cache_threshold}, file='{settings.semantic_cache_file}').")
    else:
        app.state.semantic_cache = None
        print("Semantic Cache is disabled via settings.")

    # 3. Instantiate LLM via swappable Adapter
    llm = LLMAdapter.create_llm(
        provider=settings.llm_provider,
        model=settings.llm_model,
        api_base=settings.llm_api_base,
        api_key=settings.llm_api_key or settings.gemini_api_key or settings.openai_api_key
    )
    
    # 4. Instantiate LlamaIndex RAG Engine sharing the embedding model
    rag_engine = LlamaIndexRAGAdapter(
        llm=llm,
        embed_model_name=settings.embed_model,
        embed_model=embed_model,
        pdf_path=settings.pdf_path
    )
    
    # Store it in app state so dependency injection can find it
    app.state.rag_engine = rag_engine
    print(f"RAG Engine loaded with provider '{settings.llm_provider}' ({settings.llm_model}). Ready for requests.")
    
    yield
    
    print("Shutting down application...")
    if getattr(app.state, "semantic_cache", None) is not None:
        app.state.semantic_cache.persist()
    app.state.rag_engine = None
    app.state.semantic_cache = None

app = FastAPI(
    title="Resume RAG API",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static folder
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Serve the Interactive WebUI at the root URL / and /chat
@app.get("/", response_class=HTMLResponse)
@app.head("/", include_in_schema=False)
@app.get("/chat", response_class=HTMLResponse)
@app.head("/chat", include_in_schema=False)
async def serve_web_ui():
    if os.path.exists(index_html_path):
        return FileResponse(index_html_path)
    return HTMLResponse("<h2>Resume Agent Web UI loading... Please ensure index.html exists in app/static.</h2>")

# API Endpoints
app.include_router(chat_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
