from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM Settings (Swappable Provider & Model)
    llm_provider: str = "openai_like"  # "openai_like", "ollama", "gemini"
    llm_model: str = "gemini-2.5-flash"
    llm_api_base: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_api_key: Optional[str] = None
    
    # Specific API key aliases
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    
    # RAG Settings
    embed_model: str = "BAAI/bge-base-en-v1.5"
    pdf_path: str = "data/Utkarsh_Ujwal_Gen_AI_Engineer_Resume.pdf"
    
    # Semantic Caching Settings
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.85
    semantic_cache_file: str = "data/semantic_cache.json"
    semantic_cache_max_entries: int = 1000
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
