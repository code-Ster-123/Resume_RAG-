import os
from typing import Optional
from llama_index.core.llms.llm import LLM

class LLMAdapter:
    """
    Adapter / Factory for creating LLM instances.
    Adheres to the OpenAI-compatible pattern to ensure seamless swapping between:
    - Gemini (via Google's OpenAI-compatible endpoint)
    - OpenAI
    - Groq, DeepSeek, vLLM, or any OpenAI-compatible provider
    - Ollama (local)
    """

    @staticmethod
    def create_llm(
        provider: str = "openai_like",
        model: str = "gemini-3.1-flash-lite",
        api_base: Optional[str] = "https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key: Optional[str] = None,
        context_window: int = 32768,
        temperature: float = 0.1,
        **kwargs
    ) -> LLM:
        provider = (provider or "openai_like").lower().strip()

        # Resolve API Key from parameter or environment variables
        resolved_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("LLM_API_KEY")
        )

        if provider in ("openai_like", "openai"):
            from llama_index.llms.openai_like import OpenAILike

            base_url = api_base or "https://generativelanguage.googleapis.com/v1beta/openai/"
            
            return OpenAILike(
                model=model,
                api_base=base_url,
                api_key=resolved_key or "no-key-provided",
                is_chat_model=True,
                is_function_calling_model=True,
                context_window=context_window,
                temperature=temperature,
                timeout=kwargs.pop("timeout", 60.0),
                **kwargs
            )

        elif provider == "ollama":
            from llama_index.llms.ollama import Ollama

            return Ollama(
                model=model,
                base_url=api_base or "http://localhost:11434",
                context_window=kwargs.pop("context_window", 2048),
                additional_kwargs={"num_ctx": kwargs.pop("num_ctx", 2048)},
                request_timeout=kwargs.pop("request_timeout", 120.0),
                **kwargs
            )

        elif provider == "gemini":
            from llama_index.llms.gemini import Gemini

            model_name = model if model.startswith("models/") else f"models/{model}"
            return Gemini(
                model=model_name,
                api_key=resolved_key,
                temperature=temperature,
                **kwargs
            )

        else:
            raise ValueError(
                f"Unsupported LLM provider '{provider}'. "
                f"Supported options are: 'openai_like', 'ollama', 'gemini'."
            )
