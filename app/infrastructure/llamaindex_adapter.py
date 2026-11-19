import os
from typing import Any, Optional, AsyncGenerator
from pydantic import BaseModel, Field
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool
from llama_index.core.llms.llm import LLM
from llama_index.core.agent.workflow.workflow_events import AgentStream
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader
from app.core.interfaces import RAGEngine
from app.infrastructure.llm_factory import LLMAdapter

class SearchResumeInput(BaseModel):
    query: str = Field(description="The search question or keywords to look up in the resume, e.g. candidate name or skills")

class LlamaIndexRAGAdapter:
    def __init__(
        self,
        embed_model_name: str,
        pdf_path: str,
        llm: Optional[LLM] = None,
        embed_model: Optional[BaseEmbedding] = None,
        model_name: Optional[str] = None,
        llm_provider: str = "openai_like",
        llm_api_base: Optional[str] = None,
        llm_api_key: Optional[str] = None,
    ):
        # Resolve LLM: Use injected LLM or create via LLMAdapter
        if llm is not None:
            self.llm = llm
        else:
            self.llm = LLMAdapter.create_llm(
                provider=llm_provider,
                model=model_name or "gemini-2.5-flash",
                api_base=llm_api_base,
                api_key=llm_api_key,
            )

        if embed_model is not None:
            self.embed_model = embed_model
        else:
            self.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
        
        # Safe document loading
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Resume PDF not found at: {pdf_path}")
            
        loader = PyMuPDFReader()
        documents = loader.load(file_path=pdf_path)
        
        # Build in-memory index
        self.index = VectorStoreIndex.from_documents(documents, embed_model=self.embed_model)
        self.query_engine = self.index.as_query_engine(llm=self.llm)

        async def search_resume(query: Any = None, **kwargs) -> str:
            print(f"\n[DEBUG] >>> Tool raw query: {query}, kwargs: {kwargs}")
            
            # Extract plain string if LLM passed a dict (e.g. {'query': '...', 'type': 'string'})
            search_str = query
            if isinstance(search_str, dict):
                search_str = search_str.get("query") or search_str.get("description") or list(search_str.values())[0]
            elif not search_str and kwargs:
                search_str = kwargs.get("query") or list(kwargs.values())[0]
                
            search_str = str(search_str).strip() if search_str else "candidate resume"
            print(f"[DEBUG] >>> Tool sanitized query: '{search_str}'")

            try:
                response = await self.query_engine.aquery(search_str)
                print(f"[DEBUG] >>> Tool RESULT: {response}\n")
                return str(response)
            except Exception as e:
                print(f"[DEBUG] >>> Tool ERROR: {e}\n")
                return f"Error searching resume: {e}"

        resume_tool = FunctionTool.from_defaults(
            async_fn=search_resume,
            name="search_resume",
            description="Search the candidate resume to find the person's name, contact information, skills, experience, and education.",
            fn_schema=SearchResumeInput
        )

        self.agent = AgentWorkflow.from_tools_or_functions(
            [resume_tool],
            llm=self.llm,
            system_prompt=(
                "You are an assistant for answering questions about candidate resumes. "
                "You MUST ALWAYS call the 'search_resume' tool to look up information "
                "before answering any question, including the candidate's name or background. "
                "Never say you do not have a database without searching the resume first."
            )
        )

    async def query(self, text: str) -> str:
        handler = self.agent.run(user_msg=text)
        async for event in handler.stream_events():
            print(f"[DEBUG EVENT] {type(event).__name__}")
        response = await handler
        return str(response)

    async def stream(self, text: str) -> AsyncGenerator[str, None]:
        handler = self.agent.run(user_msg=text)
        async for event in handler.stream_events():
            if isinstance(event, AgentStream) and event.delta:
                yield event.delta
        await handler
