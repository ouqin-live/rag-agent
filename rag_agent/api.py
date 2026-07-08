"""FastAPI service exposing the RAG Agent capabilities."""

from __future__ import annotations

import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag_agent.agent import Agent, AgentConfig, ChatResponse
from rag_agent.config import Settings, get_settings
from rag_agent.embedder import get_embedder
from rag_agent.evaluation import Evaluator
from rag_agent.knowledge import KnowledgeBase
from rag_agent.knowledge.reranker import EmbeddingReranker
from rag_agent.llm import get_llm_client
from rag_agent.memory import LongTermMemory, MediumTermMemory, ShortTermMemory

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Request / response schemas
# ------------------------------------------------------------------
class ChatRequest(BaseModel):
    user_id: str = Field(default="default", description="Unique user identifier")
    question: str = Field(..., description="User question")
    stream: bool = Field(default=False, description="Whether to stream the response")


class ChatResponseModel(BaseModel):
    answer: str
    contexts: list[str] = Field(default_factory=list)
    long_term_facts: list[str] = Field(default_factory=list)
    overall_score: float | None = None
    failed_rules: list[str] = Field(default_factory=list)


class DocumentAddResponse(BaseModel):
    doc_id: str | None = None
    chunk_ids: list[str]
    message: str


# ------------------------------------------------------------------
# Application state
# ------------------------------------------------------------------
class AppState:
    """Shared state bound to the FastAPI application."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.agent: Agent | None = None


# ------------------------------------------------------------------
# Lifespan: initialise the agent once on startup
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _setup_logging(settings)

    state = AppState(settings)
    state.agent = _build_agent(settings)
    app.state.app_state = state

    logger.info(
        "RAG Agent API ready (KB=%d chunks, LTM=%d facts)",
        len(state.agent.config.knowledge_base),
        len(state.agent.config.long_term_memory) if state.agent.config.long_term_memory else 0,
    )
    yield
    # Cleanup if needed


def _setup_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _build_agent(settings: Settings) -> Agent:
    """Build the default Agent from application settings."""
    embedder = get_embedder()

    kb = KnowledgeBase.from_chroma_store(
        store_path=settings.kb_store_path,
        embedder=embedder,
    )
    kb.reranker = EmbeddingReranker(embedder)

    ltm = LongTermMemory.from_chroma_store(
        store_path=settings.memory_store_path,
        embedder=embedder,
    )

    llm = get_llm_client()

    evaluator = Evaluator.with_llm(
        llm=llm,
        db_path=settings.eval_db_path,
    )

    config = AgentConfig(
        knowledge_base=kb,
        short_term_memory=ShortTermMemory(),
        medium_term_memory=MediumTermMemory(llm_client=llm),
        long_term_memory=ltm,
        evaluator=evaluator,
        llm_client=llm,
    )
    return Agent(config)


app = FastAPI(
    title="RAG Agent API",
    description="Memory + Knowledge Base + Auto-evaluation RAG Agent",
    version="0.2.0",
    lifespan=lifespan,
)


def _get_state() -> AppState:
    return app.state.app_state  # type: ignore[return-value]


def _to_chat_response_model(resp: ChatResponse) -> ChatResponseModel:
    return ChatResponseModel(
        answer=resp.answer,
        contexts=[r.text for r in resp.contexts],
        long_term_facts=resp.long_term_facts,
        overall_score=resp.evaluation.overall_score if resp.evaluation else None,
        failed_rules=resp.evaluation.failed_rules if resp.evaluation else [],
    )


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponseModel)
async def chat(request: ChatRequest):
    """Run a single conversational turn."""
    state = _get_state()
    if state.agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised")

    resp = await state.agent.achat(request.user_id, request.question)
    return _to_chat_response_model(resp)


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream the answer using Server-Sent Events."""
    state = _get_state()
    if state.agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised")

    async def event_generator():
        async for chunk in state.agent.achat_stream(request.user_id, request.question):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


@app.post("/documents", response_model=DocumentAddResponse)
async def add_document(
    file: UploadFile | None = File(default=None),
    source: str | None = Form(default=None),
    metadata: str | None = Form(default=None),
):
    """Add a document to the knowledge base.

    Either upload a file directly or provide a local path via ``source``.
    """
    state = _get_state()
    if state.agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised")

    if file is not None:
        suffix = Path(file.filename or "upload").suffix or ".txt"
        tmp_path = Path(tempfile.gettempdir()) / f"rag_agent_upload_{file.filename or 'tmp'}{suffix}"
        try:
            with tmp_path.open("wb") as f:
                shutil.copyfileobj(file.file, f)
        finally:
            file.file.close()
        source_path = str(tmp_path)
    elif source:
        source_path = source
    else:
        raise HTTPException(
            status_code=400, detail="Either 'file' or 'source' must be provided"
        )

    meta: dict[str, Any] = {}
    if metadata:
        import json

        try:
            meta = json.loads(metadata)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400, detail=f"metadata must be valid JSON: {exc}"
            ) from exc

    try:
        chunk_ids = state.agent.config.knowledge_base.add_document(
            source_path, metadata=meta
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to add document")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if file is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    return DocumentAddResponse(
        doc_id=meta.get("doc_id"),
        chunk_ids=chunk_ids,
        message=f"Added {len(chunk_ids)} chunks from {source_path}",
    )


@app.delete("/documents/{doc_id}")
async def remove_document(doc_id: str):
    """Remove a document and all its chunks from the knowledge base."""
    state = _get_state()
    if state.agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised")

    try:
        state.agent.config.knowledge_base.remove_document(doc_id)
    except Exception as exc:
        logger.exception("Failed to remove document")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"doc_id": doc_id, "message": "Document removed"}


@app.get("/memory/{user_id}")
async def get_memory(user_id: str, query: str = "", top_k: int = 20):
    """Retrieve long-term memory facts for a user."""
    state = _get_state()
    ltm = state.agent.config.long_term_memory if state.agent else None
    if ltm is None:
        raise HTTPException(status_code=503, detail="Long-term memory not configured")

    facts = ltm.recall(user_id, query, top_k=top_k)
    return {
        "user_id": user_id,
        "facts": [
            {"id": f.id, "content": f.content, "created_at": f.created_at.isoformat()}
            for f in facts
        ],
    }


@app.get("/evaluations/reports")
async def get_evaluation_report(threshold: float | None = None, limit: int = 20):
    """Generate a text report of recent failure cases."""
    state = _get_state()
    if state.agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised")

    report = state.agent.generate_failure_report(threshold=threshold, limit=limit)
    return {"report": report or "No failures found"}


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "rag_agent.api:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
