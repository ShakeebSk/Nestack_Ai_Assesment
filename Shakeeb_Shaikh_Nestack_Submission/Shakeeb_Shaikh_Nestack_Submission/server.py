"""Retrieval API. No LLM, no chain: query -> embed -> similarity_search_with_score -> raw chunks."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from pydantic import BaseModel, Field

from common import INDEX_DIR, get_embeddings, load_meta

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    meta = load_meta(INDEX_DIR)                      # model used at ingestion time
    embeddings, _ = get_embeddings(meta["provider"])  # -> same model for queries
    state["store"] = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        allow_dangerous_deserialization=True,  # index is produced by our own ingest.py
    )
    state["meta"] = meta
    yield
    state.clear()


app = FastAPI(title="PDF Vector Retrieval", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=4, ge=1, le=50)


class ChunkResult(BaseModel):
    chunk_text: str
    page_number: int
    score: float  # cosine similarity, higher = more similar


@app.get("/health")
def health():
    return {"status": "ok", "embedding": state.get("meta")}


@app.post("/query", response_model=list[ChunkResult])
def query(req: QueryRequest):
    if "store" not in state:
        raise HTTPException(503, "Vector store not loaded")
    hits = state["store"].similarity_search_with_score(req.query, k=req.top_k)
    return [
        ChunkResult(
            chunk_text=doc.page_content,
            page_number=int(doc.metadata.get("page", 0)) + 1,  # PyPDFLoader is 0-based
            score=round(float(score), 4),
        )
        for doc, score in hits
    ]
