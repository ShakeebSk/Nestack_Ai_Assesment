"""Shared config so ingestion and retrieval ALWAYS use the same embedding model."""
import json
import os
from pathlib import Path

from langchain_core.embeddings import Embeddings

INDEX_DIR = os.getenv("INDEX_DIR", "faiss_index")
META_FILE = "embedding_meta.json"

HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_MODEL = "text-embedding-3-small"


def get_embeddings(provider: str | None = None) -> tuple[Embeddings, dict]:
    """Return (embeddings, meta). Provider: huggingface (default) | openai | fake."""
    provider = (provider or os.getenv("EMBEDDING_PROVIDER", "huggingface")).lower()
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        model = os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_MODEL)
        return OpenAIEmbeddings(model=model), {"provider": "openai", "model": model}
    if provider == "fake":  # offline smoke tests only - NOT semantically meaningful
        from langchain_core.embeddings import DeterministicFakeEmbedding

        return DeterministicFakeEmbedding(size=384), {"provider": "fake", "model": "fake-384"}
    from langchain_huggingface import HuggingFaceEmbeddings

    # normalize_embeddings=True -> unit vectors -> inner product == cosine similarity
    emb = HuggingFaceEmbeddings(model_name=HF_MODEL, encode_kwargs={"normalize_embeddings": True})
    return emb, {"provider": "huggingface", "model": HF_MODEL}


def save_meta(index_dir: str, meta: dict) -> None:
    Path(index_dir, META_FILE).write_text(json.dumps(meta, indent=2))


def load_meta(index_dir: str) -> dict:
    return json.loads(Path(index_dir, META_FILE).read_text())
