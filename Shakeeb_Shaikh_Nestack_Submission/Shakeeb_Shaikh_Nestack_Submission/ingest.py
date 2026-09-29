"""Ingestion pipeline modelled as a LangGraph:  load -> split -> embed -> store

Usage:  python ingest.py --file document.pdf
"""
import argparse
import sys
from pathlib import Path
from typing import TypedDict

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

from common import INDEX_DIR, get_embeddings, save_meta


class IngestState(TypedDict, total=False):
    file: str
    chunk_size: int
    chunk_overlap: int
    index_dir: str
    documents: list[Document]
    chunks: list[Document]
    vectors: list[list[float]]
    stored: int


# ---------------------------------------------------------------- nodes
def load_node(state: IngestState) -> IngestState:
    docs = PyPDFLoader(state["file"]).load()  # one Document per page, metadata["page"] is 0-based
    print(f"[load]  {len(docs)} pages loaded from {state['file']}")
    return {"documents": docs}


def split_node(state: IngestState) -> IngestState:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=state["chunk_size"],
        chunk_overlap=state["chunk_overlap"],
        add_start_index=True,
    )
    chunks = [c for c in splitter.split_documents(state["documents"]) if c.page_content.strip()]
    print(f"[split] {len(chunks)} chunks (size={state['chunk_size']}, overlap={state['chunk_overlap']})")
    return {"chunks": chunks}


def embed_node(state: IngestState) -> IngestState:
    embeddings, _ = get_embeddings()
    vectors = embeddings.embed_documents([c.page_content for c in state["chunks"]])
    print(f"[embed] {len(vectors)} vectors, dim={len(vectors[0])}")
    return {"vectors": vectors}


def store_node(state: IngestState) -> IngestState:
    embeddings, meta = get_embeddings()
    chunks, vectors = state["chunks"], state["vectors"]
    store = FAISS.from_embeddings(
        text_embeddings=list(zip([c.page_content for c in chunks], vectors)),
        embedding=embeddings,
        metadatas=[c.metadata for c in chunks],
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,  # cosine on unit vectors
    )
    store.save_local(state["index_dir"])  # FAISS built-in persistence
    save_meta(state["index_dir"], meta)   # remember which model built the index
    print(f"[store] saved {len(chunks)} chunks to '{state['index_dir']}/'")
    return {"stored": len(chunks)}


# ---------------------------------------------------------------- graph
def build_graph():
    g = StateGraph(IngestState)
    g.add_node("load", load_node)
    g.add_node("split", split_node)
    g.add_node("embed", embed_node)
    g.add_node("store", store_node)
    g.add_edge(START, "load")
    g.add_edge("load", "split")
    g.add_edge("split", "embed")
    g.add_edge("embed", "store")
    g.add_edge("store", END)
    return g.compile()


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest a PDF into a FAISS vector store.")
    ap.add_argument("--file", required=True, help="Path to PDF")
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--chunk-overlap", type=int, default=200)
    ap.add_argument("--index-dir", default=INDEX_DIR)
    ap.add_argument("--show-graph", action="store_true", help="Print node list + Mermaid diagram and exit")
    args = ap.parse_args()

    app = build_graph()
    graph = app.get_graph()
    print("Graph nodes:", " -> ".join(n for n in graph.nodes if n not in ("__start__", "__end__")))
    if args.show_graph:
        print(graph.draw_mermaid())
        return 0

    if not Path(args.file).is_file():
        print(f"File not found: {args.file}", file=sys.stderr)
        return 1

    app.invoke({
        "file": args.file,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "index_dir": args.index_dir,
    })
    print("Ingestion complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
