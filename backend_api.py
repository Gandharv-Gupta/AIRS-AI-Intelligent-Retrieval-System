from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from RAG import chunk_text, extract_text_from_pdf, store_chunks_in_chroma
from retrieval import build_retrieval_metadata_json, retrieve_enhanced_query, retrieve_exact_query

app = FastAPI(title="AIRS Backend API")

DEFAULT_COLLECTION = "airs_docs"
DEFAULT_PERSIST_DIR = Path("./chroma_db")
UPLOAD_DIR = Path("./uploaded_pdfs")


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    collection_name: str = DEFAULT_COLLECTION
    persist_dir: str = str(DEFAULT_PERSIST_DIR)


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    collection_name: str = DEFAULT_COLLECTION,
    persist_dir: str = str(DEFAULT_PERSIST_DIR),
    chunk_size: int = 800,
    chunk_overlap: int = 120,
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    saved_path = UPLOAD_DIR / file.filename
    content = await file.read()
    saved_path.write_bytes(content)

    try:
        text = extract_text_from_pdf(saved_path)
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            raise ValueError("No chunks created from uploaded PDF.")

        stored_count = store_chunks_in_chroma(
            chunks=chunks,
            persist_directory=persist_path,
            collection_name=collection_name,
            source_path=saved_path,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to ingest PDF: {exc}") from exc

    return {
        "message": "PDF ingested successfully",
        "file": str(saved_path),
        "collection": collection_name,
        "persist_dir": str(persist_path),
        "chunks_stored": stored_count,
    }


@app.post("/retrieve/exact")
def retrieve_exact(payload: RetrievalRequest):
    try:
        result = retrieve_exact_query(
            query=payload.query,
            persist_dir=Path(payload.persist_dir),
            collection_name=payload.collection_name,
            top_k=payload.top_k,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Exact retrieval failed: {exc}") from exc


@app.post("/retrieve/enhanced")
def retrieve_enhanced(payload: RetrievalRequest):
    try:
        result = retrieve_enhanced_query(
            query=payload.query,
            persist_dir=Path(payload.persist_dir),
            collection_name=payload.collection_name,
            top_k=payload.top_k,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Enhanced retrieval failed: {exc}") from exc


@app.post("/benchmark")
def benchmark_retrieval(payload: RetrievalRequest):
    try:
        result = build_retrieval_metadata_json(
            query=payload.query,
            persist_dir=Path(payload.persist_dir),
            collection_name=payload.collection_name,
            top_k=payload.top_k,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Benchmark generation failed: {exc}") from exc
