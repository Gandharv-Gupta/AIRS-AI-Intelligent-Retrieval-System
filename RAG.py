from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PDF_PATH = Path("/Users/Gandharv.x.Gupta/Desktop/AIRS-AI Intelligent Retrieval System/atomic_habits.pdf")


def extract_text_from_pdf(pdf_path: Path) -> str:
	reader = PdfReader(str(pdf_path))
	pages = [page.extract_text() or "" for page in reader.pages]
	text = "\n".join(pages).strip()
	if not text:
		raise ValueError(f"No extractable text found in PDF: {pdf_path}")
	return text


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> List[str]:
	clean_text = " ".join(text.split())
	if chunk_overlap >= chunk_size:
		raise ValueError("chunk_overlap must be smaller than chunk_size")
	chunks: List[str] = []
	start = 0
	text_length = len(clean_text)

	while start < text_length:
		end = min(start + chunk_size, text_length)
		chunk = clean_text[start:end].strip()
		if chunk:
			chunks.append(chunk)
		if end == text_length:
			break
		start = end - chunk_overlap
	return chunks


def store_chunks_in_chroma(
	chunks: List[str],
	persist_directory: Path,
	collection_name: str,
	source_path: Path,
) -> int:
	model = SentenceTransformer(MODEL_NAME)
	embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

	client = chromadb.PersistentClient(path=str(persist_directory))
	collection = client.get_or_create_collection(
		name=collection_name,
		metadata={"hnsw:space": "cosine"},
	)

	ids = [f"{source_path.stem}-{i}" for i in range(len(chunks))]
	metadatas = [{"source": str(source_path), "chunk_index": i} for i in range(len(chunks))]

	collection.upsert(
		ids=ids,
		documents=chunks,
		embeddings=embeddings.tolist(),
		metadatas=metadatas,
	)
	return len(chunks)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Ingest a PDF, chunk text, embed with all-MiniLM-L6-v2, and store vectors in ChromaDB.",
	)
	parser.add_argument("--collection", default="airs_docs", help="Chroma collection name")
	parser.add_argument("--persist-dir", type=Path, default=Path("./chroma_db"), help="Chroma persistence directory")
	parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size in characters")
	parser.add_argument("--chunk-overlap", type=int, default=120, help="Chunk overlap in characters")
	return parser


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()

	pdf_path: Path = PDF_PATH
	if not pdf_path.exists() or not pdf_path.is_file():
		raise FileNotFoundError(f"PDF file not found: {pdf_path}")

	text = extract_text_from_pdf(pdf_path)
	chunks = chunk_text(text, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
	if not chunks:
		raise ValueError("No chunks were created from the input PDF")

	args.persist_dir.mkdir(parents=True, exist_ok=True)
	total = store_chunks_in_chroma(
		chunks=chunks,
		persist_directory=args.persist_dir,
		collection_name=args.collection,
		source_path=pdf_path,
	)

	print(f"Ingested {total} chunks from {pdf_path.name} into collection '{args.collection}'.")
	print(f"Model used: {MODEL_NAME}")
	print(f"Chroma DB persisted at: {args.persist_dir.resolve()}")


if __name__ == "__main__":
	main()
