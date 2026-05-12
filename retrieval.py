from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "airs_docs"
DEFAULT_PERSIST_DIR = Path("./chroma_db")


def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


def _get_collection(persist_dir: Path, collection_name: str):
	client = chromadb.PersistentClient(path=str(persist_dir))
	return client.get_or_create_collection(
		name=collection_name,
		metadata={"hnsw:space": "cosine"},
	)


def _query_collection(
	query_text: str,
	persist_dir: Path,
	collection_name: str,
	top_k: int,
	model_name: str,
) -> Dict[str, Any]:
	model = SentenceTransformer(model_name)
	query_embedding = model.encode([query_text], convert_to_numpy=True)
	collection = _get_collection(persist_dir=persist_dir, collection_name=collection_name)

	result = collection.query(
		query_embeddings=query_embedding.tolist(),
		n_results=top_k,
		include=["documents", "metadatas", "distances"],
	)

	ids = result.get("ids", [[]])[0]
	documents = result.get("documents", [[]])[0]
	metadatas = result.get("metadatas", [[]])[0]
	distances = result.get("distances", [[]])[0]

	hits: List[Dict[str, Any]] = []
	for index, doc_id in enumerate(ids):
		distance = float(distances[index]) if index < len(distances) else None
		similarity = None if distance is None else max(0.0, 1.0 - distance)
		hits.append(
			{
				"rank": index + 1,
				"id": doc_id,
				"document": documents[index] if index < len(documents) else None,
				"metadata": metadatas[index] if index < len(metadatas) else {},
				"distance": distance,
				"similarity": similarity,
			}
		)
	return {"hits": hits, "raw": result}


def retrieve_exact_query(
	query: str,
	persist_dir: Path = DEFAULT_PERSIST_DIR,
	collection_name: str = DEFAULT_COLLECTION,
	top_k: int = 5,
	model_name: str = MODEL_NAME,
) -> Dict[str, Any]:
	query_result = _query_collection(
		query_text=query,
		persist_dir=persist_dir,
		collection_name=collection_name,
		top_k=top_k,
		model_name=model_name,
	)
	return {
		"strategy": "exact_query_retrieval",
		"original_query": query,
		"used_query": query,
		"model": model_name,
		"collection": collection_name,
		"persist_dir": str(persist_dir),
		"top_k": top_k,
		"retrieved_at": _now_iso(),
		"results": query_result["hits"],
	}


def enhance_query_with_gpt4o(
	query: str,
	api_key: Optional[str] = None,
	llm_model: str = "gpt-4o",
) -> str:
	resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
	if not resolved_api_key:
		raise ValueError("OPENAI_API_KEY not found. Set env var or pass api_key.")

	client = OpenAI(api_key=resolved_api_key)
	completion = client.chat.completions.create(
		model=llm_model,
		messages=[
			{
				"role": "system",
				"content": (
					"You are an expert retrieval query rewriter for benchmark evaluation of semantic RAG systems. "
					"Your goal is to improve recall and precision against technical document chunks while preserving user intent. "
					"Rewrite the query into a retrieval-optimized form by: "
					"(1) resolving vague wording into explicit intent, "
					"(2) adding likely technical synonyms/paraphrases, "
					"(3) expanding with concise context terms that help embedding match, and "
					"(4) keeping the query neutral and faithful to the original meaning. "
					"Do not answer the question. Do not add facts not implied by the user query. "
					"Output exactly one rewritten query string only, no bullets, no quotes, no explanation. "
					"Keep length between 12 and 40 words."
				),
			},
			{
				"role": "user",
				"content": (
					f"Original query: {query}\n"
					"Return one optimized retrieval query."
				),
			},
		],
		temperature=0.1,
	)
	return completion.choices[0].message.content.strip()


def retrieve_enhanced_query(
	query: str,
	persist_dir: Path = DEFAULT_PERSIST_DIR,
	collection_name: str = DEFAULT_COLLECTION,
	top_k: int = 5,
	model_name: str = MODEL_NAME,
	api_key: Optional[str] = None,
	llm_model: str = "gpt-4o",
) -> Dict[str, Any]:
	used_fallback = False
	fallback_reason = None
	try:
		enhanced_query = enhance_query_with_gpt4o(query=query, api_key=api_key, llm_model=llm_model)
	except Exception as exc:
		enhanced_query = query
		used_fallback = True
		fallback_reason = str(exc)

	query_result = _query_collection(
		query_text=enhanced_query,
		persist_dir=persist_dir,
		collection_name=collection_name,
		top_k=top_k,
		model_name=model_name,
	)
	return {
		"strategy": "enhanced_query_retrieval",
		"original_query": query,
		"used_query": enhanced_query,
		"query_rewriter_model": llm_model,
		"embedding_model": model_name,
		"collection": collection_name,
		"persist_dir": str(persist_dir),
		"top_k": top_k,
		"retrieved_at": _now_iso(),
		"enhancement_fallback_used": used_fallback,
		"enhancement_fallback_reason": fallback_reason,
		"results": query_result["hits"],
	}


def build_retrieval_metadata_json(
	query: str,
	persist_dir: Path = DEFAULT_PERSIST_DIR,
	collection_name: str = DEFAULT_COLLECTION,
	top_k: int = 5,
	model_name: str = MODEL_NAME,
	api_key: Optional[str] = None,
	llm_model: str = "gpt-4o",
	output_json_path: Optional[Path] = None,
) -> Dict[str, Any]:
	exact_metadata = retrieve_exact_query(
		query=query,
		persist_dir=persist_dir,
		collection_name=collection_name,
		top_k=top_k,
		model_name=model_name,
	)
	enhanced_metadata = retrieve_enhanced_query(
		query=query,
		persist_dir=persist_dir,
		collection_name=collection_name,
		top_k=top_k,
		model_name=model_name,
		api_key=api_key,
		llm_model=llm_model,
	)

	benchmark_ready_json = {
		"query": query,
		"created_at": _now_iso(),
		"exact_retrieval_metadata": exact_metadata,
		"enhanced_retrieval_metadata": enhanced_metadata,
	}

	if output_json_path is not None:
		output_json_path.parent.mkdir(parents=True, exist_ok=True)
		output_json_path.write_text(json.dumps(benchmark_ready_json, indent=2), encoding="utf-8")

	return benchmark_ready_json
