from __future__ import annotations

from typing import Any, Dict, List

import requests
import streamlit as st

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
DEFAULT_COLLECTION = "airs_docs"
DEFAULT_PERSIST_DIR = "./chroma_db"


def call_upload_api(
	backend_url: str,
	file_name: str,
	file_bytes: bytes,
	collection_name: str,
	persist_dir: str,
	chunk_size: int,
	chunk_overlap: int,
) -> Dict[str, Any]:
	url = f"{backend_url.rstrip('/')}/upload"
	files = {"file": (file_name, file_bytes, "application/pdf")}
	params = {
		"collection_name": collection_name,
		"persist_dir": persist_dir,
		"chunk_size": chunk_size,
		"chunk_overlap": chunk_overlap,
	}
	response = requests.post(url, files=files, params=params, timeout=180)
	response.raise_for_status()
	return response.json()


def call_retrieval_api(
	backend_url: str,
	endpoint: str,
	query: str,
	collection_name: str,
	persist_dir: str,
	top_k: int,
) -> Dict[str, Any]:
	url = f"{backend_url.rstrip('/')}{endpoint}"
	payload = {
		"query": query,
		"top_k": top_k,
		"collection_name": collection_name,
		"persist_dir": persist_dir,
	}
	response = requests.post(url, json=payload, timeout=180)
	response.raise_for_status()
	return response.json()


def call_benchmark_api(
	backend_url: str,
	query: str,
	collection_name: str,
	persist_dir: str,
	top_k: int,
) -> Dict[str, Any]:
	url = f"{backend_url.rstrip('/')}/benchmark"
	payload = {
		"query": query,
		"top_k": top_k,
		"collection_name": collection_name,
		"persist_dir": persist_dir,
	}
	response = requests.post(url, json=payload, timeout=240)
	response.raise_for_status()
	return response.json()


def render_hits(result: Dict[str, Any]) -> None:
	hits: List[Dict[str, Any]] = result.get("results", [])
	if not hits:
		st.warning("No results returned.")
		return

	for item in hits:
		rank = item.get("rank")
		similarity = item.get("similarity")
		distance = item.get("distance")
		metadata = item.get("metadata", {})
		document = item.get("document", "")

		header = f"Rank {rank}"
		if similarity is not None:
			header += f" | similarity={similarity:.4f}"
		if distance is not None:
			header += f" | distance={distance:.4f}"

		with st.expander(header, expanded=(rank == 1)):
			st.write(document)
			st.caption(f"Metadata: {metadata}")


def render_benchmark_charts(benchmark_result: Dict[str, Any]) -> None:
	if not benchmark_result:
		st.info("No benchmark data available yet.")
		return

	exact = benchmark_result.get("exact_retrieval_metadata", {})
	enhanced = benchmark_result.get("enhanced_retrieval_metadata", {})

	exact_hits: List[Dict[str, Any]] = exact.get("results", [])
	enhanced_hits: List[Dict[str, Any]] = enhanced.get("results", [])

	exact_map = {item.get("rank"): item for item in exact_hits if isinstance(item.get("rank"), int)}
	enhanced_map = {item.get("rank"): item for item in enhanced_hits if isinstance(item.get("rank"), int)}
	ranks = sorted(set(exact_map.keys()) | set(enhanced_map.keys()))

	if not ranks:
		st.info("Benchmark returned no ranked results.")
		return

	similarity_values: List[Dict[str, Any]] = []
	distance_values: List[Dict[str, Any]] = []

	for rank in ranks:
		exact_item = exact_map.get(rank, {})
		enhanced_item = enhanced_map.get(rank, {})

		exact_similarity = exact_item.get("similarity")
		enhanced_similarity = enhanced_item.get("similarity")
		exact_distance = exact_item.get("distance")
		enhanced_distance = enhanced_item.get("distance")

		if exact_similarity is not None:
			similarity_values.append({"rank": rank, "approach": "Exact", "score": exact_similarity})
		if enhanced_similarity is not None:
			similarity_values.append({"rank": rank, "approach": "Enhanced", "score": enhanced_similarity})

		if exact_distance is not None:
			distance_values.append({"rank": rank, "approach": "Exact", "score": exact_distance})
		if enhanced_distance is not None:
			distance_values.append({"rank": rank, "approach": "Enhanced", "score": enhanced_distance})

	col1, col2 = st.columns(2)

	with col1:
		st.markdown("#### Similarity: Exact vs Enhanced")
		st.vega_lite_chart(
			{
				"mark": {"type": "line", "point": True},
				"encoding": {
					"x": {"field": "rank", "type": "ordinal", "title": "Rank"},
					"y": {"field": "score", "type": "quantitative", "title": "Similarity"},
					"color": {"field": "approach", "type": "nominal", "title": "Approach"},
				},
				"data": {"values": similarity_values},
			},
			use_container_width=True,
		)

	with col2:
		st.markdown("#### Distance: Exact vs Enhanced")
		st.vega_lite_chart(
			{
				"mark": {"type": "line", "point": True},
				"encoding": {
					"x": {"field": "rank", "type": "ordinal", "title": "Rank"},
					"y": {"field": "score", "type": "quantitative", "title": "Distance"},
					"color": {"field": "approach", "type": "nominal", "title": "Approach"},
				},
				"data": {"values": distance_values},
			},
			use_container_width=True,
		)


def main() -> None:
	st.set_page_config(page_title="AIRS Frontend", layout="wide")
	st.title("AIRS — AI Intelligent Retrieval System")

	if "exact_result" not in st.session_state:
		st.session_state["exact_result"] = None
	if "enhanced_result" not in st.session_state:
		st.session_state["enhanced_result"] = None
	if "exact_error" not in st.session_state:
		st.session_state["exact_error"] = None
	if "enhanced_error" not in st.session_state:
		st.session_state["enhanced_error"] = None
	if "benchmark_result" not in st.session_state:
		st.session_state["benchmark_result"] = None
	if "benchmark_error" not in st.session_state:
		st.session_state["benchmark_error"] = None

	with st.sidebar:
		st.subheader("Retrieval Settings")
		top_k = st.slider("Top K", min_value=1, max_value=20, value=5)
		chunk_size = st.number_input("Chunk Size", min_value=100, max_value=5000, value=800, step=50)
		chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=1000, value=120, step=10)

	backend_url = DEFAULT_BACKEND_URL
	collection_name = DEFAULT_COLLECTION
	persist_dir = DEFAULT_PERSIST_DIR

	st.subheader("1) Upload PDF")
	uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

	if st.button("Upload and Process PDF", type="primary", use_container_width=True):
		if uploaded_file is None:
			st.error("Please select a PDF first.")
		else:
			st.info("Upload started. Processing chunking, embeddings, and Chroma storage...")
			try:
				result = call_upload_api(
					backend_url=backend_url,
					file_name=uploaded_file.name,
					file_bytes=uploaded_file.getvalue(),
					collection_name=collection_name,
					persist_dir=persist_dir,
					chunk_size=int(chunk_size),
					chunk_overlap=int(chunk_overlap),
				)
				st.success("Processing complete.")
				st.json(result)
			except requests.RequestException as exc:
				st.error(f"Upload failed: {exc}")

	st.subheader("2) Ask Query")
	query = st.text_area("Enter your query", height=120, placeholder="Ask about the uploaded PDF...")

	col1, col2 = st.columns(2)

	with col1:
		if st.button("Exact Query Search", use_container_width=True):
			if not query.strip():
				st.error("Please enter a query.")
			else:
				with st.spinner("Running exact retrieval..."):
					try:
						st.session_state["exact_result"] = call_retrieval_api(
							backend_url=backend_url,
							endpoint="/retrieve/exact",
							query=query.strip(),
							collection_name=collection_name,
							persist_dir=persist_dir,
							top_k=int(top_k),
						)
						st.session_state["exact_error"] = None
					except requests.RequestException as exc:
						st.session_state["exact_error"] = str(exc)

	with col2:
		if st.button("Enhanced Query Search", use_container_width=True):
			if not query.strip():
				st.error("Please enter a query.")
			else:
				with st.spinner("Running enhanced retrieval..."):
					try:
						st.session_state["enhanced_result"] = call_retrieval_api(
							backend_url=backend_url,
							endpoint="/retrieve/enhanced",
							query=query.strip(),
							collection_name=collection_name,
							persist_dir=persist_dir,
							top_k=int(top_k),
						)
						st.session_state["enhanced_error"] = None
					except requests.RequestException as exc:
						st.session_state["enhanced_error"] = str(exc)

	if st.button("Generate Benchmark Results", use_container_width=True):
		if not query.strip():
			st.error("Please enter a query.")
		else:
			with st.spinner("Generating benchmark results..."):
				try:
					st.session_state["benchmark_result"] = call_benchmark_api(
						backend_url=backend_url,
						query=query.strip(),
						collection_name=collection_name,
						persist_dir=persist_dir,
						top_k=int(top_k),
					)
					st.session_state["benchmark_error"] = None
				except requests.RequestException as exc:
					st.session_state["benchmark_error"] = str(exc)

	results_col1, results_col2 = st.columns(2)

	with results_col1:
		st.markdown("### Exact Query Results")
		if st.session_state["exact_error"]:
			st.error(f"Exact retrieval failed: {st.session_state['exact_error']}")
		exact_result = st.session_state["exact_result"]
		if exact_result:
			st.success("Exact retrieval complete.")
			st.json({
				"strategy": exact_result.get("strategy"),
				"used_query": exact_result.get("used_query"),
				"top_k": exact_result.get("top_k"),
			})
			render_hits(exact_result)

	with results_col2:
		st.markdown("### Enhanced Query Results")
		if st.session_state["enhanced_error"]:
			st.error(f"Enhanced retrieval failed: {st.session_state['enhanced_error']}")
		enhanced_result = st.session_state["enhanced_result"]
		if enhanced_result:
			st.success("Enhanced retrieval complete.")
			st.json({
				"strategy": enhanced_result.get("strategy"),
				"used_query": enhanced_result.get("used_query"),
				"query_rewriter_model": enhanced_result.get("query_rewriter_model"),
				"top_k": enhanced_result.get("top_k"),
			})
			render_hits(enhanced_result)

	st.markdown("### Benchmark Comparison")
	if st.session_state["benchmark_error"]:
		st.error(f"Benchmark generation failed: {st.session_state['benchmark_error']}")
	benchmark_result = st.session_state["benchmark_result"]
	if benchmark_result:
		st.json(
			{
				"query": benchmark_result.get("query"),
				"created_at": benchmark_result.get("created_at"),
				"exact_strategy": benchmark_result.get("exact_retrieval_metadata", {}).get("strategy"),
				"enhanced_strategy": benchmark_result.get("enhanced_retrieval_metadata", {}).get("strategy"),
			}
		)
		render_benchmark_charts(benchmark_result)


if __name__ == "__main__":
	main()
