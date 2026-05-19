from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from rag.loader import get_vectorstore


def _rrf_merge(
    vector_results: list,
    bm25_results: list,
    k: int = 60,
) -> list[str]:
    """Reciprocal Rank Fusion: score = 1 / (k + rank), ranks are 1-indexed."""
    scores: dict[str, float] = {}

    for rank, doc in enumerate(vector_results, start=1):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    for rank, doc in enumerate(bm25_results, start=1):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    return [text for text, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def hybrid_search(query: str, collection_name: str, top_k: int = 5) -> list[str]:
    vectorstore: Chroma = get_vectorstore(collection_name)
    fetch_k = top_k * 2

    # --- 向量检索 ---
    vector_results = vectorstore.similarity_search(query, k=fetch_k)

    # --- BM25 检索 ---
    # 从向量库拉取全量文档文本用于 BM25（适合小规模知识库）
    raw_collection = vectorstore._collection.get(include=["documents"])
    all_texts: list[str] = raw_collection["documents"] or []

    bm25_results = []
    if all_texts:
        tokenized = [text.split() for text in all_texts]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(query.split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:fetch_k]

        class _Doc:
            def __init__(self, text: str):
                self.page_content = text

        bm25_results = [_Doc(all_texts[i]) for i in top_indices]

    # --- RRF 融合 ---
    merged = _rrf_merge(vector_results, bm25_results)
    return merged[:top_k]
