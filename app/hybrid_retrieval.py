"""
hybrid_retrieval.py - Combine BM25 and FAISS results using Reciprocal Rank Fusion.

Reciprocal Rank Fusion (RRF) is a simple, effective method to combine ranked
lists from different retrieval systems. For each document, RRF computes:

    RRF_score = sum(1 / (k + rank_i)) for each retrieval system i

where k is a constant (typically 60) that dampens the impact of high ranks.

Why RRF over other fusion methods?
- No score normalization needed (BM25 and cosine similarity have different scales)
- No training required (unlike learned fusion)
- Robust — if one system misses a result, the other can still surface it
- Well-studied in IR literature (Cormack et al., 2009)
"""
from typing import Optional


# RRF constant k — standard value from the literature
RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """Combine multiple ranked result lists using Reciprocal Rank Fusion.
    
    Args:
        ranked_lists: List of ranked lists. Each ranked list contains
                      (recipe_id, score) tuples sorted by score descending.
        k: RRF damping constant.
        
    Returns:
        Fused list of (recipe_id, rrf_score) tuples, sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    
    for ranked_list in ranked_lists:
        for rank, (recipe_id, _original_score) in enumerate(ranked_list):
            # RRF formula: 1 / (k + rank + 1)
            # rank is 0-indexed, so rank+1 gives 1-indexed rank
            rrf_scores[recipe_id] = rrf_scores.get(recipe_id, 0.0) + 1.0 / (k + rank + 1)
    
    # Sort by RRF score descending
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    return fused


def deduplicate_results(results: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Remove duplicate recipe IDs, keeping the highest-scoring occurrence.
    
    This is technically handled by RRF (which aggregates scores for the same ID),
    but useful as a safety net for single-source results.
    
    Args:
        results: List of (recipe_id, score) tuples.
        
    Returns:
        Deduplicated list preserving order.
    """
    seen = set()
    deduped = []
    for recipe_id, score in results:
        if recipe_id not in seen:
            seen.add(recipe_id)
            deduped.append((recipe_id, score))
    return deduped


def hybrid_search(
    query: str,
    bm25_retriever,
    faiss_retriever,
    top_k: int = 20,
    bm25_top_k: int = 30,
    faiss_top_k: int = 30,
) -> list[tuple[str, float]]:
    """Perform hybrid retrieval combining BM25 and FAISS results.
    
    Pipeline:
    1. Get top results from BM25 (keyword matching)
    2. Get top results from FAISS (semantic similarity)
    3. Combine using Reciprocal Rank Fusion
    4. Deduplicate
    5. Return top-k fused results
    
    Args:
        query: User search query.
        bm25_retriever: BM25Retriever instance.
        faiss_retriever: FAISSRetriever instance (can be None).
        top_k: Number of final results to return.
        bm25_top_k: Number of BM25 candidates to fetch.
        faiss_top_k: Number of FAISS candidates to fetch.
        
    Returns:
        List of (recipe_id, rrf_score) tuples.
    """
    ranked_lists = []
    
    # BM25 keyword retrieval
    bm25_results = bm25_retriever.search(query, top_k=bm25_top_k)
    if bm25_results:
        ranked_lists.append(bm25_results)
    
    # FAISS semantic retrieval (optional — may not be available)
    if faiss_retriever is not None:
        faiss_results = faiss_retriever.search(query, top_k=faiss_top_k)
        if faiss_results:
            ranked_lists.append(faiss_results)
    
    if not ranked_lists:
        return []
    
    # Fuse results
    fused = reciprocal_rank_fusion(ranked_lists)
    
    # Deduplicate (safety net)
    fused = deduplicate_results(fused)
    
    return fused[:top_k]

