"""
reranker.py - Lightweight Python-based candidate reranker.

After hybrid retrieval returns ~20 candidates, the reranker scores them
based on query-specific features to improve final ranking.

Design decision: We use a simple feature-based scorer rather than a
cross-encoder model. This keeps the system fast, explainable, and
dependency-light for an internship project.

Features used for reranking:
1. Title match boost — exact or partial title match with query
2. Ingredient coverage — how many query ingredients appear in the recipe
3. Instruction relevance — keyword overlap with instructions
4. Recipe length penalty — penalize very short or empty content
"""
import re
from typing import Optional


def _tokenize(text: str) -> set[str]:
    """Tokenize text into a set of lowercase words."""
    return set(re.findall(r'[a-z0-9]+', text.lower()))


def compute_rerank_score(
    query: str,
    recipe: dict,
    query_ingredients: list[str] = None,
) -> float:
    """Compute a reranking score for a single recipe given a query.
    
    Higher scores = better matches.
    
    Args:
        query: User search query.
        recipe: Full recipe dictionary.
        query_ingredients: Pre-extracted ingredient list from query (optional).
        
    Returns:
        Reranking score (float, higher is better).
    """
    score = 0.0
    query_tokens = _tokenize(query)
    
    if not query_tokens:
        return 0.0
    
    # --- Feature 1: Title match (weight: 3.0) ---
    title_tokens = _tokenize(recipe["title"])
    title_overlap = len(query_tokens & title_tokens)
    title_coverage = title_overlap / max(len(query_tokens), 1)
    score += title_coverage * 3.0
    
    # Exact title match bonus
    if query.lower().strip() in recipe["title"].lower():
        score += 2.0
    
    # --- Feature 2: Ingredient coverage (weight: 2.0) ---
    recipe_ingredients_text = " ".join(recipe["ingredients"]).lower()
    recipe_ing_tokens = _tokenize(recipe_ingredients_text)
    
    if query_ingredients:
        # Use extracted ingredients for more precise matching
        matched = 0
        for ing in query_ingredients:
            ing_lower = ing.lower()
            if ing_lower in recipe_ingredients_text:
                matched += 1
        ingredient_coverage = matched / max(len(query_ingredients), 1)
        score += ingredient_coverage * 2.0
    else:
        # Fallback: token overlap with ingredients
        ing_overlap = len(query_tokens & recipe_ing_tokens)
        score += (ing_overlap / max(len(query_tokens), 1)) * 1.5
    
    # --- Feature 3: Instruction keyword overlap (weight: 0.5) ---
    instruction_tokens = _tokenize(recipe["instructions"][:500])
    inst_overlap = len(query_tokens & instruction_tokens)
    score += (inst_overlap / max(len(query_tokens), 1)) * 0.5
    
    # --- Feature 4: Content quality bonus (weight: 0.3) ---
    # Prefer recipes with more complete information
    if len(recipe["ingredients"]) >= 3:
        score += 0.2
    if len(recipe["instructions"]) >= 100:
        score += 0.1
    
    return round(score, 4)


def rerank_results(
    query: str,
    candidate_ids: list[str],
    recipe_index: dict[str, dict],
    rrf_scores: dict[str, float] = None,
    query_ingredients: list[str] = None,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Rerank retrieval candidates based on query-recipe features.
    
    Combines the original retrieval score (RRF) with feature-based
    reranking to produce a final ordering.
    
    Args:
        query: User search query.
        candidate_ids: List of candidate recipe IDs from hybrid retrieval.
        recipe_index: Dictionary mapping recipe ID -> recipe dict.
        rrf_scores: Optional dict of recipe_id -> RRF score from retrieval.
        query_ingredients: Pre-extracted ingredients from the query.
        top_k: Number of top results to return after reranking.
        
    Returns:
        List of (recipe_id, final_score) tuples, sorted by score descending.
    """
    scored = []
    
    for recipe_id in candidate_ids:
        recipe = recipe_index.get(recipe_id)
        if recipe is None:
            continue
        
        # Compute feature-based rerank score
        rerank_score = compute_rerank_score(query, recipe, query_ingredients)
        
        # Combine with RRF score (if available)
        # RRF scores are small (~0.01-0.03), so we scale them up
        rrf = 0.0
        if rrf_scores and recipe_id in rrf_scores:
            rrf = rrf_scores[recipe_id] * 10.0
        
        final_score = rerank_score + rrf
        scored.append((recipe_id, final_score))
    
    # Sort by final score descending
    scored.sort(key=lambda x: x[1], reverse=True)
    
    return scored[:top_k]

