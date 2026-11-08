"""
recipe_search.py - Orchestrate the full retrieval pipeline for recipe search.

This module ties together:
1. Query parsing
2. BM25 + FAISS hybrid retrieval
3. RRF fusion
4. Hard constraint filtering
5. Reranking
6. Original record retrieval

It enforces a critical principle: NEVER trust similarity scores as proof
that a recipe satisfies a constraint. Always verify against original records.
"""
from typing import Optional

from app.query_parser import parse_query, INTENT_SEARCH, INTENT_FOLLOWUP
from app.hybrid_retrieval import hybrid_search
from app.reranker import rerank_results
from app.data_processor import normalize_ingredient_name


def filter_by_ingredients(
    recipe: dict,
    required_ingredients: list[str],
    excluded_ingredients: list[str] = None,
) -> bool:
    """Check if a recipe satisfies ingredient-based hard constraints.
    
    This is a DETERMINISTIC check against the original recipe record.
    We never rely on FAISS similarity to determine ingredient presence.
    
    Args:
        recipe: Full recipe dictionary with original ingredients.
        required_ingredients: Ingredients that MUST be in the recipe.
        excluded_ingredients: Ingredients that MUST NOT be in the recipe.
        
    Returns:
        True if the recipe passes all constraints.
    """
    recipe_text = " ".join(recipe["ingredients"]).lower()
    
    # Check required ingredients
    for ing in required_ingredients:
        if ing.lower() not in recipe_text:
            return False
    
    # Check excluded ingredients
    if excluded_ingredients:
        for ing in excluded_ingredients:
            if ing.lower() in recipe_text:
                return False
    
    return True


def filter_by_dietary(recipe: dict, restrictions: list[str]) -> bool:
    """Check if a recipe likely satisfies dietary restrictions.
    
    This is a best-effort filter since the dataset has no explicit
    dietary labels. We check for the ABSENCE of restricted ingredients.
    
    Known limitation: This can produce false positives (e.g., a recipe
    might not list "butter" explicitly but use "cream sauce" which
    contains dairy).
    
    Args:
        recipe: Full recipe dictionary.
        restrictions: List of dietary restriction labels.
        
    Returns:
        True if the recipe likely satisfies the restrictions.
    """
    recipe_text = " ".join(recipe["ingredients"]).lower()
    
    # Map dietary restrictions to ingredient exclusion lists
    _RESTRICTION_EXCLUSIONS = {
        "vegetarian": ["chicken", "beef", "pork", "lamb", "fish", "salmon",
                       "tuna", "shrimp", "bacon", "sausage", "turkey",
                       "steak", "ground meat", "veal", "duck", "crab",
                       "lobster", "anchov"],
        "vegan": ["chicken", "beef", "pork", "lamb", "fish", "salmon",
                  "egg", "milk", "cream", "butter", "cheese", "yogurt",
                  "honey", "gelatin", "bacon", "sausage", "turkey",
                  "shrimp", "steak", "whey", "casein"],
        "gluten-free": ["flour", "bread", "pasta", "noodle", "wheat",
                        "breadcrumb", "cracker", "tortilla", "couscous",
                        "barley", "rye"],
        "dairy-free": ["milk", "cream", "butter", "cheese", "yogurt",
                       "whey", "casein", "sour cream", "ice cream"],
        "nut-free": ["almond", "walnut", "pecan", "cashew", "pistachio",
                     "hazelnut", "macadamia", "peanut", "pine nut"],
    }
    
    for restriction in restrictions:
        exclusions = _RESTRICTION_EXCLUSIONS.get(restriction, [])
        for excluded in exclusions:
            if excluded in recipe_text:
                return False
    
    return True


def search_recipes(
    query: str,
    bm25_retriever,
    faiss_retriever,
    recipe_index: dict[str, dict],
    parsed_query: dict = None,
    top_k: int = 5,
) -> dict:
    """Execute the full recipe search pipeline.
    
    Pipeline:
    1. Parse query -> intent + constraints
    2. Hybrid retrieval (BM25 + FAISS) -> ~30 candidates
    3. RRF fusion -> ~20 ranked candidates
    4. Hard constraint filtering (ingredients, dietary) -> filtered set
    5. Reranking -> final top-k
    6. Fetch original records -> response
    
    Args:
        query: User search query.
        bm25_retriever: BM25Retriever instance.
        faiss_retriever: FAISSRetriever instance (can be None).
        recipe_index: Dictionary mapping recipe ID -> recipe dict.
        parsed_query: Pre-parsed query (if already available from router).
        top_k: Number of final results to return.
        
    Returns:
        Dictionary with:
            - recipes: list of full recipe dicts
            - recipe_ids: list of matched recipe IDs
            - query_info: parsed query details
            - total_candidates: number before filtering
            - total_after_filter: number after constraint filtering
    """
    # Step 1: Parse query
    if parsed_query is None:
        parsed_query = parse_query(query)
    
    # Step 2-3: Hybrid retrieval with RRF
    fused_results = hybrid_search(
        query=query,
        bm25_retriever=bm25_retriever,
        faiss_retriever=faiss_retriever,
        top_k=30,
        bm25_top_k=40,
        faiss_top_k=40,
    )
    
    total_candidates = len(fused_results)
    
    if not fused_results:
        return {
            "recipes": [],
            "recipe_ids": [],
            "query_info": parsed_query,
            "total_candidates": 0,
            "total_after_filter": 0,
        }
    
    # Step 4: Hard constraint filtering against ORIGINAL records
    rrf_scores = {rid: score for rid, score in fused_results}
    filtered_ids = []
    
    for recipe_id, score in fused_results:
        recipe = recipe_index.get(recipe_id)
        if recipe is None:
            continue
        
        # Apply dietary filters
        if parsed_query.get("dietary_restrictions"):
            if not filter_by_dietary(recipe, parsed_query["dietary_restrictions"]):
                continue
        
        filtered_ids.append(recipe_id)
    
    total_after_filter = len(filtered_ids)
    
    # Step 5: Rerank filtered candidates
    reranked = rerank_results(
        query=query,
        candidate_ids=filtered_ids,
        recipe_index=recipe_index,
        rrf_scores=rrf_scores,
        query_ingredients=parsed_query.get("ingredients"),
        top_k=top_k,
    )
    
    # Step 6: Fetch original records
    final_ids = [rid for rid, score in reranked]
    final_recipes = [recipe_index[rid] for rid in final_ids if rid in recipe_index]
    
    return {
        "recipes": final_recipes,
        "recipe_ids": final_ids,
        "query_info": parsed_query,
        "total_candidates": total_candidates,
        "total_after_filter": total_after_filter,
    }

