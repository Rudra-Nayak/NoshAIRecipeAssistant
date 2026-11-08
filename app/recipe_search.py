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
from app.dietary_filter import is_recipe_compliant, check_ambiguous_ingredients


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


def filter_by_dietary(
    recipe: dict,
    restrictions: list[str],
    region: str = "India",
    is_explicit_beef_request: bool = False,
    custom_exclusions: list[str] = None,
) -> bool:
    """Check if a recipe satisfies cultural and dietary restrictions.
    
    Enforces deterministic filtering through app.dietary_filter.
    
    Args:
        recipe: Full recipe dictionary.
        restrictions: List of dietary restriction labels.
        region: Geographic region ('India' by default).
        is_explicit_beef_request: Whether user specifically searched for beef.
        custom_exclusions: Custom user-specified excluded ingredients.
        
    Returns:
        True if the recipe satisfies all dietary constraints.
    """
    ok, _violations = is_recipe_compliant(
        recipe=recipe,
        restrictions=restrictions,
        region=region,
        is_explicit_beef_request=is_explicit_beef_request,
        custom_exclusions=custom_exclusions,
    )
    return ok


def _check_dish_distinctions(recipe: dict, query_lower: str) -> bool:
    """Verify dish-specific constraints to avoid conflating similar-sounding dishes.
    
    e.g., 'rice kheer' must contain rice, and should not return purely vermicelli/seviyan.
    """
    title_lower = recipe.get("title", "").lower()
    recipe_text = (" " + " ".join(recipe.get("ingredients", []))).lower()
    
    # Distinction 0: Kheer queries must match actual kheer, payasam, or rice pudding dishes
    if "kheer" in query_lower:
        is_kheer = (
            "kheer" in title_lower 
            or "payasam" in title_lower 
            or "rice pudding" in title_lower
            or "kheer" in recipe_text
        )
        if not is_kheer:
            return False

    # Distinction 1: Rice Kheer vs Vermicelli Kheer
    if "rice kheer" in query_lower or "chawal kheer" in query_lower or ("rice" in query_lower and "kheer" in query_lower):
        # Must contain rice in ingredients
        has_rice = any(r in recipe_text for r in ["rice", "basmati", "chawal"])
        if not has_rice:
            return False
            
    # Explicit exclusion of vermicelli if user asked for "only rice kheer"
    if ("only rice" in query_lower or "not vermicelli" in query_lower) and "kheer" in query_lower:
        has_vermicelli = any(v in recipe_text or v in title_lower for v in ["vermicelli", "seviyan", "semiya"])
        if has_vermicelli and not any(r in recipe_text for r in ["rice", "basmati"]):
            return False

    return True


def extract_clean_search_query(query: str) -> str:
    """Strip conversational filler, intent triggers, and stop-phrases to leave the core dish / ingredient keywords."""
    import re
    q = query.lower().strip()
    
    prefixes = [
        r"^(?:how\s+do\s+i\s+make|how\s+do\s+i\s+prepare|how\s+to\s+make|how\s+to\s+cook|how\s+to\s+prepare)\s+",
        r"^(?:can\s+you\s+guide\s+me\s+to\s+make|can\s+you\s+guide\s+me\s+on\s+how\s+to\s+make|guide\s+me\s+through\s+the\s+recipe\s+for|guide\s+me\s+through|guide\s+me\s+to\s+make)\s+",
        r"^(?:give\s+me\s+step\s+by\s+step\s+instructions\s+for|give\s+me\s+instructions\s+for|give\s+me\s+recipe\s+for)\s+",
        r"^(?:find\s+recipes\s+for|find\s+recipe\s+for|search\s+for\s+recipes\s+for|search\s+for\s+recipe\s+for|show\s+me\s+recipes\s+for|show\s+me\s+recipe\s+for|find\s+me\s+recipes\s+for|find\s+me\s+a\s+recipe\s+for|find\s+recipes\s+with)\s+",
        r"^(?:tell\s+me\s+how\s+to\s+make|explain\s+how\s+to\s+make|explain\s+the\s+cooking\s+process\s+for)\s+",
        r"^(?:i\s+want\s+to\s+make|i\s+want\s+to\s+cook|i\s+would\s+like\s+to\s+make|i\s+would\s+like\s+to\s+cook)\s+",
    ]
    for p in prefixes:
        q = re.sub(p, "", q).strip()
        
    suffixes = [
        r"\s+step\s+by\s+step\??$",
        r"\s+step-by-step\??$",
        r"\s+for\s+diwali\??$",
        r"\s+for\s+dinner\??$",
        r"\s+for\s+lunch\??$",
        r"\s+for\s+breakfast\??$",
        r"\s+for\s+dessert\??$",
        r"\s+recipes?\??$",
    ]
    for s in suffixes:
        q = re.sub(s, "", q).strip()
        
    q = re.sub(r"^(?:traditional|authentic|classic|delicious|easy|simple|best)\s+", "", q).strip()
    return q if q else query


def search_recipes(
    query: str,
    bm25_retriever,
    faiss_retriever,
    recipe_index: dict[str, dict],
    parsed_query: dict = None,
    top_k: int = 5,
    region: str = "India",
    custom_exclusions: list[str] = None,
) -> dict:
    """Execute the full recipe search pipeline with cultural and dietary guardrails.
    
    Pipeline:
    1. Parse query -> intent + constraints
    2. Hybrid retrieval (BM25 + FAISS) -> ~30 candidates
    3. RRF fusion -> ~20 ranked candidates
    4. Hard constraint filtering (cultural beef-default, Jain, dietary, custom exclusions)
    5. Dish distinction checks (e.g. rice kheer != vermicelli kheer)
    6. Reranking -> final top-k
    7. Fetch original records + ambiguous ingredient annotations
    
    Args:
        query: User search query.
        bm25_retriever: BM25Retriever instance.
        faiss_retriever: FAISSRetriever instance (can be None).
        recipe_index: Dictionary mapping recipe ID -> recipe dict.
        parsed_query: Pre-parsed query (if already available from router).
        top_k: Number of final results to return.
        region: Geographic region ('India' by default).
        custom_exclusions: List of user-specified excluded ingredients.
        
    Returns:
        Dictionary with:
            - recipes: list of full recipe dicts
            - recipe_ids: list of matched recipe IDs
            - query_info: parsed query details
            - total_candidates: number before filtering
            - total_after_filter: number after constraint filtering
            - ambiguous_ingredients: list of flagged ambiguous items
    """
    # Step 1: Parse query
    if parsed_query is None:
        parsed_query = parse_query(query)
    
    query_lower = query.lower()
    is_explicit_beef = parsed_query.get("is_explicit_beef_request", False)
    
    # Merge custom exclusions from parsed query and arguments
    all_exclusions = []
    if custom_exclusions:
        all_exclusions.extend(custom_exclusions)
    if parsed_query.get("missing_ingredients"):
        all_exclusions.extend(parsed_query["missing_ingredients"])
    
    # Clean conversational wrappers to extract the core culinary terms
    search_query = extract_clean_search_query(query)
    
    # Step 2-3: Hybrid retrieval with RRF
    fused_results = hybrid_search(
        query=search_query,
        bm25_retriever=bm25_retriever,
        faiss_retriever=faiss_retriever,
        top_k=35,
        bm25_top_k=45,
        faiss_top_k=45,
    )
    
    # Fallback to raw query if cleaned query yielded no results
    if not fused_results and search_query != query:
        fused_results = hybrid_search(
            query=query,
            bm25_retriever=bm25_retriever,
            faiss_retriever=faiss_retriever,
            top_k=35,
            bm25_top_k=45,
            faiss_top_k=45,
        )
    
    total_candidates = len(fused_results)
    
    if not fused_results:
        return {
            "recipes": [],
            "recipe_ids": [],
            "query_info": parsed_query,
            "total_candidates": 0,
            "total_after_filter": 0,
            "ambiguous_ingredients": [],
        }
    
    # Step 4: Hard constraint filtering against ORIGINAL records
    rrf_scores = {rid: score for rid, score in fused_results}
    filtered_ids = []
    dietary_restrictions = parsed_query.get("dietary_restrictions", [])
    
    for recipe_id, score in fused_results:
        recipe = recipe_index.get(recipe_id)
        if recipe is None:
            continue
        
        # Apply deterministic dietary & cultural filters
        compliant = filter_by_dietary(
            recipe=recipe,
            restrictions=dietary_restrictions,
            region=region,
            is_explicit_beef_request=is_explicit_beef,
            custom_exclusions=all_exclusions if all_exclusions else None,
        )
        if not compliant:
            continue
            
        # Apply fine-grained dish distinctions
        if not _check_dish_distinctions(recipe, query_lower):
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
    
    # Step 6: Fetch original records and check for ambiguous ingredients
    final_ids = [rid for rid, score in reranked]
    final_recipes = [recipe_index[rid] for rid in final_ids if rid in recipe_index]
    
    ambiguous_notes = []
    for r in final_recipes:
        flags = check_ambiguous_ingredients(r)
        if flags:
            ambiguous_notes.append({
                "recipe_id": r["id"],
                "recipe_title": r["title"],
                "flags": flags
            })
    
    return {
        "recipes": final_recipes,
        "recipe_ids": final_ids,
        "query_info": parsed_query,
        "total_candidates": total_candidates,
        "total_after_filter": total_after_filter,
        "ambiguous_ingredients": ambiguous_notes,
    }

