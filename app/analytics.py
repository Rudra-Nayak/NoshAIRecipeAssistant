"""
analytics.py - Deterministic recipe analytics and statistics.

All analytics are computed PROGRAMMATICALLY using Python, never by the LLM.
The LLM is only used to understand the query and format the response.

This prevents hallucinated statistics — every number comes from actual
computation over the dataset.
"""
from collections import Counter
from typing import Optional

from app.data_processor import (
    normalize_ingredient_name,
    get_normalized_ingredients,
    count_instruction_steps,
)


def total_recipe_count(recipes: list[dict], filters: dict = None) -> int:
    """Count total recipes, optionally filtered.
    
    Args:
        recipes: Full recipe list.
        filters: Optional dict with filter criteria.
        
    Returns:
        Number of matching recipes.
    """
    if not filters:
        return len(recipes)
    
    return len(filter_recipes(recipes, filters))


def filter_recipes(recipes: list[dict], filters: dict) -> list[dict]:
    """Filter recipes by criteria.
    
    Supported filter keys:
    - source: str ("ar", "epi", "fn")
    - ingredient_contains: list[str] — recipes containing ALL these ingredients
    - ingredient_excludes: list[str] — recipes NOT containing ANY of these
    - title_contains: str — substring match on title
    
    Args:
        recipes: Full recipe list.
        filters: Dictionary of filter criteria.
        
    Returns:
        Filtered list of recipes.
    """
    result = recipes
    
    if "source" in filters:
        result = [r for r in result if r["source"] == filters["source"]]
    
    if "ingredient_contains" in filters:
        for ingredient in filters["ingredient_contains"]:
            ing_lower = ingredient.lower()
            result = [
                r for r in result
                if any(ing_lower in i.lower() for i in r["ingredients"])
            ]
    
    if "ingredient_excludes" in filters:
        for ingredient in filters["ingredient_excludes"]:
            ing_lower = ingredient.lower()
            result = [
                r for r in result
                if not any(ing_lower in i.lower() for i in r["ingredients"])
            ]
    
    if "title_contains" in filters:
        title_lower = filters["title_contains"].lower()
        result = [r for r in result if title_lower in r["title"].lower()]
    
    return result


def most_common_ingredients(
    recipes: list[dict],
    top_n: int = 20,
    filters: dict = None,
) -> list[tuple[str, int]]:
    """Find the most common ingredients across recipes.
    
    Operates on the FULL filtered dataset, not just retrieved top-k.
    
    Args:
        recipes: Full recipe list.
        top_n: Number of top ingredients to return.
        filters: Optional filter criteria.
        
    Returns:
        List of (ingredient_name, count) tuples.
    """
    target = filter_recipes(recipes, filters) if filters else recipes
    
    counter = Counter()
    for recipe in target:
        normalized = get_normalized_ingredients(recipe)
        counter.update(normalized)
    
    return counter.most_common(top_n)


def ingredient_count_stats(
    recipes: list[dict],
    filters: dict = None,
) -> dict:
    """Compute ingredient count statistics.
    
    Args:
        recipes: Full recipe list.
        filters: Optional filter criteria.
        
    Returns:
        Dict with average, median, min, max ingredient counts.
    """
    target = filter_recipes(recipes, filters) if filters else recipes
    
    if not target:
        return {"average": 0, "median": 0, "min": 0, "max": 0, "total_recipes": 0}
    
    counts = sorted(len(r["ingredients"]) for r in target)
    n = len(counts)
    
    return {
        "average": round(sum(counts) / n, 2),
        "median": counts[n // 2],
        "min": counts[0],
        "max": counts[-1],
        "total_recipes": n,
    }


def step_count_stats(
    recipes: list[dict],
    filters: dict = None,
) -> dict:
    """Compute instruction step count statistics.
    
    Args:
        recipes: Full recipe list.
        filters: Optional filter criteria.
        
    Returns:
        Dict with average, median, min, max step counts.
    """
    target = filter_recipes(recipes, filters) if filters else recipes
    
    if not target:
        return {"average": 0, "median": 0, "min": 0, "max": 0, "total_recipes": 0}
    
    counts = sorted(count_instruction_steps(r) for r in target)
    n = len(counts)
    
    return {
        "average": round(sum(counts) / n, 2),
        "median": counts[n // 2],
        "min": counts[0],
        "max": counts[-1],
        "total_recipes": n,
    }


def rank_by_ingredient_count(
    recipes: list[dict],
    ascending: bool = True,
    top_n: int = 10,
    filters: dict = None,
) -> list[dict]:
    """Rank recipes by number of ingredients.
    
    Args:
        recipes: Full recipe list.
        ascending: If True, fewest ingredients first.
        top_n: Number of results.
        filters: Optional filter criteria.
        
    Returns:
        List of recipe dicts with added 'ingredient_count' field.
    """
    target = filter_recipes(recipes, filters) if filters else recipes
    
    ranked = sorted(
        target,
        key=lambda r: len(r["ingredients"]),
        reverse=not ascending,
    )
    
    result = []
    for r in ranked[:top_n]:
        entry = {**r, "ingredient_count": len(r["ingredients"])}
        result.append(entry)
    
    return result


def rank_by_step_count(
    recipes: list[dict],
    ascending: bool = True,
    top_n: int = 10,
    filters: dict = None,
) -> list[dict]:
    """Rank recipes by number of instruction steps.
    
    Args:
        recipes: Full recipe list.
        ascending: If True, fewest steps first.
        top_n: Number of results.
        filters: Optional filter criteria.
        
    Returns:
        List of recipe dicts with added 'step_count' field.
    """
    target = filter_recipes(recipes, filters) if filters else recipes
    
    ranked = sorted(
        target,
        key=lambda r: count_instruction_steps(r),
        reverse=not ascending,
    )
    
    result = []
    for r in ranked[:top_n]:
        entry = {**r, "step_count": count_instruction_steps(r)}
        result.append(entry)
    
    return result


def compute_analytics(
    query: str,
    recipes: list[dict],
    filters: dict = None,
) -> dict:
    """Dispatch an analytics query to the appropriate function.
    
    This is a simple keyword-based dispatcher. The LLM integration
    layer can provide more nuanced intent parsing.
    
    Args:
        query: User's analytics query.
        recipes: Full recipe list.
        filters: Active filters from conversation memory.
        
    Returns:
        Dict with analytics results and a human-readable summary.
    """
    query_lower = query.lower()
    
    result = {"query": query, "type": "analytics"}
    
    if any(k in query_lower for k in ["how many", "count", "total"]):
        count = total_recipe_count(recipes, filters)
        result["answer"] = count
        result["summary"] = f"There are {count:,} recipes"
        if filters:
            result["summary"] += f" matching the current filters"
        result["summary"] += "."
    
    elif any(k in query_lower for k in ["most common", "most popular", "top ingredient"]):
        top_n = 10
        common = most_common_ingredients(recipes, top_n=top_n, filters=filters)
        result["answer"] = common
        lines = [f"Top {top_n} most common ingredients:"]
        for name, count in common:
            lines.append(f"  {name}: {count:,} recipes")
        result["summary"] = "\n".join(lines)
    
    elif any(k in query_lower for k in ["average ingredient", "median ingredient"]):
        stats = ingredient_count_stats(recipes, filters)
        result["answer"] = stats
        result["summary"] = (
            f"Ingredient count statistics:\n"
            f"  Average: {stats['average']}\n"
            f"  Median: {stats['median']}\n"
            f"  Min: {stats['min']}, Max: {stats['max']}\n"
            f"  Total recipes: {stats['total_recipes']:,}"
        )
    
    elif any(k in query_lower for k in ["fewest ingredient", "least ingredient", "simplest"]):
        ranked = rank_by_ingredient_count(recipes, ascending=True, top_n=5, filters=filters)
        result["answer"] = ranked
        lines = ["Recipes with fewest ingredients:"]
        for r in ranked:
            lines.append(f"  [{r['id'][:8]}] {r['title']} — {r['ingredient_count']} ingredients")
        result["summary"] = "\n".join(lines)
    
    elif any(k in query_lower for k in ["most ingredient"]):
        ranked = rank_by_ingredient_count(recipes, ascending=False, top_n=5, filters=filters)
        result["answer"] = ranked
        lines = ["Recipes with most ingredients:"]
        for r in ranked:
            lines.append(f"  [{r['id'][:8]}] {r['title']} — {r['ingredient_count']} ingredients")
        result["summary"] = "\n".join(lines)
    
    elif any(k in query_lower for k in ["fewest step", "least step", "quickest", "simplest recipe"]):
        ranked = rank_by_step_count(recipes, ascending=True, top_n=5, filters=filters)
        result["answer"] = ranked
        lines = ["Recipes with fewest steps:"]
        for r in ranked:
            lines.append(f"  [{r['id'][:8]}] {r['title']} — {r['step_count']} steps")
        result["summary"] = "\n".join(lines)
    
    elif any(k in query_lower for k in ["most step"]):
        ranked = rank_by_step_count(recipes, ascending=False, top_n=5, filters=filters)
        result["answer"] = ranked
        lines = ["Recipes with most steps:"]
        for r in ranked:
            lines.append(f"  [{r['id'][:8]}] {r['title']} — {r['step_count']} steps")
        result["summary"] = "\n".join(lines)
    
    else:
        # Generic stats
        stats_ing = ingredient_count_stats(recipes, filters)
        stats_step = step_count_stats(recipes, filters)
        result["answer"] = {"ingredient_stats": stats_ing, "step_stats": stats_step}
        result["summary"] = (
            f"Dataset statistics:\n"
            f"  Total recipes: {stats_ing['total_recipes']:,}\n"
            f"  Avg ingredients: {stats_ing['average']} (median: {stats_ing['median']})\n"
            f"  Avg steps: {stats_step['average']} (median: {stats_step['median']})"
        )
    
    return result

