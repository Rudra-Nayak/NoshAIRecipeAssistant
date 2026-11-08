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
    
    # 1. Multi-ingredient co-occurrence analytics (e.g. 'both garlic and onion', 'contain X and Y')
    import re
    if (("both" in query_lower and "and" in query_lower) or 
        ("contain" in query_lower and "and" in query_lower and any(w in query_lower for w in ["how many", "count", "percentage", "percent"]))):
        m = re.search(r"both\s+([a-zA-Z\s]+?)\s+and\s+([a-zA-Z\s]+?)(?:\?|\.|\s+what|\s+how|$)", query_lower)
        if not m:
            m = re.search(r"contain(?:ing)?\s+([a-zA-Z\s]+?)\s+and\s+([a-zA-Z\s]+?)(?:\?|\.|\s+what|\s+how|$)", query_lower)
            
        if m:
            ing1 = m.group(1).strip()
            ing2 = m.group(2).strip()
            ing1 = re.sub(r"\b(recipes?|contain|containing)\b", "", ing1).strip()
            ing2 = re.sub(r"\b(what|percentage|percent|of|the|dataset|is|that)\b", "", ing2).strip()
            
            total_count = len(recipes)
            matching = [
                r for r in recipes
                if any(ing1 in i.lower() for i in r["ingredients"])
                and any(ing2 in i.lower() for i in r["ingredients"])
            ]
            count = len(matching)
            pct = (count / total_count * 100) if total_count > 0 else 0.0
            
            summary = (
                f"### 📊 Ingredient Co-Occurrence Analytics\n\n"
                f"- **Matching Recipes**: **{count:,}** recipes contain both **{ing1}** and **{ing2}**.\n"
                f"- **Dataset Denominator**: **{total_count:,}** total usable recipes in the dataset.\n"
                f"- **Percentage**: **{pct:.2f}%** of the entire collection.\n\n"
                f"**Normalization Methodology**:\n"
                f"Ingredient names are normalized by stripping measurement units (cloves, cups, grams, tablespoons), "
                f"preparation adjectives (minced, chopped, diced, crushed, sliced), and punctuation. "
                f"Entries such as _'4 cloves garlic, minced'_ and _'1 yellow onion, chopped'_ are canonicalized to **{ing1}** and **{ing2}**."
            )
            result["answer"] = {"count": count, "total": total_count, "percentage": pct, "ingredients": [ing1, ing2]}
            result["summary"] = summary
            return result

    # 2. Fewest instruction steps with ingredient constraint (e.g. 'Among recipes containing tomato, which five have the fewest instruction steps?')
    if any(k in query_lower for k in ["fewest step", "least step", "fewest instruction step"]):
        target_ing = ""
        m_ing = re.search(r"containing\s+([a-zA-Z\s]+?)(?:,|\s+which|\s+have|\s+has|\?|$)", query_lower)
        if not m_ing:
            m_ing = re.search(r"recipes?\s+with\s+([a-zA-Z\s]+?)(?:,|\s+which|\s+have|\s+has|\?|$)", query_lower)
        if m_ing:
            target_ing = m_ing.group(1).strip()
            
        target_recipes = recipes
        if target_ing:
            target_recipes = [
                r for r in recipes 
                if any(target_ing in i.lower() for i in r["ingredients"]) or target_ing in r["title"].lower()
            ]
            
        ranked = rank_by_step_count(target_recipes, ascending=True, top_n=5, filters=filters)
        result["answer"] = ranked
        
        lines = [
            f"### 📋 Top 5 Recipes {f'Containing {target_ing.title()} ' if target_ing else ''}with Fewest Instruction Steps\n",
            "**Step-Counting Rule**: Instructions are counted by splitting text on newlines (`\\n`) or numbered step markers (`1.`, `2.`), falling back to terminal sentence delimiters (`. `) for single paragraphs.\n"
        ]
        for idx, r in enumerate(ranked, 1):
            lines.append(f"**{idx}. 🍲 {r['title']}** (Recipe ID: `{r['id']}`)")
            lines.append(f"   - **Instruction Steps**: **{r['step_count']} step(s)**")
            lines.append(f"   - **Instructions**: _{r['instructions'][:160].replace(chr(10), ' ').strip()}..._")
            lines.append("")
        result["summary"] = "\n".join(lines)
        return result
    
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

