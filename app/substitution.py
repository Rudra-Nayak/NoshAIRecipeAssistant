"""
substitution.py - Evidence-based ingredient substitution.

Finds substitution evidence from the LOCAL dataset by looking for recipes
that use ingredient A and recipes that use ingredient B in similar contexts.

Key principles:
- The dataset is the PRIMARY evidence source
- Never claim medical safety
- Distinguish direct evidence from indirect/weak evidence
- Respect dietary restrictions
- Clearly communicate uncertainty
"""
from typing import Optional
from collections import Counter, defaultdict

from app.data_processor import normalize_ingredient_name


def find_substitution_evidence(
    original_ingredient: str,
    substitute_ingredient: str,
    recipes: list[dict],
    recipe_index: dict[str, dict],
) -> dict:
    """Find evidence for substituting one ingredient with another.
    
    Evidence types:
    - DIRECT: Both ingredients appear in the same recipe (strongest)
    - INDIRECT: Both appear in recipes with the same title (moderate)
    - CONTEXTUAL: Both appear in recipes with similar other ingredients (weak)
    
    Args:
        original_ingredient: The ingredient to replace.
        substitute_ingredient: The proposed replacement.
        recipes: Full recipe list.
        recipe_index: Recipe ID -> recipe dict mapping.
        
    Returns:
        Dictionary with evidence details and confidence assessment.
    """
    orig_norm = normalize_ingredient_name(original_ingredient)
    sub_norm = normalize_ingredient_name(substitute_ingredient)
    
    # Find recipes containing each ingredient
    recipes_with_original = []
    recipes_with_substitute = []
    recipes_with_both = []
    
    for recipe in recipes:
        ingredients_text = " ".join(recipe["ingredients"]).lower()
        has_original = orig_norm in ingredients_text
        has_substitute = sub_norm in ingredients_text
        
        if has_original and has_substitute:
            recipes_with_both.append(recipe["id"])
        elif has_original:
            recipes_with_original.append(recipe["id"])
        elif has_substitute:
            recipes_with_substitute.append(recipe["id"])
    
    # --- Assess evidence strength ---
    evidence = {
        "original": original_ingredient,
        "substitute": substitute_ingredient,
        "original_normalized": orig_norm,
        "substitute_normalized": sub_norm,
        "direct_evidence": [],
        "indirect_evidence": [],
        "contextual_evidence": [],
        "confidence": "none",
        "explanation": "",
        "warnings": [],
    }
    
    # Direct evidence: both in same recipe
    if recipes_with_both:
        evidence["direct_evidence"] = recipes_with_both[:10]
        evidence["confidence"] = "high"
        evidence["explanation"] = (
            f"Found {len(recipes_with_both)} recipe(s) that use BOTH "
            f"'{original_ingredient}' and '{substitute_ingredient}', "
            f"suggesting they work well together in similar dishes."
        )
    
    # Indirect evidence: same title, different ingredients
    if recipes_with_original and recipes_with_substitute:
        title_map = defaultdict(list)
        for rid in recipes_with_original:
            r = recipe_index.get(rid)
            if r:
                title_map[r["title"].lower()].append(("original", rid))
        for rid in recipes_with_substitute:
            r = recipe_index.get(rid)
            if r:
                title_map[r["title"].lower()].append(("substitute", rid))
        
        for title, entries in title_map.items():
            types = {t for t, _ in entries}
            if "original" in types and "substitute" in types:
                evidence["indirect_evidence"].append({
                    "title": title,
                    "recipe_ids": [rid for _, rid in entries],
                })
        
        if evidence["indirect_evidence"] and evidence["confidence"] == "none":
            evidence["confidence"] = "moderate"
            evidence["explanation"] = (
                f"Found recipes with the same name that use either "
                f"'{original_ingredient}' or '{substitute_ingredient}', "
                f"suggesting they may be interchangeable in those contexts."
            )
    
    # Contextual evidence: similar co-occurring ingredients
    if evidence["confidence"] == "none":
        # Check if the two ingredients appear with similar companion ingredients
        orig_companions = _get_companion_ingredients(
            orig_norm, recipes_with_original[:100], recipe_index
        )
        sub_companions = _get_companion_ingredients(
            sub_norm, recipes_with_substitute[:100], recipe_index
        )
        
        # Jaccard similarity of companion ingredients
        if orig_companions and sub_companions:
            overlap = orig_companions & sub_companions
            union = orig_companions | sub_companions
            similarity = len(overlap) / len(union) if union else 0
            
            if similarity > 0.3:
                evidence["contextual_evidence"].append({
                    "similarity": round(similarity, 3),
                    "shared_companions": list(overlap)[:10],
                })
                evidence["confidence"] = "low"
                evidence["explanation"] = (
                    f"'{original_ingredient}' and '{substitute_ingredient}' appear "
                    f"with similar other ingredients ({len(overlap)} shared), "
                    f"suggesting they may serve similar roles. However, this is "
                    f"indirect evidence and may not guarantee a good substitution."
                )
    
    if evidence["confidence"] == "none":
        evidence["explanation"] = (
            f"No evidence found in the dataset to support substituting "
            f"'{original_ingredient}' with '{substitute_ingredient}'. "
            f"This does not mean the substitution is impossible — "
            f"it just isn't supported by the recipes in our database."
        )
    
    # Add usage counts for context
    evidence["original_recipe_count"] = len(recipes_with_original) + len(recipes_with_both)
    evidence["substitute_recipe_count"] = len(recipes_with_substitute) + len(recipes_with_both)
    evidence["both_recipe_count"] = len(recipes_with_both)
    
    # Warnings
    if evidence["confidence"] in ("low", "none"):
        evidence["warnings"].append(
            "This substitution has limited evidence in our dataset. "
            "Results may vary depending on the specific recipe."
        )
    
    return evidence


def _get_companion_ingredients(
    ingredient_norm: str,
    recipe_ids: list[str],
    recipe_index: dict[str, dict],
) -> set[str]:
    """Get the set of ingredients that commonly appear alongside a given ingredient.
    
    Args:
        ingredient_norm: Normalized ingredient name.
        recipe_ids: IDs of recipes containing this ingredient.
        recipe_index: Recipe ID -> recipe dict mapping.
        
    Returns:
        Set of normalized companion ingredient names.
    """
    companions = set()
    for rid in recipe_ids:
        recipe = recipe_index.get(rid)
        if recipe:
            for ing in recipe["ingredients"]:
                norm = normalize_ingredient_name(ing)
                if norm != ingredient_norm:
                    companions.add(norm)
    return companions


def generate_substitution_response(
    original_ingredient: str,
    substitute_ingredient: str,
    recipe_id: str,
    evidence: dict,
    recipe_index: dict[str, dict],
) -> dict:
    """Generate a substitution recommendation for a specific recipe.
    
    Args:
        original_ingredient: Ingredient to replace.
        substitute_ingredient: Proposed replacement.
        recipe_id: ID of the recipe to modify.
        evidence: Evidence dict from find_substitution_evidence.
        recipe_index: Recipe ID -> recipe dict mapping.
        
    Returns:
        Dictionary with the substitution recommendation.
    """
    recipe = recipe_index.get(recipe_id)
    if not recipe:
        return {"error": f"Recipe {recipe_id} not found."}
    
    # Build modified ingredient list
    modified_ingredients = []
    substituted = False
    orig_norm = normalize_ingredient_name(original_ingredient)
    
    for ing in recipe["ingredients"]:
        ing_norm = normalize_ingredient_name(ing)
        if orig_norm in ing_norm or ing_norm in orig_norm:
            # Replace this ingredient
            modified_ingredients.append(f"{substitute_ingredient} (substituted for {ing})")
            substituted = True
        else:
            modified_ingredients.append(ing)
    
    if not substituted:
        return {
            "error": f"'{original_ingredient}' not found in recipe '{recipe['title']}'.",
            "recipe_ingredients": recipe["ingredients"],
        }
    
    return {
        "recipe_id": recipe_id,
        "recipe_title": recipe["title"],
        "original_ingredient": original_ingredient,
        "substitute_ingredient": substitute_ingredient,
        "confidence": evidence["confidence"],
        "evidence_summary": evidence["explanation"],
        "modified_ingredients": modified_ingredients,
        "original_ingredients": recipe["ingredients"],
        "instructions": recipe["instructions"],
        "warnings": evidence["warnings"],
    }

