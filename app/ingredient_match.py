"""
ingredient_match.py - Ingredient-based recipe matching and recommendation.

Given a list of ingredients the user has, finds recipes that can be made
with those ingredients (or close to it).

Key features:
- Ingredient normalization for fuzzy matching
- Equivalent ingredient dictionary (transparent, not hidden)
- Match/missing/extra ingredient breakdown
- Explainable recommendation scores
- Hard constraint enforcement (dietary, excluded ingredients)
"""
import re
from typing import Optional
from collections import Counter

from app.data_processor import normalize_ingredient_name, is_valid_recipe_record


# --- Ingredient Equivalents ---
# These are transparent, hand-curated equivalences.
# We do NOT assume two ingredients are interchangeable just because
# their names are similar.

INGREDIENT_EQUIVALENTS = {
    # Oils
    "vegetable oil": {"canola oil", "corn oil", "sunflower oil"},
    "canola oil": {"vegetable oil", "corn oil", "sunflower oil"},
    "olive oil": {"extra virgin olive oil", "extra-virgin olive oil"},
    "extra virgin olive oil": {"olive oil"},
    
    # Dairy
    "butter": {"unsalted butter", "salted butter"},
    "unsalted butter": {"butter"},
    "whole milk": {"milk"},
    "milk": {"whole milk", "2% milk"},
    "heavy cream": {"heavy whipping cream", "whipping cream"},
    "sour cream": {"crème fraîche"},
    
    # Sugars
    "sugar": {"granulated sugar", "white sugar"},
    "granulated sugar": {"sugar", "white sugar"},
    "brown sugar": {"light brown sugar", "dark brown sugar"},
    "light brown sugar": {"brown sugar"},
    
    # Flours
    "flour": {"all-purpose flour", "plain flour"},
    "all-purpose flour": {"flour", "plain flour"},
    
    # Herbs (dried <-> fresh, with caveat)
    "garlic": {"garlic cloves", "minced garlic"},
    "garlic cloves": {"garlic"},
    
    # Common substitutions
    "lemon juice": {"lime juice"},
    "lime juice": {"lemon juice"},
    "green onion": {"scallion", "spring onion"},
    "scallion": {"green onion", "spring onion"},
}


def _word_match(target: str, text: str) -> bool:
    """Check if target exists in text respecting word boundaries."""
    if not target or not text:
        return False
    pattern = r'(?:\b|_)' + re.escape(target.strip()) + r'(?:\b|_)'
    return bool(re.search(pattern, text, re.IGNORECASE))


def normalize_and_match(
    user_ingredient: str,
    recipe_ingredient: str,
) -> bool:
    """Check if a user ingredient matches a recipe ingredient.
    
    Uses normalization and the equivalents dictionary, matching tokens on
    word boundaries to prevent false positives like 'butter' in 'butterflies'
    or 'egg' in 'eggplant'.
    
    Args:
        user_ingredient: Normalized ingredient the user has.
        recipe_ingredient: Raw ingredient string from the recipe.
        
    Returns:
        True if the ingredients match.
    """
    user_norm = normalize_ingredient_name(user_ingredient)
    recipe_norm = normalize_ingredient_name(recipe_ingredient)
    
    # Direct match on word boundaries
    if _word_match(user_norm, recipe_norm) or _word_match(recipe_norm, user_norm):
        return True
    
    # Check equivalents
    equivalents = INGREDIENT_EQUIVALENTS.get(user_norm, set())
    for equiv in equivalents:
        if _word_match(equiv, recipe_norm) or _word_match(recipe_norm, equiv):
            return True
    
    return False


def compute_match_score(
    user_ingredients: list[str],
    recipe: dict,
) -> dict:
    """Compute how well a recipe matches the user's available ingredients.
    
    Returns a detailed breakdown of matched, missing, and extra ingredients
    with an explainable score.
    
    Args:
        user_ingredients: List of normalized ingredient names the user has.
        recipe: Full recipe dictionary.
        
    Returns:
        Dictionary with:
            - score: float (0.0 to 1.0, proportion of recipe ingredients matched)
            - matched: list of matched ingredient pairs
            - missing: list of recipe ingredients not matched
            - extra: list of user ingredients not used
            - total_recipe_ingredients: int
            - total_user_ingredients: int
    """
    matched = []
    missing = []
    used_user_ingredients = set()
    
    for recipe_ing in recipe["ingredients"]:
        found = False
        for user_ing in user_ingredients:
            if normalize_and_match(user_ing, recipe_ing):
                matched.append({
                    "user_ingredient": user_ing,
                    "recipe_ingredient": recipe_ing,
                })
                used_user_ingredients.add(user_ing)
                found = True
                break
        
        if not found:
            missing.append(recipe_ing)
    
    extra = [ing for ing in user_ingredients if ing not in used_user_ingredients]
    
    total_recipe = len(recipe["ingredients"])
    score = len(matched) / total_recipe if total_recipe > 0 else 0.0
    
    return {
        "score": round(score, 4),
        "matched": matched,
        "missing": missing,
        "extra": extra,
        "total_recipe_ingredients": total_recipe,
        "total_user_ingredients": len(user_ingredients),
        "matched_count": len(matched),
        "missing_count": len(missing),
    }


def find_matching_recipes(
    user_ingredients: list[str],
    recipes: list[dict],
    dietary_restrictions: list[str] = None,
    excluded_ingredients: list[str] = None,
    min_match_score: float = 0.2,
    max_steps: Optional[int] = None,
    prefer_fewest_missing: bool = False,
    top_n: int = 10,
) -> list[dict]:
    """Find recipes that best match the user's available ingredients.
    
    Pipeline:
    1. Normalize user ingredients
    2. Score every recipe by ingredient coverage
    3. Apply hard constraints (dietary, excluded ingredients, max instruction steps)
    4. Sort by match score or fewest missing ingredients
    5. Return top-N with full match breakdown and step counts
    
    Hard constraints are NEVER relaxed. If no recipe passes all constraints,
    we return an empty list with a clear message.
    """
    from app.recipe_search import filter_by_dietary
    from app.data_processor import count_instruction_steps
    
    # Step 1: Normalize user ingredients
    normalized_user = [normalize_ingredient_name(ing) for ing in user_ingredients]
    
    results = []
    
    for recipe in recipes:
        # Step 1.5: Validate that recipe is not corrupt / non-culinary dummy data
        if not is_valid_recipe_record(recipe):
            continue

        # Step 2: Compute match score
        match = compute_match_score(normalized_user, recipe)
        
        # Skip recipes with zero or negligible matches early
        if match["matched_count"] == 0:
            continue
            
        step_count = count_instruction_steps(recipe)
        
        # Step 3: Apply hard constraints
        # Max instruction steps hard constraint
        if max_steps is not None and step_count > max_steps:
            continue
            
        # Dietary restrictions
        if dietary_restrictions:
            if not filter_by_dietary(recipe, dietary_restrictions):
                continue
        
        # Excluded ingredients
        if excluded_ingredients:
            recipe_text = " ".join(recipe["ingredients"]).lower()
            skip = False
            for excluded in excluded_ingredients:
                if excluded.lower() in recipe_text:
                    skip = True
                    break
            if skip:
                continue
        
        results.append({
            "recipe": recipe,
            "match": match,
            "step_count": step_count,
        })
    
    # Step 4: Sort
    if prefer_fewest_missing:
        # Maximize matched user ingredients first, then minimize missing additional items, then fewer steps
        results.sort(key=lambda x: (-x["match"]["matched_count"], len(x["match"]["missing"]), x["step_count"]))
    else:
        results.sort(key=lambda x: (-x["match"]["score"], len(x["match"]["missing"]), x["step_count"]))
    
    # Step 5: Return top-N
    return results[:top_n]

