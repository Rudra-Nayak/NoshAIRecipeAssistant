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
from typing import Optional
from collections import Counter

from app.data_processor import normalize_ingredient_name


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


def normalize_and_match(
    user_ingredient: str,
    recipe_ingredient: str,
) -> bool:
    """Check if a user ingredient matches a recipe ingredient.
    
    Uses normalization and the equivalents dictionary.
    
    Args:
        user_ingredient: Normalized ingredient the user has.
        recipe_ingredient: Raw ingredient string from the recipe.
        
    Returns:
        True if the ingredients match.
    """
    recipe_norm = normalize_ingredient_name(recipe_ingredient)
    
    # Direct match
    if user_ingredient in recipe_norm or recipe_norm in user_ingredient:
        return True
    
    # Check equivalents
    equivalents = INGREDIENT_EQUIVALENTS.get(user_ingredient, set())
    for equiv in equivalents:
        if equiv in recipe_norm or recipe_norm in equiv:
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
    min_match_score: float = 0.3,
    top_n: int = 10,
) -> list[dict]:
    """Find recipes that best match the user's available ingredients.
    
    Pipeline:
    1. Normalize user ingredients
    2. Score every recipe by ingredient coverage
    3. Apply hard constraints (dietary, excluded ingredients)
    4. Sort by match score
    5. Return top-N with full match breakdown
    
    Hard constraints are NEVER relaxed. If no recipe passes all constraints,
    we return an empty list with a clear message.
    
    Args:
        user_ingredients: Raw ingredient strings from the user.
        recipes: Full recipe list.
        dietary_restrictions: List of dietary restriction labels.
        excluded_ingredients: Ingredients to exclude.
        min_match_score: Minimum match score threshold.
        top_n: Number of top results to return.
        
    Returns:
        List of result dicts, each containing the recipe and match details.
    """
    from app.recipe_search import filter_by_dietary
    
    # Step 1: Normalize user ingredients
    normalized_user = [normalize_ingredient_name(ing) for ing in user_ingredients]
    
    results = []
    
    for recipe in recipes:
        # Step 2: Compute match score
        match = compute_match_score(normalized_user, recipe)
        
        # Skip low-score matches early (optimization for 123K recipes)
        if match["score"] < min_match_score:
            continue
        
        # Step 3: Apply hard constraints
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
        })
    
    # Step 4: Sort by match score (highest first)
    results.sort(key=lambda x: x["match"]["score"], reverse=True)
    
    # Step 5: Return top-N
    return results[:top_n]

