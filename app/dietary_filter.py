"""
dietary_filter.py - Deterministic cultural and dietary restriction filtering.

Responsibilities:
- Strict deterministic filtering against raw recipe ingredient records.
- Support for regional defaults (India-aware default exclusion of beef).
- Support for multi-preference restrictions:
    * vegetarian
    * vegan
    * jain_vegetarian (no root vegetables, onions, garlic, potatoes, carrots, etc.)
    * no_beef (with neutral respect for explicit user requests)
    * no_pork
    * no_seafood
    * no_eggs
    * custom user-specified excluded ingredients
- Identification and explanation of ambiguous or uncertain ingredients.
- Non-destructive query-time filtering (never mutates the source dataset).
"""
import re
from typing import Optional, Tuple


# --- Exclusion Keyword Dictionaries ---

# Beef and bovine derivatives
BEEF_EXCLUSIONS = frozenset([
    "beef", "veal", "steak", "brisket", "corned beef", "ground chuck",
    "ground round", "ground sirloin", "sirloin", "tenderloin", "ribeye",
    "flank steak", "skirt steak", "t-bone", "roast beef", "ox", "oxtail",
    "suet", "tallow", "beef broth", "beef stock", "beef bouillon",
    "beef consomme", "veal stock", "veal chops", "pastrami", "bison"
])

# Pork and porcine derivatives
PORK_EXCLUSIONS = frozenset([
    "pork", "bacon", "ham", "prosciutto", "pancetta", "lard", "pork sausage",
    "pork chops", "pork loin", "pork belly", "pork shoulder", "chorizo",
    "pepperoni", "salami", "bratwurst", "pork ribs", "pork cutlet"
])

# Seafood and marine derivatives
SEAFOOD_EXCLUSIONS = frozenset([
    "fish", "salmon", "tuna", "cod", "halibut", "trout", "tilapia", "bass",
    "snapper", "anchov", "sardine", "mackerel", "swordfish", "haddock",
    "shrimp", "prawn", "crab", "lobster", "crawfish", "crayfish", "clam",
    "mussel", "oyster", "scallop", "squid", "calamari", "octopus",
    "fish sauce", "worcestershire", "oyster sauce", "clam juice", "bonito"
])

# Poultry & other meats
POULTRY_OTHER_MEAT_EXCLUSIONS = frozenset([
    "chicken", "turkey", "duck", "goose", "quail", "pheasant", "lamb",
    "mutton", "goat", "venison", "rabbit", "poultry", "chicken broth",
    "chicken stock", "chicken bouillon", "turkey stock", "bone broth"
])

# Eggs
EGG_EXCLUSIONS = frozenset([
    "egg", "eggs", "egg white", "egg whites", "egg yolk", "egg yolks",
    "mayonnaise", "meringue", "albumin", "ovalbumin"
])

# Dairy products
DAIRY_EXCLUSIONS = frozenset([
    "milk", "cream", "heavy cream", "butter", "cheese", "yogurt", "yoghurt",
    "curd", "paneer", "ghee", "whey", "casein", "sour cream", "ice cream",
    "half-and-half", "buttermilk", "mascarpone", "ricotta", "mozzarella",
    "cheddar", "parmesan", "condensed milk", "evaporated milk"
])

# Jain specific: root vegetables & underground produce
JAIN_ROOT_VEGETABLE_EXCLUSIONS = frozenset([
    "onion", "onions", "green onion", "scallion", "scallions", "shallot",
    "shallots", "leek", "leeks", "chive", "chives", "garlic", "garlic cloves",
    "potato", "potatoes", "carrot", "carrots", "radish", "radishes", "mooli",
    "beet", "beets", "beetroot", "turnip", "turnips", "ginger", "fresh ginger",
    "horseradish", "sweet potato", "sweet potatoes", "yam", "yams", "taro",
    "colocasia", "arbi", "lotus root"
])

# General meat/slaughter byproducts
ANIMAL_BYPRODUCT_EXCLUSIONS = frozenset([
    "gelatin", "gelatine", "lard", "suet", "tallow", "animal rennet",
    "rennet", "marshmallow"
])

# Honey
HONEY_EXCLUSIONS = frozenset(["honey", "honeycomb"])


# --- Ambiguous Ingredients ---
# Ingredients whose source or preparation may be animal or plant derived
AMBIGUOUS_INGREDIENTS = {
    "gelatin": "May be derived from pork, beef, or vegetable sources (agar-agar).",
    "gelatine": "May be derived from pork, beef, or vegetable sources (agar-agar).",
    "rennet": "May be animal-derived or microbial/vegetable rennet.",
    "shortening": "May contain animal fats or 100% vegetable oil.",
    "worcestershire sauce": "Traditionally contains anchovies unless labeled vegan.",
    "marshmallows": "Traditionally contains animal gelatin.",
    "cheese": "Some cheeses use animal rennet for curdling.",
    "parmesan": "Traditional authentic Parmesan uses animal rennet.",
}


def _contains_any(text: str, exclusion_set: frozenset) -> Optional[str]:
    """Check if any term from the exclusion set appears as a word in text.
    
    Uses word boundaries where possible to avoid false matches.
    """
    for term in exclusion_set:
        # Regex word boundary check for short terms, substring for multi-word
        if " " in term or "-" in term:
            if term in text:
                return term
        else:
            # Word boundary regex
            pattern = rf"\b{re.escape(term)}(?:s|es)?\b"
            if re.search(pattern, text):
                return term
    return None


def is_recipe_compliant(
    recipe: dict,
    restrictions: list[str],
    region: str = "India",
    is_explicit_beef_request: bool = False,
    custom_exclusions: list[str] = None,
) -> Tuple[bool, list[str]]:
    """Determine whether a recipe satisfies all active cultural and dietary restrictions.
    
    Args:
        recipe: Recipe dictionary containing 'ingredients' and 'title'.
        restrictions: List of restriction tags:
            'vegetarian', 'vegan', 'jain_vegetarian', 'no_beef',
            'no_pork', 'no_seafood', 'no_eggs'.
        region: Selected geographic region (defaults to 'India').
        is_explicit_beef_request: True if user explicitly asked for beef recipes.
        custom_exclusions: User-specified list of custom excluded ingredients.
        
    Returns:
        Tuple of (is_compliant: bool, violations: list[str]).
    """
    ingredients_text = " ".join(recipe.get("ingredients", [])).lower()
    title_text = recipe.get("title", "").lower()
    full_text = f"{title_text} {ingredients_text}"
    
    violations = []
    
    # 1. India Region Default: Exclude beef unless user explicitly requested beef
    if region.lower() == "india" and not is_explicit_beef_request:
        violation = _contains_any(full_text, BEEF_EXCLUSIONS)
        if violation:
            violations.append(f"Contains beef ({violation}) - excluded by Indian regional preference")
    
    # 2. Explicit no_beef restriction (even outside India or if user set it)
    if "no_beef" in restrictions and not is_explicit_beef_request:
        violation = _contains_any(full_text, BEEF_EXCLUSIONS)
        if violation and not any("beef" in v for v in violations):
            violations.append(f"Contains beef ({violation})")

    # 3. No Pork
    if "no_pork" in restrictions:
        violation = _contains_any(full_text, PORK_EXCLUSIONS)
        if violation:
            violations.append(f"Contains pork ({violation})")

    # 4. No Seafood
    if "no_seafood" in restrictions:
        violation = _contains_any(full_text, SEAFOOD_EXCLUSIONS)
        if violation:
            violations.append(f"Contains seafood ({violation})")

    # 5. No Eggs
    if "no_eggs" in restrictions:
        violation = _contains_any(full_text, EGG_EXCLUSIONS)
        if violation:
            violations.append(f"Contains egg ({violation})")

    # 6. Vegetarian
    # Excludes all meat, poultry, seafood, and animal slaughter byproducts
    if "vegetarian" in restrictions or "jain_vegetarian" in restrictions or "vegan" in restrictions:
        violation = _contains_any(full_text, BEEF_EXCLUSIONS)
        if not violation:
            violation = _contains_any(full_text, PORK_EXCLUSIONS)
        if not violation:
            violation = _contains_any(full_text, SEAFOOD_EXCLUSIONS)
        if not violation:
            violation = _contains_any(full_text, POULTRY_OTHER_MEAT_EXCLUSIONS)
        if not violation:
            violation = _contains_any(full_text, ANIMAL_BYPRODUCT_EXCLUSIONS)
        
        if violation:
            violations.append(f"Contains non-vegetarian ingredient ({violation})")

    # 7. Jain Vegetarian
    # Vegetarian rules PLUS exclusion of underground root vegetables
    if "jain_vegetarian" in restrictions:
        violation = _contains_any(full_text, JAIN_ROOT_VEGETABLE_EXCLUSIONS)
        if violation:
            violations.append(f"Contains root vegetable restricted in Jain diet ({violation})")

    # 8. Vegan
    # Vegetarian rules PLUS exclusion of dairy, eggs, honey
    if "vegan" in restrictions:
        violation_egg = _contains_any(full_text, EGG_EXCLUSIONS)
        if violation_egg:
            violations.append(f"Contains egg ({violation_egg})")
        
        violation_dairy = _contains_any(full_text, DAIRY_EXCLUSIONS)
        if violation_dairy:
            violations.append(f"Contains dairy ({violation_dairy})")
            
        violation_honey = _contains_any(full_text, HONEY_EXCLUSIONS)
        if violation_honey:
            violations.append(f"Contains honey ({violation_honey})")

    # 9. Custom User Excluded Ingredients
    if custom_exclusions:
        for custom_item in custom_exclusions:
            custom_item_clean = custom_item.strip().lower()
            if not custom_item_clean:
                continue
            # Multi-word or single word match
            if custom_item_clean in full_text:
                violations.append(f"Contains excluded ingredient ({custom_item_clean})")

    is_compliant = len(violations) == 0
    return is_compliant, violations


def check_ambiguous_ingredients(recipe: dict) -> list[dict]:
    """Inspect recipe ingredients for items with uncertain dietary status.
    
    Returns list of dicts with ingredient name, explanation, and note.
    """
    ingredients_text = " ".join(recipe.get("ingredients", [])).lower()
    flags = []
    
    for item, note in AMBIGUOUS_INGREDIENTS.items():
        pattern = rf"\b{re.escape(item)}\b"
        if re.search(pattern, ingredients_text):
            flags.append({
                "ingredient": item,
                "note": note
            })
            
    return flags


def filter_recipes_dietary(
    recipes: list[dict],
    restrictions: list[str],
    region: str = "India",
    is_explicit_beef_request: bool = False,
    custom_exclusions: list[str] = None,
) -> list[dict]:
    """Filter a list of recipes deterministically based on all active cultural and dietary settings.
    
    Does NOT mutate original recipe dicts.
    """
    compliant_recipes = []
    for recipe in recipes:
        ok, _ = is_recipe_compliant(
            recipe=recipe,
            restrictions=restrictions,
            region=region,
            is_explicit_beef_request=is_explicit_beef_request,
            custom_exclusions=custom_exclusions,
        )
        if ok:
            compliant_recipes.append(recipe)
    return compliant_recipes

