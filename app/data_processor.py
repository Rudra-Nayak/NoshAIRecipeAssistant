"""
data_processor.py - Build recipe index and text representations for retrieval.

Responsibilities:
- Create a recipe lookup dictionary (ID -> recipe) for O(1) access
- Build searchable text representations for BM25 and embeddings
- Parse ingredient strings into normalized forms for matching
- Compute dataset-wide statistics for analytics
"""
import re
from collections import Counter
from typing import Optional


def build_recipe_index(recipes: list[dict]) -> dict[str, dict]:
    """Build a dictionary mapping recipe ID to recipe for O(1) lookup.
    
    This is critical: after retrieval returns a list of IDs, we need to
    fetch the original recipe records to verify constraints and build responses.
    
    Args:
        recipes: List of cleaned recipe dictionaries.
        
    Returns:
        Dictionary mapping recipe ID -> full recipe dict.
    """
    return {recipe["id"]: recipe for recipe in recipes}


def build_search_text(recipe: dict) -> str:
    """Create a single searchable text from a recipe's fields.
    
    Combines title, ingredients, and a snippet of instructions into one
    string used for both BM25 tokenization and embedding generation.
    
    Why combine? Because a user might search for "chicken pasta" and the
    word "chicken" is in ingredients while "pasta" is in the title.
    
    Args:
        recipe: A cleaned recipe dictionary.
        
    Returns:
        Combined text string for indexing.
    """
    title = recipe["title"]
    ingredients = ", ".join(recipe["ingredients"])
    # Use first 300 chars of instructions to keep text manageable
    # (full instructions can be very long and dilute semantic signal)
    instructions_snippet = recipe["instructions"][:300]
    
    return f"{title}. Ingredients: {ingredients}. Instructions: {instructions_snippet}"


def build_all_search_texts(recipes: list[dict]) -> list[str]:
    """Build search texts for all recipes, maintaining index alignment.
    
    The returned list is index-aligned with the input recipes list:
    search_texts[i] corresponds to recipes[i].
    
    Args:
        recipes: List of cleaned recipe dictionaries.
        
    Returns:
        List of search text strings.
    """
    return [build_search_text(recipe) for recipe in recipes]


# --- Ingredient Normalization ---

# Common words to strip when normalizing ingredient names
_STRIP_WORDS = {
    "fresh", "freshly", "dried", "ground", "chopped", "minced", "diced",
    "sliced", "grated", "crushed", "melted", "softened", "frozen", "canned",
    "large", "medium", "small", "thin", "thick", "finely", "coarsely",
    "roughly", "lightly", "firmly", "packed", "sifted", "toasted",
    "optional", "divided", "to taste", "for garnish", "as needed",
}

# Pattern to extract the ingredient name from a quantity string like
# "2 tablespoons butter" or "1/2 cup all-purpose flour"
_QUANTITY_PATTERN = re.compile(
    r'^[\d\s./½¼¾⅓⅔⅛⅜⅝⅞-]*'          # Leading numbers and fractions
    r'(?:cups?|tablespoons?|tbsp|teaspoons?|tsp|ounces?|oz|pounds?|lbs?|'
    r'cloves?|cans?|packages?|pinch(?:es)?|dash(?:es)?|pieces?|stalks?|'
    r'heads?|bunche?s?|sprigs?|slices?|sticks?|quarts?|gallons?|liters?|'
    r'milliliters?|ml|grams?|\bg\b|kilograms?|\bkg\b)\s*',
    re.IGNORECASE
)


def normalize_ingredient_name(ingredient_str: str) -> str:
    """Extract and normalize the core ingredient name from a raw string.
    
    Example:
        "2 tablespoons unsalted butter, melted" -> "butter"
        "1/2 cup all-purpose flour" -> "all-purpose flour"
        "salt and pepper to taste" -> "salt and pepper"
    
    This is a heuristic parser — not perfect, but good enough for matching.
    
    Args:
        ingredient_str: Raw ingredient string with quantity.
        
    Returns:
        Normalized ingredient name (lowercase, stripped of quantities and modifiers).
    """
    text = ingredient_str.lower().strip()
    
    # Remove parenthetical notes like "(about 2 cups)" or "(10.75 ounce)"
    text = re.sub(r'\([^)]*\)', '', text)
    
    # Remove quantity and unit prefix
    text = _QUANTITY_PATTERN.sub('', text)
    
    # Remove trailing comma-separated notes like ", melted" or ", finely chopped"
    if ',' in text:
        text = text.split(',')[0]
    
    # Remove modifier words
    words = text.split()
    words = [w for w in words if w not in _STRIP_WORDS]
    
    name = ' '.join(words).strip()
    
    # Remove any remaining leading/trailing punctuation
    name = name.strip('.,;:!? ')
    
    return name if name else ingredient_str.lower().strip()


def get_normalized_ingredients(recipe: dict) -> list[str]:
    """Get normalized ingredient names for a recipe.
    
    Args:
        recipe: A cleaned recipe dictionary.
        
    Returns:
        List of normalized ingredient name strings.
    """
    return [normalize_ingredient_name(ing) for ing in recipe["ingredients"]]


def count_instruction_steps(recipe: dict) -> int:
    """Count the number of instruction steps in a recipe.
    
    Heuristic: split by newlines first; if only one block, split by
    sentence-ending periods followed by a capital letter.
    
    Args:
        recipe: A cleaned recipe dictionary.
        
    Returns:
        Number of instruction steps.
    """
    instructions = recipe["instructions"].strip()
    
    # Try splitting by newlines first (ar and epi use this format)
    steps = [s.strip() for s in instructions.split('\n') if s.strip()]
    
    if len(steps) > 1:
        return len(steps)
    
    # Fallback: count sentences (rough heuristic for fn's prose format)
    # Split on period followed by space and capital letter
    sentences = re.split(r'\.(?:\s+)(?=[A-Z])', instructions)
    return max(len([s for s in sentences if len(s.strip()) > 10]), 1)


def compute_dataset_stats(recipes: list[dict]) -> dict:
    """Compute aggregate statistics over the full recipe dataset.
    
    These stats are precomputed once and used by the analytics module
    to answer statistical queries without re-scanning the data.
    
    Args:
        recipes: List of cleaned recipe dictionaries.
        
    Returns:
        Dictionary of computed statistics.
    """
    total = len(recipes)
    
    # Ingredient counts per recipe
    ingredient_counts = [len(r["ingredients"]) for r in recipes]
    ingredient_counts_sorted = sorted(ingredient_counts)
    
    # Step counts per recipe
    step_counts = [count_instruction_steps(r) for r in recipes]
    step_counts_sorted = sorted(step_counts)
    
    # Most common normalized ingredients across all recipes
    all_ingredients = []
    for recipe in recipes:
        all_ingredients.extend(get_normalized_ingredients(recipe))
    ingredient_freq = Counter(all_ingredients)
    
    # Source distribution
    source_counts = Counter(r["source"] for r in recipes)
    
    stats = {
        "total_recipes": total,
        "source_counts": dict(source_counts),
        "ingredient_count_avg": round(sum(ingredient_counts) / total, 2),
        "ingredient_count_median": ingredient_counts_sorted[total // 2],
        "ingredient_count_min": ingredient_counts_sorted[0],
        "ingredient_count_max": ingredient_counts_sorted[-1],
        "step_count_avg": round(sum(step_counts) / total, 2),
        "step_count_median": step_counts_sorted[total // 2],
        "top_ingredients_50": ingredient_freq.most_common(50),
        "ingredient_frequencies": ingredient_freq,
    }
    
    return stats

