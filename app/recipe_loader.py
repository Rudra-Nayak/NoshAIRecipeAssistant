"""
recipe_loader.py - Load and clean the raw Eight Portions recipe dataset.

Responsibilities:
- Load all three JSON files (ar, epi, fn)
- Clean ADVERTISEMENT noise from AllRecipes ingredients
- Filter out unusable records (missing title/ingredients/instructions)
- Produce a unified list of recipe dictionaries
- Save/load processed data for fast startup
"""
import json
import os
import pickle
from pathlib import Path
from typing import Optional


# --- Constants ---

RAW_DATA_DIR = Path("recipes_raw")
PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_FILE = PROCESSED_DATA_DIR / "recipes.pkl"

# Source file mapping: short label -> filename
SOURCE_FILES = {
    "ar": "recipes_raw_nosource_ar.json",
    "epi": "recipes_raw_nosource_epi.json",
    "fn": "recipes_raw_nosource_fn.json",
}


def load_raw_json(filepath: str | Path) -> dict:
    """Load a single raw JSON recipe file.
    
    Args:
        filepath: Path to the JSON file.
        
    Returns:
        Dictionary mapping recipe IDs to recipe objects.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Recipe file not found: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data


def clean_ingredient(ingredient: str) -> Optional[str]:
    """Clean a single ingredient string.
    
    Removes ADVERTISEMENT noise from AllRecipes data.
    Returns None if the ingredient is empty or only contains ADVERTISEMENT.
    
    Args:
        ingredient: Raw ingredient string.
        
    Returns:
        Cleaned ingredient string, or None if it should be discarded.
    """
    if not isinstance(ingredient, str):
        return None
    
    # Remove trailing ADVERTISEMENT (AllRecipes-specific noise)
    cleaned = ingredient.replace(" ADVERTISEMENT", "").strip()
    
    # Discard standalone ADVERTISEMENT entries and empty strings
    if cleaned == "" or cleaned == "ADVERTISEMENT":
        return None
    
    return cleaned


def clean_recipe(recipe_id: str, recipe: dict, source: str) -> Optional[dict]:
    """Clean and validate a single recipe record.
    
    A recipe is usable if it has a non-empty title, at least one
    valid ingredient, and non-empty instructions.
    
    Args:
        recipe_id: The original recipe ID from the JSON file.
        recipe: Raw recipe dictionary.
        source: Source label ("ar", "epi", or "fn").
        
    Returns:
        Cleaned recipe dict, or None if the recipe is unusable.
    """
    # --- Validate title ---
    title = recipe.get("title")
    if not title or not isinstance(title, str) or not title.strip():
        return None
    
    # --- Validate and clean ingredients ---
    raw_ingredients = recipe.get("ingredients")
    if not raw_ingredients or not isinstance(raw_ingredients, list):
        return None
    
    cleaned_ingredients = []
    for ing in raw_ingredients:
        cleaned = clean_ingredient(ing)
        if cleaned:
            cleaned_ingredients.append(cleaned)
    
    if not cleaned_ingredients:
        return None
    
    # --- Validate instructions ---
    instructions = recipe.get("instructions")
    if not instructions or not isinstance(instructions, str) or not instructions.strip():
        return None
    
    return {
        "id": recipe_id,
        "source": source,
        "title": title.strip(),
        "ingredients": cleaned_ingredients,
        "instructions": instructions.strip(),
    }


def load_all_recipes(raw_dir: str | Path = RAW_DATA_DIR) -> list[dict]:
    """Load and clean all recipes from the three source files.
    
    Args:
        raw_dir: Directory containing the raw JSON files.
        
    Returns:
        List of cleaned recipe dictionaries.
    """
    raw_dir = Path(raw_dir)
    all_recipes = []
    stats = {}
    
    for source, filename in SOURCE_FILES.items():
        filepath = raw_dir / filename
        raw_data = load_raw_json(filepath)
        
        source_total = len(raw_data)
        source_usable = 0
        
        for recipe_id, recipe in raw_data.items():
            cleaned = clean_recipe(recipe_id, recipe, source)
            if cleaned:
                all_recipes.append(cleaned)
                source_usable += 1
        
        stats[source] = {"total": source_total, "usable": source_usable}
        print(f"  [{source}] Loaded {source_usable}/{source_total} recipes from {filename}")
    
    print(f"  Total usable recipes: {len(all_recipes)}")
    return all_recipes


def save_processed(recipes: list[dict], output_path: str | Path = PROCESSED_FILE) -> None:
    """Save processed recipes to disk using pickle for fast loading.
    
    Args:
        recipes: List of cleaned recipe dictionaries.
        output_path: Path to save the processed data.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump(recipes, f)
    
    print(f"  Saved {len(recipes)} recipes to {output_path}")


def load_processed(input_path: str | Path = PROCESSED_FILE) -> Optional[list[dict]]:
    """Load previously processed recipes from disk.
    
    Args:
        input_path: Path to the processed data file.
        
    Returns:
        List of recipe dicts, or None if the file doesn't exist.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        return None
    
    with open(input_path, "rb") as f:
        recipes = pickle.load(f)
    
    return recipes


def get_recipes(raw_dir: str | Path = RAW_DATA_DIR,
                processed_path: str | Path = PROCESSED_FILE,
                force_reload: bool = False) -> list[dict]:
    """Get recipes, loading from cache if available.
    
    This is the main entry point for other modules. It:
    1. Tries to load from processed cache first (fast)
    2. Falls back to loading/cleaning raw data (slow, first time only)
    3. Saves the processed data for next time
    
    Args:
        raw_dir: Directory containing raw JSON files.
        processed_path: Path for the processed cache file.
        force_reload: If True, re-process from raw data even if cache exists.
        
    Returns:
        List of cleaned recipe dictionaries.
    """
    if not force_reload:
        recipes = load_processed(processed_path)
        if recipes is not None:
            print(f"  Loaded {len(recipes)} recipes from cache")
            return recipes
    
    print("  Processing raw recipe data (first time setup)...")
    recipes = load_all_recipes(raw_dir)
    save_processed(recipes, processed_path)
    return recipes

