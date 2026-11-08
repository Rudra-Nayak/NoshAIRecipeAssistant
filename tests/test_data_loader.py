"""
test_data_loader.py - Tests for recipe loading and data processing.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.recipe_loader import clean_ingredient, clean_recipe, load_all_recipes
from app.data_processor import (
    build_recipe_index, build_search_text,
    normalize_ingredient_name, count_instruction_steps,
)


class TestCleanIngredient:
    """Test ingredient cleaning logic."""
    
    def test_removes_advertisement_suffix(self):
        result = clean_ingredient("2 tablespoons butter ADVERTISEMENT")
        assert result == "2 tablespoons butter"
    
    def test_removes_standalone_advertisement(self):
        result = clean_ingredient("ADVERTISEMENT")
        assert result is None
    
    def test_preserves_normal_ingredient(self):
        result = clean_ingredient("1 cup flour")
        assert result == "1 cup flour"
    
    def test_handles_empty_string(self):
        result = clean_ingredient("")
        assert result is None
    
    def test_handles_none(self):
        result = clean_ingredient(None)
        assert result is None
    
    def test_strips_whitespace(self):
        result = clean_ingredient("  1 cup sugar  ")
        assert result == "1 cup sugar"


class TestCleanRecipe:
    """Test recipe validation and cleaning."""
    
    def test_valid_recipe_passes(self):
        recipe = {
            "title": "Test Recipe",
            "ingredients": ["1 cup flour", "2 eggs"],
            "instructions": "Mix and bake.",
        }
        result = clean_recipe("test123", recipe, "ar")
        assert result is not None
        assert result["id"] == "test123"
        assert result["source"] == "ar"
        assert result["title"] == "Test Recipe"
    
    def test_null_title_rejected(self):
        recipe = {"title": None, "ingredients": ["flour"], "instructions": "Bake."}
        assert clean_recipe("id1", recipe, "ar") is None
    
    def test_empty_recipe_rejected(self):
        assert clean_recipe("id1", {}, "ar") is None
    
    def test_empty_ingredients_rejected(self):
        recipe = {"title": "Test", "ingredients": [], "instructions": "Bake."}
        assert clean_recipe("id1", recipe, "ar") is None
    
    def test_empty_instructions_rejected(self):
        recipe = {"title": "Test", "ingredients": ["flour"], "instructions": ""}
        assert clean_recipe("id1", recipe, "ar") is None
    
    def test_advertisement_cleaned_from_ingredients(self):
        recipe = {
            "title": "Test",
            "ingredients": ["1 cup flour ADVERTISEMENT", "ADVERTISEMENT"],
            "instructions": "Bake.",
        }
        result = clean_recipe("id1", recipe, "ar")
        assert result is not None
        assert result["ingredients"] == ["1 cup flour"]
    
    def test_all_advertisement_ingredients_rejected(self):
        recipe = {
            "title": "Test",
            "ingredients": ["ADVERTISEMENT", "ADVERTISEMENT"],
            "instructions": "Bake.",
        }
        assert clean_recipe("id1", recipe, "ar") is None


class TestNormalizeIngredient:
    """Test ingredient name normalization."""
    
    def test_removes_quantity(self):
        result = normalize_ingredient_name("2 tablespoons butter")
        assert "butter" in result
        assert "tablespoons" not in result
    
    def test_removes_modifiers(self):
        result = normalize_ingredient_name("freshly ground black pepper")
        assert "pepper" in result
    
    def test_handles_parenthetical_notes(self):
        result = normalize_ingredient_name("1 (14 ounce) can tomatoes")
        assert "tomatoes" in result
    
    def test_handles_fraction(self):
        result = normalize_ingredient_name("1/2 cup flour")
        assert "flour" in result
    
    def test_preserves_compound_names(self):
        result = normalize_ingredient_name("1 cup all-purpose flour")
        assert "flour" in result


class TestCountInstructionSteps:
    """Test instruction step counting."""
    
    def test_newline_separated(self):
        recipe = {"instructions": "Step one.\nStep two.\nStep three."}
        assert count_instruction_steps(recipe) == 3
    
    def test_single_paragraph(self):
        recipe = {"instructions": "Mix the flour and sugar. Add the eggs. Bake at 350F."}
        assert count_instruction_steps(recipe) >= 1
    
    def test_empty_lines_ignored(self):
        recipe = {"instructions": "Step one.\n\n\nStep two."}
        assert count_instruction_steps(recipe) == 2


class TestBuildRecipeIndex:
    """Test recipe index building."""
    
    def test_builds_correct_index(self):
        recipes = [
            {"id": "a1", "title": "Recipe A", "ingredients": [], "instructions": ""},
            {"id": "b2", "title": "Recipe B", "ingredients": [], "instructions": ""},
        ]
        index = build_recipe_index(recipes)
        assert "a1" in index
        assert "b2" in index
        assert index["a1"]["title"] == "Recipe A"
    
    def test_empty_list(self):
        assert build_recipe_index([]) == {}


class TestBuildSearchText:
    """Test search text generation."""
    
    def test_combines_all_fields(self):
        recipe = {
            "title": "Chicken Soup",
            "ingredients": ["chicken", "noodles", "broth"],
            "instructions": "Boil the chicken in broth. Add noodles.",
        }
        text = build_search_text(recipe)
        assert "Chicken Soup" in text
        assert "chicken" in text
        assert "noodles" in text
        assert "Boil" in text

