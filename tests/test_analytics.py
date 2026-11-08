"""
test_analytics.py - Tests for recipe analytics computations.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.analytics import (
    total_recipe_count, filter_recipes, most_common_ingredients,
    ingredient_count_stats, step_count_stats,
    rank_by_ingredient_count, rank_by_step_count,
)


@pytest.fixture
def sample_recipes():
    """Fixture providing a small set of test recipes."""
    return [
        {
            "id": "r1", "source": "ar", "title": "Simple Pasta",
            "ingredients": ["pasta", "olive oil", "garlic"],
            "instructions": "Boil pasta.\nAdd oil and garlic.",
        },
        {
            "id": "r2", "source": "epi", "title": "Chicken Soup",
            "ingredients": ["chicken", "broth", "carrots", "celery", "noodles"],
            "instructions": "Simmer chicken in broth.\nAdd vegetables.\nCook until tender.",
        },
        {
            "id": "r3", "source": "fn", "title": "Vegetable Stir Fry",
            "ingredients": ["broccoli", "bell pepper", "soy sauce", "garlic", "rice"],
            "instructions": "Stir fry vegetables. Serve over rice.",
        },
        {
            "id": "r4", "source": "ar", "title": "Garlic Bread",
            "ingredients": ["bread", "butter", "garlic"],
            "instructions": "Spread butter and garlic on bread. Bake.",
        },
    ]


class TestTotalRecipeCount:
    def test_unfiltered(self, sample_recipes):
        assert total_recipe_count(sample_recipes) == 4
    
    def test_filtered_by_source(self, sample_recipes):
        count = total_recipe_count(sample_recipes, {"source": "ar"})
        assert count == 2


class TestFilterRecipes:
    def test_filter_by_source(self, sample_recipes):
        result = filter_recipes(sample_recipes, {"source": "ar"})
        assert len(result) == 2
        assert all(r["source"] == "ar" for r in result)
    
    def test_filter_by_ingredient_contains(self, sample_recipes):
        result = filter_recipes(sample_recipes, {"ingredient_contains": ["garlic"]})
        assert len(result) == 3  # r1, r3, r4
    
    def test_filter_by_ingredient_excludes(self, sample_recipes):
        result = filter_recipes(sample_recipes, {"ingredient_excludes": ["chicken"]})
        assert len(result) == 3  # Everything except r2
    
    def test_filter_by_title(self, sample_recipes):
        result = filter_recipes(sample_recipes, {"title_contains": "pasta"})
        assert len(result) == 1
        assert result[0]["id"] == "r1"
    
    def test_combined_filters(self, sample_recipes):
        result = filter_recipes(sample_recipes, {
            "source": "ar",
            "ingredient_contains": ["garlic"],
        })
        assert len(result) == 2  # r1 and r4


class TestMostCommonIngredients:
    def test_basic(self, sample_recipes):
        result = most_common_ingredients(sample_recipes, top_n=3)
        assert len(result) <= 3
        # garlic appears in 3 recipes, should be most common
        top_name = result[0][0]
        assert "garlic" in top_name


class TestIngredientCountStats:
    def test_basic_stats(self, sample_recipes):
        stats = ingredient_count_stats(sample_recipes)
        assert stats["total_recipes"] == 4
        assert stats["min"] == 3  # r1 and r4 have 3
        assert stats["max"] == 5  # r2 and r3 have 5
        assert stats["average"] == 4.0  # (3+5+5+3)/4
    
    def test_empty_list(self):
        stats = ingredient_count_stats([])
        assert stats["total_recipes"] == 0


class TestRankByIngredientCount:
    def test_ascending(self, sample_recipes):
        ranked = rank_by_ingredient_count(sample_recipes, ascending=True, top_n=2)
        assert len(ranked) == 2
        assert ranked[0]["ingredient_count"] <= ranked[1]["ingredient_count"]
    
    def test_descending(self, sample_recipes):
        ranked = rank_by_ingredient_count(sample_recipes, ascending=False, top_n=2)
        assert len(ranked) == 2
        assert ranked[0]["ingredient_count"] >= ranked[1]["ingredient_count"]


class TestRankByStepCount:
    def test_ascending(self, sample_recipes):
        ranked = rank_by_step_count(sample_recipes, ascending=True, top_n=2)
        assert len(ranked) == 2
        assert ranked[0]["step_count"] <= ranked[1]["step_count"]

