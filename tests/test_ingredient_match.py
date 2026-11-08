"""
test_ingredient_match.py - Tests for ingredient matching.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.ingredient_match import (
    normalize_and_match, compute_match_score,
    find_matching_recipes, INGREDIENT_EQUIVALENTS,
)


@pytest.fixture
def sample_recipes():
    return [
        {
            "id": "r1", "source": "ar", "title": "Buttered Pasta",
            "ingredients": ["1 lb pasta", "2 tablespoons butter", "3 cloves garlic"],
            "instructions": "Boil pasta. Melt butter with garlic. Toss.",
        },
        {
            "id": "r2", "source": "epi", "title": "Chicken Stir Fry",
            "ingredients": ["2 chicken breasts", "1 cup rice", "2 tablespoons soy sauce",
                          "1 bell pepper", "2 cloves garlic"],
            "instructions": "Cook chicken. Stir fry with vegetables.",
        },
        {
            "id": "r3", "source": "fn", "title": "Simple Salad",
            "ingredients": ["lettuce", "tomato", "olive oil", "lemon juice"],
            "instructions": "Chop vegetables. Dress with oil and lemon.",
        },
    ]


class TestNormalizeAndMatch:
    def test_direct_match(self):
        assert normalize_and_match("garlic", "3 cloves garlic, minced")
    
    def test_no_match(self):
        assert not normalize_and_match("chicken", "2 tablespoons butter")
    
    def test_equivalent_match(self):
        # butter and unsalted butter are equivalents
        assert normalize_and_match("butter", "2 tablespoons unsalted butter")
    
    def test_partial_match(self):
        assert normalize_and_match("chicken", "2 chicken breasts")


class TestComputeMatchScore:
    def test_full_match(self):
        recipe = {
            "ingredients": ["pasta", "butter", "garlic"],
        }
        result = compute_match_score(["pasta", "butter", "garlic"], recipe)
        assert result["score"] == 1.0
        assert result["missing_count"] == 0
    
    def test_partial_match(self):
        recipe = {
            "ingredients": ["pasta", "butter", "garlic", "cheese"],
        }
        result = compute_match_score(["pasta", "butter"], recipe)
        assert 0 < result["score"] < 1.0
        assert result["missing_count"] > 0
    
    def test_no_match(self):
        recipe = {
            "ingredients": ["pasta", "butter", "garlic"],
        }
        result = compute_match_score(["sushi", "seaweed"], recipe)
        assert result["score"] == 0.0
    
    def test_extra_user_ingredients(self):
        recipe = {
            "ingredients": ["pasta", "butter"],
        }
        result = compute_match_score(["pasta", "butter", "chicken", "rice"], recipe)
        assert result["score"] == 1.0  # All recipe ingredients matched
        assert len(result["extra"]) == 2  # chicken and rice are extra


class TestFindMatchingRecipes:
    def test_basic_matching(self, sample_recipes):
        results = find_matching_recipes(
            user_ingredients=["pasta", "butter", "garlic"],
            recipes=sample_recipes,
            min_match_score=0.3,
            top_n=5,
        )
        assert len(results) > 0
        # Buttered Pasta should be top result
        assert results[0]["recipe"]["id"] == "r1"
    
    def test_no_match_returns_empty(self, sample_recipes):
        results = find_matching_recipes(
            user_ingredients=["sushi", "seaweed", "wasabi"],
            recipes=sample_recipes,
            min_match_score=0.5,
        )
        assert len(results) == 0
    
    def test_dietary_filter(self, sample_recipes):
        results = find_matching_recipes(
            user_ingredients=["rice", "garlic", "soy sauce"],
            recipes=sample_recipes,
            dietary_restrictions=["vegetarian"],
            min_match_score=0.2,
        )
        # Should exclude chicken stir fry
        for r in results:
            assert r["recipe"]["id"] != "r2"


class TestIngredientEquivalents:
    def test_equivalents_are_bidirectional_for_key_pairs(self):
        """Check that key equivalents have reasonable bidirectionality."""
        # butter -> unsalted butter
        assert "unsalted butter" in INGREDIENT_EQUIVALENTS["butter"]
        # unsalted butter -> butter
        assert "butter" in INGREDIENT_EQUIVALENTS["unsalted butter"]

