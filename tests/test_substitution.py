"""
test_substitution.py - Tests for ingredient substitution evidence.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.substitution import (
    find_substitution_evidence, generate_substitution_response,
)


@pytest.fixture
def recipe_corpus():
    recipes = [
        {
            "id": "r1", "source": "ar", "title": "Baking with Butter",
            "ingredients": ["1 cup butter", "2 cups flour", "1 cup sugar"],
            "instructions": "Cream butter and sugar. Add flour. Bake.",
        },
        {
            "id": "r2", "source": "epi", "title": "Baking with Margarine",
            "ingredients": ["1 cup margarine", "2 cups flour", "1 cup sugar"],
            "instructions": "Cream margarine and sugar. Add flour. Bake.",
        },
        {
            "id": "r3", "source": "fn", "title": "Cookies",
            "ingredients": ["1 cup butter", "1/2 cup margarine", "flour"],
            "instructions": "Mix both fats. Bake.",
        },
        {
            "id": "r4", "source": "ar", "title": "Fish Tacos",
            "ingredients": ["fish", "lime juice", "tortillas"],
            "instructions": "Cook fish. Top with lime.",
        },
    ]
    index = {r["id"]: r for r in recipes}
    return recipes, index


class TestFindSubstitutionEvidence:
    def test_direct_evidence(self, recipe_corpus):
        recipes, index = recipe_corpus
        # r3 contains BOTH butter and margarine
        evidence = find_substitution_evidence(
            "butter", "margarine", recipes, index
        )
        assert evidence["confidence"] in ("high", "moderate")
        assert "r3" in evidence["direct_evidence"]
    
    def test_no_evidence(self, recipe_corpus):
        recipes, index = recipe_corpus
        evidence = find_substitution_evidence(
            "butter", "dragonfruit", recipes, index
        )
        assert evidence["confidence"] == "none"
        assert len(evidence["direct_evidence"]) == 0
    
    def test_evidence_structure(self, recipe_corpus):
        recipes, index = recipe_corpus
        evidence = find_substitution_evidence(
            "butter", "margarine", recipes, index
        )
        assert "original" in evidence
        assert "substitute" in evidence
        assert "confidence" in evidence
        assert "explanation" in evidence
        assert "warnings" in evidence


class TestGenerateSubstitutionResponse:
    def test_successful_substitution(self, recipe_corpus):
        recipes, index = recipe_corpus
        evidence = find_substitution_evidence("butter", "margarine", recipes, index)
        response = generate_substitution_response(
            "butter", "margarine", "r1", evidence, index
        )
        assert "error" not in response
        assert response["recipe_id"] == "r1"
        assert any("margarine" in ing for ing in response["modified_ingredients"])
    
    def test_ingredient_not_in_recipe(self, recipe_corpus):
        recipes, index = recipe_corpus
        evidence = find_substitution_evidence("butter", "margarine", recipes, index)
        # r4 is Fish Tacos — no butter
        response = generate_substitution_response(
            "butter", "margarine", "r4", evidence, index
        )
        assert "error" in response
    
    def test_recipe_not_found(self, recipe_corpus):
        recipes, index = recipe_corpus
        evidence = find_substitution_evidence("butter", "margarine", recipes, index)
        response = generate_substitution_response(
            "butter", "margarine", "nonexistent_id", evidence, index
        )
        assert "error" in response
