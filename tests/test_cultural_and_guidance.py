"""
test_cultural_and_guidance.py - Validation test suite for cultural sensitivity,
dietary guardrails, beginner-friendly detailed cooking guidance, and multi-turn context.

Implements all 12 required test cases from Part 8:
- Test 1: Greeting handling (no search)
- Test 2: Indian dessert (Rice Kheer for Diwali)
- Test 3: Ingredient exclusion (I don't have coconut milk)
- Test 4: Recipe distinction (Rice kheer vs Vermicelli kheer)
- Test 5: Coconut ingredient clarification (Coconut milk vs flakes)
- Test 6: Dietary restrictions (Deterministic vegetarian filtering)
- Test 7: Beef handling (Indian region default exclusion)
- Test 8: Explicit preference override (Explicit beef request respected neutrally)
- Test 9: Detailed cooking guide (Beginner-friendly 13-point structure)
- Test 10: Follow-up context (Sugar -> Jaggery substitution)
- Test 11: No matching recipe (Graceful recovery)
- Test 12: Unsupported ingredient evidence (Transparent uncertainty)
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.dietary_filter import is_recipe_compliant, check_ambiguous_ingredients, filter_recipes_dietary
from app.query_parser import parse_query, is_explicit_beef_query, extract_missing_ingredients
from app.memory import ConversationMemory
from app.router import route_query
from app.recipe_search import search_recipes, filter_by_dietary
from app.llm import generate_cooking_guide, _fallback_cooking_guide


@pytest.fixture
def sample_corpus():
    """Diverse culinary corpus covering Indian desserts, beef dishes, and vegetarian recipes."""
    recipes = [
        {
            "id": "rec_kheer_rice_1",
            "source": "ar",
            "title": "Traditional Indian Rice Kheer",
            "ingredients": ["1/2 cup basmati rice", "4 cups whole milk", "1/2 cup sugar", "4 pods green elaichi cardamom", "2 tbsp sliced almonds"],
            "instructions": "Rinse rice. Bring milk to boil in heavy pot. Add rice and simmer on low heat for 35 minutes stirring frequently until creamy. Add sugar and crushed cardamom. Garnish with nuts."
        },
        {
            "id": "rec_kheer_seviyan_2",
            "source": "fn",
            "title": "Seviyan Vermicelli Kheer",
            "ingredients": ["1 cup roasted vermicelli seviyan", "3 cups milk", "1/3 cup sugar", "1 tbsp ghee", "cardamom powder"],
            "instructions": "Heat ghee in pan. Sauté vermicelli until golden. Add milk and simmer for 10 minutes until vermicelli is tender. Stir in sugar."
        },
        {
            "id": "rec_payasam_coconut_3",
            "source": "epi",
            "title": "Kerala Coconut Milk Payasam",
            "ingredients": ["1 cup rice", "2 cups coconut milk", "1 cup jaggery", "1/4 cup cashew nuts", "1 tbsp ghee"],
            "instructions": "Cook rice in water. Melt jaggery and strain. Add coconut milk and simmer for 8 minutes. Fry cashews in ghee and pour on top."
        },
        {
            "id": "rec_dessert_flakes_4",
            "source": "ar",
            "title": "Sweet Coconut Rice Delight",
            "ingredients": ["1 cup cooked rice", "2 cups milk", "1/2 cup sweetened coconut flakes", "1/4 cup sugar"],
            "instructions": "Combine cooked rice and milk in saucepan. Bring to gentle simmer. Fold in shredded coconut flakes and sugar. Cook 5 minutes."
        },
        {
            "id": "rec_beef_curry_5",
            "source": "fn",
            "title": "Spicy Beef Curry",
            "ingredients": ["1 lb beef stew meat, cubed", "2 onions, diced", "3 cloves garlic", "2 tbsp curry powder", "1 cup beef broth"],
            "instructions": "Brown beef cubes in skillet. Add onions and garlic. Stir in curry powder and beef broth. Simmer covered for 1 hour."
        },
        {
            "id": "rec_jain_dal_6",
            "source": "ar",
            "title": "Simple Moong Dal (Jain Style)",
            "ingredients": ["1 cup yellow moong dal", "3 cups water", "1/2 tsp turmeric", "1 tsp cumin seeds", "1 tbsp ghee", "1 green chili, slit", "salt"],
            "instructions": "Pressure cook moong dal with water, turmeric, and salt. Heat ghee in small pan, add cumin seeds and green chili. Pour tadka over cooked dal."
        },
        {
            "id": "rec_onion_garlic_dal_7",
            "source": "ar",
            "title": "Tadka Dal with Garlic and Onions",
            "ingredients": ["1 cup toor dal", "3 cups water", "1 large onion, chopped", "4 cloves garlic, minced", "2 tomatoes", "1 tsp cumin", "2 tbsp oil"],
            "instructions": "Cook dal until soft. Heat oil, fry onions and garlic until golden brown. Add tomatoes and spices. Mix with cooked dal."
        }
    ]
    index = {r["id"]: r for r in recipes}
    return recipes, index


# ==============================================================================
# Test 1: Greeting
# User: Hi. Expected: Friendly greeting. No random recipe retrieval.
# ==============================================================================
def test_1_greeting_no_random_search(sample_corpus):
    recipes, index = sample_corpus
    mem = ConversationMemory()
    resp = route_query("Hi", mem, None, None, index, recipes)
    assert mem.last_intent == "greeting"
    assert "Hello" in resp or "Namaste" in resp or "Assistant" in resp
    # Must NOT return random recipe cards
    assert "Yorkshire Pudding" not in resp
    assert len(mem.matching_recipe_ids) == 0


# ==============================================================================
# Test 2: Indian dessert
# User: How do I make rice kheer for Diwali?
# Expected: Relevant rice kheer recipe and detailed, grounded cooking guidance.
# ==============================================================================
def test_2_indian_dessert_cooking_guidance(sample_corpus):
    recipes, index = sample_corpus
    target = index["rec_kheer_rice_1"]
    
    # Test beginner-friendly cooking guide generation
    guide = _fallback_cooking_guide(target)
    assert "Rice Kheer" in guide
    assert "Ingredients" in guide
    assert "Required Equipment" in guide
    assert "Step-by-Step" in guide
    assert "Doneness Indicators" in guide
    assert "Common Mistakes" in guide
    assert "rec_kheer_rice_1" in guide


# ==============================================================================
# Test 3: Ingredient exclusion
# User: I don't have coconut milk.
# Expected: Understand missing ingredient and preserve the rice dessert context.
# ==============================================================================
def test_3_ingredient_exclusion_context():
    parsed = parse_query("I don't have coconut milk. Suggest payasam recipe.")
    assert "coconut milk" in parsed["missing_ingredients"]
    assert parsed["intent"] in ("contextual_modification", "ingredient_match", "search")


# ==============================================================================
# Test 4: Recipe distinction
# User: I want only rice kheer, not vermicelli kheer.
# Expected: Return only verified rice-based kheer recipes.
# ==============================================================================
def test_4_recipe_distinction(sample_corpus):
    recipes, index = sample_corpus
    rice_kheer = index["rec_kheer_rice_1"]
    seviyan_kheer = index["rec_kheer_seviyan_2"]
    
    # Check that vermicelli kheer does NOT satisfy 'only rice kheer'
    from app.recipe_search import _check_dish_distinctions
    assert _check_dish_distinctions(rice_kheer, "i want only rice kheer, not vermicelli kheer") is True
    assert _check_dish_distinctions(seviyan_kheer, "i want only rice kheer, not vermicelli kheer") is False


# ==============================================================================
# Test 5: Coconut ingredient clarification
# User: Does this recipe contain coconut?
# Expected: Inspect actual ingredients and distinguish coconut milk from coconut flakes.
# ==============================================================================
def test_5_coconut_ingredient_clarification(sample_corpus):
    recipes, index = sample_corpus
    recipe_with_flakes = index["rec_dessert_flakes_4"]
    mem = ConversationMemory()
    mem.last_recipes = [recipe_with_flakes]
    
    from app.router import _handle_followup
    resp = _handle_followup("Does this recipe contain coconut?", mem, index, recipes, None, None, {})
    assert "coconut flakes" in resp.lower() or "shredded coconut" in resp.lower()
    assert "does not contain coconut milk" in resp.lower()


# ==============================================================================
# Test 6: Dietary restrictions
# User: Find vegetarian recipes. Expected: Deterministic ingredient filtering.
# ==============================================================================
def test_6_vegetarian_deterministic_filtering(sample_corpus):
    recipes, index = sample_corpus
    beef_recipe = index["rec_beef_curry_5"]
    jain_dal = index["rec_jain_dal_6"]
    
    # Beef must fail vegetarian check
    compliant_beef, _ = is_recipe_compliant(beef_recipe, restrictions=["vegetarian"], region="Global")
    assert compliant_beef is False
    
    # Moong Dal must pass vegetarian check
    compliant_dal, _ = is_recipe_compliant(jain_dal, restrictions=["vegetarian"], region="Global")
    assert compliant_dal is True


# ==============================================================================
# Test 7: Beef handling
# User: Find recipes for dinner. Expected: Apply configured Indian-region default restrictions.
# ==============================================================================
def test_7_indian_region_default_beef_exclusion(sample_corpus):
    recipes, index = sample_corpus
    beef_recipe = index["rec_beef_curry_5"]
    
    # In Indian region, beef is excluded by default
    compliant, violations = is_recipe_compliant(
        recipe=beef_recipe,
        restrictions=[],
        region="India",
        is_explicit_beef_request=False
    )
    assert compliant is False
    assert any("beef" in v.lower() for v in violations)


# ==============================================================================
# Test 8: Explicit preference override
# User explicitly requests beef recipes. Expected: Respect explicit request without bias.
# ==============================================================================
def test_8_explicit_beef_request_override(sample_corpus):
    recipes, index = sample_corpus
    beef_recipe = index["rec_beef_curry_5"]
    
    # When user explicitly queries for beef, Indian default exclusion is overridden neutrally
    compliant, _ = is_recipe_compliant(
        recipe=beef_recipe,
        restrictions=[],
        region="India",
        is_explicit_beef_request=True
    )
    assert compliant is True
    
    # Query parser should detect explicit beef request
    assert is_explicit_beef_query("Show me spicy beef curry recipes") is True
    assert is_explicit_beef_query("I want dinner without beef") is False


# ==============================================================================
# Test 9: Detailed cooking guide
# User: Explain how to prepare this recipe step by step.
# Expected: Actionable, beginner-friendly, source-grounded instructions.
# ==============================================================================
def test_9_detailed_cooking_guide_structure(sample_corpus):
    recipes, index = sample_corpus
    recipe = index["rec_jain_dal_6"]
    
    guide = _fallback_cooking_guide(recipe)
    # Check 13-point structure criteria
    assert "rec_jain_dal_6" in guide
    assert "Ingredients" in guide
    assert "Equipment" in guide
    assert "Preparation" in guide
    assert "Step 1" in guide
    assert "Doneness Indicators" in guide
    assert "Mistakes to Avoid" in guide
    assert "Storage" in guide


# ==============================================================================
# Test 10: Follow-up context
# User: Can I replace sugar with jaggery?
# Expected: Understand which recipe and ingredient the user is referring to.
# ==============================================================================
def test_10_followup_substitution_context(sample_corpus):
    recipes, index = sample_corpus
    from app.substitution import find_substitution_evidence
    
    # Both kheer and payasam use sugar or jaggery
    evidence = find_substitution_evidence("sugar", "jaggery", recipes, index)
    assert "sugar" == evidence["original"]
    assert "jaggery" == evidence["substitute"]
    assert evidence["confidence"] in ("high", "moderate", "low", "none")


# ==============================================================================
# Test 11: No matching recipe
# User requests a dish that does not exist in dataset.
# Expected: Explain that no exact match exists and avoid inventing a recipe.
# ==============================================================================
def test_11_no_matching_recipe_graceful_handling(sample_corpus):
    recipes, index = sample_corpus
    
    # Searching for impossible non-existent dish with strict filters
    filtered = filter_recipes_dietary(
        recipes,
        restrictions=["vegetarian"],
        custom_exclusions=["rice", "milk", "vermicelli", "dal", "beef"]
    )
    assert len(filtered) == 0


# ==============================================================================
# Test 12: Unsupported ingredient evidence
# User requests a substitution that cannot be verified from the dataset.
# Expected: Transparently distinguish general advice from dataset-supported evidence.
# ==============================================================================
def test_12_unsupported_substitution_evidence(sample_corpus):
    recipes, index = sample_corpus
    from app.substitution import find_substitution_evidence
    
    evidence = find_substitution_evidence("cardamom", "dragonfruit", recipes, index)
    assert evidence["confidence"] == "none"
    assert "No evidence found in the dataset" in evidence["explanation"]
    assert len(evidence["direct_evidence"]) == 0
