"""
query_parser.py - Extract structured intent and constraints from user queries.

Responsibilities:
- Classify query intent (search, analytics, ingredient_match, substitution, followup)
- Extract ingredients, dietary restrictions, and other constraints
- Provide a deterministic fallback when the LLM is unavailable
"""
import re
from typing import Optional


# --- Intent Types ---
INTENT_SEARCH = "search"
INTENT_ANALYTICS = "analytics"
INTENT_INGREDIENT_MATCH = "ingredient_match"
INTENT_SUBSTITUTION = "substitution"
INTENT_FOLLOWUP = "followup"
INTENT_GREETING = "greeting"
INTENT_GENERAL = "general"


# --- Greeting / Conversational patterns ---
_GREETING_EXACT = {
    "hi", "hello", "hey", "hiya", "howdy", "hola", "yo",
    "good morning", "good afternoon", "good evening", "good day",
    "what's up", "whats up", "sup", "help", "who are you",
    "what can you do", "thanks", "thank you", "thx", "bye", "goodbye"
}

_GREETING_PREFIXES = (
    "hi ", "hello ", "hey ", "good morning ", "good afternoon ",
    "good evening ", "who are you", "what can you do"
)


# --- Keyword patterns for deterministic intent classification ---

_ANALYTICS_KEYWORDS = [
    "how many", "count", "total", "average", "median", "most common",
    "most popular", "least common", "statistics", "stats", "ranking",
    "top", "longest", "shortest", "fewest", "most ingredients",
    "least ingredients", "most steps", "fewest steps",
]

_SUBSTITUTION_KEYWORDS = [
    "substitute", "substitution", "replace", "replacement", "swap",
    "instead of", "alternative to", "use instead", "in place of",
    "can i use", "what can i use instead",
]

_INGREDIENT_MATCH_KEYWORDS = [
    "i have", "what can i make with", "recipes with",
    "using these ingredients", "ingredients i have",
    "cook with", "make with",
]

_DIETARY_KEYWORDS = {
    "vegetarian": ["vegetarian", "no meat", "meatless", "meat-free"],
    "vegan": ["vegan", "no dairy", "no eggs", "plant-based", "dairy-free"],
    "gluten-free": ["gluten-free", "gluten free", "no gluten", "celiac"],
    "dairy-free": ["dairy-free", "dairy free", "no dairy", "lactose-free"],
    "nut-free": ["nut-free", "nut free", "no nuts", "peanut-free"],
}


def classify_intent_deterministic(query: str) -> str:
    """Classify query intent using keyword matching.
    
    This is the fallback when the LLM is unavailable. It checks for
    known keyword patterns in order of specificity.
    
    Args:
        query: The user's raw query string.
        
    Returns:
        One of the INTENT_* constants.
    """
    query_clean = query.lower().strip().strip('!?.,')
    
    # Check greeting / conversational opening
    if query_clean in _GREETING_EXACT or query.lower().strip().startswith(_GREETING_PREFIXES):
        # Only treat as greeting if it doesn't also contain explicit search/recipe keywords
        if not any(k in query_clean for k in ["recipe", "cook", "make", "substitute", "ingredient"]):
            return INTENT_GREETING
    
    query_lower = query.lower().strip()
    
    # Check analytics first (most specific patterns)
    for keyword in _ANALYTICS_KEYWORDS:
        if keyword in query_lower:
            return INTENT_ANALYTICS
    
    # Check substitution
    for keyword in _SUBSTITUTION_KEYWORDS:
        if keyword in query_lower:
            return INTENT_SUBSTITUTION
    
    # Check ingredient matching
    for keyword in _INGREDIENT_MATCH_KEYWORDS:
        if keyword in query_lower:
            return INTENT_INGREDIENT_MATCH
    
    # Default to search (most common intent)
    return INTENT_SEARCH


def extract_dietary_restrictions(query: str) -> list[str]:
    """Extract dietary restrictions from a query string.
    
    Args:
        query: The user's raw query string.
        
    Returns:
        List of detected dietary restriction labels.
    """
    query_lower = query.lower()
    restrictions = []
    
    for label, keywords in _DIETARY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query_lower:
                restrictions.append(label)
                break
    
    return restrictions


def extract_ingredients_from_query(query: str) -> list[str]:
    """Extract ingredient names from a natural language query.
    
    Handles patterns like:
    - "I have chicken, rice, and onions"
    - "recipes with tomatoes and basil"
    - "chicken, garlic, olive oil"
    
    Args:
        query: The user's raw query string.
        
    Returns:
        List of extracted ingredient strings.
    """
    query_lower = query.lower().strip()
    
    # Remove common prefixes
    prefixes = [
        "i have ", "what can i make with ", "recipes with ",
        "recipes using ", "using ", "cook with ", "make with ",
        "find recipes with ", "show me recipes with ",
        "search for recipes with ", "find me recipes with ",
    ]
    text = query_lower
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    
    # Remove trailing question marks and periods
    text = text.rstrip('?.! ')
    
    # Split by commas and "and"
    # "chicken, rice, and onions" -> ["chicken", "rice", "onions"]
    parts = re.split(r',\s*|\s+and\s+', text)
    
    # Clean each part
    ingredients = []
    for part in parts:
        part = part.strip()
        if part and len(part) > 1:  # Skip single characters
            ingredients.append(part)
    
    return ingredients


def parse_query(query: str, llm_parse_fn=None) -> dict:
    """Parse a user query into a structured representation.
    
    If an LLM parse function is provided, it's used for intent classification
    and entity extraction. Otherwise, falls back to deterministic parsing.
    
    Args:
        query: The user's raw query string.
        llm_parse_fn: Optional async function for LLM-based parsing.
        
    Returns:
        Dictionary with keys:
            - intent: str (one of INTENT_* constants)
            - query: str (original query)
            - ingredients: list[str] (extracted ingredients)
            - dietary_restrictions: list[str] (detected restrictions)
            - constraints: dict (additional constraints)
    """
    intent = classify_intent_deterministic(query)
    dietary = extract_dietary_restrictions(query)
    
    # Extract ingredients for relevant intents
    ingredients = []
    if intent in (INTENT_INGREDIENT_MATCH, INTENT_SEARCH):
        ingredients = extract_ingredients_from_query(query)
    
    return {
        "intent": intent,
        "query": query,
        "ingredients": ingredients,
        "dietary_restrictions": dietary,
        "constraints": {},
    }

