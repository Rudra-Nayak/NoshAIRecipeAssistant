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
INTENT_COOKING_GUIDANCE = "cooking_guidance"
INTENT_CONTEXTUAL_MOD = "contextual_modification"
INTENT_DIETARY_UPDATE = "dietary_update"
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

_COOKING_GUIDANCE_KEYWORDS = [
    "how do i make", "how do i prepare", "how to make", "how to cook",
    "guide me through", "guide me", "step by step", "step-by-step",
    "explain the cooking process", "cooking instructions", "how to prepare",
    "explain how to prepare", "recipe for diwali", "for diwali",
    "detailed instructions", "instructions for"
]

_CONTEXTUAL_MOD_KEYWORDS = [
    "i don't have", "i dont have", "don't have", "dont have",
    "out of", "ran out of", "without using", "without", "no coconut milk",
    "cannot use", "can't use", "cant use"
]

_DIETARY_UPDATE_KEYWORDS = [
    "i am vegetarian", "i'm vegetarian", "i am vegan", "i'm vegan",
    "i am jain", "i'm jain", "set region", "change region", "switch region"
]

_DIETARY_KEYWORDS = {
    "vegetarian": ["vegetarian", "no meat", "meatless", "meat-free"],
    "vegan": ["vegan", "no dairy", "no eggs", "plant-based", "dairy-free"],
    "jain_vegetarian": ["jain", "jain vegetarian", "no root vegetables", "no onion", "no garlic"],
    "no_beef": ["no beef", "without beef", "avoid beef", "beef-free", "beef free"],
    "no_pork": ["no pork", "without pork", "avoid pork", "pork-free", "pork free"],
    "no_seafood": ["no seafood", "no fish", "without seafood", "without fish"],
    "no_eggs": ["no egg", "no eggs", "without egg", "without eggs", "eggless", "egg-free"],
    "gluten-free": ["gluten-free", "gluten free", "no gluten", "celiac"],
    "dairy-free": [
        "dairy-free", "dairy free", "no dairy", "lactose-free",
        "cannot have dairy", "can't have dairy", "cant have dairy",
        "avoid dairy", "without dairy", "allergic to dairy", "no milk", "no cheese"
    ],
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
    
    # Check cooking guidance ("How do I make...", "Guide me through...")
    for keyword in _COOKING_GUIDANCE_KEYWORDS:
        if keyword in query_lower:
            return INTENT_COOKING_GUIDANCE

    # Check contextual modifications ("I don't have coconut milk")
    for keyword in _CONTEXTUAL_MOD_KEYWORDS:
        if keyword in query_lower:
            return INTENT_CONTEXTUAL_MOD

    # Check dietary preference updates ("I am vegetarian", "Set region...")
    for keyword in _DIETARY_UPDATE_KEYWORDS:
        if keyword in query_lower:
            return INTENT_DIETARY_UPDATE

    # Check ingredient matching ("I have...", "What can I make with...")
    for keyword in _INGREDIENT_MATCH_KEYWORDS:
        if keyword in query_lower:
            return INTENT_INGREDIENT_MATCH

    # Check analytics (count, average, statistics)
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
    """Extract ingredient names from a natural language query."""
    query_lower = query.lower().strip()
    
    # Check if query matches "i have ..." clause
    m_have = re.search(r"(?:i\s+have|using|cook\s+with|make\s+with)\s+([^.!?]+)", query_lower)
    text = m_have.group(1).strip() if m_have else query_lower
    
    # Remove common prefixes
    prefixes = [
        "what can i make with ", "recipes with ",
        "recipes using ", "using ", "cook with ", "make with ",
        "find recipes with ", "show me recipes with ",
        "search for recipes with ", "find me recipes with ",
    ]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
            
    # Cut off at suggestion or constraint phrases if present
    text = re.split(r"\b(?:suggest|find|show|with\s+no|without|at\s+most|prefer)\b", text)[0]
    text = text.rstrip('?.! ')
    
    parts = [p.strip() for p in re.split(r",?\s+and\s+|,\s*", text) if p.strip()]
    ingredients = []
    for part in parts:
        part = part.strip()
        if part and len(part) > 1 and part not in ("and", "or", "with"):
            ingredients.append(part)
            
    return ingredients


def is_explicit_beef_query(query: str) -> bool:
    """Check if the user explicitly requested beef recipes.
    
    Returns True only if beef/steak/veal is requested without negation words like 'no', 'without', 'free'.
    """
    query_lower = query.lower()
    
    # Check for negative modifiers
    negatives = ["no beef", "without beef", "avoid beef", "beef-free", "beef free", "no steak", "no veal"]
    if any(neg in query_lower for neg in negatives):
        return False
        
    beef_terms = ["beef", "steak", "veal", "brisket", "flank steak", "roast beef"]
    return any(re.search(rf"\b{re.escape(term)}\b", query_lower) for term in beef_terms)


def extract_missing_ingredients(query: str) -> list[str]:
    """Extract ingredients the user states they do not have or want excluded."""
    query_lower = query.lower()
    missing = []
    
    patterns = [
        r"(?:i\s+)?(?:don'?t|dont)\s+have\s+([^.,;!?]+)",
        r"(?:out\s+of|ran\s+out\s+of)\s+([^.,;!?]+)",
        r"(?:without\s+using|without)\s+([^.,;!?]+)",
        r"(?:no)\s+([a-zA-Z\s]+?)(?:,\s*suggest|\.\s*suggest|\s+and|\s+recipe|\s+steps?|$)",
        r"(?:exclude|remove)\s+([a-zA-Z\s]+?)(?:\s+too|\s+as\s+well|\.|$)",
        r"(?:cannot|can'?t)\s+(?:have|eat|use)\s+([a-zA-Z\s]+?)(?:\.|$)",
    ]
    for pat in patterns:
        matches = re.finditer(pat, query_lower)
        for m in matches:
            raw_match = m.group(1).strip()
            raw_match = re.sub(r"\b(suggest|recipe|recipes|please|some|any|can|you|instead|too|steps?)\b.*", "", raw_match).strip()
            if raw_match and len(raw_match) > 1:
                parts = re.split(r",\s*|\s+or\s+|\s+and\s+", raw_match)
                for p in parts:
                    clean_p = p.strip()
                    if clean_p and len(clean_p) > 1 and clean_p not in ("and", "or"):
                        missing.append(clean_p)
                        
    return list(dict.fromkeys(missing))


def extract_max_steps(query: str) -> Optional[int]:
    """Extract maximum instruction steps constraint if specified in query."""
    query_lower = query.lower()
    word_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
    }
    patterns = [
        r"(?:at\s+most|maximum|max|under|no\s+more\s+than|less\s+than|up\s+to)\s+(\w+)\s+(?:instruction\s+)?steps?",
        r"(\w+)\s+(?:instruction\s+)?steps?\s+(?:or\s+less|or\s+fewer|max|maximum)",
    ]
    for p in patterns:
        m = re.search(p, query_lower)
        if m:
            val_str = m.group(1).strip()
            if val_str.isdigit():
                return int(val_str)
            elif val_str in word_to_num:
                return word_to_num[val_str]
    return None


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
    if intent in (INTENT_INGREDIENT_MATCH, INTENT_SEARCH, INTENT_COOKING_GUIDANCE):
        ingredients = extract_ingredients_from_query(query)
        
    missing = extract_missing_ingredients(query)
    explicit_beef = is_explicit_beef_query(query)
    
    constraints = {}
    max_steps = extract_max_steps(query)
    if max_steps is not None:
        constraints["max_steps"] = max_steps
        
    if any(w in query.lower() for w in ["fewest additional", "fewest missing", "least additional"]):
        constraints["prefer_fewest_missing"] = True
    
    return {
        "intent": intent,
        "query": query,
        "ingredients": ingredients,
        "missing_ingredients": missing,
        "dietary_restrictions": dietary,
        "is_explicit_beef_request": explicit_beef,
        "constraints": constraints,
    }

