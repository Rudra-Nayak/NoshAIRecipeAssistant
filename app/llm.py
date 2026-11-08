"""
llm.py - Gemini API integration for natural language understanding and generation.

Responsibilities:
- Parse user queries into structured intents (LLM-enhanced)
- Generate grounded responses using retrieved recipe evidence
- Explain substitution recommendations
- Format analytics results in natural language

Key principles:
- The LLM receives relevant recipe records and evidence, NOT the entire dataset
- The LLM must never fabricate recipe IDs, ingredients, or statistics
- All factual content comes from the retrieval pipeline
- Graceful degradation when API is unavailable
"""
import os
import json
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# --- Client Initialization ---

_client = None


def _get_client():
    """Get or create the Gemini API client.
    
    Returns:
        google.genai.Client instance, or None if API key is missing.
    """
    global _client
    if _client is not None:
        return _client
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        return None
    
    try:
        from google import genai
        _client = genai.Client(api_key=api_key)
        return _client
    except Exception as e:
        print(f"  WARNING: Failed to initialize Gemini client: {e}")
        return None


def get_model_name() -> str:
    """Get the configured Gemini model name."""
    return os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


def is_available() -> bool:
    """Check if the Gemini API is available and configured."""
    return _get_client() is not None


def parse_query_with_llm(query: str, conversation_context: str = "") -> Optional[dict]:
    """Use Gemini to parse a user query into structured intent.
    
    This enhances the deterministic parser with better understanding
    of natural language variations and ambiguity.
    
    Args:
        query: User's raw query.
        conversation_context: Recent conversation history for context.
        
    Returns:
        Parsed query dict, or None if LLM is unavailable.
    """
    client = _get_client()
    if client is None:
        return None
    
    prompt = f"""You are a recipe chatbot query parser. Analyze the user's query and extract structured information.

Previous conversation:
{conversation_context if conversation_context else "None (first message)"}

Current user query: "{query}"

Respond with a JSON object containing:
{{
    "intent": "search" | "analytics" | "ingredient_match" | "substitution" | "followup" | "general",
    "ingredients": ["list", "of", "ingredients mentioned"],
    "dietary_restrictions": ["list of dietary restrictions like vegetarian, vegan, gluten-free"],
    "search_terms": "key search terms extracted from query",
    "is_followup": true/false (does this reference previous conversation context?),
    "original_ingredient": "ingredient to substitute (for substitution intent only)",
    "substitute_ingredient": "replacement ingredient (for substitution intent only)",
    "analytics_type": "count | common_ingredients | avg_ingredients | ranking | stats (for analytics intent only)"
}}

Rules:
- If the query references "those", "them", "which one", "the previous", it's a followup
- Extract ALL ingredients mentioned, even in phrases like "I have chicken and rice"
- Be precise about intent classification
- For substitution queries like "can I use X instead of Y", extract both ingredients

Respond with ONLY the JSON object, no markdown formatting."""

    try:
        response = client.models.generate_content(
            model=get_model_name(),
            contents=prompt,
        )
        
        # Parse JSON from response
        text = response.text.strip()
        # Remove markdown code blocks if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        
        parsed = json.loads(text)
        return parsed
    
    except json.JSONDecodeError:
        return None
    except Exception as e:
        print(f"  WARNING: LLM query parsing failed: {e}")
        return None


def generate_response(
    query: str,
    recipes: list[dict],
    context: dict,
    conversation_history: str = "",
) -> str:
    """Generate a grounded response using retrieved recipe evidence.
    
    The LLM is given specific recipe records and must base its response
    entirely on this evidence. It must not invent recipes or details.
    
    Args:
        query: User's query.
        recipes: Retrieved recipe records (evidence).
        context: Additional context (intent, filters, analytics results, etc).
        conversation_history: Recent chat history.
        
    Returns:
        Generated response string.
    """
    client = _get_client()
    if client is None:
        return _fallback_response(query, recipes, context)
    
    # Build evidence text from retrieved recipes
    evidence_lines = []
    for i, recipe in enumerate(recipes[:5], 1):  # Limit to 5 recipes for context window
        evidence_lines.append(f"\nRecipe {i}:")
        evidence_lines.append(f"  ID: {recipe['id']}")
        evidence_lines.append(f"  Title: {recipe['title']}")
        evidence_lines.append(f"  Source: {recipe['source']}")
        evidence_lines.append(f"  Ingredients: {', '.join(recipe['ingredients'][:15])}")
        instructions_preview = recipe['instructions'][:300]
        evidence_lines.append(f"  Instructions: {instructions_preview}")
    
    evidence_text = "\n".join(evidence_lines)
    
    # Build context text
    context_text = ""
    if context.get("type") == "analytics":
        context_text = f"\nAnalytics Result: {context.get('summary', '')}"
    elif context.get("type") == "ingredient_match":
        context_text = f"\nIngredient Matching Results provided above."
    elif context.get("type") == "substitution":
        context_text = f"\nSubstitution Evidence: {context.get('explanation', '')}"
        context_text += f"\nConfidence: {context.get('confidence', 'unknown')}"
    
    prompt = f"""You are a helpful recipe assistant. Answer the user's question using ONLY the recipe evidence provided below.

CONVERSATION HISTORY:
{conversation_history if conversation_history else "None"}

USER QUERY: {query}

RETRIEVED RECIPE EVIDENCE:
{evidence_text}
{context_text}

RULES:
1. Only reference recipes from the evidence above. Never invent recipes.
2. Always include the recipe ID and title when mentioning a recipe.
3. Do not fabricate ingredients, instructions, or nutritional information.
4. If the evidence doesn't answer the question, say so honestly.
5. For analytics questions, use the provided computed statistics — do not guess.
6. Be concise but informative. Use a friendly, professional tone.
7. If no recipes were found, acknowledge this and suggest alternative searches.
8. Never claim medical safety or make unsupported nutritional claims.

Respond naturally to the user's question:"""

    try:
        response = client.models.generate_content(
            model=get_model_name(),
            contents=prompt,
        )
        return response.text
    
    except Exception as e:
        print(f"  WARNING: LLM generation failed: {e}")
        return _fallback_response(query, recipes, context)


def _fallback_response(query: str, recipes: list[dict], context: dict) -> str:
    """Generate a structured response without the LLM.
    
    Used when the Gemini API is unavailable. Produces a readable
    but template-based response.
    """
    if context.get("type") == "analytics":
        return context.get("summary", "Analytics computation completed.")
    
    if not recipes:
        return (
            "I couldn't find any recipes matching your query. "
            "Try different keywords or broaden your search criteria."
        )
    
    lines = [f"I found {len(recipes)} recipe(s) matching your query:\n"]
    
    for i, recipe in enumerate(recipes[:5], 1):
        lines.append(f"**{i}. {recipe['title']}**")
        lines.append(f"   ID: `{recipe['id']}`")
        lines.append(f"   Source: {recipe['source']}")
        lines.append(f"   Ingredients ({len(recipe['ingredients'])}): "
                     f"{', '.join(recipe['ingredients'][:5])}...")
        instructions_preview = recipe['instructions'][:150].replace('\n', ' ')
        lines.append(f"   Instructions: {instructions_preview}...")
        lines.append("")
    
    return "\n".join(lines)


def generate_substitution_explanation(
    evidence: dict,
    recipe: dict = None,
) -> str:
    """Generate a natural language explanation of substitution evidence.
    
    Args:
        evidence: Evidence dict from substitution module.
        recipe: Optional specific recipe for context.
        
    Returns:
        Explanation string.
    """
    client = _get_client()
    
    if client is None:
        # Fallback: use the pre-built explanation
        lines = [evidence["explanation"]]
        if evidence["warnings"]:
            lines.append("\n⚠️ " + " ".join(evidence["warnings"]))
        lines.append(f"\nEvidence strength: {evidence['confidence']}")
        lines.append(f"Recipes with '{evidence['original']}': {evidence['original_recipe_count']}")
        lines.append(f"Recipes with '{evidence['substitute']}': {evidence['substitute_recipe_count']}")
        lines.append(f"Recipes with both: {evidence['both_recipe_count']}")
        return "\n".join(lines)
    
    prompt = f"""Explain this ingredient substitution evidence to a home cook in a friendly, helpful way:

Original ingredient: {evidence['original']}
Substitute ingredient: {evidence['substitute']}
Confidence level: {evidence['confidence']}
Evidence: {evidence['explanation']}

Recipes using the original: {evidence['original_recipe_count']}
Recipes using the substitute: {evidence['substitute_recipe_count']}
Recipes using both: {evidence['both_recipe_count']}

Warnings: {', '.join(evidence['warnings']) if evidence['warnings'] else 'None'}

Rules:
- Be honest about the confidence level
- Don't make medical or nutritional safety claims
- If confidence is low or none, clearly state the uncertainty
- Suggest trying a small batch first if evidence is limited
- Keep it concise (3-5 sentences)"""

    try:
        response = client.models.generate_content(
            model=get_model_name(),
            contents=prompt,
        )
        return response.text
    except Exception:
        return evidence["explanation"]

