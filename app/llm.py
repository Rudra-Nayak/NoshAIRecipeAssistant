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
    "intent": "search" | "analytics" | "ingredient_match" | "substitution" | "followup" | "greeting" | "general",
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


def generate_cooking_guide(
    query: str,
    recipe: dict,
    context: dict = None,
    conversation_history: str = "",
) -> str:
    """Generate a beginner-friendly, grounded, 13-point detailed cooking guide for a recipe.
    
    Preserves all original ingredients, quantities, and cooking sequence, but expands
    vague instructions with sensory cues, heat levels, equipment, prep, pitfalls, and storage.
    """
    client = _get_client()
    if client is None:
        return _fallback_cooking_guide(recipe)
        
    recipe_id = recipe.get("id", "Unknown")
    title = recipe.get("title", "Recipe")
    source = recipe.get("source", "Unknown").upper()
    ingredients_list = "\n".join(f"- {ing}" for ing in recipe.get("ingredients", []))
    instructions = recipe.get("instructions", "")
    
    prompt = f"""You are a patient, expert chef and conversational cooking guide for Nosh Robotics.
The user is asking for cooking guidance for this recipe:

USER QUERY: {query}

OFFICIAL RETRIEVED RECIPE RECORD:
Recipe Title: {title}
Recipe ID: {recipe_id}
Dataset Source: {source}
Ingredients:
{ingredients_list}

Original Instructions:
{instructions}

TASK:
Produce an actionable, beginner-friendly, and structured step-by-step cooking guide.

MANDATORY RULES:
1. Grounding: You must use the retrieved recipe as the primary source of truth. Preserve all original ingredients and their exact quantities. Do not add mandatory ingredients that are not in the recipe.
2. Label Estimates: If preparation times, cooking times, or servings are not explicitly stated in the source, label them clearly with '(Estimated)'.
3. Beginner Explanations: Expand brief instructions into clear, actionable advice. Instead of just 'simmer until thickened', explain heat level (e.g. low flame), stirring cadence, and sensory/texture indicators (e.g., 'until it coats the back of a spoon').
4. Cultural & Indian Culinary Context: When the recipe involves Indian cuisine or desserts (like Kheer or Payasam), naturally reference familiar terms where appropriate (e.g., Kadai, Ghee, Elaichi/Cardamom, Chawal, Simmer, Tadka) without altering the dish into something else.
5. Format the output cleanly with the following markdown structure:

# 🍲 {title}
> **Recipe ID**: `{recipe_id}` | **Source**: {source}  
> ⏱️ **Prep Time**: [Value or '15 mins (Estimated)'] | 🔥 **Cook Time**: [Value or '30 mins (Estimated)'] | 🍽️ **Servings**: [Value or '4 servings (Estimated)']

### 🛒 Ingredients (Original & Preserved)
[List all ingredients exactly as provided in the recipe]

### 🍳 Required Equipment
[List common kitchen tools needed, e.g., heavy-bottomed pot/kadai, wooden spatula, measuring cups]

### 🥣 Preparation Before Cooking (Mise en Place)
[Essential prep steps before turning on heat: washing, soaking, chopping, crushing spices]

### 👩‍🍳 Step-by-Step Cooking Instructions
[Numbered, beginner-friendly steps with heat levels, stirring tips, and timing cues]

### 👁️ Visual & Texture Doneness Indicators
[Sensory cues indicating when it's done: consistency, aroma, color change]

### ⚠️ Common Mistakes to Avoid
[Practical tips on preventing scorching, overcooking, or curdling]

### 🌿 Serving & Garnishing Suggestions
[Serving temperature, garnishing touches]

### 🧊 Storage Suggestions
[How to store leftovers and shelf life]

Generate the complete guide now:"""

    try:
        response = client.models.generate_content(
            model=get_model_name(),
            contents=prompt,
        )
        return response.text
    except Exception as e:
        print(f"  WARNING: LLM cooking guide generation failed: {e}")
        return _fallback_cooking_guide(recipe)


def _fallback_cooking_guide(recipe: dict) -> str:
    """Generate a structured, beginner-friendly cooking guide deterministically without LLM."""
    title = recipe.get("title", "Delicious Dish")
    recipe_id = recipe.get("id", "N/A")
    source = recipe.get("source", "N/A").upper()
    ingredients = recipe.get("ingredients", [])
    raw_instructions = recipe.get("instructions", "").strip()
    
    # Split instructions into steps
    from app.data_processor import count_instruction_steps
    steps = [s.strip() for s in raw_instructions.split('\n') if s.strip()]
    if len(steps) <= 1:
        import re
        steps = [s.strip() for s in re.split(r'\.(?:\s+)(?=[A-Z])', raw_instructions) if len(s.strip()) > 15]
    if not steps:
        steps = [raw_instructions]
        
    lines = [
        f"# 🍲 {title}",
        f"> **Recipe ID**: `{recipe_id}` | **Source**: {source}",
        f"> ⏱️ **Prep Time**: 15 mins (Estimated) | 🔥 **Cook Time**: 25-35 mins (Estimated) | 🍽️ **Servings**: 4 (Estimated)",
        "",
        "### 🛒 Ingredients (Original & Preserved)",
    ]
    for ing in ingredients:
        lines.append(f"- {ing}")
        
    lines.extend([
        "",
        "### 🍳 Required Equipment",
        "- Heavy-bottomed cooking pan or kadai",
        "- Wooden spoon or heat-resistant spatula",
        "- Measuring spoons & cups",
        "",
        "### 🥣 Preparation Before Cooking (Mise en Place)",
        "- Measure out all ingredients before turning on the heat.",
        "- Rinse and prepare any produce or grains as required.",
        "",
        "### 👩‍🍳 Step-by-Step Cooking Instructions",
    ])
    
    for i, step in enumerate(steps, 1):
        step_text = step.rstrip('.') + '.'
        lines.append(f"{i}. **Step {i}**: {step_text}")
        
    lines.extend([
        "",
        "### 👁️ Visual & Texture Doneness Indicators",
        "- Check for a cohesive texture, aromatic fragrance, and uniform cooking.",
        "- Liquids should gently simmer without rapid boiling to avoid burning.",
        "",
        "### ⚠️ Common Mistakes to Avoid",
        "- Keep the heat on low-to-medium flame to prevent sticking or scorching.",
        "- Stir periodically, especially when cooking dairy or grain-based dishes.",
        "",
        "### 🌿 Serving & Garnishing Suggestions",
        "- Serve hot or at room temperature as preferred.",
        "- Garnish with fresh herbs, roasted nuts, or appropriate spices.",
        "",
        "### 🧊 Storage Suggestions",
        "- Cool completely and store in an airtight container in the refrigerator for up to 3 days."
    ])
    
    return "\n".join(lines)


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
    
    lines = [f"I found {len(recipes)} matching recipe(s) for your request:\n"]
    
    for i, recipe in enumerate(recipes[:5], 1):
        lines.append(f"### {i}. 🍲 {recipe['title']}")
        lines.append(f"> **ID**: `{recipe['id']}` | **Source**: `{recipe['source'].upper()}`")
        lines.append(f"**Key Ingredients ({len(recipe['ingredients'])} total):**")
        for ing in recipe['ingredients'][:4]:
            lines.append(f"- {ing}")
        if len(recipe['ingredients']) > 4:
            lines.append(f"- *...and {len(recipe['ingredients']) - 4} more*")
            
        instructions_preview = recipe['instructions'][:180].replace('\n', ' ').strip()
        lines.append(f"**Cooking Overview:** {instructions_preview}...")
        lines.append("*💡 Tip: Ask me 'How do I prepare this step-by-step?' for detailed cooking guidance!*")
        lines.append("\n---\n")
    
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

