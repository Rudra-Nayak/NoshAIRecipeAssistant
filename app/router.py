"""
router.py - Query routing and orchestration.

Central dispatcher that:
1. Parses the user query (deterministic + optional LLM enhancement)
2. Routes to the appropriate handler (search, analytics, ingredient match, substitution)
3. Manages conversation memory updates
4. Returns the final response

This is the main entry point that the Streamlit UI calls.
"""
from typing import Optional

from app.query_parser import (
    parse_query, classify_intent_deterministic,
    INTENT_SEARCH, INTENT_ANALYTICS, INTENT_INGREDIENT_MATCH,
    INTENT_SUBSTITUTION, INTENT_FOLLOWUP, INTENT_GREETING, INTENT_GENERAL,
)
from app.recipe_search import search_recipes
from app.analytics import compute_analytics, filter_recipes
from app.ingredient_match import find_matching_recipes
from app.substitution import find_substitution_evidence, generate_substitution_response
from app.memory import ConversationMemory
from app import llm


def route_query(
    query: str,
    memory: ConversationMemory,
    bm25_retriever,
    faiss_retriever,
    recipe_index: dict[str, dict],
    all_recipes: list[dict],
) -> str:
    """Route a user query to the appropriate handler and return a response.
    
    This is the main orchestration function. It:
    1. Tries LLM-enhanced parsing, falls back to deterministic
    2. Detects follow-ups and applies previous context
    3. Routes to the right handler
    4. Updates conversation memory
    5. Generates a response (LLM-enhanced or template-based)
    
    Args:
        query: User's raw query string.
        memory: ConversationMemory instance for this session.
        bm25_retriever: BM25Retriever instance.
        faiss_retriever: FAISSRetriever instance (can be None).
        recipe_index: Recipe ID -> recipe dict mapping.
        all_recipes: Full list of all recipes.
        
    Returns:
        Response string to display to the user.
    """
    memory.add_user_message(query)
    
    # --- Step 1: Parse query ---
    # Fast check for conversational greetings / pleasantries
    deterministic_intent = classify_intent_deterministic(query)
    if deterministic_intent == INTENT_GREETING:
        response = _handle_greeting(query)
        memory.last_intent = INTENT_GREETING
        memory.add_assistant_message(response)
        return response

    # Try LLM parsing for complex queries
    llm_parsed = None
    if llm.is_available():
        conversation_context = memory.get_chat_history_text(max_turns=3)
        llm_parsed = llm.parse_query_with_llm(query, conversation_context)
    
    # Deterministic parsing as primary/fallback
    parsed = parse_query(query)
    
    # Merge LLM insights if available
    if llm_parsed:
        if llm_parsed.get("intent"):
            parsed["intent"] = llm_parsed["intent"]
        if llm_parsed.get("ingredients"):
            parsed["ingredients"] = llm_parsed["ingredients"]
        if llm_parsed.get("dietary_restrictions"):
            parsed["dietary_restrictions"] = llm_parsed["dietary_restrictions"]
        if llm_parsed.get("is_followup"):
            parsed["intent"] = INTENT_FOLLOWUP
        if llm_parsed.get("original_ingredient"):
            parsed["original_ingredient"] = llm_parsed["original_ingredient"]
        if llm_parsed.get("substitute_ingredient"):
            parsed["substitute_ingredient"] = llm_parsed["substitute_ingredient"]
    
    # --- Step 2: Handle follow-ups ---
    if parsed["intent"] == INTENT_FOLLOWUP or _is_followup_heuristic(query):
        context = memory.get_context_for_followup()
        if context["has_previous_results"]:
            parsed["intent"] = _infer_followup_intent(query, context)
            # Carry forward filters and restrictions
            if context["dietary_restrictions"]:
                parsed["dietary_restrictions"] = context["dietary_restrictions"]
            if context["active_filters"]:
                parsed["constraints"].update(context["active_filters"])
    
    # --- Step 3: Route to handler ---
    intent = parsed["intent"]
    
    if intent == INTENT_GREETING:
        response = _handle_greeting(query)

    elif intent == INTENT_ANALYTICS:
        response = _handle_analytics(query, all_recipes, memory, parsed)
    
    elif intent == INTENT_INGREDIENT_MATCH:
        response = _handle_ingredient_match(query, all_recipes, memory, parsed)
    
    elif intent == INTENT_SUBSTITUTION:
        response = _handle_substitution(query, all_recipes, recipe_index, memory, parsed)
    
    elif intent == INTENT_FOLLOWUP and memory.last_recipes:
        response = _handle_followup(query, memory, recipe_index, all_recipes,
                                     bm25_retriever, faiss_retriever, parsed)
    
    else:  # INTENT_SEARCH or INTENT_GENERAL
        response = _handle_search(query, bm25_retriever, faiss_retriever,
                                   recipe_index, memory, parsed)
    
    memory.add_assistant_message(response)
    return response


def _is_followup_heuristic(query: str) -> bool:
    """Detect follow-up queries using simple heuristics."""
    followup_indicators = [
        "which one", "which of", "those", "them", "the previous",
        "from those", "among those", "of these", "the first",
        "the second", "the last", "that one", "this one",
        "tell me more", "more about", "show me more",
        "what about", "how about", "and also",
    ]
    query_lower = query.lower()
    return any(indicator in query_lower for indicator in followup_indicators)


def _infer_followup_intent(query: str, context: dict) -> str:
    """Infer the intent of a follow-up query."""
    query_lower = query.lower()
    
    # Check if this is an analytics question about previous results
    analytics_words = ["how many", "count", "fewest", "most", "average", "simplest"]
    if any(w in query_lower for w in analytics_words):
        return INTENT_ANALYTICS
    
    # Default to search (operating on previous results)
    return INTENT_FOLLOWUP


def _handle_greeting(query: str) -> str:
    """Handle conversational greetings and pleasantries."""
    query_lower = query.lower().strip()
    if any(thx in query_lower for thx in ["thank", "thanks", "thx"]):
        return (
            "You're very welcome! Let me know if you need anything else—whether it's finding "
            "another recipe, checking ingredient substitutions, or calculating cooking statistics!"
        )
    
    return (
        "Hello! 👋 I'm your Nosh Recipe Assistant. How can I help you in the kitchen today?\n\n"
        "Here are a few things you can ask me:\n"
        "- 🔍 **Search recipes**: *\"Find slow cooker beef recipes\"* or *\"Show me vegetarian pasta dishes\"*\n"
        "- 🥘 **Cook with what you have**: *\"I have chicken, bell peppers, and soy sauce\"*\n"
        "- 🔄 **Substitutions**: *\"Can I use Greek yogurt instead of sour cream?\"*\n"
        "- 📊 **Dataset statistics**: *\"How many recipes are there?\"* or *\"What is the average number of ingredients?\"*\n\n"
        "What would you like to cook today?"
    )


def _handle_search(
    query: str,
    bm25_retriever,
    faiss_retriever,
    recipe_index: dict,
    memory: ConversationMemory,
    parsed: dict,
) -> str:
    """Handle a recipe search query."""
    result = search_recipes(
        query=query,
        bm25_retriever=bm25_retriever,
        faiss_retriever=faiss_retriever,
        recipe_index=recipe_index,
        parsed_query=parsed,
        top_k=5,
    )
    
    # Update memory
    memory.update_search_results(
        query=query,
        intent=INTENT_SEARCH,
        recipe_ids=result["recipe_ids"],
        recipes=result["recipes"],
        dietary=parsed.get("dietary_restrictions"),
        ingredients=parsed.get("ingredients"),
    )
    
    # Generate response
    context = {"type": "search", "total_candidates": result["total_candidates"]}
    conversation_history = memory.get_chat_history_text(max_turns=3)
    
    response = llm.generate_response(
        query=query,
        recipes=result["recipes"],
        context=context,
        conversation_history=conversation_history,
    )
    
    return response


def _handle_analytics(
    query: str,
    all_recipes: list[dict],
    memory: ConversationMemory,
    parsed: dict,
) -> str:
    """Handle an analytics query.
    
    Analytics operate on the FULL dataset (or filtered subset),
    not just the retrieved top-k.
    """
    # Build filters from context
    filters = {}
    if parsed.get("dietary_restrictions"):
        filters["ingredient_excludes"] = []
    if memory.active_filters:
        filters.update(memory.active_filters)
    
    # If there are previous results and query seems to reference them,
    # operate on the filtered set
    if memory.matching_recipe_ids and _is_followup_heuristic(query):
        # Create a filtered recipe list from previous results
        filtered = [r for r in all_recipes if r["id"] in set(memory.matching_recipe_ids)]
        if filtered:
            result = compute_analytics(query, filtered, filters if filters else None)
        else:
            result = compute_analytics(query, all_recipes, filters if filters else None)
    else:
        result = compute_analytics(query, all_recipes, filters if filters else None)
    
    # Update memory
    memory.update_search_results(
        query=query,
        intent=INTENT_ANALYTICS,
        recipe_ids=memory.matching_recipe_ids,  # Keep previous results
        recipes=memory.last_recipes,
    )
    
    # Generate response
    conversation_history = memory.get_chat_history_text(max_turns=3)
    response = llm.generate_response(
        query=query,
        recipes=[],
        context=result,
        conversation_history=conversation_history,
    )
    
    return response


def _handle_ingredient_match(
    query: str,
    all_recipes: list[dict],
    memory: ConversationMemory,
    parsed: dict,
) -> str:
    """Handle an ingredient matching query."""
    ingredients = parsed.get("ingredients", [])
    if not ingredients:
        return ("Please tell me what ingredients you have! "
                "For example: 'I have chicken, rice, and garlic'")
    
    results = find_matching_recipes(
        user_ingredients=ingredients,
        recipes=all_recipes,
        dietary_restrictions=parsed.get("dietary_restrictions"),
        min_match_score=0.2,
        top_n=5,
    )
    
    if not results:
        response = (
            f"I couldn't find any recipes that match your ingredients "
            f"({', '.join(ingredients)})"
        )
        if parsed.get("dietary_restrictions"):
            response += f" with {', '.join(parsed['dietary_restrictions'])} restrictions"
        response += ". Try adding more common ingredients or relaxing your filters."
        
        memory.update_search_results(
            query=query,
            intent=INTENT_INGREDIENT_MATCH,
            recipe_ids=[],
            recipes=[],
            ingredients=ingredients,
            dietary=parsed.get("dietary_restrictions"),
        )
        return response
    
    # Format response
    recipe_ids = [r["recipe"]["id"] for r in results]
    recipes = [r["recipe"] for r in results]
    
    memory.update_search_results(
        query=query,
        intent=INTENT_INGREDIENT_MATCH,
        recipe_ids=recipe_ids,
        recipes=recipes,
        ingredients=ingredients,
        dietary=parsed.get("dietary_restrictions"),
    )
    
    # Build response with match details
    if llm.is_available():
        # Add match details to context for LLM
        match_details = []
        for r in results[:5]:
            match_details.append({
                "title": r["recipe"]["title"],
                "id": r["recipe"]["id"],
                "score": r["match"]["score"],
                "matched": len(r["match"]["matched"]),
                "missing": len(r["match"]["missing"]),
                "missing_items": r["match"]["missing"][:5],
            })
        context = {"type": "ingredient_match", "matches": match_details}
        return llm.generate_response(query, recipes, context,
                                      memory.get_chat_history_text(3))
    
    # Fallback template response
    lines = [f"Here are recipes matching your ingredients ({', '.join(ingredients)}):\n"]
    for i, r in enumerate(results[:5], 1):
        recipe = r["recipe"]
        match = r["match"]
        lines.append(f"**{i}. {recipe['title']}** (ID: `{recipe['id']}`)")
        lines.append(f"   Match: {match['score']:.0%} "
                     f"({match['matched_count']}/{match['total_recipe_ingredients']} ingredients)")
        if match["missing"]:
            lines.append(f"   Missing: {', '.join(match['missing'][:5])}")
        lines.append("")
    
    return "\n".join(lines)


def _handle_substitution(
    query: str,
    all_recipes: list[dict],
    recipe_index: dict,
    memory: ConversationMemory,
    parsed: dict,
) -> str:
    """Handle a substitution query."""
    original = parsed.get("original_ingredient", "")
    substitute = parsed.get("substitute_ingredient", "")
    
    if not original or not substitute:
        # Try to extract from query directly
        import re
        # Pattern: "substitute X for Y" or "use X instead of Y"
        patterns = [
            r'(?:substitute|replace|swap)\s+(.+?)\s+(?:for|with)\s+(.+?)(?:\?|$)',
            r'(?:use|try)\s+(.+?)\s+instead\s+of\s+(.+?)(?:\?|$)',
            r'(?:can i use|what about)\s+(.+?)\s+(?:instead of|in place of)\s+(.+?)(?:\?|$)',
        ]
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                substitute = match.group(1).strip()
                original = match.group(2).strip()
                break
    
    if not original or not substitute:
        return ("I'd be happy to help with ingredient substitutions! "
                "Please ask something like: 'Can I use almond milk instead of regular milk?'")
    
    evidence = find_substitution_evidence(
        original_ingredient=original,
        substitute_ingredient=substitute,
        recipes=all_recipes,
        recipe_index=recipe_index,
    )
    
    # Update memory
    memory.update_search_results(
        query=query,
        intent=INTENT_SUBSTITUTION,
        recipe_ids=evidence.get("direct_evidence", [])[:5],
        recipes=[recipe_index[rid] for rid in evidence.get("direct_evidence", [])[:5]
                 if rid in recipe_index],
    )
    
    # Generate explanation
    response = llm.generate_substitution_explanation(evidence)
    
    return response


def _handle_followup(
    query: str,
    memory: ConversationMemory,
    recipe_index: dict,
    all_recipes: list[dict],
    bm25_retriever,
    faiss_retriever,
    parsed: dict,
) -> str:
    """Handle a follow-up question about previous results."""
    context = memory.get_context_for_followup()
    previous_recipes = memory.last_recipes
    
    if not previous_recipes:
        # No previous context — treat as new search
        return _handle_search(query, bm25_retriever, faiss_retriever,
                               recipe_index, memory, parsed)
    
    # Check if it's an analytics question about previous results
    query_lower = query.lower()
    
    if any(w in query_lower for w in ["fewest step", "least step", "simplest", "easiest"]):
        from app.data_processor import count_instruction_steps
        ranked = sorted(previous_recipes, key=lambda r: count_instruction_steps(r))
        if ranked:
            best = ranked[0]
            from app.data_processor import count_instruction_steps as cs
            steps = cs(best)
            context_dict = {
                "type": "analytics",
                "summary": (f"Among the previous results, '{best['title']}' "
                           f"(ID: {best['id']}) has the fewest steps ({steps}).")
            }
            return llm.generate_response(query, [best], context_dict,
                                          memory.get_chat_history_text(3))
    
    if any(w in query_lower for w in ["fewest ingredient", "least ingredient"]):
        ranked = sorted(previous_recipes, key=lambda r: len(r["ingredients"]))
        if ranked:
            best = ranked[0]
            context_dict = {
                "type": "analytics",
                "summary": (f"Among the previous results, '{best['title']}' "
                           f"(ID: {best['id']}) has the fewest ingredients "
                           f"({len(best['ingredients'])}).")
            }
            return llm.generate_response(query, [best], context_dict,
                                          memory.get_chat_history_text(3))
    
    # General follow-up — pass previous results as context
    return llm.generate_response(
        query=query,
        recipes=previous_recipes[:5],
        context={"type": "followup", "previous_query": context["last_query"]},
        conversation_history=memory.get_chat_history_text(5),
    )

