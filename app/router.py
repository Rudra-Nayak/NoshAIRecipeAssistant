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
    INTENT_SUBSTITUTION, INTENT_FOLLOWUP, INTENT_GREETING,
    INTENT_COOKING_GUIDANCE, INTENT_CONTEXTUAL_MOD, INTENT_DIETARY_UPDATE,
    INTENT_GENERAL,
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

    elif intent == INTENT_COOKING_GUIDANCE:
        response = _handle_cooking_guidance(query, bm25_retriever, faiss_retriever,
                                            recipe_index, all_recipes, memory, parsed)

    elif intent == INTENT_CONTEXTUAL_MOD:
        response = _handle_contextual_mod(query, bm25_retriever, faiss_retriever,
                                          recipe_index, all_recipes, memory, parsed)

    elif intent == INTENT_DIETARY_UPDATE:
        response = _handle_dietary_update(query, memory, parsed)

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
        "- 🔍 **Search recipes**: *\"Find Chicken Tikka recipes\"* or *\"Show me pasta recipes\"*\n"
        "- 🥘 **Cook with what you have**: *\"I have chicken, bell peppers, and soy sauce\"*\n"
        "- 🔄 **Substitutions**: *\"Can I use Greek yogurt instead of sour cream?\"*\n"
        "- 📊 **Dataset statistics**: *\"How many recipes are there?\"* or *\"What is the average number of ingredients?\"*\n\n"
        "What would you like to cook today?"
    )


def _handle_cooking_guidance(
    query: str,
    bm25_retriever,
    faiss_retriever,
    recipe_index: dict,
    all_recipes: list[dict],
    memory: ConversationMemory,
    parsed: dict,
) -> str:
    """Handle structured, beginner-friendly step-by-step cooking guide requests."""
    # Determine the target recipe:
    target_recipe = None
    
    # If the user specifically mentions a dish in the query, search for it
    query_lower = query.lower()
    dish_indicators = ["kheer", "payasam", "curry", "pasta", "cookie", "cake", "soup", "stew", "bread"]
    has_specific_dish = any(d in query_lower for d in dish_indicators) or len(query.split()) > 4
    
    if has_specific_dish:
        # Search for best candidate
        search_res = search_recipes(
            query=query,
            bm25_retriever=bm25_retriever,
            faiss_retriever=faiss_retriever,
            recipe_index=recipe_index,
            parsed_query=parsed,
            top_k=3,
            region=memory.region,
            custom_exclusions=memory.excluded_ingredients,
        )
        if search_res["recipes"]:
            target_recipe = search_res["recipes"][0]
            memory.update_search_results(
                query=query,
                intent=INTENT_COOKING_GUIDANCE,
                recipe_ids=search_res["recipe_ids"],
                recipes=search_res["recipes"],
            )

    # Fallback to previously discussed active recipe
    if not target_recipe and memory.last_recipes:
        target_recipe = memory.last_recipes[0]
        
    if not target_recipe:
        # No recipe available or found
        return (
            "I couldn't find a matching recipe in the dataset to guide you through. "
            "Could you please specify the name of the dish you'd like to prepare (e.g. *'How do I make Rice Kheer?'*)?"
        )
        
    memory.set_current_dish(target_recipe["title"], target_recipe["id"])
    conversation_history = memory.get_chat_history_text(max_turns=3)
    
    return llm.generate_cooking_guide(
        query=query,
        recipe=target_recipe,
        context={"dish": target_recipe["title"]},
        conversation_history=conversation_history,
    )


def _handle_contextual_mod(
    query: str,
    bm25_retriever,
    faiss_retriever,
    recipe_index: dict,
    all_recipes: list[dict],
    memory: ConversationMemory,
    parsed: dict,
) -> str:
    """Handle ingredient omission ('I don't have coconut milk') while preserving dish intent."""
    missing = parsed.get("missing_ingredients", [])
    for m in missing:
        memory.add_excluded_ingredient(m)
        
    query_lower = query.lower()
    
    # Determine target dish: explicitly mentioned in query or preserved in memory
    target_dish = ""
    for kw in ["payasam", "kheer", "curry", "soup", "pasta", "cake", "pudding", "rice"]:
        if kw in query_lower:
            target_dish = kw
            break
            
    if not target_dish and memory.current_dish:
        target_dish = memory.current_dish
    elif not target_dish and memory.last_recipes:
        target_dish = memory.last_recipes[0]["title"]
        
    if not target_dish:
        target_dish = "dessert"
        
    # Search for alternative recipe without the excluded ingredients
    search_query = f"{target_dish}"
    search_res = search_recipes(
        query=search_query,
        bm25_retriever=bm25_retriever,
        faiss_retriever=faiss_retriever,
        recipe_index=recipe_index,
        parsed_query=parsed,
        top_k=3,
        region=memory.region,
        custom_exclusions=memory.excluded_ingredients,
    )
    
    missing_str = ", ".join(memory.excluded_ingredients) if memory.excluded_ingredients else "that ingredient"
    
    if search_res["recipes"]:
        memory.update_search_results(
            query=query,
            intent=INTENT_CONTEXTUAL_MOD,
            recipe_ids=search_res["recipe_ids"],
            recipes=search_res["recipes"],
        )
        
        lines = [
            f"Understood! Since you don't have **{missing_str}**, I preserved your goal to prepare {target_dish.title()} "
            f"and found recipes that do not require it:\n"
        ]
        for i, r in enumerate(search_res["recipes"], 1):
            lines.append(f"**{i}. 🍲 {r['title']}**")
            lines.append(f"   Ingredients: {', '.join(r['ingredients'][:5])}...")
            lines.append(f"   *Instructions overview:* {r['instructions'][:140].replace(chr(10), ' ')}...")
            lines.append("")
        lines.append("💡 *Would you like me to guide you step-by-step through any of these?*")
        return "\n".join(lines)
    else:
        return (
            f"I noted that you don't have {missing_str}. However, I could not find a {target_dish} recipe "
            f"in our dataset completely free of those ingredients. Would you like to explore alternative substitutions?"
        )


def _handle_dietary_update(query: str, memory: ConversationMemory, parsed: dict) -> str:
    """Handle explicit user statements about dietary preferences or regions."""
    dietary = parsed.get("dietary_restrictions", [])
    for d in dietary:
        if d not in memory.dietary_restrictions:
            memory.dietary_restrictions.append(d)
            
    query_lower = query.lower()
    if "region to india" in query_lower:
        memory.set_region("India")
        return "✅ Region preference set to **India 🇮🇳**. Default beef-exclusion and cultural guidelines are now active."
    elif "region to global" in query_lower:
        memory.set_region("Global")
        return "✅ Region preference set to **Global 🌐**."
        
    if memory.dietary_restrictions:
        prefs_str = ", ".join(memory.dietary_restrictions)
        return (
            f"✅ Got it! I've updated your dietary preferences to: **{prefs_str}**.\n"
            f"All upcoming recipe searches and recommendations will deterministically respect these restrictions."
        )
    return "I've noted your dietary preferences!"


def _handle_search(
    query: str,
    bm25_retriever,
    faiss_retriever,
    recipe_index: dict,
    memory: ConversationMemory,
    parsed: dict,
) -> str:
    """Handle a recipe search query with region and custom exclusions."""
    result = search_recipes(
        query=query,
        bm25_retriever=bm25_retriever,
        faiss_retriever=faiss_retriever,
        recipe_index=recipe_index,
        parsed_query=parsed,
        top_k=5,
        region=memory.region,
        custom_exclusions=memory.excluded_ingredients,
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
        lines.append(f"**{i}. 🍲 {recipe['title']}**")
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
                           f"has the fewest steps ({steps}).")
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
                           f"has the fewest ingredients "
                           f"({len(best['ingredients'])}).")
            }
            return llm.generate_response(query, [best], context_dict,
                                          memory.get_chat_history_text(3))
    
    # Check if this is an ingredient inspection question (e.g. 'Does this recipe contain coconut?')
    if any(q in query_lower for q in ["contain coconut", "has coconut", "have coconut", "use coconut", "is there coconut"]) or "contain" in query_lower:
        active_rec = previous_recipes[0]
        ing_text = " ".join(active_rec.get("ingredients", [])).lower()
        
        has_coconut_milk = "coconut milk" in ing_text or "coconut cream" in ing_text
        has_coconut_flakes = "coconut flake" in ing_text or "shredded coconut" in ing_text or "flaked coconut" in ing_text or "desiccated coconut" in ing_text
        has_any_coconut = "coconut" in ing_text
        
        if "coconut" in query_lower:
            if has_coconut_flakes and not has_coconut_milk:
                matching_lines = [i for i in active_rec["ingredients"] if "coconut" in i.lower()]
                return (
                    f"🔍 **Ingredient Inspection for '{active_rec['title']}'**:\n\n"
                    f"- This recipe contains: **{', '.join(matching_lines)}**.\n"
                    f"- **Important Distinction**: It uses **coconut flakes/shredded coconut**, but **does NOT contain coconut milk**.\n\n"
                    f"Coconut flakes add dry texture and sweetness, whereas coconut milk provides liquid creaminess. "
                    f"If you need a recipe that is completely coconut-free, just let me know!"
                )
            elif has_coconut_milk:
                matching_lines = [i for i in active_rec["ingredients"] if "coconut" in i.lower()]
                return (
                    f"🔍 Yes, **'{active_rec['title']}'** contains **{', '.join(matching_lines)}**."
                )
            elif not has_any_coconut:
                return (
                    f"🔍 **Ingredient Inspection for '{active_rec['title']}'**:\n\n"
                    f"I verified the complete ingredient list: **There is no coconut of any kind in this recipe.**\n\n"
                    f"Ingredients listed are: {', '.join(active_rec['ingredients'])}."
                )

    # General follow-up — pass previous results as context
    return llm.generate_response(
        query=query,
        recipes=previous_recipes[:5],
        context={"type": "followup", "previous_query": context["last_query"]},
        conversation_history=memory.get_chat_history_text(5),
    )

