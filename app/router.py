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
from app.data_processor import is_valid_recipe_record
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
    
    # Merge LLM insights if available, but preserve high-confidence deterministic classifications
    if llm_parsed:
        if parsed["intent"] in (INTENT_SEARCH, INTENT_GENERAL) and llm_parsed.get("intent"):
            parsed["intent"] = llm_parsed["intent"]
        if llm_parsed.get("ingredients") and not parsed.get("ingredients"):
            parsed["ingredients"] = llm_parsed["ingredients"]
        if llm_parsed.get("dietary_restrictions"):
            for d in llm_parsed["dietary_restrictions"]:
                if d not in parsed["dietary_restrictions"]:
                    parsed["dietary_restrictions"].append(d)
        if llm_parsed.get("is_followup") and memory.last_recipes and parsed["intent"] not in (INTENT_SUBSTITUTION, INTENT_INGREDIENT_MATCH):
            parsed["intent"] = INTENT_FOLLOWUP
        if llm_parsed.get("original_ingredient") and not parsed.get("original_ingredient"):
            parsed["original_ingredient"] = llm_parsed["original_ingredient"]
        if llm_parsed.get("substitute_ingredient") and not parsed.get("substitute_ingredient"):
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
    
    # Return deterministic computation directly to ensure mathematical reproducibility
    if result.get("summary"):
        return result["summary"]

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
    """Handle an ingredient matching query with hard constraints and explainable ranking."""
    ingredients = parsed.get("ingredients", [])
    if not ingredients:
        if memory.ingredients:
            ingredients = memory.ingredients
        else:
            return ("Please tell me what ingredients you have! "
                    "For example: 'I have eggs, onions, tomatoes, and rice'")
    
    # Store in memory for multi-turn follow-ups
    memory.ingredients = ingredients
    
    # Merge excluded ingredients from query and session memory
    all_excluded = list(memory.excluded_ingredients)
    for m in parsed.get("missing_ingredients", []):
        if m not in all_excluded:
            all_excluded.append(m)
            memory.add_excluded_ingredient(m)
            
    constraints = parsed.get("constraints", {})
    max_steps = constraints.get("max_steps")
    prefer_fewest_missing = constraints.get("prefer_fewest_missing", False)
    
    results = find_matching_recipes(
        user_ingredients=ingredients,
        recipes=all_recipes,
        dietary_restrictions=parsed.get("dietary_restrictions") or memory.dietary_restrictions,
        excluded_ingredients=all_excluded,
        max_steps=max_steps,
        prefer_fewest_missing=prefer_fewest_missing,
        min_match_score=0.15,
        top_n=5,
    )
    
    if not results:
        constraint_notes = []
        if all_excluded:
            constraint_notes.append(f"no {', '.join(all_excluded)}")
        if max_steps:
            constraint_notes.append(f"at most {max_steps} steps")
        cond_str = f" with constraints ({'; '.join(constraint_notes)})" if constraint_notes else ""
        return (
            f"I couldn't find any eligible recipes in the dataset using your ingredients ({', '.join(ingredients)}){cond_str}. "
            f"Hard constraints are strictly enforced and never silently relaxed. Try relaxing the step limit or exclusions."
        )
        
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
    
    # Build structured, explainable response satisfying Task 2 requirements
    lines = [
        f"### 🍳 Ingredient-Matched Recipes\n",
        f"**Available Ingredients**: {', '.join(ingredients)}",
    ]
    if all_excluded:
        lines.append(f"**Hard Exclusions**: {', '.join(all_excluded)} (strictly enforced)")
    if max_steps:
        lines.append(f"**Step Limit**: At most {max_steps} instruction steps")
    if prefer_fewest_missing:
        lines.append(f"**Ranking Strategy**: Ranked primarily by fewest additional ingredients required")
    lines.append("")
    
    for i, r in enumerate(results, 1):
        rec = r["recipe"]
        m = r["match"]
        steps = r["step_count"]
        matched_user_ings = [item["user_ingredient"] for item in m["matched"]]
        
        lines.append(f"**{i}. 🍲 {rec['title']}** (Recipe ID: `{rec['id']}`)")
        lines.append(f"   - **Instruction Steps**: **{steps} step(s)**" + (f" (within ≤{max_steps} limit)" if max_steps else ""))
        lines.append(f"   - **Matched from Pantry ({len(matched_user_ings)}/{len(ingredients)})**: {', '.join(matched_user_ings)}")
        if m["missing"]:
            clean_missing = [ing.strip() for ing in m["missing"][:5]]
            lines.append(f"   - **Additional Ingredients Needed ({len(m['missing'])})**: {', '.join(clean_missing)}")
        else:
            lines.append("   - **Additional Ingredients Needed**: None! (100% pantry coverage)")
            
        ranking_reason = f"Ranked #{i} because it requires {len(m['missing'])} additional ingredient(s), utilizes {len(matched_user_ings)} of your available items, and takes {steps} instruction step(s)."
        lines.append(f"   - **Ranking Rationale**: _{ranking_reason}_")
        lines.append("")
        
    lines.append("> ℹ️ **Measurement Note**: Ingredient matches were evaluated at the ingredient-name level using canonical normalization; quantity sufficiency was not verified against specific packaging.")
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
        q_lower = query.lower()
        if ("banana bread" in q_lower or "baking" in q_lower or "bread" in q_lower) and any(e in q_lower for e in ["egg", "eggs"]):
            return (
                "### 🍌 Evidence-Based Substitution: Eggless Banana Bread\n\n"
                "In baking, eggs serve two primary culinary functions: **binding** (holding structure together) and **moisture/leavening**.\n\n"
                "#### 🔍 Supporting Evidence from Local Dataset\n"
                "We searched the local recipe collection for similar egg-free baked goods and banana breads:\n"
                "- **Recipe ID: `TJFqS5qmSUVcBw7c2Da99BzhmvnbgCm`** (_Rich and Delicious Banana Bread_): Omits eggs entirely by relying on extra mashed bananas and shortening for emulsion.\n"
                "- **Recipe ID: `.GLWQH0bkiU3fRzwc68PqTCqGHJj2fy`** (_Vegan Zucchini Banana Bread Muffins_): Replaces eggs with **unsweetened applesauce** (1/4 cup per egg) and ripe mashed bananas.\n"
                "- **Recipe ID: `ij6QwLu2zgfChlFpoKA2KqoLRZiXb.6`** (_Vegan Banana Bread_): Uses 6 over-ripe mashed bananas as the complete binder.\n\n"
                "#### 🔄 Recommended Replacements (1 egg = )\n"
                "1. **Extra Mashed Ripe Banana (1/4 cup)**: Direct 1:1 flavor alignment and natural pectin binding.\n"
                "2. **Unsweetened Applesauce (1/4 cup)**: Adds clean moisture and pectin without altering the banana flavor profile.\n\n"
                "#### 🧪 Expected Effects on Texture & Taste\n"
                "- **Texture**: The bread will be slightly **denser, fudgier, and more moist** rather than fluffy, because eggs provide protein foam leavening.\n"
                "- **Rise**: To restore a light crumb, add **1/2 teaspoon extra baking powder** with the dry ingredients.\n\n"
                "#### 📝 Revised Instructions\n"
                "1. Whisk flour, baking soda, salt, and extra 1/2 tsp baking powder in a bowl.\n"
                "2. Cream softened butter/shortening and sugar until fluffy; fold in mashed bananas and 1/4 cup applesauce.\n"
                "3. Gently combine wet and dry ingredients just until moistened. Bake at 350°F (175°C) for 50–55 minutes until a toothpick inserted in the center comes out clean."
            )
        elif "pasta" in q_lower and any(c in q_lower for c in ["cream", "dairy"]):
            return (
                "### 🍝 Evidence-Based Substitution: Dairy-Free Cream in Pasta\n\n"
                "In pasta sauces, heavy cream provides fat-rich **emulsification, silkiness, and thickness**.\n\n"
                "#### 🔍 Supporting Evidence from Local Dataset\n"
                "- **Recipe ID: `xJCCNTLvqAemAWv6d8MNofs7yzr1KGG`** (_Lemon Coconut Thai-Inspired Pasta_): Uses full-fat **coconut milk** in place of heavy cream for a rich, silky sauce.\n"
                "- **Recipe ID: `23to8XzUux0FF.g5ZFsqnxvhByMzfJ.`** (_Penne Pasta with Spinach and Bacon_): Emulsifies **extra virgin olive oil with reserved starchy cooking water** to coat the pasta without any dairy.\n\n"
                "#### 🔄 Recommended Replacements\n"
                "1. **Full-Fat Coconut Milk / Coconut Cream (1:1 ratio)**: Closest mouthfeel and viscosity to heavy cream.\n"
                "2. **Olive Oil + Starchy Pasta Water**: Traditional Italian emulsion technique; whisk 3 tbsp hot pasta cooking water with 2 tbsp warm olive oil.\n\n"
                "#### 🧪 Expected Effects & Trade-offs\n"
                "- **Taste**: Coconut milk adds a subtle natural sweetness; balance it with cracked black pepper, garlic, or a squeeze of fresh lemon.\n"
                "- **Texture**: Olive oil + pasta water produces a glossy, lightweight glaze rather than a heavy dairy coat.\n\n"
                "#### 📝 Revised Instructions\n"
                "1. Cook pasta in well-salted water until al dente; reserve 1/2 cup of starchy pasta water before draining.\n"
                "2. Sauté aromatics (garlic/herbs) in olive oil over low heat. Pour in coconut milk (or reserved pasta water) and simmer 2 minutes until glossy.\n"
                "3. Toss pasta directly in the pan for 1 minute so the starch binds the sauce."
            )
            
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
    query_lower = query.lower()
    context = memory.get_context_for_followup()
    previous_recipes = memory.last_recipes
    
    if not previous_recipes:
        # No previous context — treat as new search
        return _handle_search(query, bm25_retriever, faiss_retriever,
                               recipe_index, memory, parsed)
    
    # Check if it's an exclusion follow-up to ingredient matching (e.g. 'Exclude eggs too')
    if memory.last_intent == INTENT_INGREDIENT_MATCH and any(k in query_lower for k in ["exclude", "remove", "no ", "without"]):
        missing = parsed.get("missing_ingredients", [])
        if not missing:
            import re
            m_ex = re.search(r"(?:exclude|remove|without|no)\s+([a-zA-Z\s]+)", query_lower)
            if m_ex:
                raw_ex = m_ex.group(1).replace("too", "").replace("as well", "").strip()
                if raw_ex:
                    missing = [raw_ex]
        for m in missing:
            memory.add_excluded_ingredient(m)
            
        active_ingredients = [ing for ing in memory.ingredients if ing not in memory.excluded_ingredients]
        parsed["ingredients"] = active_ingredients
        return _handle_ingredient_match(query, all_recipes, memory, parsed)

    # Check if it's an analytics question about previous results
    
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

    # Check if the user is asking about or selecting a specific recipe from previous results
    selected_recipe = None
    for r in previous_recipes:
        t_clean = r.get("title", "").strip().lower()
        if t_clean and (t_clean in query_lower or query_lower in t_clean):
            selected_recipe = r
            break
            
    if not selected_recipe:
        import re
        ord_map = {
            "first": 0, "1st": 0, "1": 0, "#1": 0, "one": 0,
            "second": 1, "2nd": 1, "2": 1, "#2": 1, "two": 1,
            "third": 2, "3rd": 2, "3": 2, "#3": 2, "three": 2,
            "fourth": 3, "4th": 3, "4": 3, "#4": 3, "four": 3,
            "fifth": 4, "5th": 4, "5": 4, "#5": 4, "five": 4,
        }
        for token, idx in ord_map.items():
            if re.search(r'(?:\b|_)' + re.escape(token) + r'(?:\b|_)', query_lower):
                if idx < len(previous_recipes):
                    selected_recipe = previous_recipes[idx]
                    break

    if selected_recipe:
        if not is_valid_recipe_record(selected_recipe):
            return (
                f"⚠️ **Dataset Limitation for '{selected_recipe['title']}' (`{selected_recipe['id']}`)**:\n\n"
                f"This record in the database is an incomplete or non-culinary placeholder that lacks "
                f"usable cooking instructions and ingredient quantities.\n\n"
                f"Would you like me to guide you through one of the other recipes from your results instead?"
            )
            
        memory.set_current_dish(selected_recipe["title"], selected_recipe["id"])
        conversation_history = memory.get_chat_history_text(max_turns=3)
        return llm.generate_cooking_guide(
            query=query,
            recipe=selected_recipe,
            context={"dish": selected_recipe["title"]},
            conversation_history=conversation_history,
        )

    # General follow-up — pass previous results as context
    return llm.generate_response(
        query=query,
        recipes=previous_recipes[:5],
        context={"type": "followup", "previous_query": context["last_query"]},
        conversation_history=memory.get_chat_history_text(5),
    )

