"""
memory.py - Lightweight session-based conversation memory.

Stores the minimum context needed for multi-turn conversations:
- Previous queries and intents
- Active recipe filters
- Matching recipe IDs from the last search
- Dietary restrictions
- User-provided ingredients

Design decision: We use a simple dict-based memory per session,
not a database or vector-based memory system. This is sufficient
for a single-session chatbot and keeps the architecture simple.
"""
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ConversationMemory:
    """Stores conversation state for multi-turn interactions.
    
    Attributes:
        history: List of (role, message) tuples for display.
        last_query: The previous user query.
        last_intent: The intent of the previous query.
        active_filters: Currently active recipe filters (carry forward).
        matching_recipe_ids: Recipe IDs from the last search result.
        dietary_restrictions: Active dietary restrictions (persist across turns).
        user_ingredients: Ingredients the user has mentioned having.
        last_recipes: The actual recipe dicts from the last result.
        turn_count: Number of conversation turns.
    """
    history: list = field(default_factory=list)
    last_query: str = ""
    last_intent: str = ""
    active_filters: dict = field(default_factory=dict)
    matching_recipe_ids: list = field(default_factory=list)
    dietary_restrictions: list = field(default_factory=list)
    user_ingredients: list = field(default_factory=list)
    last_recipes: list = field(default_factory=list)
    turn_count: int = 0
    
    def add_user_message(self, message: str) -> None:
        """Record a user message."""
        self.history.append(("user", message))
        self.turn_count += 1
    
    def add_assistant_message(self, message: str) -> None:
        """Record an assistant response."""
        self.history.append(("assistant", message))
    
    def update_search_results(
        self,
        query: str,
        intent: str,
        recipe_ids: list[str],
        recipes: list[dict],
        filters: dict = None,
        dietary: list[str] = None,
        ingredients: list[str] = None,
    ) -> None:
        """Update memory after a search operation.
        
        This is called after every successful search to maintain
        context for follow-up questions.
        
        Args:
            query: The current query.
            intent: Detected intent.
            recipe_ids: IDs of matched recipes.
            recipes: Full recipe dicts of matches.
            filters: Active filters used.
            dietary: Dietary restrictions applied.
            ingredients: Ingredients mentioned by user.
        """
        self.last_query = query
        self.last_intent = intent
        self.matching_recipe_ids = recipe_ids
        self.last_recipes = recipes
        
        if filters:
            self.active_filters.update(filters)
        
        if dietary:
            # Merge new dietary restrictions with existing ones
            for d in dietary:
                if d not in self.dietary_restrictions:
                    self.dietary_restrictions.append(d)
        
        if ingredients:
            self.user_ingredients = ingredients
    
    def get_context_for_followup(self) -> dict:
        """Get the current conversation context for follow-up handling.
        
        Returns:
            Dictionary with all active context that should inform
            the next response.
        """
        return {
            "last_query": self.last_query,
            "last_intent": self.last_intent,
            "active_filters": self.active_filters,
            "matching_recipe_ids": self.matching_recipe_ids,
            "dietary_restrictions": self.dietary_restrictions,
            "user_ingredients": self.user_ingredients,
            "has_previous_results": len(self.matching_recipe_ids) > 0,
            "previous_result_count": len(self.matching_recipe_ids),
        }
    
    def clear(self) -> None:
        """Reset all memory (new conversation)."""
        self.history.clear()
        self.last_query = ""
        self.last_intent = ""
        self.active_filters.clear()
        self.matching_recipe_ids.clear()
        self.dietary_restrictions.clear()
        self.user_ingredients.clear()
        self.last_recipes.clear()
        self.turn_count = 0
    
    def get_chat_history_text(self, max_turns: int = 5) -> str:
        """Get recent chat history as formatted text for LLM context.
        
        Args:
            max_turns: Maximum number of recent turns to include.
            
        Returns:
            Formatted string of recent conversation.
        """
        recent = self.history[-(max_turns * 2):]  # Each turn = user + assistant
        lines = []
        for role, message in recent:
            prefix = "User" if role == "user" else "Assistant"
            # Truncate long messages for context window efficiency
            truncated = message[:500] + "..." if len(message) > 500 else message
            lines.append(f"{prefix}: {truncated}")
        return "\n".join(lines)

