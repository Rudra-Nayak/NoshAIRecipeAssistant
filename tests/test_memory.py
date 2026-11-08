"""
test_memory.py - Tests for conversation memory.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.memory import ConversationMemory


class TestConversationMemory:
    def test_initial_state(self):
        mem = ConversationMemory()
        assert mem.last_query == ""
        assert mem.last_intent == ""
        assert mem.matching_recipe_ids == []
        assert mem.active_filters == {}
        assert mem.dietary_restrictions == []
        assert mem.turn_count == 0
    
    def test_add_messages(self):
        mem = ConversationMemory()
        mem.add_user_message("Hello")
        mem.add_assistant_message("Hi there!")
        assert len(mem.history) == 2
        assert mem.turn_count == 1
    
    def test_update_search_results(self):
        mem = ConversationMemory()
        recipes = [{"id": "r1", "title": "Test"}]
        mem.update_search_results(
            query="chicken pasta",
            intent="search",
            recipe_ids=["r1"],
            recipes=recipes,
            dietary=["vegetarian"],
            ingredients=["chicken", "pasta"],
        )
        assert mem.last_query == "chicken pasta"
        assert mem.last_intent == "search"
        assert mem.matching_recipe_ids == ["r1"]
        assert mem.dietary_restrictions == ["vegetarian"]
        assert mem.user_ingredients == ["chicken", "pasta"]
    
    def test_dietary_restrictions_accumulate(self):
        mem = ConversationMemory()
        mem.update_search_results("q1", "search", [], [], dietary=["vegetarian"])
        mem.update_search_results("q2", "search", [], [], dietary=["gluten-free"])
        assert "vegetarian" in mem.dietary_restrictions
        assert "gluten-free" in mem.dietary_restrictions
    
    def test_clear_resets_everything(self):
        mem = ConversationMemory()
        mem.add_user_message("test")
        mem.update_search_results("q", "search", ["r1"], [{"id": "r1"}], dietary=["vegan"])
        mem.clear()
        assert mem.last_query == ""
        assert mem.turn_count == 0
        assert len(mem.history) == 0
        assert len(mem.dietary_restrictions) == 0
        assert len(mem.matching_recipe_ids) == 0
    
    def test_get_context_for_followup(self):
        mem = ConversationMemory()
        mem.update_search_results("q1", "search", ["r1", "r2"], [], dietary=["vegetarian"])
        context = mem.get_context_for_followup()
        assert context["has_previous_results"] is True
        assert context["previous_result_count"] == 2
        assert context["dietary_restrictions"] == ["vegetarian"]
    
    def test_chat_history_text(self):
        mem = ConversationMemory()
        mem.add_user_message("Hello")
        mem.add_assistant_message("How can I help?")
        text = mem.get_chat_history_text()
        assert "User: Hello" in text
        assert "Assistant: How can I help?" in text
