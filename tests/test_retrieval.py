"""
test_retrieval.py - Tests for BM25, hybrid retrieval, and reranking.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.keyword_retrieval import BM25Retriever, tokenize
from app.hybrid_retrieval import reciprocal_rank_fusion, deduplicate_results
from app.reranker import compute_rerank_score, rerank_results


class TestTokenize:
    """Test tokenization."""
    
    def test_basic_tokenization(self):
        tokens = tokenize("Hello World 123")
        assert "hello" in tokens
        assert "world" in tokens
        assert "123" in tokens
    
    def test_handles_punctuation(self):
        tokens = tokenize("chicken, rice, and onions!")
        assert "chicken" in tokens
        assert "rice" in tokens
        assert "onions" in tokens
    
    def test_empty_string(self):
        assert tokenize("") == []


class TestBM25Retriever:
    """Test BM25 search functionality."""
    
    @pytest.fixture
    def retriever(self):
        texts = [
            "Chicken Parmesan with pasta and tomato sauce",
            "Vegetable stir fry with rice and soy sauce",
            "Chocolate cake with frosting and sprinkles",
            "Grilled chicken with lemon and herbs",
            "Pasta primavera with vegetables",
        ]
        ids = ["r1", "r2", "r3", "r4", "r5"]
        return BM25Retriever(texts, ids)
    
    def test_basic_search(self, retriever):
        results = retriever.search("chicken", top_k=5)
        assert len(results) > 0
        # Chicken recipes should rank highest
        result_ids = [r[0] for r in results]
        assert "r1" in result_ids or "r4" in result_ids
    
    def test_multi_term_search(self, retriever):
        results = retriever.search("chicken pasta", top_k=5)
        assert len(results) > 0
        # r1 has both chicken and pasta
        assert results[0][0] == "r1"
    
    def test_no_match(self, retriever):
        results = retriever.search("sushi", top_k=5)
        # May return some results with low scores but shouldn't crash
        assert isinstance(results, list)
    
    def test_empty_query(self, retriever):
        results = retriever.search("", top_k=5)
        assert results == []
    
    def test_top_k_limit(self, retriever):
        results = retriever.search("chicken", top_k=2)
        assert len(results) <= 2


class TestRRF:
    """Test Reciprocal Rank Fusion."""
    
    def test_single_list(self):
        ranked = [("r1", 10.0), ("r2", 8.0), ("r3", 5.0)]
        fused = reciprocal_rank_fusion([ranked])
        # Should maintain order
        assert fused[0][0] == "r1"
    
    def test_two_lists_agreement(self):
        list1 = [("r1", 10.0), ("r2", 8.0)]
        list2 = [("r1", 0.9), ("r2", 0.7)]
        fused = reciprocal_rank_fusion([list1, list2])
        # r1 is top in both, should be top in fused
        assert fused[0][0] == "r1"
    
    def test_two_lists_disagreement(self):
        list1 = [("r1", 10.0), ("r2", 8.0)]
        list2 = [("r2", 0.9), ("r3", 0.7)]
        fused = reciprocal_rank_fusion([list1, list2])
        # r2 appears in both lists, should benefit
        fused_ids = [r[0] for r in fused]
        assert "r2" in fused_ids
    
    def test_empty_lists(self):
        fused = reciprocal_rank_fusion([])
        assert fused == []
    
    def test_unique_items_combined(self):
        list1 = [("r1", 10.0)]
        list2 = [("r2", 10.0)]
        fused = reciprocal_rank_fusion([list1, list2])
        fused_ids = {r[0] for r in fused}
        assert fused_ids == {"r1", "r2"}


class TestDeduplicate:
    """Test result deduplication."""
    
    def test_no_duplicates(self):
        results = [("r1", 1.0), ("r2", 0.8)]
        assert deduplicate_results(results) == results
    
    def test_removes_duplicates(self):
        results = [("r1", 1.0), ("r2", 0.8), ("r1", 0.5)]
        deduped = deduplicate_results(results)
        assert len(deduped) == 2
        # First occurrence kept
        assert deduped[0] == ("r1", 1.0)


class TestReranker:
    """Test the reranking scorer."""
    
    def test_title_match_boost(self):
        recipe = {
            "title": "Chicken Parmesan",
            "ingredients": ["chicken", "cheese", "sauce"],
            "instructions": "Bread the chicken. Add sauce and cheese. Bake.",
        }
        score = compute_rerank_score("chicken parmesan", recipe)
        assert score > 0
    
    def test_ingredient_match_boost(self):
        recipe = {
            "title": "Mystery Dish",
            "ingredients": ["chicken breast", "garlic", "olive oil"],
            "instructions": "Cook together.",
        }
        score = compute_rerank_score(
            "chicken garlic", recipe,
            query_ingredients=["chicken", "garlic"]
        )
        assert score > 0
    
    def test_rerank_ordering(self):
        recipes = {
            "r1": {
                "title": "Chicken Pasta",
                "ingredients": ["chicken", "pasta", "sauce"],
                "instructions": "Cook pasta. Add chicken and sauce.",
            },
            "r2": {
                "title": "Chocolate Cake",
                "ingredients": ["flour", "sugar", "cocoa"],
                "instructions": "Mix and bake.",
            },
        }
        reranked = rerank_results(
            query="chicken pasta",
            candidate_ids=["r1", "r2"],
            recipe_index=recipes,
            top_k=2,
        )
        # Chicken Pasta should rank higher for "chicken pasta" query
        assert reranked[0][0] == "r1"

