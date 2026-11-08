"""
keyword_retrieval.py - BM25-based keyword retrieval.

BM25 (Best Matching 25) is a bag-of-words retrieval function that ranks
documents based on query term frequency, document length, and corpus statistics.

Why BM25?
- Fast, no GPU needed
- Great for exact keyword matches ("chicken parmesan")
- Complements semantic search which may miss exact terms
"""
from rank_bm25 import BM25Okapi
from typing import Optional


# Conversational and general stop words that dilute specific culinary keywords in BM25
SEARCH_STOP_WORDS = {
    "how", "do", "i", "can", "you", "we", "me", "make", "cook", "prepare", "get", "give", "show",
    "find", "search", "recipe", "recipes", "step", "by", "steps", "a", "an", "the", "in", "on", "at",
    "to", "for", "of", "with", "from", "and", "or", "is", "are", "was", "were", "it", "this", "that",
    "these", "those", "my", "your", "our", "their", "please", "help", "guide", "explain", "instructions",
    "traditional", "authentic", "classic", "diwali"
}


def tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer.
    
    We keep it simple here. More sophisticated tokenization (stemming,
    lemmatization) could improve recall but adds complexity.
    
    Args:
        text: Input text string.
        
    Returns:
        List of lowercase tokens.
    """
    # Split on non-alphanumeric characters, lowercase, filter empty
    import re
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    return tokens


class BM25Retriever:
    """BM25-based keyword retrieval over recipe search texts.
    
    Built once at startup, then queried repeatedly.
    """
    
    def __init__(self, search_texts: list[str], recipe_ids: list[str]):
        """Initialize BM25 index.
        
        Args:
            search_texts: List of search text strings (one per recipe).
            recipe_ids: List of recipe IDs, index-aligned with search_texts.
        """
        assert len(search_texts) == len(recipe_ids), \
            "search_texts and recipe_ids must be the same length"
        
        self.recipe_ids = recipe_ids
        
        # Tokenize all documents
        print("  Building BM25 index...")
        self.tokenized_corpus = [tokenize(text) for text in search_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"  BM25 index built with {len(recipe_ids)} documents")
    
    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Search for recipes matching the query.
        
        Args:
            query: User search query.
            top_k: Number of top results to return.
            
        Returns:
            List of (recipe_id, bm25_score) tuples, sorted by score descending.
        """
        query_tokens = tokenize(query)
        
        if not query_tokens:
            return []
        
        # Filter common English conversational search stop words that dilute specific culinary keywords
        filtered_tokens = [t for t in query_tokens if t not in SEARCH_STOP_WORDS]
        if filtered_tokens:
            query_tokens = filtered_tokens
        
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices sorted by score descending
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        # Return only results with positive scores
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.recipe_ids[idx], float(scores[idx])))
        
        return results

