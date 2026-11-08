# Internal API Design & Module Interface Specification

## 1. Module Hierarchy
The application is structured into decoupled, single-responsibility Python modules under `app/`:

| Module | Core Responsibility | Key Exported Functions |
|---|---|---|
| `recipe_loader.py` | Ingestion, validation, noise removal, and disk caching | `get_recipes()`, `clean_recipe()`, `clean_ingredient()` |
| `data_processor.py` | Text assembly, normalization, step parsing | `build_recipe_index()`, `build_all_search_texts()`, `normalize_ingredient_name()`, `count_instruction_steps()` |
| `keyword_retrieval.py` | Tokenization and BM25 index operations | `BM25Retriever.search()` |
| `semantic_retrieval.py` | Embedding generation and FAISS vector index | `FAISSRetriever.search()`, `build_faiss_index()` |
| `hybrid_retrieval.py` | RRF ranking and candidate deduplication | `reciprocal_rank_fusion()`, `hybrid_search()` |
| `reranker.py` | Feature-based scoring and ordering | `compute_rerank_score()`, `rerank_results()` |
| `recipe_search.py` | Orchestrates search and hard constraint enforcement | `search_recipes()`, `filter_by_dietary()`, `filter_by_ingredients()` |
| `analytics.py` | Dataset-wide deterministic computations | `compute_analytics()`, `total_recipe_count()`, `most_common_ingredients()` |
| `ingredient_match.py` | Ingredient overlap scoring and coverage breakdown | `find_matching_recipes()`, `compute_match_score()` |
| `substitution.py` | Evidence-based substitution assessment | `find_substitution_evidence()`, `generate_substitution_response()` |
| `memory.py` | Session context state management | `ConversationMemory` dataclass |
| `llm.py` | Grounded generation with Gemini API & templates | `generate_response()`, `parse_query_with_llm()`, `is_available()` |
| `router.py` | Master request dispatcher | `route_query()` |

---

## 2. Key Data Contracts & Signatures

### 2.1 Search Function Contract (`app/recipe_search.py`)
```python
def search_recipes(
    query: str,
    bm25_retriever: BM25Retriever,
    faiss_retriever: Optional[FAISSRetriever],
    recipe_index: dict[str, dict],
    parsed_query: Optional[dict] = None,
    top_k: int = 5,
) -> dict:
    """Returns:
    {
        "recipes": list[dict],            # Full verified recipe dictionaries
        "recipe_ids": list[str],          # Ordered list of stable IDs
        "query_info": dict,               # Extracted intents and constraints
        "total_candidates": int,          # Count pre-filtering
        "total_after_filter": int         # Count post-constraint verification
    }
    """
```

### 2.2 Ingredient Match Contract (`app/ingredient_match.py`)
```python
def find_matching_recipes(
    user_ingredients: list[str],
    recipes: list[dict],
    dietary_restrictions: Optional[list[str]] = None,
    excluded_ingredients: Optional[list[str]] = None,
    min_match_score: float = 0.3,
    top_n: int = 10,
) -> list[dict]:
    """Returns list of elements:
    {
        "recipe": dict,
        "match": {
            "score": float,               # Jaccard coverage [0.0 - 1.0]
            "matched": list[dict],        # Matched ingredient pairs
            "missing": list[str],         # Ingredients required but not owned
            "extra": list[str],           # Owned ingredients not needed
            "matched_count": int,
            "missing_count": int
        }
    }
    """
```

### 2.3 Substitution Evidence Contract (`app/substitution.py`)
```python
def find_substitution_evidence(
    original_ingredient: str,
    substitute_ingredient: str,
    recipes: list[dict],
    recipe_index: dict[str, dict],
) -> dict:
    """Returns:
    {
        "original": str,
        "substitute": str,
        "direct_evidence": list[str],     # Recipe IDs containing BOTH ingredients
        "indirect_evidence": list[dict],  # Co-occurring recipe titles
        "confidence": "high" | "moderate" | "low" | "none",
        "explanation": str,
        "warnings": list[str]
    }
    """
```

