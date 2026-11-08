"""
build_index.py - Build the FAISS vector index from recipe data.

Run this ONCE after initial setup or when the dataset changes:
    python -m app.build_index

This generates embeddings for all recipes using the Gemini API
and saves the FAISS index to disk. Subsequent app startups will
load the pre-built index instead of regenerating embeddings.
"""
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.recipe_loader import get_recipes
from app.data_processor import build_all_search_texts
from app.semantic_retrieval import build_faiss_index


def main():
    """Build the FAISS index."""
    import argparse
    parser = argparse.ArgumentParser(description="Build FAISS vector index")
    parser.add_argument("--limit", type=int, default=1000,
                        help="Number of recipes to embed (default: 1000 for quick build, 0 for all)")
    args = parser.parse_args()

    print("=" * 60)
    print("FAISS Index Builder - Nosh Recipe Chatbot")
    print("=" * 60)
    
    # Load recipes
    print("\n[1/3] Loading recipes...")
    recipes = get_recipes()
    if args.limit and args.limit > 0:
        recipes = recipes[:args.limit]
        print(f"  Using subset of {len(recipes)} recipes for FAISS vector index")
    else:
        print(f"  Loaded {len(recipes)} recipes (full dataset)")
    
    # Build search texts
    print("\n[2/3] Building search texts...")
    search_texts = build_all_search_texts(recipes)
    recipe_ids = [r["id"] for r in recipes]
    print(f"  Built {len(search_texts)} search texts")
    
    # Build FAISS index
    print(f"\n[3/3] Building FAISS index with {len(search_texts)} recipes...")
    start = time.time()
    success = build_faiss_index(search_texts, recipe_ids)
    elapsed = time.time() - start
    
    if success:
        print(f"\n✅ FAISS index built successfully in {elapsed:.1f}s")
        print("  Index saved to vector_store/")
    else:
        print(f"\n❌ Failed to build FAISS index.")
        print("  Make sure GEMINI_API_KEY is set in your .env file.")
        print("  The chatbot will still work with BM25-only retrieval.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

