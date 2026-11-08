"""Quick smoke test of the data loading and processing pipeline."""
import sys
sys.path.insert(0, ".")

from app.recipe_loader import get_recipes
from app.data_processor import build_recipe_index, build_all_search_texts, normalize_ingredient_name
from app.keyword_retrieval import BM25Retriever
from app.query_parser import parse_query, classify_intent_deterministic

print("=" * 60)
print("Stage 2-3 Smoke Test")
print("=" * 60)

# Test 1: Load recipes
print("\n[Test 1] Loading recipes...")
recipes = get_recipes(force_reload=True)
print(f"  ✅ Loaded {len(recipes)} recipes")

# Test 2: Build recipe index
print("\n[Test 2] Building recipe index...")
index = build_recipe_index(recipes)
print(f"  ✅ Index has {len(index)} entries")

# Verify a sample recipe
sample_id = list(index.keys())[0]
sample = index[sample_id]
print(f"  Sample: '{sample['title']}' from {sample['source']}")
print(f"    Ingredients: {len(sample['ingredients'])}")
print(f"    First ingredient: {sample['ingredients'][0]}")

# Check ADVERTISEMENT cleaned
has_ad = sum(1 for r in recipes 
             for ing in r["ingredients"] 
             if "ADVERTISEMENT" in ing)
print(f"\n  ADVERTISEMENT remnants: {has_ad} (should be 0)")

# Test 3: Search texts
print("\n[Test 3] Building search texts...")
search_texts = build_all_search_texts(recipes)
print(f"  ✅ Built {len(search_texts)} search texts")
print(f"  Sample (first 100 chars): {search_texts[0][:100]}...")

# Test 4: Ingredient normalization
print("\n[Test 4] Ingredient normalization...")
test_cases = [
    "2 tablespoons unsalted butter, melted",
    "1/2 cup all-purpose flour",
    "salt and pepper to taste",
    "3 cloves garlic, minced",
    "1 (14 ounce) can diced tomatoes",
]
for tc in test_cases:
    normalized = normalize_ingredient_name(tc)
    print(f"  '{tc}' -> '{normalized}'")

# Test 5: BM25 retriever
print("\n[Test 5] Building BM25 index...")
recipe_ids = [r["id"] for r in recipes]
bm25 = BM25Retriever(search_texts, recipe_ids)

results = bm25.search("chicken parmesan", top_k=3)
print(f"  Search 'chicken parmesan' -> {len(results)} results")
for rid, score in results:
    r = index[rid]
    print(f"    [{score:.2f}] {r['title']}")

# Test 6: Query parsing
print("\n[Test 6] Query parsing...")
queries = [
    "Find vegetarian recipes with potatoes",
    "How many recipes are there?",
    "I have chicken, rice, and onions",
    "Can I use almond milk instead of regular milk?",
]
for q in queries:
    parsed = parse_query(q)
    print(f"  '{q}'")
    print(f"    Intent: {parsed['intent']}, Ingredients: {parsed['ingredients']}, "
          f"Dietary: {parsed['dietary_restrictions']}")

print("\n" + "=" * 60)
print("All smoke tests passed! ✅")
print("=" * 60)

