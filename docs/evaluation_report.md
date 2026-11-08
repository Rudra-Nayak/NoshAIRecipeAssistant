# Evaluation Report: Performance, Metrics & Latency

## 1. Evaluation Methodology
The evaluation suite (`evaluation/run_evaluation.py`) measures retrieval quality, multi-turn retention, hallucination resistance, and latency over the full 122,938 recipe database.

---

## 2. Test Scenarios & Execution Results

All 7 benchmark test cases were executed automatically against the active system:

| Test ID | Task / Capability | Query / Flow | Intent | Retrieval Latency | End-to-End Latency | Status |
|---|---|---|---|---|---|---|
| **TC-01** | Basic Recipe Search | *"Find recipes for chocolate chip cookies"* | `search` | 417.3 ms | 785.8 ms | **PASSED** |
| **TC-02 (T1)** | Multi-turn Search | *"Find vegetarian recipes containing potatoes"* | `search` | 374.8 ms | 1,567.3 ms | **PASSED** |
| **TC-02 (T2)** | Memory & Follow-up | *"Which one has the fewest steps?"* | `analytics` | 1,294.4 ms | 1,388.1 ms | **PASSED** |
| **TC-03 (T1)** | Statistical Analytics | *"How many total recipes are there?"* | `analytics` | 414.2 ms | 414.3 ms | **PASSED** |
| **TC-03 (T2)** | Dataset Aggregate | *"What are the most common ingredients?"* | `analytics` | 442.8 ms | 6,504.7 ms | **PASSED** |
| **TC-04** | Ingredient Match | *"I have eggs, flour, sugar, and butter. What can I make?"* | `ingredient_match` | 846.4 ms | 33,359.3 ms | **PASSED** |
| **TC-05** | Substitution Evidence | *"Can I use almond milk instead of regular milk?"* | `substitution` | 726.2 ms | 1,061.6 ms | **PASSED** |
| **TC-06** | No-Match Query | *"Find recipes for martian space rocks with liquid nitrogen"* | `search` | 773.8 ms | 1,413.5 ms | **PASSED** |
| **TC-07** | Hallucination Resistance | *"What is the exact recipe for Dragon Breath Soup with ID 99999999?"* | `search` | 903.2 ms | 1,785.4 ms | **PASSED** |

---

## 3. Retrieval Performance Comparison

| Metric / Dimension | Keyword-Only (BM25) | Semantic-Only (Dense FAISS) | Hybrid (BM25 + FAISS + RRF) |
|---|---|---|---|
| **Exact Dish Lookup** (*"Chicken Parmesan"*) | **High Precision**: Directly matches title tokens. | Medium: May prioritize dishes with similar ingredients. | **High Precision & Recall**: Surfaces exact title matches first while retaining conceptual variants. |
| **Descriptive Culinary Query** (*"Comfort food for cold winter night"*) | Low Recall: Query terms rarely appear verbatim in ingredients. | **High Recall**: Maps conceptual warmth/stew embeddings. | **High Recall & Robustness**: Retrieves conceptual candidates with RRF boosting. |
| **Dietary Negation** (*"Vegetarian lasagna without meat"*) | Low: Frequently surfaces meat lasagna due to high "lasagna" term score. | Poor: Semantic embedding is heavily dominated by traditional lasagna recipes. | **Guaranteed Correct**: Candidates from both retrievers pass through deterministic hard-constraint filtering. |
| **Average Query Latency** | ~400–800 ms (123K corpus) | ~20–30 ms (FAISS inner product) | ~400–800 ms |

---

## 4. Latency Summary (Measured on 122,938 Recipes)

- **Average Retrieval Latency**: **688.12 ms**
- **Average End-to-End Turn Latency (standard queries)**: **1,250 ms**
- **Full Corpus Statistical Exhaustion (TC-04 single-pass)**: ~33 s
- **Unit Test Execution (80 Tests)**: **26.25 s** (100% Pass)

