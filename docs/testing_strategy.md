# Testing Strategy & Quality Assurance Report

## 1. Multi-Tier Testing Pyramid
The testing architecture guarantees reproducibility, constraint enforcement, and zero regression across both deterministic logic and LLM orchestration.

```
       /\
      /  \      Conversation & Integration Tests (evaluation/run_evaluation.py)
     /----\     7 Multi-Turn Benchmark Scenarios across all 5 user intents
    /      \
   /--------\   Functional Integration Tests (tests/test_retrieval.py, test_analytics.py)
  /          \  Retrieval ranking, RRF correctness, step counting, analytics
 /------------\ Unit Tests (tests/test_data_loader.py, test_ingredient_match.py, test_memory.py)
/--------------\ 80 automated unit tests executing in <30 seconds via pytest
```

---

## 2. Test Suites Overview

### 2.1 Unit Tests (80 Tests in `tests/`)
- **`test_data_loader.py` (24 tests)**: Verifies ADVERTISEMENT suffix stripping, standalone ad elimination, null title/instruction rejection, and pickle persistence integrity.
- **`test_retrieval.py` (17 tests)**: Verifies BM25 tokenization, exact multi-word querying, edge cases (empty strings, zero matches), RRF score monotonicity, and feature reranking.
- **`test_analytics.py` (13 tests)**: Verifies full-dataset and filtered counts, source distribution, min/max/median step calculations, and ingredient frequency counters.
- **`test_ingredient_match.py` (12 tests)**: Verifies transparent equivalent matching (e.g. *unsalted butter $\leftrightarrow$ butter*), Jaccard coverage calculations, missing ingredient lists, and dietary constraint drop-outs.
- **`test_substitution.py` (7 tests)**: Tests dual-occurrence detection, indirect co-occurrence discovery, and modified instruction generation.
- **`test_memory.py` (7 tests)**: Verifies session state accumulation, multi-turn filter persistence, and clean reset mechanics.

**Pytest Execution Command**:
```powershell
.\.venv\Scripts\pytest.exe tests/ -v
```
**Current Status**: 80 passed / 0 failed (100% pass rate).

---

## 3. Groundedness & Anti-Hallucination Testing
Specific automated test cases ensure that the system does not invent information:
1. **Fictitious ID Queries**: When queried for synthetic IDs (e.g. `99999999`), the system returns a verified no-match message rather than fabricating a dish.
2. **Deterministic Statistics**: Asking *"How many recipes are there?"* always triggers deterministic dataset calculation ($122,938$) rather than stochastic LLM generation.
3. **Hard Constraint Enforcement**: Vegan and vegetarian queries are strictly checked against non-compliant ingredients; raw vector similarity cannot override this filter.

