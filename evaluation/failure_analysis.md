# Failure Analysis & Edge Case Report

## 1. Overview
This failure analysis identifies edge cases, observed failures, and boundary conditions encountered during system testing across all 122,938 recipes, detailing the root causes and mitigations implemented.

---

## 2. Documented Failure Modes & Mitigations

### 2.1 Failure Mode 1: Regex Unit Extraction Colliding with Ingredient Names
- **Observed Behavior**: The normalization of `"garlic"` resulted in `"arlic"`.
- **Root Cause**: In `data_processor.py`, the unit regex contained `g` (abbreviation for grams) without word boundary anchors:
  ```python
  # Problematic:
  r'(?:...|grams?|g|kilograms?|kg)\s*'
  ```
  The single character `g` matched the first letter of any word beginning with `g`.
- **Remediation**: Added word boundaries `\bg\b` and `\bkg\b` to prevent partial prefix matching on common ingredients like garlic, ginger, or gelatin.
- **Verification**: Validated by unit test `test_analytics.py::TestMostCommonIngredients::test_basic` (Passed).

### 2.2 Failure Mode 2: Monolithic Instruction Prose without Newlines
- **Observed Behavior**: In the Food Network (`fn`) subset, recipe instruction strings are formatted as continuous text paragraphs without newline characters (`\n`). Simple line-splitting yielded a step count of 1 for complex 500-word recipes.
- **Root Cause**: Inconsistent dataset formatting across the three web source providers.
- **Remediation**: Implemented an intelligent sentence boundary fallback regex `re.split(r'\.(?:\s+)(?=[A-Z])', instructions)` in `data_processor.py: count_instruction_steps`.
- **Verification**: Validated by `test_data_loader.py::TestCountInstructionSteps`.

### 2.3 Failure Mode 3: Semantic Overriding of Dietary Restrictions
- **Observed Behavior**: In vector similarity search, querying *"vegetarian bolognese"* retrieves traditional beef/pork bolognese recipes with high cosine similarity because the semantic topic remains "bolognese".
- **Root Cause**: Dense embeddings cluster items by semantic topic rather than logical negation or restriction constraints.
- **Remediation**: Decoupled constraint enforcement from similarity scoring. The system strictly applies deterministic Python filtering (`recipe_search.py: filter_by_dietary`) against original recipe ingredients, discarding non-compliant candidates before final reranking.

### 2.4 Failure Mode 4: Ingredient Matching Scanning Latency over 123K Items
- **Observed Behavior**: Full exhaustive scanning of 122,938 recipes with nested string matching for ingredient coverage in TC-04 took ~33 seconds on CPU.
- **Remediation**:
  1. Implemented early score thresholding (`min_match_score = 0.2`) to exit recipe loops early.
  2. For interactive multi-turn sessions, the search candidate retrieval narrows candidate bounds before applying exhaustive ingredient coverage breakdown.

### 2.5 Failure Mode 5: Ungrounded Numerical Queries (Hallucination Risk)
- **Observed Behavior**: General LLMs asked *"What is the median ingredient count in AllRecipes?"* generate plausible-sounding but fictitious figures (e.g. 7 or 8.5).
- **Remediation**: The system classifies statistical queries into `INTENT_ANALYTICS`, routing them to `app/analytics.py` which computes exact metrics over the full dataset (Average: 9.14, Median: 9). The LLM is supplied with precomputed results and instructed never to fabricate figures.

