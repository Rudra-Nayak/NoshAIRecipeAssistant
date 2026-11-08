# System Limitations & Future Roadmap

## 1. Known Dataset Limitations

### 1.1 Unstructured Ingredient Strings
- **Observation**: The raw Eight Portions dataset stores ingredients as flat strings (e.g. `"2 (10.75 ounce) cans condensed cream of chicken soup"`).
- **Current Mitigation**: Heuristic regex-based quantity and modifier stripping.
- **Limitation**: Complex compound ingredients (e.g., *"1 cup chopped cooked chicken or turkey"*) may fail to isolate both meat types cleanly without a dedicated NER parser.

### 1.2 Lack of Explicit Dietary Metadata
- **Observation**: The dataset contains zero structured labels for cuisines, allergens, preparation times, or dietary categories (e.g., vegetarian, keto, halal).
- **Current Mitigation**: Deterministic negative-keyword exclusions (e.g. checking for absence of poultry, red meat, and seafood for vegetarian).
- **Limitation**: Subtle non-vegetarian components (e.g. gelatin, animal rennet in cheese, oyster sauce) might escape keyword matching if not explicitly named.

### 1.3 Instruction Format Variance
- **Observation**: Food Network (`fn`) instructions are predominantly monolithic single-paragraph blocks with zero newline markers, unlike AllRecipes and Epicurious.
- **Current Mitigation**: Sentence-boundary punctuation heuristics are used to approximate step counts.

---

## 2. Technical Limitations & Edge Cases

### 2.1 Embedding Generation at Full Scale
- Building dense embeddings for all 122,938 recipes through public Gemini API endpoints requires API quota allowances and time.
- **Mitigation**: The system is fully functional in BM25-only mode out of the box and seamlessly activates FAISS vector search when an index file is present.

### 2.2 Cold Start
- Initial indexing of BM25 takes approximately 10–15 seconds at application launch for 123K recipes.
- In Streamlit, this is mitigated via `@st.cache_resource`, ensuring it runs only once per server lifetime.

---

## 3. Future Roadmap for Nosh Robotics

1. **Structured Recipe Parsing Model**: Fine-tune a lightweight BERT NER model specifically on culinary ingredient strings to extract `(quantity, unit, food_name, preparation_state)`.
2. **Robotic Action Sequencing**: Deconstruct instructions into an Action-Object sequence (e.g., `Chop(onion, dice)`, `Heat(pan, medium)`, `Add(butter, pan)`) for robotic execution.
3. **Pantry Inventory Integration**: Integrate with smart kitchen camera feeds or barcode scanners to update available user ingredients automatically.

