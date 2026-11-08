# User Flow & Interaction Specification

## 1. Overview of Interaction Flows
The Nosh Recipe Assistant supports 7 distinct interaction pathways:
1. **First-Time User Initialization Flow**
2. **Recipe Search & Factual Q&A Flow**
3. **Statistical Analytics Flow**
4. **Ingredient-Based Matching Flow**
5. **Evidence-Based Substitution Flow**
6. **Multi-Turn Context & Follow-Up Flow**
7. **No-Match & Graceful Recovery Flow**

---

## 2. Interaction Flowcharts

### 2.1 Complete Conversational Lifecycle

```mermaid
flowchart TD
    Start([User Opens Web App]) --> Init[Initialize Session & Load Cached Dataset]
    Init --> DisplayUI[Render Streamlit Interface & Metrics Sidebar]
    DisplayUI --> InputWait[/User Submits Message/]

    InputWait --> IntentClassify{Determine Intent & Extract Constraints}
    
    %% Branches
    IntentClassify -->|Recipe Search| FlowSearch[Retrieve Candidates via BM25 + FAISS]
    IntentClassify -->|Dataset Analytics| FlowAnalytics[Compute Over Full 122K Dataset]
    IntentClassify -->|I Have X, Y, Z| FlowMatch[Score Recipes by Ingredient Coverage]
    IntentClassify -->|Substitute A for B| FlowSub[Analyze Co-Occurrence Evidence]
    IntentClassify -->|Follow-up Question| FlowFollowup[Apply Memory Filter to Prior Results]

    %% Search Path
    FlowSearch --> HardConstraints{Hard Constraints Satisfied?}
    HardConstraints -->|No Candidates| FlowNoMatch[Display Informative No-Match Guidance]
    HardConstraints -->|Valid Matches| Rerank[Feature Reranker & Fetch Recipe IDs]
    Rerank --> GenResponse[Ground LLM on Retrieved Recipes]

    %% Analytics Path
    FlowAnalytics --> GenAnalytics[Format Precise Statistics into Natural Prose]

    %% Matching Path
    FlowMatch --> GenMatch[Present Match %, Missing Ingredients, & Instructions]

    %% Sub Path
    FlowSub --> GenSub[State Evidence Strength & Cooking Implications]

    %% Follow-up Path
    FlowFollowup --> GenFollowup[Operate Contextually on Active Recipe Subset]

    %% Convergence
    GenResponse --> UpdateMem[Update Session Conversation Memory]
    GenAnalytics --> UpdateMem
    GenMatch --> UpdateMem
    GenSub --> UpdateMem
    GenFollowup --> UpdateMem
    FlowNoMatch --> UpdateMem

    UpdateMem --> RenderOutput[/Render Assistant Response in Chat/]
    RenderOutput --> InputWait
```

---

### 2.2 Multi-Turn Context Follow-Up Flow

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Streamlit as Streamlit UI
    participant Router as Router & Parser
    participant Memory as Session Memory
    participant Engine as Search & Analytics Engine
    participant LLM as Grounded LLM

    User->>Streamlit: "Find vegetarian recipes containing potatoes"
    Streamlit->>Router: route_query(query, memory)
    Router->>Engine: search_recipes(dietary=['vegetarian'], ingredient='potato')
    Engine-->>Router: [Recipe A (ID: 01), Recipe B (ID: 02), Recipe C (ID: 03)]
    Router->>Memory: save(ids=[01, 02, 03], filters={dietary: vegetarian})
    Router->>LLM: generate_response(candidates=[A, B, C])
    LLM-->>Streamlit: "Here are 3 vegetarian potato recipes..."
    Streamlit-->>User: Displays recipes with IDs and step summaries

    User->>Streamlit: "Which one has the fewest steps?"
    Streamlit->>Router: route_query("Which one has the fewest steps?", memory)
    Router->>Memory: get_context_for_followup()
    Memory-->>Router: active_recipes=[01, 02, 03]
    Router->>Engine: count_instruction_steps for [01, 02, 03]
    Engine-->>Router: Recipe B has 3 steps (fewest)
    Router->>LLM: format answer referencing Recipe B (ID: 02)
    LLM-->>Streamlit: "Among the 3 vegetarian recipes, Recipe B has only 3 steps..."
    Streamlit-->>User: Grounded follow-up answer displayed
```

---

## 3. Failure & Edge Case Handling

### 3.1 Zero-Match Scenarios
When a query produces no results (e.g. impossible ingredient combination or contradictory constraints):
- **Never fabricate** synthetic recipes.
- Clarify the active constraint that caused zero matches (e.g., *"No recipes found combining salmon with vegan constraint"*).
- Suggest actionable relaxation paths (e.g., broaden ingredient keywords or remove dietary filters).

### 3.2 Gemini API Unavailability / Rate Limits
If the external Gemini API is unreachable or fails:
- System transitions to **Deterministic Template Fallback** (`app/llm.py: _fallback_response`).
- Exact recipe IDs, ingredient lists, instructions, and statistical results are rendered via deterministic Markdown formatting without degradation of functional correctness.

