"""
main.py - Streamlit chatbot interface for the Nosh Recipe Assistant.

An intelligent, culturally-conscious conversational recipe assistant
featuring hybrid retrieval (BM25 + FAISS), India-aware cultural guardrails,
and grounded, beginner-friendly step-by-step cooking guides.

Run with:
    streamlit run app/main.py
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path when launched via 'streamlit run app/main.py'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Nosh Recipe Assistant | Culinary AI",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Culinary Styling ---
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #0E1117;
    }
    
    /* Header styling */
    .culinary-header {
        background: linear-gradient(135deg, #1E232A 0%, #2A1B14 100%);
        border: 1px solid #3E2723;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .culinary-title {
        color: #FFA726;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .culinary-subtitle {
        color: #CFD8DC;
        font-size: 1.0rem;
        line-height: 1.5;
        margin: 0;
    }
    
    /* Region & Dietary Badge Tags */
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        margin-top: 8px;
    }
    .badge-region {
        background-color: #E65100;
        color: white;
    }
    .badge-diet {
        background-color: #2E7D32;
        color: white;
    }
    
    /* Recipe Card Styling */
    .recipe-card-box {
        background-color: #1A1F26;
        border-left: 4px solid #FF9800;
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
    }
    
    /* Prompt suggestion chips */
    .stButton>button.suggestion-chip {
        background-color: #262C36;
        color: #ECEFF1;
        border: 1px solid #37474F;
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 0.88rem;
        transition: all 0.2s;
    }
    .stButton>button.suggestion-chip:hover {
        background-color: #E65100;
        border-color: #FF9800;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="🍳 Warming up the kitchen and loading recipe database...")
def initialize_system():
    """Initialize all components once and cache them.
    
    Returns:
        Tuple of (all_recipes, recipe_index, bm25_retriever, faiss_retriever)
    """
    from app.recipe_loader import get_recipes
    from app.data_processor import build_recipe_index, build_all_search_texts
    from app.keyword_retrieval import BM25Retriever
    from app.semantic_retrieval import FAISSRetriever
    
    # Load recipes (from cache or raw data)
    all_recipes = get_recipes()
    
    # Build recipe index for O(1) lookup
    recipe_index = build_recipe_index(all_recipes)
    
    # Build search texts
    search_texts = build_all_search_texts(all_recipes)
    recipe_ids = [r["id"] for r in all_recipes]
    
    # Initialize BM25 retriever
    bm25_retriever = BM25Retriever(search_texts, recipe_ids)
    
    # Try to load FAISS retriever
    faiss_retriever = FAISSRetriever.from_disk()
    if faiss_retriever is None:
        print("  NOTE: FAISS index not found. Running with BM25 only.")
        print("  Run 'python -m app.build_index' to build the FAISS index.")
    
    return all_recipes, recipe_index, bm25_retriever, faiss_retriever


def get_memory():
    """Get or create session-level conversation memory."""
    from app.memory import ConversationMemory
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory()
    return st.session_state.memory


def main():
    """Main Streamlit application."""
    
    # --- Initialize system ---
    try:
        all_recipes, recipe_index, bm25_retriever, faiss_retriever = initialize_system()
    except FileNotFoundError as e:
        st.error(f"❌ Dataset not found: {e}")
        st.info("Please place the recipe JSON files in the `recipes_raw/` directory.")
        return
    except Exception as e:
        st.error(f"❌ Failed to initialize: {e}")
        return
    
    memory = get_memory()
    
    # --- Sidebar: Kitchen Controls & Cultural Settings ---
    with st.sidebar:
        st.markdown("### 👨‍🍳 Kitchen Control Panel")
        
        # 1. Regional Setting
        st.markdown("#### 🌍 Regional Preference")
        selected_region = st.selectbox(
            "Select Target Region:",
            ["India (Default 🇮🇳)", "Global 🌐"],
            index=0,
            help="In India mode, beef is excluded from default searches and casual recommendations unless explicitly requested."
        )
        region_clean = "India" if "India" in selected_region else "Global"
        memory.set_region(region_clean)
        
        if region_clean == "India":
            st.caption("🛡️ *Culturally-conscious: Beef excluded from default discovery.*")
        
        st.divider()
        
        # 2. Dietary Preferences (Multi-Select)
        st.markdown("#### 🥗 Dietary Preferences")
        dietary_options = [
            "Vegetarian",
            "Jain Vegetarian (No Onion/Garlic/Roots)",
            "Vegan (Plant-Based)",
            "No Beef",
            "No Pork",
            "No Seafood",
            "No Eggs",
        ]
        selected_diets = st.multiselect(
            "Active Dietary Filters:",
            dietary_options,
            default=[],
            help="Strict deterministic filtering is applied at query time."
        )
        
        # Map to internal labels
        diet_mapping = {
            "Vegetarian": "vegetarian",
            "Jain Vegetarian (No Onion/Garlic/Roots)": "jain_vegetarian",
            "Vegan (Plant-Based)": "vegan",
            "No Beef": "no_beef",
            "No Pork": "no_pork",
            "No Seafood": "no_seafood",
            "No Eggs": "no_eggs",
        }
        active_diet_labels = [diet_mapping[d] for d in selected_diets if d in diet_mapping]
        memory.dietary_restrictions = active_diet_labels
        
        # 3. Custom Excluded Ingredients
        st.markdown("#### 🚫 Custom Excluded Ingredients")
        custom_excluded_input = st.text_input(
            "Allergies or ingredients you don't have:",
            placeholder="e.g. coconut milk, peanuts, mushrooms",
            help="Comma-separated items that must be excluded from recipe results."
        )
        if custom_excluded_input:
            items = [i.strip().lower() for i in custom_excluded_input.split(",") if i.strip()]
            for item in items:
                memory.add_excluded_ingredient(item)
                
        st.divider()
        
        # 4. System Status Metrics
        st.markdown("#### ⚙️ Engine Diagnostics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Recipes", f"{len(all_recipes):,}")
        with col2:
            st.metric("Region", region_clean)
            
        faiss_status = "✅ Hybrid (BM25 + FAISS)" if faiss_retriever else "⚠️ BM25 Lexical"
        st.caption(f"**Retrieval:** {faiss_status}")
        
        from app.llm import is_available
        llm_status = "✅ Connected (Gemini 3.5)" if is_available() else "⚠️ Template Mode (Offline)"
        st.caption(f"**Culinary LLM:** {llm_status}")
        
        st.divider()
        
        if st.button("🗑️ Reset Kitchen Session", use_container_width=True):
            memory.clear()
            st.session_state.messages = []
            st.rerun()

    # --- Main Chat Area Header ---
    active_badges_html = f"<span class='badge-pill badge-region'>Region: {region_clean}</span>"
    for d in selected_diets:
        short_name = d.split(" (")[0]
        active_badges_html += f"<span class='badge-pill badge-diet'>{short_name}</span>"
    if memory.excluded_ingredients:
        active_badges_html += f"<span class='badge-pill' style='background:#C62828; color:white;'>Excluded: {', '.join(memory.excluded_ingredients[:3])}</span>"

    st.markdown(f"""
    <div class="culinary-header">
        <div class="culinary-title">🍳 Nosh Recipe Assistant</div>
        <p class="culinary-subtitle">
            Grounded Recipe Discovery, Zero-Hallucination Analytics, Pantry Matching & Step-by-Step Cooking Guidance.
        </p>
        <div>{active_badges_html}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        welcome = (
            "Namaste & Welcome! 🙏 I'm your **Nosh Recipe Assistant**.\n\n"
            "I'm here to help you discover authentic recipes, calculate kitchen statistics, "
            "cook with available pantry ingredients, find safe substitutions, and guide you **step-by-step** through any dish.\n\n"
            "What would you like to prepare today?"
        )
        st.session_state.messages.append({"role": "assistant", "content": welcome})
    
    # Display message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑‍🍳" if message["role"] == "assistant" else "👤"):
            st.markdown(message["content"])
            
            # If message contains recipe inspection cards
            if message.get("recipes"):
                with st.expander(f"📖 Inspect {len(message['recipes'])} Recipe Cards (IDs, Ingredients & Original Steps)"):
                    for r in message["recipes"]:
                        st.markdown(f"#### 🍲 {r['title']}")
                        st.caption(f"**Stable ID**: `{r['id']}` | **Source**: `{r.get('source', 'N/A').upper()}`")
                        
                        col_a, col_b = st.columns([1, 1])
                        with col_a:
                            st.markdown("**🛒 Ingredients Required:**")
                            for ing in r.get('ingredients', []):
                                st.markdown(f"- {ing}")
                        with col_b:
                            st.markdown("**📝 Original Instructions:**")
                            st.text(r.get('instructions', ''))
                        st.divider()

    # --- Quick Inspiration Chips (When conversation is short) ---
    if len(st.session_state.messages) <= 2:
        st.markdown("**💡 Quick Inspiration:**")
        chip_cols = st.columns(4)
        quick_prompts = [
            ("🍚 Rice Kheer for Diwali", "How do I make rice kheer for Diwali step by step?"),
            ("🥘 Pantry Pasta Cooking", "I have pasta, butter, garlic, and parmesan. What can I make?"),
            ("🔄 Jaggery for Sugar", "Can I replace sugar with jaggery in dessert recipes?"),
            ("📊 Recipe Statistics", "How many total recipes are there, and what are the top ingredients?"),
        ]
        
        for idx, (label, prompt_text) in enumerate(quick_prompts):
            with chip_cols[idx]:
                if st.button(label, key=f"chip_{idx}", use_container_width=True):
                    # Inject as user input
                    st.session_state.injected_prompt = prompt_text
                    st.rerun()

    # --- User Input & Processing ---
    prompt_to_process = None
    if "injected_prompt" in st.session_state and st.session_state.injected_prompt:
        prompt_to_process = st.session_state.injected_prompt
        st.session_state.injected_prompt = None
    elif user_input := st.chat_input("Ask for a recipe, cooking guidance, substitutions, or pantry matching..."):
        prompt_to_process = user_input

    if prompt_to_process:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt_to_process})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt_to_process)
        
        # Generate response
        with st.chat_message("assistant", avatar="🧑‍🍳"):
            with st.spinner("🍳 Checking ingredients, cultural guidelines, and preparing cooking guidance..."):
                from app.router import route_query
                
                start_time = time.time()
                response = route_query(
                    query=prompt_to_process,
                    memory=memory,
                    bm25_retriever=bm25_retriever,
                    faiss_retriever=faiss_retriever,
                    recipe_index=recipe_index,
                    all_recipes=all_recipes,
                )
                elapsed = time.time() - start_time
            
            st.markdown(response)
            
            # Show recipe card expander if search or matching intent returned recipes
            current_turn_recipes = list(memory.last_recipes[:5]) if memory.last_recipes else []
            if current_turn_recipes and memory.last_intent in ("search", "ingredient_match", "followup", "contextual_modification"):
                with st.expander(f"📖 Inspect {len(current_turn_recipes)} Recipe Cards (IDs, Ingredients & Original Steps)"):
                    for r in current_turn_recipes:
                        st.markdown(f"#### 🍲 {r['title']}")
                        st.caption(f"**Stable ID**: `{r['id']}` | **Source**: `{r.get('source', 'N/A').upper()}`")
                        
                        col_a, col_b = st.columns([1, 1])
                        with col_a:
                            st.markdown("**🛒 Ingredients Required:**")
                            for ing in r.get('ingredients', []):
                                st.markdown(f"- {ing}")
                        with col_b:
                            st.markdown("**📝 Original Instructions:**")
                            st.text(r.get('instructions', ''))
                        st.divider()
            
            st.caption(f"⏱️ Response generated in {elapsed:.2f}s | Region: **{memory.region}**")
        
        msg_payload = {"role": "assistant", "content": response}
        if current_turn_recipes and memory.last_intent in ("search", "ingredient_match", "followup", "contextual_modification"):
            msg_payload["recipes"] = current_turn_recipes
        st.session_state.messages.append(msg_payload)


if __name__ == "__main__":
    main()
