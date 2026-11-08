"""
main.py - Streamlit chatbot interface for the Nosh Recipe Assistant.

This is the UI entry point. Run with:
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
    page_title="Nosh Recipe Assistant",
    page_icon="🍳",
    layout="centered",
)


@st.cache_resource(show_spinner="Loading recipe database...")
def initialize_system():
    """Initialize all components once and cache them.
    
    This runs only on the first request. Subsequent requests
    reuse the cached objects.
    
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
    
    # Try to load FAISS retriever (may not be available)
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
    
    # --- Header ---
    st.title("🍳 Nosh Recipe Assistant")
    st.caption("Search recipes, get cooking analytics, match ingredients, and find substitutions.")
    
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
    
    # --- Sidebar ---
    with st.sidebar:
        st.header("ℹ️ System Info")
        st.metric("Total Recipes", f"{len(all_recipes):,}")
        
        faiss_status = "✅ Active" if faiss_retriever else "⚠️ BM25 Only"
        st.text(f"Retrieval: {faiss_status}")
        
        from app.llm import is_available
        llm_status = "✅ Connected" if is_available() else "⚠️ Offline (template mode)"
        st.text(f"Gemini LLM: {llm_status}")
        
        st.divider()
        
        st.markdown("**Try asking:**")
        st.markdown("- Find chicken pasta recipes")
        st.markdown("- How many recipes are there?")
        st.markdown("- I have eggs, flour, sugar, and butter")
        st.markdown("- Can I use almond milk instead of regular milk?")
        st.markdown("- Which one has the fewest steps?")
        
        st.divider()
        
        if st.button("🗑️ Clear Conversation"):
            memory = get_memory()
            memory.clear()
            st.session_state.messages = []
            st.rerun()
    
    # --- Chat Interface ---
    memory = get_memory()
    
    # Initialize message history for display
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Welcome message
        welcome = (
            "Hello! I'm the Nosh Recipe Assistant. I can help you:\n\n"
            "🔍 **Search** for recipes by name or ingredients\n"
            "📊 **Analyze** recipe statistics and trends\n"
            "🥘 **Match** recipes to ingredients you have\n"
            "🔄 **Suggest** ingredient substitutions\n\n"
            "What would you like to cook today?"
        )
        st.session_state.messages.append({"role": "assistant", "content": welcome})
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("recipes"):
                with st.expander(f"📖 Inspect {len(message['recipes'])} Recipe Details (IDs, Ingredients & Steps)"):
                    for r in message["recipes"]:
                        st.markdown(f"#### {r['title']}")
                        st.caption(f"Stable ID: `{r['id']}` | Source: `{r.get('source', 'N/A').upper()}`")
                        st.markdown("**Ingredients:**")
                        for ing in r.get('ingredients', []):
                            st.markdown(f"- {ing}")
                        st.markdown("**Instructions:**")
                        st.text(r.get('instructions', ''))
                        st.divider()
    
    # --- User Input ---
    if user_input := st.chat_input("Ask about recipes..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                from app.router import route_query
                
                start_time = time.time()
                response = route_query(
                    query=user_input,
                    memory=memory,
                    bm25_retriever=bm25_retriever,
                    faiss_retriever=faiss_retriever,
                    recipe_index=recipe_index,
                    all_recipes=all_recipes,
                )
                elapsed = time.time() - start_time
            
            st.markdown(response)
            
            # Show recipe cards if available for this turn
            current_turn_recipes = list(memory.last_recipes[:5]) if memory.last_recipes else []
            if current_turn_recipes and memory.last_intent in ("search", "ingredient_match", "followup"):
                with st.expander(f"📖 Inspect {len(current_turn_recipes)} Recipe Details (IDs, Ingredients & Steps)"):
                    for r in current_turn_recipes:
                        st.markdown(f"#### {r['title']}")
                        st.caption(f"Stable ID: `{r['id']}` | Source: `{r.get('source', 'N/A').upper()}`")
                        st.markdown("**Ingredients:**")
                        for ing in r.get('ingredients', []):
                            st.markdown(f"- {ing}")
                        st.markdown("**Instructions:**")
                        st.text(r.get('instructions', ''))
                        st.divider()
            
            st.caption(f"⏱️ Response time: {elapsed:.2f}s")
        
        msg_payload = {"role": "assistant", "content": response}
        if current_turn_recipes and memory.last_intent in ("search", "ingredient_match", "followup"):
            msg_payload["recipes"] = current_turn_recipes
        st.session_state.messages.append(msg_payload)


if __name__ == "__main__":
    main()

