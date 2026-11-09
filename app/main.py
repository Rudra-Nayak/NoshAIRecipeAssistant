"""
main.py - Minimalist, ChatGPT-inspired UI for Nosh Recipe Assistant.

Featuring:
- Clean ChatGPT aesthetic with Light / Dark mode toggle
- Cultural and dietary filters (India-aware default, Jain, Vegan, Vegetarian, etc.)
- Zero irrelevant developer/engine diagnostics on the frontend
- Fast, intuitive conversational flow and step-by-step cooking guides
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path when launched via 'streamlit run app/main.py'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import platform
import asyncio

if platform.system() == "Windows":
    # Prevent Windows ProactorEventLoop WinError 10054 noise when browser refreshes or closes
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="Nosh Recipe Assistant",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Initialize Theme State ---
if "theme" not in st.session_state:
    st.session_state.theme = "dark"


def get_chatgpt_css(theme: str) -> str:
    """Return ChatGPT-matching CSS styling for dark or light theme."""
    is_dark = theme == "dark"
    
    bg_main = "#212121" if is_dark else "#FFFFFF"
    bg_sidebar = "#171717" if is_dark else "#F9F9F9"
    text_primary = "#ECECEC" if is_dark else "#0D0D0D"
    text_secondary = "#B4B4B4" if is_dark else "#5D5D5D"
    card_bg = "#2F2F2F" if is_dark else "#F4F4F4"
    card_border = "#383838" if is_dark else "#E5E5E5"
    card_hover = "#3A3A3A" if is_dark else "#EAEAEA"
    input_bg = "#2F2F2F" if is_dark else "#FFFFFF"
    input_border = "#424242" if is_dark else "#D1D5DB"
    user_bubble = "#2F2F2F" if is_dark else "#F4F4F4"
    accent = "#10A37F" if is_dark else "#10A37F"  # Subtle ChatGPT green accent

    return f"""
    <style>
        /* Base page reset */
        .stApp {{
            background-color: {bg_main} !important;
            color: {text_primary} !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }}

        /* Top header / toolbar cleanup */
        header[data-testid="stHeader"] {{
            background-color: transparent !important;
        }}
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {{
            background-color: {bg_sidebar} !important;
            border-right: 1px solid {card_border} !important;
            padding-top: 1rem;
        }}
        section[data-testid="stSidebar"] * {{
            color: {text_primary} !important;
        }}
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stMultiSelect label,
        section[data-testid="stSidebar"] .stTextInput label {{
            color: {text_secondary} !important;
            font-size: 0.82rem !important;
            font-weight: 500 !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }}

        /* ChatGPT-style New Chat button */
        .new-chat-btn div.stButton > button {{
            width: 100%;
            background-color: transparent;
            color: {text_primary} !important;
            border: 1px solid {card_border};
            border-radius: 8px;
            padding: 8px 12px;
            font-size: 0.9rem;
            font-weight: 500;
            display: flex;
            align-items: center;
            justify-content: flex-start;
            gap: 8px;
            transition: all 0.15s ease-in-out;
        }}
        .new-chat-btn div.stButton > button:hover {{
            background-color: {card_bg} !important;
            border-color: {card_hover} !important;
        }}

        /* Subtle Clear / Action Buttons */
        .subtle-btn div.stButton > button {{
            background-color: transparent;
            color: {text_secondary} !important;
            border: none;
            font-size: 0.85rem;
            padding: 6px 8px;
            width: 100%;
            text-align: left;
            transition: color 0.15s ease;
        }}
        .subtle-btn div.stButton > button:hover {{
            color: {text_primary} !important;
            background-color: {card_bg} !important;
            border-radius: 6px;
        }}

        /* Main content container */
        .block-container {{
            max-width: 840px !important;
            padding-top: 2rem !important;
            padding-bottom: 6rem !important;
            margin: 0 auto !important;
        }}

        /* Hero / Welcome screen */
        .chatgpt-hero {{
            text-align: center;
            margin-top: 5vh;
            margin-bottom: 2.5rem;
        }}
        .chatgpt-hero-icon {{
            font-size: 2.8rem;
            margin-bottom: 12px;
            display: inline-block;
        }}
        .chatgpt-hero-title {{
            font-size: 1.85rem;
            font-weight: 600;
            color: {text_primary};
            margin-bottom: 6px;
        }}
        .chatgpt-hero-sub {{
            font-size: 0.98rem;
            color: {text_secondary};
        }}

        /* Starter prompt cards */
        .starter-card div.stButton > button {{
            background-color: {card_bg} !important;
            border: 1px solid {card_border} !important;
            border-radius: 14px !important;
            padding: 14px 16px !important;
            text-align: left !important;
            height: 95px !important;
            width: 100% !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            transition: all 0.2s ease-in-out !important;
        }}
        .starter-card div.stButton > button:hover {{
            background-color: {card_hover} !important;
            border-color: {text_secondary} !important;
            transform: translateY(-1px);
        }}
        .starter-card div.stButton > button p {{
            color: {text_primary} !important;
            font-size: 0.88rem !important;
            line-height: 1.35 !important;
            margin: 0 !important;
        }}

        /* Chat messages */
        div[data-testid="stChatMessage"] {{
            background-color: transparent !important;
            padding: 1.1rem 0.5rem !important;
            border-radius: 8px !important;
        }}
        div[data-testid="stChatMessage"]:nth-child(even) {{
            background-color: {user_bubble} !important;
            border: 1px solid {card_border} !important;
            border-radius: 16px !important;
            padding: 0.9rem 1.2rem !important;
        }}
        div[data-testid="stChatMessage"] p, 
        div[data-testid="stChatMessage"] li,
        div[data-testid="stChatMessage"] h1,
        div[data-testid="stChatMessage"] h2,
        div[data-testid="stChatMessage"] h3,
        div[data-testid="stChatMessage"] h4 {{
            color: {text_primary} !important;
        }}

        /* Chat input bar */
        div[data-testid="stChatInput"] {{
            background-color: {input_bg} !important;
            border: 1px solid {input_border} !important;
            border-radius: 24px !important;
            box-shadow: 0 4px 16px {'rgba(0,0,0,0.35)' if is_dark else 'rgba(0,0,0,0.06)'} !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        }}
        div[data-testid="stChatInput"]:focus-within {{
            border-color: {accent} !important;
            box-shadow: 0 0 0 1px {accent}, 0 4px 16px {'rgba(16,163,127,0.2)' if is_dark else 'rgba(16,163,127,0.08)'} !important;
        }}
        div[data-testid="stChatInput"] textarea {{
            color: {text_primary} !important;
            font-size: 0.95rem !important;
            outline: none !important;
            box-shadow: none !important;
        }}
        div[data-testid="stChatInput"] textarea:focus {{
            outline: none !important;
            box-shadow: none !important;
        }}
        div[data-testid="stChatInput"] textarea::placeholder {{
            color: {text_secondary} !important;
        }}
        div[data-testid="stBottomBlockContainer"] {{
            background-color: {bg_main} !important;
        }}

        /* Streamlit widget controls (selectbox, text input, tags) */
        div[data-baseweb="select"] > div {{
            background-color: {card_bg} !important;
            border-color: {card_border} !important;
            border-radius: 8px !important;
            color: {text_primary} !important;
        }}
        div[data-baseweb="tag"] {{
            background-color: {'#3A3A3A' if is_dark else '#E5E7EB'} !important;
            border-radius: 6px !important;
        }}
        div[data-baseweb="tag"] span {{
            color: {text_primary} !important;
            font-size: 0.8rem !important;
        }}
        div[data-baseweb="input"] > div {{
            background-color: {card_bg} !important;
            border-color: {card_border} !important;
            border-radius: 8px !important;
            color: {text_primary} !important;
        }}
        div[data-baseweb="input"] input {{
            color: {text_primary} !important;
        }}

        /* Clean scrollbars */
        ::-webkit-scrollbar {{
            width: 6px;
            height: 6px;
        }}
        ::-webkit-scrollbar-thumb {{
            background: {card_border};
            border-radius: 3px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: {text_secondary};
        }}
    </style>
    """


@st.cache_resource(show_spinner=False)
def initialize_system():
    """Initialize recipe catalog and retrieval engines."""
    from app.recipe_loader import get_recipes
    from app.data_processor import build_recipe_index, build_all_search_texts
    from app.keyword_retrieval import BM25Retriever
    from app.semantic_retrieval import FAISSRetriever
    
    all_recipes = get_recipes()
    recipe_index = build_recipe_index(all_recipes)
    search_texts = build_all_search_texts(all_recipes)
    recipe_ids = [r["id"] for r in all_recipes]
    
    bm25_retriever = BM25Retriever(search_texts, recipe_ids)
    faiss_retriever = FAISSRetriever.from_disk()
    
    return all_recipes, recipe_index, bm25_retriever, faiss_retriever


def get_memory():
    """Get or create session-level conversation memory."""
    from app.memory import ConversationMemory
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory()
    return st.session_state.memory


def main():
    """Main ChatGPT-styled Streamlit application."""
    # Apply ChatGPT dynamic theme styling
    st.markdown(get_chatgpt_css(st.session_state.theme), unsafe_allow_html=True)
    
    # Initialize system backend
    try:
        all_recipes, recipe_index, bm25_retriever, faiss_retriever = initialize_system()
    except Exception as e:
        st.error(f"Unable to load recipe catalog: {e}")
        return

    memory = get_memory()

    # --- Sidebar: ChatGPT style navigation & filters ---
    with st.sidebar:
        # Top action: + New Chat
        st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
        if st.button("➕  New chat", key="new_chat_btn", use_container_width=True):
            memory.clear()
            st.session_state.messages = []
            st.session_state.pending_prompt = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        
        # Dietary & Region Preferences (Clean minimalist form)
        st.markdown("<p style='font-size: 0.85rem; font-weight: 600; margin-bottom: 8px;'>CULINARY PREFERENCES</p>", unsafe_allow_html=True)
        
        # 1. Region selection
        region_display = st.selectbox(
            "Region",
            ["India (Default 🇮🇳)", "Global 🌐"],
            index=0 if memory.region == "India" else 1,
            label_visibility="visible",
        )
        selected_region = "India" if "India" in region_display else "Global"
        memory.set_region(selected_region)
        
        # 2. Dietary restrictions
        diet_options = [
            "Vegetarian",
            "Jain Vegetarian",
            "Vegan",
            "No Beef",
            "No Pork",
            "No Seafood",
            "No Eggs",
        ]
        
        # Map user-friendly labels to internal keys
        diet_mapping = {
            "Vegetarian": "vegetarian",
            "Jain Vegetarian": "jain_vegetarian",
            "Vegan": "vegan",
            "No Beef": "no_beef",
            "No Pork": "no_pork",
            "No Seafood": "no_seafood",
            "No Eggs": "no_eggs",
        }
        reverse_diet_mapping = {v: k for k, v in diet_mapping.items()}
        current_diet_defaults = [reverse_diet_mapping[d] for d in memory.dietary_restrictions if d in reverse_diet_mapping]

        selected_diets = st.multiselect(
            "Dietary Restrictions",
            diet_options,
            default=current_diet_defaults,
            label_visibility="visible",
        )
        memory.dietary_restrictions = [diet_mapping[d] for d in selected_diets if d in diet_mapping]
        
        # 3. Custom excluded ingredients
        current_excluded_str = ", ".join(memory.excluded_ingredients)
        custom_excluded = st.text_input(
            "Excluded Ingredients",
            value=current_excluded_str,
            placeholder="e.g. peanuts, dairy, mushrooms",
            label_visibility="visible",
        )
        if custom_excluded != current_excluded_str:
            memory.excluded_ingredients = [
                i.strip().lower() for i in custom_excluded.split(",") if i.strip()
            ]

        # Spacer to push Theme Switcher & Footer to bottom
        st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
        st.divider()

        # Theme Selector (Dark / Light Mode)
        theme_col1, theme_col2 = st.columns([1, 1])
        with theme_col1:
            st.markdown("<span style='font-size: 0.85rem;'>Appearance</span>", unsafe_allow_html=True)
        with theme_col2:
            is_dark = st.session_state.theme == "dark"
            new_theme_label = "🌙 Dark" if is_dark else "☀️ Light"
            if st.button(new_theme_label, key="theme_toggle_btn", use_container_width=True):
                st.session_state.theme = "light" if is_dark else "dark"
                st.rerun()

        # Subtle Clear Conversation action
        st.markdown('<div class="subtle-btn">', unsafe_allow_html=True)
        if st.button("🗑️  Clear conversation", key="clear_chat_btn"):
            memory.clear()
            st.session_state.messages = []
            st.session_state.pending_prompt = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Main Chat Canvas ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display ChatGPT Welcome Hero & Starter Cards when conversation has not started
    if len(st.session_state.messages) == 0 and not st.session_state.get("pending_prompt"):
        st.markdown("""
        <div class="chatgpt-hero">
            <div class="chatgpt-hero-icon">🍳</div>
            <div class="chatgpt-hero-title">Nosh Recipe Assistant</div>
            <div class="chatgpt-hero-sub">What would you like to cook today?</div>
        </div>
        """, unsafe_allow_html=True)

        # 2x2 Starter Suggestion Cards
        starter_cards = [
            ("🍚  Rice Kheer for Diwali", "How do I make traditional rice kheer for Diwali step by step?"),
            ("🥘  Pantry Pasta", "I have pasta, garlic, butter, and parmesan. What can I cook?"),
            ("🔄  Jaggery for Sugar", "Can I replace refined sugar with jaggery in dessert recipes?"),
            ("🥗  High-Protein Dinner", "Suggest healthy high-protein vegetarian recipes with lentils or paneer"),
        ]

        col1, col2 = st.columns(2)
        for idx, (label, prompt_text) in enumerate(starter_cards):
            target_col = col1 if idx % 2 == 0 else col2
            with target_col:
                st.markdown('<div class="starter-card">', unsafe_allow_html=True)
                if st.button(f"**{label}**\n\n_{prompt_text}_", key=f"start_card_{idx}", use_container_width=True):
                    st.session_state.pending_prompt = prompt_text
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Render Conversation Messages
    for msg in st.session_state.messages:
        avatar = "🧑‍🍳" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Always render the chat input at the bottom so it is NEVER missing
    user_input = st.chat_input("Ask about any recipe, cooking guidance, or ingredients...")

    # Determine prompt to process
    prompt_to_handle = None
    if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
        prompt_to_handle = st.session_state.pop("pending_prompt")
    elif user_input:
        prompt_to_handle = user_input

    if prompt_to_handle:
        # 1. Record & render user message
        st.session_state.messages.append({"role": "user", "content": prompt_to_handle})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt_to_handle)

        # 2. Process query & generate culinary guidance
        with st.chat_message("assistant", avatar="🧑‍🍳"):
            with st.spinner("Preparing culinary guidance..."):
                from app.router import route_query
                response = route_query(
                    query=prompt_to_handle,
                    memory=memory,
                    bm25_retriever=bm25_retriever,
                    faiss_retriever=faiss_retriever,
                    recipe_index=recipe_index,
                    all_recipes=all_recipes,
                )
            st.markdown(response)

        # 3. Save assistant message to conversation history and rerun cleanly
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


if __name__ == "__main__":
    main()
