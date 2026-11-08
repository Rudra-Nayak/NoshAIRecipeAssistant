# Architectural Design Decisions & Trade-Off Analysis

## 1. Technology Choices & Justification

### 1.1 Python 3.11+ Backend
- **Choice**: Modern Python with strict type hinting, dataclasses, and path handling via `pathlib`.
- **Justification**: Python is the lingua franca of GenAI and data science. Native compatibility with FAISS, Rank-BM25, Pandas, and the Google GenAI SDK.
- **Alternatives Considered**: Go or Rust for raw speed. *Rejected* because data science ecosystem maturity and rapid interview inspectability take precedence.

### 1.2 Streamlit User Interface
- **Choice**: Streamlit (`streamlit>=1.40.0`).
- **Justification**: Eliminates frontend-backend decoupling overhead. Provides reactive component rerendering, chat message abstractions (`st.chat_message`, `st.chat_input`), and simple session state management (`st.session_state`).
- **Alternatives Considered**: React / Next.js + FastAPI. *Rejected* due to unneeded complexity, CORS configuration, and container orchestration requirements for an interview assignment.

### 1.3 FAISS CPU for Vector Search
- **Choice**: `faiss-cpu` with flat inner-product index (`IndexFlatIP`).
- **Justification**: For $N=122,938$ vectors of dimension 768, an exact flat scan requires $<25$ms on modern CPUs while consuming only $\approx 370$MB of RAM.
- **Alternatives Considered**: Hosted vector databases (Pinecone, Weaviate, Milvus). *Rejected* because they introduce network round-trips, external API token limits, and third-party credential dependencies.

### 1.4 BM25 for Lexical Search
- **Choice**: `rank-bm25` (Okapi variant).
- **Justification**: Vector search frequently experiences vocabulary mismatch on specific, rare culinary entities (e.g., *"guanciale"*, *"za'atar"*). BM25 guarantees that rare exact terms receive heavy IDF weighting.
- **Alternatives Considered**: Elasticsearch. *Rejected* because it requires running a separate Java daemon, violating the lightweight local-run requirement.

### 1.5 Reciprocal Rank Fusion (RRF)
- **Choice**: RRF with standard $k=60$.
- **Justification**: Enables merging lexical and semantic rankings without normalizing incompatible score distributions (unbounded BM25 vs cosine similarities).
- **Alternatives Considered**: Learned rankers / linear interpolation ($\alpha \cdot \text{BM25} + (1-\alpha) \cdot \text{Dense}$). *Rejected* because score distribution shift across query types leads to instability without extensive supervised training data.

### 1.6 Lightweight Python Feature Reranker
- **Choice**: Explicit feature-based scorer (Title match + Ingredient coverage + Instruction overlap).
- **Justification**: Runs in $<2$ms in pure Python. Highly transparent, debuggable during live interviews, and requires zero heavy deep-learning dependencies (e.g. PyTorch, HuggingFace Transformers).
- **Alternatives Considered**: Cross-Encoder BERT (`cross-encoder/ms-marco-MiniLM-L-6-v2`). *Rejected* due to high CPU latency (300-600ms per 20 candidates) and multi-gigabyte package footprints.

