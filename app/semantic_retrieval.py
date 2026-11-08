"""
semantic_retrieval.py - FAISS-based semantic similarity search.

Uses Google's text-embedding-004 model to create dense vector representations
of recipes, then uses FAISS (Facebook AI Similarity Search) for fast
approximate nearest-neighbor lookup.

Key design decisions:
- Build the FAISS index ONCE and persist to disk
- Only generate embeddings for new/changed recipes
- Never regenerate all embeddings per query — only embed the query
"""
import os
import pickle
import numpy as np
from pathlib import Path
from typing import Optional

# Lazy imports — only load heavy libraries when needed
_faiss = None


def _get_faiss():
    """Lazy import of faiss to speed up module loading."""
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss


# --- Constants ---

VECTOR_STORE_DIR = Path("vector_store")
FAISS_INDEX_FILE = VECTOR_STORE_DIR / "recipe_index.faiss"
METADATA_FILE = VECTOR_STORE_DIR / "recipe_metadata.pkl"

# Embedding dimension for text-embedding-004
EMBEDDING_DIM = 768

# Batch size for embedding generation (API rate limiting)
EMBEDDING_BATCH_SIZE = 100


def generate_embeddings_batch(texts: list[str], model_name: str = "models/text-embedding-004") -> Optional[np.ndarray]:
    """Generate embeddings for a batch of texts using Gemini API.
    
    Args:
        texts: List of text strings to embed.
        model_name: Embedding model name.
        
    Returns:
        numpy array of shape (len(texts), EMBEDDING_DIM), or None on failure.
    """
    try:
        from google import genai
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_api_key_here":
            print("  WARNING: GEMINI_API_KEY not set. Cannot generate embeddings.")
            return None
        
        client = genai.Client(api_key=api_key)
        
        embeddings = []
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i:i + EMBEDDING_BATCH_SIZE]
            result = client.models.embed_content(
                model=model_name,
                contents=batch,
            )
            for embedding in result.embeddings:
                embeddings.append(embedding.values)
            
            if (i + EMBEDDING_BATCH_SIZE) % 1000 == 0:
                print(f"    Embedded {min(i + EMBEDDING_BATCH_SIZE, len(texts))}/{len(texts)} texts...")
        
        return np.array(embeddings, dtype=np.float32)
    
    except Exception as e:
        print(f"  ERROR generating embeddings: {e}")
        return None


def build_faiss_index(
    search_texts: list[str],
    recipe_ids: list[str],
    index_path: Path = FAISS_INDEX_FILE,
    metadata_path: Path = METADATA_FILE,
) -> bool:
    """Build and persist a FAISS index from recipe search texts.
    
    This should only be called ONCE during initial setup or when the
    dataset changes. The index is saved to disk for fast loading.
    
    Args:
        search_texts: List of search text strings to embed.
        recipe_ids: List of recipe IDs, index-aligned with search_texts.
        index_path: Path to save the FAISS index.
        metadata_path: Path to save the ID mapping metadata.
        
    Returns:
        True if the index was built successfully, False otherwise.
    """
    faiss = _get_faiss()
    
    print(f"  Generating embeddings for {len(search_texts)} recipes...")
    embeddings = generate_embeddings_batch(search_texts)
    
    if embeddings is None:
        print("  Failed to generate embeddings. FAISS index not built.")
        return False
    
    print(f"  Building FAISS index (dim={embeddings.shape[1]})...")
    
    # Use IndexFlatIP (Inner Product) with normalized vectors = cosine similarity
    # For ~123K vectors, flat index is fast enough and exact
    dim = embeddings.shape[1]
    
    # L2-normalize for cosine similarity via inner product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    embeddings_normalized = embeddings / norms
    
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_normalized)
    
    # Save to disk
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    
    # Save ID mapping
    metadata = {"recipe_ids": recipe_ids, "dim": dim}
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"  FAISS index saved: {index.ntotal} vectors, dim={dim}")
    return True


def load_faiss_index(
    index_path: Path = FAISS_INDEX_FILE,
    metadata_path: Path = METADATA_FILE,
) -> Optional[tuple]:
    """Load a persisted FAISS index and its ID mapping.
    
    Args:
        index_path: Path to the FAISS index file.
        metadata_path: Path to the metadata file.
        
    Returns:
        Tuple of (faiss_index, recipe_ids_list), or None if not found.
    """
    if not index_path.exists() or not metadata_path.exists():
        return None
    
    faiss = _get_faiss()
    
    index = faiss.read_index(str(index_path))
    
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    
    return index, metadata["recipe_ids"]


class FAISSRetriever:
    """FAISS-based semantic similarity retriever.
    
    Wraps the FAISS index and provides a search interface matching
    the BM25Retriever API.
    """
    
    def __init__(self, index, recipe_ids: list[str]):
        """Initialize with a loaded FAISS index.
        
        Args:
            index: A FAISS index object.
            recipe_ids: List of recipe IDs, index-aligned with FAISS vectors.
        """
        self.index = index
        self.recipe_ids = recipe_ids
    
    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Search for semantically similar recipes.
        
        Args:
            query: User search query.
            top_k: Number of top results to return.
            
        Returns:
            List of (recipe_id, similarity_score) tuples, sorted by score descending.
        """
        # Embed the query (single text)
        query_embedding = generate_embeddings_batch([query])
        
        if query_embedding is None:
            return []
        
        # Normalize query embedding
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.recipe_ids):
                results.append((self.recipe_ids[idx], float(score)))
        
        return results
    
    @classmethod
    def from_disk(cls, index_path: Path = FAISS_INDEX_FILE,
                  metadata_path: Path = METADATA_FILE) -> Optional["FAISSRetriever"]:
        """Load a FAISSRetriever from persisted files.
        
        Returns:
            FAISSRetriever instance, or None if index files don't exist.
        """
        result = load_faiss_index(index_path, metadata_path)
        if result is None:
            return None
        index, recipe_ids = result
        return cls(index, recipe_ids)

