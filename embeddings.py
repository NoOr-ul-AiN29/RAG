# embeddings.py
# ---------------------------------------------------------------------------
# PURPOSE:
#   1. Convert text chunks into vector embeddings using a FREE local model.
#   2. Build a FAISS index from those embeddings so we can search them fast.
#   3. Save the FAISS index + original chunks to disk for reuse.
#   4. Load a previously saved index from disk.
#   5. Search the index to retrieve the top-k most relevant chunks for a query.
#
# MODEL USED:
#   "all-MiniLM-L6-v2" from sentence-transformers (HuggingFace).
#   - Completely FREE — runs locally, no API key needed.
#   - Downloads once (~90MB) on first run, then cached on disk.
#   - Produces 384-dimensional embeddings (smaller & faster than OpenAI).
#   - Excellent quality for semantic search tasks.
#
# HOW FAISS WORKS (simple version):
#   Each chunk of text becomes a list of numbers (a "vector"). When you ask
#   a question, that question also becomes a vector. FAISS finds the stored
#   vectors that are closest to the question vector — those are the most
#   relevant chunks.
# ---------------------------------------------------------------------------

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Load the free local embedding model
# ---------------------------------------------------------------------------
# This downloads the model on first run and caches it locally.
# No API key, no cost, no internet needed after the first download.
print("[embeddings] Loading local embedding model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("[embeddings] Embedding model loaded successfully.")

# all-MiniLM-L6-v2 produces 384-dimensional vectors
EMBEDDING_DIMENSION = 384


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of text strings into embedding vectors using the local
    sentence-transformers model. Completely free, runs on your machine.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors (each a list of 384 floats).
    """
    # encode() returns a numpy array of shape [n_texts, 384]
    # convert_to_numpy=True ensures we always get numpy arrays back
    vectors = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False   # set True if you want a progress bar
    )

    print(f"[embeddings] Generated {len(vectors)} embeddings locally.")
    # Convert numpy array rows to plain Python lists for consistency
    return vectors.tolist()


def build_faiss_index(chunks: list[str]) -> faiss.IndexFlatL2:
    """
    Generate embeddings for all chunks and load them into a new FAISS index.

    We use IndexFlatL2 which does exact nearest-neighbour search using
    Euclidean (L2) distance. Simple and reliable for small-to-medium
    document sets.

    Args:
        chunks: List of text chunks to index.

    Returns:
        A populated faiss.IndexFlatL2 object.
    """
    vectors = get_embeddings(chunks)

    # FAISS needs a 2-D float32 numpy array  [n_chunks × dimension]
    matrix = np.array(vectors, dtype="float32")

    index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
    index.add(matrix)

    print(f"[embeddings] FAISS index built with {index.ntotal} vectors.")
    return index


def save_index(
    index: faiss.IndexFlatL2,
    chunks: list[str],
    index_path: str,
    chunks_path: str
) -> None:
    """
    Persist the FAISS index and the original text chunks to disk so we
    don't have to re-embed the same document on every server restart.

    Args:
        index:       The FAISS index to save.
        chunks:      The text chunks that correspond to the indexed vectors.
        index_path:  File path where the FAISS binary will be written.
        chunks_path: File path where the chunks JSON will be written.
    """
    faiss.write_index(index, index_path)

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"[embeddings] Index saved to '{index_path}', "
          f"chunks saved to '{chunks_path}'.")


def load_index(
    index_path: str,
    chunks_path: str
) -> tuple[faiss.IndexFlatL2, list[str]]:
    """
    Load a previously saved FAISS index and its matching text chunks from
    disk.

    Args:
        index_path:  Path to the saved FAISS binary file.
        chunks_path: Path to the saved chunks JSON file.

    Returns:
        A tuple of (faiss_index, chunks_list).

    Raises:
        FileNotFoundError: If either file does not exist.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index not found at '{index_path}'. "
            "Please upload a document first."
        )
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(
            f"Chunks file not found at '{chunks_path}'. "
            "Please upload a document first."
        )

    index = faiss.read_index(index_path)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"[embeddings] Loaded index with {index.ntotal} vectors "
          f"and {len(chunks)} chunks.")
    return index, chunks


def search_index(
    query: str,
    index: faiss.IndexFlatL2,
    chunks: list[str],
    top_k: int = 5
) -> list[str]:
    """
    Embed the query locally and find the top_k most similar chunks in
    the FAISS index.

    Args:
        query:  The user's question as a plain string.
        index:  The loaded FAISS index.
        chunks: The text chunks corresponding to the indexed vectors.
        top_k:  How many relevant chunks to return.

    Returns:
        A list of the top_k most relevant text chunk strings.
    """
    query_vector = get_embeddings([query])
    query_matrix = np.array(query_vector, dtype="float32")

    # D = distances, I = indices of the nearest neighbours
    distances, indices = index.search(query_matrix, top_k)

    results = []
    for idx in indices[0]:
        if idx != -1 and idx < len(chunks):   # -1 means not enough results
            results.append(chunks[idx])

    print(f"[embeddings] Retrieved {len(results)} chunks for query.")
    return results