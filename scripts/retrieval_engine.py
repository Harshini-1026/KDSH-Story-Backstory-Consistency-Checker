import os
from typing import List, Dict

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------- SETTINGS ----------
CHUNKS_FILE = os.path.join("processed", "story_chunks.csv")
OUTPUT_FOLDER = os.path.join("processed", "retrieval_outputs")
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
# ------------------------------


def ensure_output_folder():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)


print("\nLoading embedding model...")
model = SentenceTransformer(MODEL_NAME)


def embed_text_list(text_list: List[str]) -> np.ndarray:
    """Convert list of texts to normalized embeddings (for cosine similarity)."""
    emb = model.encode(text_list, convert_to_numpy=True)
    # normalize for cosine similarity with IndexFlatIP
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms
    return emb.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def load_chunks() -> pd.DataFrame:
    if not os.path.exists(CHUNKS_FILE):
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_FILE}")

    df = pd.read_csv(CHUNKS_FILE)
    print(f"Loaded {len(df)} chunks from {CHUNKS_FILE}")
    return df


def build_search_system():
    df = load_chunks()
    texts = df["text"].fillna("").tolist()

    print("Embedding story chunks...")
    embeddings = embed_text_list(texts)

    print("Building FAISS index (cosine similarity)...")
    index = build_faiss_index(embeddings)

    return df, index, embeddings


# Build at import so callers can immediately use `get_relevant_chunks`.
DF_CHUNKS, SEARCH_INDEX, EMB_STORE = build_search_system()


def get_relevant_chunks(claim: str, top_k: int = TOP_K) -> List[Dict]:
    """Return top_k relevant chunks for a claim.

    Each result contains: chunk_id, story_id, similarity (cosine), text
    """
    if not claim or not isinstance(claim, str):
        return []

    q_emb = embed_text_list([claim])

    # FAISS returns inner product; since vectors are normalized, this is cosine.
    distances, positions = SEARCH_INDEX.search(q_emb, top_k)

    results = []
    for score, pos in zip(distances[0], positions[0]):
        if pos < 0 or pos >= len(DF_CHUNKS):
            continue
        row = DF_CHUNKS.iloc[int(pos)]
        results.append(
            {
                "chunk_id": row.get("chunk_id"),
                "story_id": row.get("story_id"),
                "similarity": float(score),
                "text": row.get("text"),
            }
        )

    return results


def save_search_results(claim: str, results: List[Dict], file_name: str = "retrieval_sample.csv"):
    ensure_output_folder()
    out_path = os.path.join(OUTPUT_FOLDER, file_name)
    df = pd.DataFrame(results)
    df.insert(0, "claim", claim)
    df.to_csv(out_path, index=False)
    print(f"\n✔ Saved retrieved evidence to: {out_path}")


def demo_example():
    ensure_output_folder()
    claim = "The character had a lonely childhood and feared abandonment"
    results = get_relevant_chunks(claim)
    for r in results:
        print("\n--- Match ---")
        print("Chunk:", r["chunk_id"])
        print("Score:", r["similarity"])
        print("Text:", (r["text"] or "")[:250], "...")
    save_search_results(claim, results)


if __name__ == "__main__":
    demo_example()
