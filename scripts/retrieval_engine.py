import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# ---------- SETTINGS ----------
CHUNKS_FILE = "../processed/train_chunks.csv"   # same logic will work for test also
OUTPUT_FOLDER = "../processed/retrieval_outputs/"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
# ------------------------------


def ensure_output_folder():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)


print("\nLoading embedding model...")
model = SentenceTransformer(MODEL_NAME)


def embed_text_list(text_list):
    """
    Convert text list -> embeddings
    """
    return model.encode(text_list, convert_to_numpy=True)


def build_faiss_index(embeddings):
    """
    FAISS index creation
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def load_chunks():
    """
    Load story chunks created by Dharanesh
    """
    print(f"\nLoading chunks from: {CHUNKS_FILE}")
    df = pd.read_csv(CHUNKS_FILE)

    texts = df["text"].tolist()
    chunk_ids = df["chunk_id"].tolist()

    print(f"Total chunks loaded: {len(df)}")

    return df, texts, chunk_ids


def build_search_system():
    df, texts, chunk_ids = load_chunks()

    print("\nEmbedding story chunks...")
    embeddings = embed_text_list(texts)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    return df, index, embeddings, chunk_ids


df_chunks, search_index, emb_store, chunk_ids = build_search_system()


def search_related_passages(query_text, top_k=TOP_K):
    """
    Input  : backstory sentence / claim
    Output : top matching story parts
    """

    query_emb = embed_text_list([query_text])

    distances, positions = search_index.search(query_emb, top_k)

    results = []

    for dist, pos in zip(distances[0], positions[0]):
        results.append({
            "chunk_id": chunk_ids[pos],
            "score": float(dist),
            "text": df_chunks.loc[pos, "text"]
        })

    return results


def save_search_results(claim, results, file_name="retrieval_sample.csv"):
    out_path = os.path.join(OUTPUT_FOLDER, file_name)

    df = pd.DataFrame(results)
    df.insert(0, "claim", claim)

    df.to_csv(out_path, index=False)

    print(f"\n✔ Saved retrieved evidence to: {out_path}")


def demo_example():
    print("\n--- Demo Search Example ---")

    claim = "The character had a lonely childhood and feared abandonment"

    results = search_related_passages(claim)

    for r in results:
        print("\n--- Match ---")
        print("Chunk:", r["chunk_id"])
        print("Score:", r["score"])
        print("Text:", r["text"][:250], "...")

    save_search_results(claim, results)


if __name__ == "__main__":
    ensure_output_folder()
    demo_example()
