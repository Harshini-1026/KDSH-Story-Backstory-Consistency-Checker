# =====================================================
# KDSH 2026 – PART 2: CLAIM → EVIDENCE RETRIEVAL
# =====================================================

# ---------------- IMPORTS ----------------
import os
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
# -----------------------------------------


# ---------------- SETTINGS ----------------
CHUNKS_FILE = "processed/train_chunks.csv"
CLAIMS_FILE = "processed/claims/backstory_claims.csv"

OUTPUT_FOLDER = "processed/retrieval_outputs"
OUTPUT_FILE = "retrieval_results.csv"

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
# -----------------------------------------


# =====================================================
# LOAD DATA
# =====================================================

def load_chunks():
    df = pd.read_csv(CHUNKS_FILE)

    required_cols = {"story_id", "chunk_id", "chunk_number", "text"}
    if not required_cols.issubset(df.columns):
        raise ValueError("Chunks file must contain story_id, chunk_id, chunk_number, text")

    return df


def load_claims():
    df = pd.read_csv(CLAIMS_FILE)

    required_cols = {"claim_id", "story_id", "claim"}
    if not required_cols.issubset(df.columns):
        raise ValueError("Claims file must contain claim_id, story_id, claim")

    return df


# =====================================================
# EMBEDDING + FAISS
# =====================================================

def embed_texts(model, texts):
    return model.encode(texts, convert_to_numpy=True)


def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


# =====================================================
# RETRIEVAL CORE
# =====================================================

def retrieve_evidence_for_claim(claim_row, model, df_chunks):
    """
    Retrieve top-K relevant chunks for a single claim
    """

    story_id = claim_row["story_id"]
    claim_text = claim_row["claim"]

    # Filter chunks for the same story
    story_chunks = df_chunks[df_chunks["story_id"] == story_id].reset_index(drop=True)

    texts = story_chunks["text"].tolist()
    chunk_ids = story_chunks["chunk_id"].tolist()

    # Embed chunks
    embeddings = embed_texts(model, texts)

    # Build FAISS index
    index = build_faiss_index(embeddings)

    # Embed claim
    query_embedding = embed_texts(model, [claim_text])

    # Search
    distances, positions = index.search(query_embedding, TOP_K)

    # Collect results
    results = []
    for dist, pos in zip(distances[0], positions[0]):
        results.append({
            "claim_id": claim_row["claim_id"],
            "story_id": story_id,
            "claim": claim_text,
            "chunk_id": chunk_ids[pos],
            "chunk_number": story_chunks.loc[pos, "chunk_number"],
            "similarity_score": float(dist),
            "text": story_chunks.loc[pos, "text"]
        })

    return results


# =====================================================
# MAIN PIPELINE
# =====================================================

def main():

    print("\n[PART 2] Loading data...")
    df_chunks = load_chunks()
    df_claims = load_claims()

    print("[PART 2] Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    all_results = []

    print("[PART 2] Retrieving evidence for claims...")
    for _, claim_row in df_claims.iterrows():
        print(f"  → Processing claim: {claim_row['claim']}")
        claim_results = retrieve_evidence_for_claim(
            claim_row, model, df_chunks
        )
        all_results.extend(claim_results)

    # Save output
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)

    pd.DataFrame(all_results).to_csv(output_path, index=False)

    print("\n✅ PART 2 COMPLETED SUCCESSFULLY")
    print(f"📄 Retrieval results saved to: {output_path}")


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()
