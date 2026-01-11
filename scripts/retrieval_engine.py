# scripts/retrieval_engine.py

import os
import numpy as np
import pathway as pw
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

CHUNKS_CSV = "processed/story_chunks.csv"
OUTPUT_DIR = "processed/retrieval_outputs"

model = SentenceTransformer("all-MiniLM-L6-v2")


class ChunkSchema(pw.Schema):
    story_id: str
    chunk_id: str
    chunk_number: int
    text: str


# Pathway ingestion
chunks_table = pw.io.csv.read(
    CHUNKS_CSV,
    schema=ChunkSchema
)

# Convert to pandas once (offline retrieval use)
chunks_df = chunks_table.to_pandas()
embeddings = model.encode(chunks_df["text"].tolist(), show_progress_bar=False)


def get_relevant_chunks(query: str, top_k: int = 5):
    q_emb = model.encode([query])
    sims = cosine_similarity(q_emb, embeddings)[0]

    top_idx = np.argsort(sims)[-top_k:][::-1]

    results = []
    for idx in top_idx:
        results.append({
            "chunk_id": chunks_df.iloc[idx]["chunk_id"],
            "similarity": float(sims[idx]),
            "text": chunks_df.iloc[idx]["text"],
        })
    return results


def save_search_results(claim, results, file_name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    import pandas as pd
    df = pd.DataFrame(results)
    df.insert(0, "claim", claim)

    out = os.path.join(OUTPUT_DIR, file_name)
    df.to_csv(out, index=False)

    print(f"✔ Saved retrieved evidence to: {out}")