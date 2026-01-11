# scripts/backstory_claim_engine.py

import os
import pandas as pd
import pathway as pw

from retrieval_engine import get_relevant_chunks, save_search_results
from consistency_checker import evaluate_backstory_consistency, save_result

BACKSTORY_CSV = "data/train.csv"
CLAIMS_DIR = "processed/claims"
TOP_K = 5


class BackstorySchema(pw.Schema):
    id: int
    content: str
    label: str


def split_claims(text):
    text = text.replace("\n", " ")
    for sep in ["? ", "! ", "; "]:
        text = text.replace(sep, ". ")
    return [p.strip() for p in text.split(".") if len(p.strip()) > 10]


def run_pipeline():
    os.makedirs(CLAIMS_DIR, exist_ok=True)

    table = pw.io.csv.read(
        BACKSTORY_CSV,
        schema=BackstorySchema
    )

    df = table.to_pandas()

    for _, row in df.iterrows():
        story_id = row["id"]
        claims = split_claims(row["content"])

        if not claims:
            save_result(story_id, 0, "No claims extracted")
            continue

        pd.DataFrame({
            "story_id": story_id,
            "claim_id": range(1, len(claims) + 1),
            "claim": claims
        }).to_csv(f"{CLAIMS_DIR}/{story_id}_claims.csv", index=False)

        evidence = []

        for i, claim in enumerate(claims, start=1):
            results = get_relevant_chunks(claim, TOP_K)
            save_search_results(claim, results, f"{story_id}_claim{i}_retrieval.csv")

            for r in results:
                evidence.append({
                    "story_id": story_id,
                    "claim": claim,
                    "chunk_id": r["chunk_id"],
                    "similarity": r["similarity"],
                    "text": r["text"],
                })

        ev_df = pd.DataFrame(evidence)
        label, rationale = evaluate_backstory_consistency(ev_df)
        save_result(story_id, label, rationale)


if __name__ == "__main__":
    run_pipeline()