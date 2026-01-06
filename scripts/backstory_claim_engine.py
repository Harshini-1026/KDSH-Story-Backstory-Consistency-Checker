import os
import pandas as pd

from retrieval_engine import search_related_passages, save_search_results
from consistency_checker import evaluate_claim_consistency, save_result


BACKSTORY_FILE = "../data/train.csv"   # later can run for test also
OUTPUT_FOLDER = "../processed/claims/"
TOP_K_PASSAGES = 5


def ensure_output_folder():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def split_backstory_into_claims(backstory_text):
    """
    Beginner-friendly claim splitter
    Breaks backstory into short meaning units
    """

    if not isinstance(backstory_text, str):
        return []

    backstory_text = backstory_text.replace("\n", " ").strip()

    # split by . , ; and and/or
    rough_parts = backstory_text.split(".")

    claims = []

    for part in rough_parts:
        part = part.strip()

        if len(part) < 10:
            continue

        claims.append(part)

    return claims


def process_story_row(row):

    story_id = row.get("story_id")
    backstory = row.get("backstory", "")

    print(f"\n==============================")
    print(f" Processing Story ID: {story_id}")
    print(f"==============================")

    claims = split_backstory_into_claims(backstory)

    if not claims:
        print("⚠ No usable claims found — skipping")
        return

    all_evidence_rows = []

    for i, claim in enumerate(claims):

        print(f"\n➡ Claim {i+1}: {claim}")

        results = search_related_passages(claim, top_k=TOP_K_PASSAGES)

        file_name = f"{story_id}_claim{i+1}_retrieval.csv"
        save_search_results(claim, results, file_name=file_name)

        for r in results:
            all_evidence_rows.append({
                "story_id": story_id,
                "claim": claim,
                "chunk_id": r["chunk_id"],
                "score": r["score"],
                "text": r["text"]
            })

    evidence_df = pd.DataFrame(all_evidence_rows)

    # pass entire gathered evidence to Barath's logic
    label, rationale = evaluate_claim_consistency(evidence_df)

    save_result(story_id, label, rationale)


def run_pipeline():

    ensure_output_folder()

    df = pd.read_csv(BACKSTORY_FILE)

    for _, row in df.iterrows():
        process_story_row(row)


if __name__ == "__main__":
    run_pipeline()
