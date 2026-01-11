import os
from typing import List

import pandas as pd

from .retrieval_engine import get_relevant_chunks, save_search_results
from .consistency_checker import evaluate_backstory_consistency, save_result

BACKSTORY_FILE = os.path.join("data", "test.csv")  # change to test.csv when needed
OUTPUT_FOLDER = os.path.join("processed", "claims")
TOP_K_PASSAGES = 5


def ensure_output_folder():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def split_backstory_into_claims(backstory_text: str) -> List[str]:
    """Simple sentence-level splitter using common punctuation.

    This is intentionally lightweight and rule-based (no ML).
    """
    if not isinstance(backstory_text, str):
        return []

    text = backstory_text.replace("\n", " ").strip()
    # split on common sentence boundaries
    parts = []
    for sep in [". ", "? ", "! ", "; "]:
        text = text.replace(sep, ". ")

    rough = [p.strip() for p in text.split(".")]
    claims = [p for p in rough if len(p) >= 10]
    return claims


def process_story_row(row: pd.Series):
    story_id = row.get("story_id") or row.get("id") or "unknown"
    backstory = row.get("content") or row.get("backstory") or ""

    print(f"\n==============================")
    print(f" Processing Story ID: {story_id}")
    print(f"==============================")

    claims = split_backstory_into_claims(backstory)

    if not claims:
        print("⚠ No usable claims found — skipping")
        save_result(story_id, 0, "No claims extracted from backstory")
        return

    all_evidence_rows = []

    for i, claim in enumerate(claims, start=1):
        print(f"\n➡ Claim {i}: {claim}")

        results = get_relevant_chunks(claim, top_k=TOP_K_PASSAGES)

        # save retrieval for inspection
        file_name = f"{story_id}_claim{i}_retrieval.csv"
        save_search_results(claim, results, file_name=file_name)

        for r in results:
            all_evidence_rows.append(
                {
                    "story_id": story_id,
                    "claim": claim,
                    "chunk_id": r.get("chunk_id"),
                    "similarity": r.get("similarity"),
                    "text": r.get("text"),
                }
            )

    evidence_df = pd.DataFrame(all_evidence_rows)

    # pass entire gathered evidence (for all claims of this backstory)
    label, rationale = evaluate_backstory_consistency(evidence_df)

    save_result(story_id, label, rationale)


def run_pipeline(use_file: str = None):
    ensure_output_folder()

    file_to_use = use_file or BACKSTORY_FILE
    if not os.path.exists(file_to_use):
        print(f"Backstory file not found: {file_to_use}")
        return

    df = pd.read_csv(file_to_use)

    for _, row in df.iterrows():
        process_story_row(row)


if __name__ == "__main__":
    run_pipeline()
