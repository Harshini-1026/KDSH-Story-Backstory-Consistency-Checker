import os
from typing import Tuple

import pandas as pd

OUTPUT_FILE = os.path.join("outputs", "results.csv")

SUPPORT_KEYWORDS = [
    "supports",
    "matches",
    "aligns",
    "consistent",
    "similar",
    "reinforces",
    "confirms",
]

CONTRADICT_KEYWORDS = [
    "contradict",
    "opposite",
    "denies",
    "conflicts",
    "inconsistent",
    "breaks",
    "contradiction",
]


def text_has_any_keyword(text: str, keywords: list) -> bool:
    if not isinstance(text, str):
        return False
    t = text.lower()
    return any(k in t for k in keywords)


def evaluate_backstory_consistency(evidence_df: pd.DataFrame) -> Tuple[int, str]:
    """Conservative rule-based decision for an entire backstory.

    - If no evidence -> contradict (0)
    - If any evidence contains contradiction keywords -> contradict (0)
    - If any evidence has high similarity (>= 0.75) or contains support keywords -> consistent (1)
    - Otherwise -> contradict (0)
    """
    if evidence_df is None or evidence_df.empty:
        return 0, "No supporting evidence found for backstory"

    # Check contradictions first (conservative)
    for _, row in evidence_df.iterrows():
        if text_has_any_keyword(row.get("text", ""), CONTRADICT_KEYWORDS):
            chunk = row.get("chunk_id")
            return 0, f"Contradiction found in evidence chunk: {chunk}"

    # Check for strong similarity signals
    if "similarity" in evidence_df.columns:
        try:
            best_sim = float(evidence_df["similarity"].max())
            if best_sim >= 0.75:
                best_row = evidence_df.loc[evidence_df["similarity"].idxmax()]
                return 1, f"Strong supporting evidence (similarity={best_sim:.2f}) in chunk {best_row.get('chunk_id')}"
        except Exception:
            pass

    # Check for support keywords in texts
    for _, row in evidence_df.iterrows():
        if text_has_any_keyword(row.get("text", ""), SUPPORT_KEYWORDS):
            chunk = row.get("chunk_id")
            return 1, f"Supportive language found in evidence chunk: {chunk}"

    # fallback conservative decision
    return 0, "Insufficient supporting evidence; leaning contradict"


def save_result(story_id: str, label: int, rationale: str):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    row = pd.DataFrame([{"story_id": story_id, "prediction": int(label), "rationale": rationale}])

    if not os.path.exists(OUTPUT_FILE):
        row.to_csv(OUTPUT_FILE, index=False)
    else:
        row.to_csv(OUTPUT_FILE, mode="a", header=False, index=False)

    print(f"\n✔ Saved decision for story {story_id} -> {label}")


if __name__ == "__main__":
    # simple smoke demo
    print("consistency_checker ready")
