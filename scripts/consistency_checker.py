# scripts/consistency_checker.py

import os
from typing import Tuple
import pandas as pd

OUTPUT_FILE = "outputs/results.csv"

SUPPORT_KEYWORDS = [
    "supports", "matches", "aligns", "consistent",
    "reinforces", "confirms"
]

CONTRADICT_KEYWORDS = [
    "contradict", "conflicts", "denies",
    "inconsistent", "opposite"
]


def has_any(text, keywords):
    if not isinstance(text, str):
        return False
    t = text.lower()
    return any(k in t for k in keywords)


def evaluate_backstory_consistency(df: pd.DataFrame) -> Tuple[int, str]:
    if df.empty:
        return 0, "No supporting evidence found"

    for _, row in df.iterrows():
        if has_any(row["text"], CONTRADICT_KEYWORDS):
            return 0, f"Contradiction found in {row['chunk_id']}"

    best_sim = df["similarity"].max()
    if best_sim >= 0.75:
        row = df.loc[df["similarity"].idxmax()]
        return 1, f"Strong support (similarity={best_sim:.2f}) in {row['chunk_id']}"

    for _, row in df.iterrows():
        if has_any(row["text"], SUPPORT_KEYWORDS):
            return 1, f"Supportive language in {row['chunk_id']}"

    return 0, "Insufficient supporting evidence; leaning contradict"


def save_result(story_id, label, rationale):
    os.makedirs("outputs", exist_ok=True)
    import pandas as pd

    row = pd.DataFrame([{
        "story_id": story_id,
        "prediction": label,
        "rationale": rationale
    }])

    if not os.path.exists(OUTPUT_FILE):
        row.to_csv(OUTPUT_FILE, index=False)
    else:
        row.to_csv(OUTPUT_FILE, mode="a", header=False, index=False)

    print(f"✔ Saved decision for story {story_id} -> {label}")