import pandas as pd
import os

RETRIEVAL_FOLDER = "../processed/retrieval_outputs/"
OUTPUT_FILE = "../outputs/results.csv"

SUPPORT_KEYWORDS = [
    "supports", "matches", "aligns", "consistent",
    "similar", "reinforces", "confirms"
]

CONTRADICT_KEYWORDS = [
    "contradict", "opposite", "denies",
    "conflicts", "inconsistent", "breaks"
]


def read_retrieval_file(file_name):
    path = os.path.join(RETRIEVAL_FOLDER, file_name)
    print(f"\nReading evidence from: {path}")
    return pd.read_csv(path)


def simple_signal_score(text):
    """
    Very simple rule-based signal system
    """

    text = str(text).lower()

    support_score = sum(k in text for k in SUPPORT_KEYWORDS)
    contradict_score = sum(k in text for k in CONTRADICT_KEYWORDS)

    return support_score, contradict_score


def evaluate_claim_consistency(df):
    support_total = 0
    contradict_total = 0

    for _, row in df.iterrows():
        text = row["text"]
        s, c = simple_signal_score(text)

        support_total += s
        contradict_total += c

    if contradict_total > support_total:
        label = 0
        rationale = "Story behavior conflicts with key backstory assumptions"
    else:
        label = 1
        rationale = "Backstory aligns with character actions and narrative flow"

    return label, rationale


def save_result(story_id, label, rationale):

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    result_row = pd.DataFrame([{
        "story_id": story_id,
        "prediction": label,
        "rationale": rationale
    }])

    if not os.path.exists(OUTPUT_FILE):
        result_row.to_csv(OUTPUT_FILE, index=False)
    else:
        result_row.to_csv(OUTPUT_FILE, mode="a", header=False, index=False)

    print(f"\n✔ Saved decision for story {story_id}")


def demo_run():

    sample_file = "retrieval_sample.csv"

    df = read_retrieval_file(sample_file)

    story_id = df["chunk_id"][0].split("_")[0]

    label, rationale = evaluate_claim_consistency(df)

    print("\nDecision:")
    print("Label:", label)
    print("Reason:", rationale)

    save_result(story_id, label, rationale)


if __name__ == "__main__":
    demo_run()
