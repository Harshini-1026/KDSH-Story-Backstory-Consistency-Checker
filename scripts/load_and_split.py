import os
import pandas as pd

# ---------- SETTINGS ----------
INPUT_FILES = [
    "../data/train.csv",
    "../data/test.csv"
]

OUTPUT_FOLDER = "../processed/"
CHUNK_SIZE = 300  # words per chunk
# ------------------------------


def ensure_output_folder():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def split_text_into_chunks(text, chunk_size=300):
    """
    Splits big story text into small readable chunks (paragraph size)
    """
    words = text.split()

    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def process_file(file_path):
    print(f"\nProcessing: {file_path}")

    df = pd.read_csv(file_path)

    # expected columns:
    # story_id , story_text , backstory , (label only in train)

    all_rows = []

    for idx, row in df.iterrows():
            story_id = row.get("story_id", idx)
            story_text = str(row.get("story_text", "")).strip()

            if not story_text:
                continue

            chunks = split_text_into_chunks(story_text, CHUNK_SIZE)

            for i, chunk in enumerate(chunks):
                all_rows.append({
                    "story_id": story_id,
                    "chunk_id": f"{story_id}_chunk{i+1}",
                    "chunk_number": i+1,
                    "text": chunk
                })

    out_df = pd.DataFrame(all_rows)

    out_name = os.path.basename(file_path).replace(".csv", "_chunks.csv")
    out_path = os.path.join(OUTPUT_FOLDER, out_name)

    out_df.to_csv(out_path, index=False)

    print(f"✔ Saved: {out_path}")
    print(f"Total chunks created: {len(out_df)}")


def main():
    ensure_output_folder()

    for file in INPUT_FILES:
        process_file(file)


if __name__ == "__main__":
    main()
