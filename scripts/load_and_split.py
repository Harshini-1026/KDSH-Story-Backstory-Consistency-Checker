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
    for i in range(0, len(words), size):
        yield " ".join(words[i:i + size])


def main():
    extract_zip(DATA_ZIP, EXTRACT_DIR)

    rows = []

    for txt_file in Path(EXTRACT_DIR).glob("*.txt"):
        story_id = txt_file.stem
        try:
            text = txt_file.read_text(encoding="utf-8")
        except Exception:
            text = txt_file.read_text(encoding="latin-1")

        for idx, chunk in enumerate(split_into_chunks(text, CHUNK_SIZE), start=1):
            rows.append([
                story_id,
                f"{story_id}_chunk{idx}",
                idx,
                chunk
            ])

    Path("processed").mkdir(exist_ok=True)

    # Write CSV safely
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["story_id", "chunk_id", "chunk_number", "text"])
        writer.writerows(rows)

    # ✅ Pathway ingestion (correct usage)
    pw.io.csv.read(
        OUTPUT_CSV,
        schema=pw.schema_from_csv(OUTPUT_CSV)
    )

    print("✅ Step 1 complete — Story chunks prepared & ingested via Pathway")


if __name__ == "__main__":
    main()