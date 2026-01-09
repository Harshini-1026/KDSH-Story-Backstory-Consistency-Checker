# scripts/load_and_split.py
import zipfile
import csv
from pathlib import Path
import pathway as pw

DATA_ZIP = "data/Books-20260106T140352Z-1-001.zip"
EXTRACT_DIR = "processed/raw_books"
OUTPUT_CSV = "processed/story_chunks.csv"
CHUNK_SIZE = 350


def extract_zip(zip_path, extract_dir):
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.filename.endswith(".txt"):
                target = extract_dir / Path(info.filename).name
                with zf.open(info.filename) as src, open(target, "wb") as dst:
                    dst.write(src.read())


def split_into_chunks(text, size):
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