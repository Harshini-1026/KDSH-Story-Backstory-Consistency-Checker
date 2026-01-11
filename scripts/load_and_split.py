# scripts/load_and_split.py

import zipfile
from pathlib import Path
import pathway as pw

# ================= SETTINGS =================
DATA_ZIP = "data/Books-20260106T140352Z-1-001.zip"
EXTRACT_DIR = "processed/raw_books"
OUTPUT_CSV = "processed/story_chunks.csv"
CHUNK_SIZE = 350
# ============================================


class StoryChunkSchema(pw.Schema):
    story_id: str
    chunk_id: str
    chunk_number: int
    text: str


def extract_zip(zip_path, extract_dir):
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.filename.endswith(".txt"):
                target = extract_dir / Path(info.filename).name
                if not target.exists():
                    with zf.open(info.filename) as src, open(target, "wb") as dst:
                        dst.write(src.read())


def generate_rows():
    extract_zip(DATA_ZIP, EXTRACT_DIR)

    for txt_file in Path(EXTRACT_DIR).glob("*.txt"):
        story_id = txt_file.stem
        try:
            text = txt_file.read_text(encoding="utf-8")
        except Exception:
            text = txt_file.read_text(encoding="latin-1")

        words = text.split()
        for i in range(0, len(words), CHUNK_SIZE):
            chunk = " ".join(words[i:i + CHUNK_SIZE])
            yield {
                "story_id": story_id,
                "chunk_id": f"{story_id}_chunk{i//CHUNK_SIZE + 1}",
                "chunk_number": i // CHUNK_SIZE + 1,
                "text": chunk,
            }


def main():
    rows = list(generate_rows())
    if not rows:
        print("❌ No chunks generated")
        return

    table = pw.debug.table_from_rows(
        rows,
        schema=StoryChunkSchema
    )

    pw.io.csv.write(table, OUTPUT_CSV)
    print(f"✅ Pathway ingestion completed → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()