import os
import zipfile
import io
from pathlib import Path
import pandas as pd

# ---------- SETTINGS ----------
DATA_ZIP = "data/Books-20260106T140352Z-1-001.zip"
OUTPUT_FOLDER = "processed"
OUTPUT_FILE = "processed/story_chunks.csv"
CHUNK_SIZE = 350  # words per chunk (between 300-400 as requested)
# ------------------------------


def ensure_output_folder():
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)


def extract_zip_to_folder(zip_path: str, extract_to: str) -> list:
    """Extracts zip and returns list of extracted text file paths."""
    extracted_files = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.filename.endswith(".txt"):
                target_path = Path(extract_to) / Path(info.filename).name
                with zf.open(info.filename) as fsrc, open(target_path, "wb") as fdst:
                    fdst.write(fsrc.read())
                extracted_files.append(str(target_path))
    return extracted_files


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def build_chunks_from_text_files(txt_paths: list) -> pd.DataFrame:
    all_rows = []

    for txt_path in txt_paths:
        story_id = Path(txt_path).stem
        try:
            with open(txt_path, "r", encoding="utf-8") as fh:
                text = fh.read().strip()
        except Exception:
            # fallback to latin-1 if utf-8 fails
            with open(txt_path, "r", encoding="latin-1") as fh:
                text = fh.read().strip()

        if not text:
            continue

        chunks = split_text_into_chunks(text, CHUNK_SIZE)

        for i, chunk in enumerate(chunks, start=1):
            all_rows.append(
                {
                    "story_id": story_id,
                    "chunk_id": f"{story_id}_chunk{i}",
                    "chunk_number": i,
                    "text": chunk,
                }
            )

    df = pd.DataFrame(all_rows)
    return df


def write_df_with_pathway_if_available(df: pd.DataFrame, out_path: str):
    """Try to use Pathway for writing CSV if available; otherwise fall back to pandas.

    The code attempts to import `pathway` and use a best-effort API. If Pathway
    is not present or the expected methods are not found, this safely falls back
    to `DataFrame.to_csv` so the pipeline remains runnable on Windows.
    """
    try:
        import pathway as pw

        # Best-effort: if Pathway provides a simple CSV write API, use it.
        # Many Pathway installs won't expose a direct write helper; in that
        # case, fall back to pandas.
        if hasattr(pw, "io") and hasattr(pw.io, "write_csv"):
            pw.io.write_csv(df, out_path)
            print(f"✔ Saved with Pathway: {out_path}")
            return
        if hasattr(pw, "write_csv"):
            pw.write_csv(df, out_path)
            print(f"✔ Saved with Pathway: {out_path}")
            return
    except Exception:
        pass

    # Fallback
    df.to_csv(out_path, index=False)
    print(f"✔ Saved with pandas fallback: {out_path}")


def main():
    ensure_output_folder()

    extract_folder = Path(OUTPUT_FOLDER) / "raw_books"
    extract_folder.mkdir(parents=True, exist_ok=True)

    zip_path = Path(DATA_ZIP)
    if not zip_path.exists():
        print(f"ZIP file not found: {zip_path}")
        return

    print(f"Extracting books from: {zip_path} -> {extract_folder}")
    txt_paths = extract_zip_to_folder(str(zip_path), str(extract_folder))

    print(f"Found {len(txt_paths)} text files; building chunks...")

    df_chunks = build_chunks_from_text_files(txt_paths)

    if df_chunks.empty:
        print("No chunks were created — nothing to save.")
        return

    out_path = Path(OUTPUT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    write_df_with_pathway_if_available(df_chunks, str(out_path))

    print(f"Total chunks created: {len(df_chunks)}")


if __name__ == "__main__":
    main()
