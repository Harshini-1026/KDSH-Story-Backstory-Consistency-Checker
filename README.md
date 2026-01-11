# KDSH Story–Backstory Consistency Checker

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Run the pipeline:
   python app.py

## Output
- Generates results.csv containing predictions for each test example.

## Notes
- Chunking and ingestion are handled using Pathway.
- Retrieval and reasoning are offline for reproducibility.