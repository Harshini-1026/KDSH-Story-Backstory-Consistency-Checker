# app.py

import os

STEPS = [
    ("Pathway Ingestion & Chunking", "python scripts/load_and_split.py"),
    ("Claim Reasoning Pipeline", "python scripts/backstory_claim_engine.py"),
]


def run():
    for name, cmd in STEPS:
        print("\n" + "=" * 70)
        print(f" RUNNING: {name}")
        print("=" * 70)

        status = os.system(cmd)
        if status != 0:
            print(f"❌ FAILED at step: {name}")
            return

    print("\n🎯 FULL PATHWAY PIPELINE COMPLETED")
    print("📁 Output → outputs/results.csv")


if __name__ == "__main__":
    run()