import os
import subprocess


def run_step(title, command):
    print("\n" + "="*60)
    print(f" RUNNING: {title}")
    print("="*60)

    result = subprocess.run(command, shell=True)

    if result.returncode != 0:
        print(f"\n❌ Failed at step: {title}")
        exit(1)

    print(f"\n✔ Completed: {title}")


def main():

    run_step("Step 1 — Story Chunk Preparation",
             "python scripts/load_and_split.py")

    run_step("Step 2 — Build Retrieval & FAISS Index",
             "python scripts/retrieval_engine.py")

    run_step("Step 3 — Backstory Claim Processing",
             "python scripts/backstory_claim_engine.py")

    print("\n🎯 Pipeline Complete — Final Results Generated at:")
    print("➡ outputs/results.csv")


if __name__ == "__main__":
    main()
