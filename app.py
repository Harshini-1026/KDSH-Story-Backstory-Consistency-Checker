import subprocess
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")


def run_step(name, script_name):
    script_path = os.path.join(SCRIPTS_DIR, script_name)

    if not os.path.exists(script_path):
        print(f"\n❌ Script not found: {script_path}")
        sys.exit(1)

    print("\n" + "="*65)
    print(f" RUNNING: {name}")
    print("="*65)

    result = subprocess.run(f"python \"{script_path}\"", shell=True)

    if result.returncode != 0:
        print(f"\n❌ FAILED at step: {name}")
        sys.exit(1)

    print(f"\n✔ COMPLETED: {name}")


def main():

    run_step("Step 1 — Story Chunk Preparation",
             "load_and_split.py")

    run_step("Step 2 — Retrieval & FAISS Index Build",
             "retrieval_engine.py")

    run_step("Step 3 — Backstory Claim Processing",
             "backstory_claim_engine.py")

    print("\n🎯 Pipeline Finished Successfully")
    print("➡ Final results available at: outputs/results.csv")


if __name__ == "__main__":
    main()
