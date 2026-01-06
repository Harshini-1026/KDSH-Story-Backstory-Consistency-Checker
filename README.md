# KDSH – Story & Backstory Consistency Checker

This project is built for the Kharagpur Data Science Hackathon (Track-A).

The system evaluates whether a hypothetical backstory of a character is
logically consistent with the events that occur in a long narrative story.

Given:
✔ a full novel (long story)
✔ a hypothetical backstory of a character

The system outputs:

1 = Backstory is consistent with story events
0 = Backstory contradicts the story

The project is designed to be beginner-friendly,
explainable, and rule-based — not a black-box ML system.

---

## 🔎 Project Workflow (Simple Explanation)

1) Story is split into small chunks (paragraph / scene level)
2) Backstory is broken into small claims
3) For each claim → related story passages are retrieved
4) Evidence from multiple parts of the narrative is collected
5) Rule-based reasoning decides:

   ✔ supports backstory → Consistent (1)
   ❌ contradicts backstory → Inconsistent (0)

Final results are saved in: outputs/results.csv

---

## 👥 Team Roles (Neutral Representation)

🟢 Team Member 1 — Story Chunk Preparation
Script: scripts/load_and_split.py

🟣 Team Member 2 — Retrieval & FAISS Search
Script: scripts/retrieval_engine.py

🔵 Team Member 3 — Claim Extraction & Pipeline Runner
Script: scripts/backstory_claim_engine.py

🟠 Team Member 4 — Consistency Decision Engine
Script: scripts/consistency_checker.py

Root-level Runner
app.py → executes full pipeline in one command

---

## 📂 Project Structure

KDSH-Story-Backstory-Consistency-Checker/

 ├─ data/
 │   ├─ train.csv
 │   ├─ test.csv

 ├─ processed/
 │   ├─ train_chunks.csv
 │   ├─ test_chunks.csv
 │   ├─ retrieval_outputs/
 │   ├─ claims/

 ├─ outputs/
 │   ├─ results.csv

 ├─ scripts/
 │   ├─ load_and_split.py
 │   ├─ retrieval_engine.py
 │   ├─ backstory_claim_engine.py
 │   ├─ consistency_checker.py

 ├─ app.py        ← Root runner (executes full pipeline)
 ├─ README.md
 ├─ requirements.txt
 ├─ .gitignore

---

## ▶️ How to Run (Single Command)

Install requirements:

pip install -r requirements.txt

Run full pipeline:

python app.py

This will automatically:

1) prepare story chunks
2) build retrieval search index
3) process backstory claims
4) generate final results

Output file generated:

outputs/results.csv

Format:

story_id , prediction , rationale

---

## 🧠 Why our system is unique

✔ Evidence from multiple story regions
✔ Meaning-based semantic retrieval
✔ Explainable rule-based reasoning
✔ Long-context narrative handling
✔ Beginner-friendly modular workflow
