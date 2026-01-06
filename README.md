# KDSH – Story & Backstory Consistency Checker

This project is built for the Kharagpur Data Science Hackathon.

Given:
✔ a full novel (long story)
✔ a hypothetical backstory of a character

Our system checks whether the backstory is:

1 = Consistent with story events
0 = Contradicting the story

The project is designed to be beginner-friendly,
explainable, and rule-based — not a black-box ML system.

---

## 🔎 Project Workflow (Simple Explanation)

1) Story is split into small chunks (paragraph-level)
2) Backstory is broken into claims
3) For each claim → related story passages are retrieved
4) Evidence from multiple story regions is collected
5) Rule-based reasoning decides:

   ✔ supports backstory → Consistent (1)
   ❌ contradicts backstory → Inconsistent (0)

Final results are saved in `outputs/results.csv`

---

## 👥 Team Roles (Who works on what)

### 🟢 Team Member 1 — Story Chunk Preparation
Script:
`scripts/load_and_split.py`

Input:
`data/train.csv`, `data/test.csv`

Output:
`processed/train_chunks.csv`
`processed/test_chunks.csv`

---

### 🟣 Team Member 2 — Retrieval & FAISS Search
Script:
`scripts/retrieval_engine.py`

Input:
`processed/train_chunks.csv`

Output:
`processed/retrieval_outputs/`

---

### 🔵 Team Member 3 — Claim Extraction & Pipeline Runner
Script:
`scripts/backstory_claim_engine.py`

Output:
`processed/claims/`

---

### 🟠 Team Member 4 — Consistency Decision Engine
Script:
`scripts/consistency_checker.py`

Final Output:
`outputs/results.csv`

---

## 📂 Project Structure

(keep same as folder tree above)

---

## ▶️ How to Run (Step-by-Step)

1) Run Team Member 1 script  
   → prepares story chunks

2) Run Team Member 2 script  
   → builds search system

3) Run Team Member 3 script  
   → sends claims through pipeline

4) Team Member 4 module automatically  
   → generates predictions & rationale

---

## 🏁 Final Output Format

story_id | prediction | rationale

Example:

1 | 1 | Backstory aligns with later character actions  
2 | 0 | Story behavior contradicts assumed traits

---

## 🧠 Why our system is unique

✔ Evidence from multiple story regions  
✔ Explainable rule-based reasoning  
✔ Long-context narrative handling  
✔ Beginner-friendly workflow
