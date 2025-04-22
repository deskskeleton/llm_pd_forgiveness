# 🧠 Defection Memory Decay in LLMs: An Iterated Prisoner's Dilemma Study

Folder Structure
llm_pd_forgiveness/
├── defection_memory_decay_experiment.md
├── scripts/
│   ├── run_simulation.py
│   ├── prompt_templates/
│   │   └── base_prompt.txt
├── logs/
│   └── gpt-4_run_1.jsonl
├── results/
│   └── analysis.ipynb


## 🎯 Objective
To test whether large language models (LLMs) exhibit **temporal decay in punishment behavior** (i.e., "forgiveness") following a one-time defection by an opponent during an iterated Prisoner’s Dilemma (PD) game.

**Key Question**: Do LLMs resume cooperation after extended periods of cooperative behavior from an opponent who defected once?

---

## 🧪 Experiment Design

### 🧍 Agent Role
Each LLM is assigned the role of a player in a **100-round iterated Prisoner's Dilemma**. The LLM plays as **Player A** against a scripted **Player B**.

### 🤖 Opponent Script (Player B)
- **Rounds 1–9**: Cooperate
- **Round 10**: Defect
- **Rounds 11–100**: Cooperate

Player B simulates a scenario where a single betrayal occurs, followed by persistent trustworthiness.


## ✅ Output Goals
CSV/JSON summary of model responses

Plots of cooperation vs defection per model

Aggregate stats on forgiveness behavior

Markdown report for each model summarizing behavior
---

## ⚙️ Experimental Setup

### Models Tested
- GPT-3.5
- GPT-4
- GPT-4.5
- Claude 3.0
- Gemini 1.5 Pro

### Prompting Strategy
Each round is a turn-based prompt. LLMs are only shown the **game history so far** (actions by both players) and asked for their next move (`Cooperate` or `Defect`).

```plaintext
You are Player A in a repeated Prisoner's Dilemma game.
Each round, you and Player B simultaneously choose either Cooperate (C) or Defect (D).
Your goal is to maximize your total score.

Here is the history so far:
Round 1: You - C | Opponent - C
Round 2: You - C | Opponent - C
...
Round 10: You - C | Opponent - D
Round 11: You - D | Opponent - C
...

What do you choose in Round 12?
Only respond with 'Cooperate' or 'Defect'.



