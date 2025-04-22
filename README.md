# LLM Prisoner's Dilemma Forgiveness Study

This project investigates how different Large Language Models (LLMs) respond to betrayal in an iterated Prisoner's Dilemma game, specifically focusing on their "forgiveness" patterns after experiencing defection.

## Models Tested
- GPT-4
- GPT-3.5-turbo
- Claude-3-opus
- Gemini-1.5-pro

## Project Structure
```
llm_pd_forgiveness/
├── scripts/
│   ├── run_simulation.py    # Main simulation script
│   └── prompt_templates/    # Contains game prompts
├── results/                 # Simulation results (JSON)
└── logs/                    # Detailed execution logs
```

## Setup
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install openai anthropic google-generativeai python-dotenv
   ```
3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_key_here
   ANTHROPIC_API_KEY=your_key_here
   GOOGLE_API_KEY=your_key_here
   ```

## Running the Simulation
```bash
cd scripts
python run_simulation.py
```

The simulation runs 100 rounds of Prisoner's Dilemma for each model, with a scripted opponent that defects in round 10. Results are saved in the `results` directory, with detailed logs in the `logs` directory.

## Analysis
The simulation tracks:
- Post-defection cooperation rate
- Number of rounds until cooperation resumes
- Complete game history with all moves

## Note
This is an experimental research project investigating LLM behavior in game theory scenarios. Results may vary based on model versions and prompting strategies. 