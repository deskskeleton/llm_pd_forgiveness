"""
Test script for the Prisoner's Dilemma experiment with a single model and fewer rounds.
"""
import os
import json
import time
import logging
from datetime import datetime
from run_simulation import PrisonersDilemmaGame

# Configure logging specifically for the test
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True  # Override the logging config from the main script
)
logger = logging.getLogger(__name__)

def run_test():
    """Run a test simulation with GPT-3.5 Turbo."""
    # Test configuration
    model_name = 'gpt-3.5-turbo'
    test_rounds = 15  # Enough rounds to see post-defection behavior
    
    # Create test results directory
    test_dir = 'results/test_runs'
    os.makedirs(test_dir, exist_ok=True)
    
    print("\n=== Starting Test Run ===")
    print(f"Model: {model_name}")
    print(f"Rounds: {test_rounds}")
    print("Expected behavior: Cooperation until round 9, opponent defects at round 10")
    print("Running test...")
    
    # Initialize and run test game
    game = PrisonersDilemmaGame(model_name, total_rounds=test_rounds)
    
    try:
        results = game.run_game()
        
        # Save test results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"{model_name}_test_{timestamp}_{game.run_id}"
        
        # Save game results
        results_file = f"{test_dir}/{base_filename}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_file}")
        
        # Save prompts
        prompts_file = f"{test_dir}/{base_filename}_prompts.json"
        prompts_data = {
            'model_name': model_name,
            'run_id': game.run_id,
            'seed': game.seed,
            'prompts': [round_data['prompt'] for round_data in results['game_history']]
        }
        with open(prompts_file, 'w') as f:
            json.dump(prompts_data, f, indent=2)
        print(f"Prompts saved to {prompts_file}")
        
        # Print test summary
        print("\n=== Test Results Summary ===")
        print(f"Total rounds played: {test_rounds}")
        print(f"Post-defection cooperation rate: {results['post_defection_cooperation_rate']:.2%}")
        print(f"Rounds until cooperation resumed: {results['rounds_until_cooperation']}")
        print("\nGame History:")
        for round_data in results['game_history']:
            print(f"Round {round_data['round']:2d}: "
                  f"Model: {round_data['model_move']:9s} | "
                  f"Opponent: {round_data['opponent_move']}")
        
        # Cost estimation (approximate)
        avg_tokens_per_round = 200  # Approximate tokens per round
        total_input_tokens = avg_tokens_per_round * test_rounds
        total_output_tokens = 2 * test_rounds  # 2 tokens per response
        
        cost_per_1k_input = 0.0015  # GPT-3.5 Turbo input cost per 1K tokens
        cost_per_1k_output = 0.002  # GPT-3.5 Turbo output cost per 1K tokens
        
        estimated_cost = (
            (total_input_tokens / 1000) * cost_per_1k_input +
            (total_output_tokens / 1000) * cost_per_1k_output
        )
        
        print("\n=== Cost Estimation ===")
        print(f"Estimated total tokens: {total_input_tokens + total_output_tokens}")
        print(f"Estimated cost: ${estimated_cost:.4f}")
        
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        print("Check the logs for more details.")
        raise

if __name__ == "__main__":
    run_test() 