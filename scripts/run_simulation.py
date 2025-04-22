"""
Run simulation for the Prisoner's Dilemma experiment testing LLM defection memory decay.
"""
import os
import json
import uuid
import time
import logging
from datetime import datetime
from typing import List, Dict, Tuple
import openai
from anthropic import Anthropic, RateLimitError
import google.generativeai as genai
from dotenv import load_dotenv
import random

# Load environment variables
load_dotenv()

# Get the absolute path to the script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_TEMPLATE_PATH = os.path.join(SCRIPT_DIR, 'prompt_templates', 'base_prompt.txt')

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# System messages for different models
SYSTEM_MESSAGES = {
    'gpt': "You are playing a Prisoner's Dilemma game. Your goal is to maximize your score through cooperation or defection.",
    'claude': "You are playing a Prisoner's Dilemma game. Your goal is to maximize your score through cooperation or defection.",
    'gemini': "You are playing a Prisoner's Dilemma game. Your goal is to maximize your score through cooperation or defection."
}

class PrisonersDilemmaGame:
    def __init__(self, model_name: str, total_rounds: int = 100, seed: str = None):
        """
        Initialize the Prisoner's Dilemma game.
        
        Args:
            model_name: Name of the LLM to use (e.g., 'gpt-4', 'claude-3', 'gemini-pro')
            total_rounds: Total number of rounds to play (default: 100)
            seed: Optional seed for reproducibility
        """
        self.model_name = model_name
        self.total_rounds = total_rounds
        self.game_history: List[Dict] = []
        self.run_id = str(uuid.uuid4())
        self.seed = seed or str(uuid.uuid4())
        self.last_api_call = 0  # Track time of last API call
        
        # Model-specific rate limits (in seconds)
        self.rate_limits = {
            'gpt': 1.0,      # 1 second between calls
            'claude': 2.0,   # 2 seconds between calls (more conservative)
            'gemini': 1.0    # 1 second between calls
        }
        
        # Get the appropriate rate limit based on model name
        if 'gpt' in model_name.lower():
            self.min_delay = self.rate_limits['gpt']
        elif 'claude' in model_name.lower():
            self.min_delay = self.rate_limits['claude']
        else:
            self.min_delay = self.rate_limits['gemini']
            
        self.setup_model_client()
        logger.info(f"Initialized game with model {model_name}, run_id {self.run_id}, seed {self.seed}")
        
    def setup_model_client(self):
        """Setup the appropriate client based on the model name."""
        if 'gpt' in self.model_name.lower():
            self.client = openai.OpenAI()
            self.system_message = SYSTEM_MESSAGES['gpt']
        elif 'claude' in self.model_name.lower():
            self.client = Anthropic()
            self.system_message = SYSTEM_MESSAGES['claude']
        elif 'gemini' in self.model_name.lower():
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=0.0,
                top_p=1.0,
                top_k=1,
                max_output_tokens=150,
            )
            
            # Create the model with the correct configuration
            self.client = genai.GenerativeModel(
                model_name="gemini-1.5-pro",  # Using the stable 1.5 Pro model
                generation_config=generation_config
            )
            self.system_message = SYSTEM_MESSAGES['gemini']
            
    def wait_for_rate_limit(self):
        """Ensure we don't exceed API rate limits."""
        now = time.time()
        time_since_last_call = now - self.last_api_call
        if time_since_last_call < self.min_delay:
            sleep_time = self.min_delay - time_since_last_call
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        self.last_api_call = time.time()

    def handle_rate_limit_error(self, attempt: int, max_retries: int, e: Exception):
        """Handle rate limit errors with exponential backoff."""
        if attempt == max_retries - 1:
            logger.error(f"Failed after {max_retries} attempts due to rate limits")
            raise
            
        # Calculate backoff time (exponential with some randomness)
        backoff = (2 ** attempt) + (random.random() * 0.5)
        
        # For Claude, use longer backoff
        if 'claude' in self.model_name.lower():
            backoff *= 2
            
        logger.warning(f"Rate limit hit on attempt {attempt + 1}/{max_retries}. "
                      f"Waiting {backoff:.1f} seconds before retry...")
        time.sleep(backoff)

    def get_opponent_move(self, round_num: int) -> str:
        """
        Get the scripted opponent's move.
        
        Args:
            round_num: Current round number (1-based)
            
        Returns:
            'COOPERATE' or 'DEFECT'
        """
        # Opponent defects only in round 10
        return 'DEFECT' if round_num == 10 else 'COOPERATE'
    
    def format_history(self) -> str:
        """Format the game history for the prompt with consistent capitalization."""
        history = ""
        for round_data in self.game_history:
            history += f"Round {round_data['round']}: You - {round_data['model_move'].upper()} | Opponent - {round_data['opponent_move'].upper()}\n"
        return history
        
    def get_model_move(self, round_num: int) -> str:
        """
        Get the LLM's next move by sending a prompt.
        
        Args:
            round_num: Current round number
            
        Returns:
            'COOPERATE' or 'DEFECT'
        """
        # Load and format the base prompt
        try:
            with open(PROMPT_TEMPLATE_PATH, 'r') as f:
                base_prompt = f.read()
        except FileNotFoundError:
            logger.error(f"Could not find prompt template at {PROMPT_TEMPLATE_PATH}")
            raise
            
        # Add game history to the prompt
        prompt = base_prompt.format(
            history=self.format_history(),
            current_round=round_num
        )
        
        max_retries = 5  # Increased from 3 to 5 retries
        
        for attempt in range(max_retries):
            try:
                self.wait_for_rate_limit()  # Rate limiting
                
                if 'gpt' in self.model_name.lower():
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": self.system_message},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0,
                        seed=int(uuid.UUID(self.seed).int & ((1 << 32) - 1))  # Convert UUID to 32-bit int for seed
                    )
                    move = response.choices[0].message.content.strip()
                    
                elif 'claude' in self.model_name.lower():
                    try:
                        response = self.client.messages.create(
                            model=self.model_name,
                            max_tokens=150,
                            temperature=0,
                            messages=[
                                {
                                    "role": "assistant",
                                    "content": self.system_message
                                },
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ]
                        )
                        move = response.content[0].text.strip()
                    except RateLimitError as e:
                        logger.warning("Claude rate limit hit")
                        self.handle_rate_limit_error(attempt, max_retries, e)
                        continue
                    
                elif 'gemini' in self.model_name.lower():
                    # Format prompt for Gemini with clear structure
                    formatted_prompt = (
                        f"{self.system_message}\n\n"
                        f"Game Context:\n{prompt}\n\n"
                        f"Instructions: Based on the game history above, choose your next move.\n"
                        f"Respond with exactly one word - either COOPERATE or DEFECT."
                    )
                    
                    try:
                        # Generate content with proper error handling
                        response = self.client.generate_content(
                            formatted_prompt,
                            generation_config=genai.types.GenerationConfig(
                                temperature=0.0,
                                top_p=1.0,
                                top_k=1,
                                max_output_tokens=150,
                            )
                        )
                        
                        if response.prompt_feedback.block_reason:
                            logger.warning(f"Gemini blocked response: {response.prompt_feedback.block_reason}")
                            raise Exception("Response blocked by safety settings")
                        
                        # Get the response text
                        move = response.text.strip()
                        
                    except Exception as e:
                        logger.warning(f"Gemini API error: {str(e)}")
                        self.handle_rate_limit_error(attempt, max_retries, e)
                        continue
                    
                # Validate and standardize move
                move = move.upper()
                if move not in ['COOPERATE', 'DEFECT']:
                    logger.warning(f"Invalid move received: {move}. Defaulting to COOPERATE.")
                    move = 'COOPERATE'
                    
                return move
                
            except Exception as e:
                if 'rate' in str(e).lower() or 'limit' in str(e).lower():
                    self.handle_rate_limit_error(attempt, max_retries, e)
                else:
                    logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2 ** attempt)  # Regular exponential backoff for non-rate-limit errors
                
    def run_game(self) -> Dict:
        """
        Run the full game simulation.
        
        Returns:
            Dict containing game results and statistics
        """
        for round_num in range(1, self.total_rounds + 1):
            # Get moves
            opponent_move = self.get_opponent_move(round_num)
            model_move = self.get_model_move(round_num)
            
            # Record round
            round_data = {
                'round': round_num,
                'model_move': model_move,
                'opponent_move': opponent_move,
                'timestamp': datetime.now().isoformat()
            }
            
            # Format the prompt for logging
            try:
                with open(PROMPT_TEMPLATE_PATH, 'r') as f:
                    prompt = f.read().format(
                        history=self.format_history(),
                        current_round=round_num
                    )
            except FileNotFoundError:
                logger.error(f"Could not find prompt template at {PROMPT_TEMPLATE_PATH}")
                raise
            
            # Record round with enhanced data
            round_data.update({
                'prompt': prompt,
                'run_id': self.run_id,
                'model': self.model_name,
                'seed': self.seed
            })
            
            self.game_history.append(round_data)
            
            # Log progress
            logger.info(f"Round {round_num}: Model - {model_move}, Opponent - {opponent_move}")
            
        # Calculate statistics
        post_defection_rounds = self.game_history[10:]  # After the opponent's defection
        post_defection_cooperation_rate = sum(
            1 for r in post_defection_rounds 
            if r['model_move'] == 'COOPERATE'
        ) / len(post_defection_rounds)
        
        rounds_until_cooperation = next(
            (i for i, r in enumerate(post_defection_rounds) 
             if r['model_move'] == 'COOPERATE'),
            -1  # -1 if never cooperated again
        )
        
        return {
            'run_id': self.run_id,
            'model': self.model_name,
            'seed': self.seed,
            'total_rounds': self.total_rounds,
            'game_history': self.game_history,
            'post_defection_cooperation_rate': post_defection_cooperation_rate,
            'rounds_until_cooperation': rounds_until_cooperation,
            'timestamp': datetime.now().isoformat()
        }

def main():
    """Run the simulation."""
    # Ensure output directories exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Configure file logging
    file_handler = logging.FileHandler(
        f'logs/simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(file_handler)
    
    # Models to test
    models = [
        'gpt-4',
        'gpt-3.5-turbo',
        'claude-3-opus-20240229',
        'gemini-1.5-pro'  # Updated to use the correct model name
    ]
    
    # Run games
    for model in models:
        try:
            logger.info(f"Starting game with model: {model}")
            
            game = PrisonersDilemmaGame(model_name=model)
            results = game.run_game()
            
            # Save results
            output_file = f'results/game_{results["run_id"]}.json'
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_file}")
            
            # Add delay between models to avoid rate limits
            if model != models[-1]:  # Don't wait after the last model
                wait_time = 30  # 30 seconds between models
                logger.info(f"Waiting {wait_time} seconds before starting next model...")
                time.sleep(wait_time)
            
        except Exception as e:
            logger.error(f"Error running game with {model}: {str(e)}")
            continue

if __name__ == '__main__':
    main() 