import os
import logging
from dotenv import load_dotenv
from agent.prompt import SYSTEM_PROMPT, STEP_PROMPT_TEMPLATE
from agent.parser import parse_action
from env.actions import Action

# Use Groq for inference (free tier available)
from groq import Groq

load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class LLMAgent:
    def __init__(self, model="llama-3.3-70b-versatile"):
        """
        Initialize Groq agent.
        
        Latest models:
        - llama-3.3-70b-versatile
        - llama-3.1-8b-instant
        """
        api_key = os.environ.get("GROK_API_KEY")
        if not api_key:
            raise ValueError("GROK_API_KEY not set in environment")
        
        logger.info(f"Initializing Groq LLMAgent with model: {model}")
        self.client = Groq(api_key=api_key)
        self.model = model
        self.step_count = 0
        logger.info("✓ Groq client initialized successfully")

    def act(self, state) -> Action:
        """
        Generate the next action based on current state.
        
        Args:
            state: Current environment state
            
        Returns:
            Action: The selected action
        """
        self.step_count += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"STEP {self.step_count}: Agent Decision Making")
        logger.info(f"{'='*60}")
        
        # Log current state
        state_dict = state.to_dict()
        logger.info(f"Current State:")
        for key, value in state_dict.items():
            logger.info(f"  {key}: {value}")
        
        # Get available actions
        available_actions = [a.value for a in Action]
        logger.info(f"Available actions: {available_actions}")
        
        # Build prompt
        prompt = STEP_PROMPT_TEMPLATE.format(
            state=state_dict,
            actions=available_actions,
        )
        logger.debug(f"Prompt sent to Groq:\n{prompt}")
        
        try:
            # Call Groq API
            logger.info(f"Calling Groq API (model: {self.model})...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=50,
            )
            logger.info(f"✓ Groq API response received")
            
            # Extract action text
            text = response.choices[0].message.content
            logger.info(f"Raw response from LLM: '{text}'")
            
            # Parse action
            action = parse_action(text)
            logger.info(f"✓ Action parsed successfully: {action.value}")
            logger.info(f"{'='*60}\n")
            
            return action
            
        except ValueError as e:
            logger.error(f"✗ Action parsing failed: {e}")
            logger.error(f"  Response was: '{text}'")
            logger.warning(f"Falling back to random action")
            import random
            fallback_action = random.choice(list(Action))
            logger.info(f"✓ Fallback action selected: {fallback_action.value}")
            logger.info(f"{'='*60}\n")
            return fallback_action
            
        except Exception as e:
            logger.error(f"✗ Groq API call failed: {e}")
            logger.warning(f"Falling back to random action")
            import random
            fallback_action = random.choice(list(Action))
            logger.info(f"✓ Fallback action selected: {fallback_action.value}")
            logger.info(f"{'='*60}\n")
            return fallback_action
