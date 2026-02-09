import logging
from env.actions import Action

logger = logging.getLogger(__name__)


def parse_action(text: str) -> Action:
    """
    Parse LLM response into an Action.
    
    Args:
        text: Raw text response from LLM
        
    Returns:
        Action: Parsed action enum
        
    Raises:
        ValueError: If text doesn't match any action
    """
    text = text.strip().lower()
    logger.debug(f"Parsing action from text: '{text}'")

    for action in Action:
        if text == action.value:
            logger.debug(f"✓ Matched action: {action.value}")
            return action

    # Log available actions for debugging
    available = [a.value for a in Action]
    logger.error(f"No match found for '{text}'")
    logger.error(f"Available actions: {available}")
    raise ValueError(f"Invalid action from agent: {text}")
