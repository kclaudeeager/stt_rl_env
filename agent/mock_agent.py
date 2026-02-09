"""Mock agent for testing RL loop without LLM calls."""

import random
from env.actions import Action


class MockAgent:
    """Simulates an agent that makes random valid actions for testing."""

    def __init__(self, seed=42):
        self.seed = seed
        random.seed(seed)

    def act(self, state) -> Action:
        """Return a random action (excluding STOP until later)."""
        # Early on, avoid STOP
        actions = [a for a in Action if a != Action.STOP]
        return random.choice(actions)
