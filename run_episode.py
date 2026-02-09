from asyncio.log import logger
import os
import json
from dotenv import load_dotenv

from env.environment import STTRLenv
from agent.llm_agent import LLMAgent
from agent.mock_agent import MockAgent
from training.model import WhisperModel

# Load .env file
load_dotenv()


def main(use_mock=False):
    """
    Run an episode.
    
    Args:
        use_mock: If True, use MockAgent. If False, use real Groq LLMAgent.
    """
    # Load FLEURS dataset with Whisper model
    print("Initializing FLEURS dataset and Whisper model...")
    try:
        from data.loader import FLEURSLoader
        loader = FLEURSLoader(repo_id="google/fleurs", config="en_us")
        loader.prepare()
        print("✓ FLEURS dataset loaded")
        train_loader = loader.get_split_samples("train")
        val_loader = loader.get_split_samples("validation")
    except Exception as e:
        print(f"✗ Failed to load FLEURS: {e}")
        print("Falling back to None loaders for mock testing")
        train_loader = None
        val_loader = None

    # Load Whisper model (FLEURS variant)
    try:
        model = WhisperModel(
            model_id="Dafisns/whisper-turbo-multilingual-fleurs",
            language="english"
        )
        print("✓ Whisper model loaded")
    except Exception as e:
        print(f"✗ Failed to load Whisper model: {e}")
        model = None

    env = STTRLenv(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        max_steps=5,
    )

    # Use mock agent for testing, real Groq agent otherwise
    if use_mock:
        print("Using MockAgent")
        agent = MockAgent(seed=42)
    else:
        print("Using real Groq LLMAgent")
        agent = LLMAgent(model="llama-3.3-70b-versatile")

    trajectory = []

    while not env.done:
        state = env.state
        action = agent.act(state)

        next_state, reward, done = env.step(action)
        logger.info(f"Reward: {reward:.3f}")
        logger.info(f"Next State: {next_state.to_dict()} | Done: {done}")
        trajectory.append(
            {
                "state": state.to_dict(),
                "action": action.value,
                "reward": reward,
            }
        )

        print(f"ACTION={action.value} | REWARD={reward:.3f}")

    print("Episode finished")
    
    # Save trajectory to JSON
    with open("trajectory.json", "w") as f:
        json.dump(trajectory, f, indent=2)
    
    print(f"Trajectory saved to trajectory.json ({len(trajectory)} steps)")
    
    return trajectory


if __name__ == "__main__":
    import sys
    # Default to real Groq agent for full episode
    # Use: uv run python run_episode.py --mock for testing
    use_mock = "--mock" in sys.argv
    main(use_mock=use_mock)
