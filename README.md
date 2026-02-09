# LLM-Driven RL Environment for Whisper STT Fine-tuning

## Overview

This is a **reinforcement learning environment** where an LLM agent (Groq) sequentially chooses hyperparameter actions to optimize Whisper speech-to-text fine-tuning on low-resource Swahili (Afrivoice_Swahili).

**Current Status:** MVP ready with:
- ✅ Complete RL loop (state → action → reward → trajectory)
- ✅ LLM agent (Groq API llama-3.3-70b-versatile)
- ✅ Mock agent for testing without API
- ✅ Judge for hidden test evaluation (government domain)
- ✅ Deterministic dataset splits (health+agri for train/val, government for hidden test)
- ⏳ Real Whisper trainer (ready to integrate)

---

## Quick Start

### 1. Mock Episode (No API calls, instant)
```bash
uv run python run_episode.py --mock
```
Output: 5-step episode with random actions, trajectory saved

### 2. Status Report
```bash
uv run python STATUS.py
```

### 3. Real Groq Agent (Requires internet)
```bash
uv run python run_episode.py
```

---

## The RL Loop

```python
episode:
  for step in range(max_steps):
    state = environment.state              # (lr, batch_size, val_wer, ...)
    action = agent.act(state)              # LLM or random
    apply action to config
    train_loss = trainer.run_training_step()
    val_wer = evaluator.evaluate()
    reward = (prev_wer - val_wer) - penalties
    log(state, action, reward)
    
save trajectory
```

**Key Ideas:**
- LLM sees hyperparameter state + recent WER
- LLM chooses 1 action (e.g., "lr_up")
- That action is applied for N training steps
- WER computed on validation set
- Reward = WER improvement (with OOM/NaN penalties)

---

## Architecture

```
run_episode.py (main)
├─ STTRLenv (env/environment.py)
│  ├─ TrainingConfig
│  ├─ run_training_step (training/trainer.py)
│  └─ evaluate (training/eval.py)
│
├─ LLMAgent (agent/llm_agent.py) or MockAgent
│  ├─ Groq API call
│  └─ Action parsing
│
└─ Trajectory → trajectory.json

judge/judge.py
├─ Load checkpoint
├─ Anti-cheat checks
└─ WER on government (hidden test)

data/loader.py
├─ Afrivoice with HF_TOKEN
└─ Deterministic splits
```

---

## Components

### Core RL
- `env/environment.py`: Main STTRLenv loop
- `env/state.py`: Hyperparameter + WER state
- `env/actions.py`: Action space (7 actions)
- `env/reward.py`: Reward computation

### LLM Agent
- `agent/llm_agent.py`: Groq API (llama-3.3-70b-versatile)
- `agent/mock_agent.py`: Random actions (no API)
- `agent/prompt.py`: System & step prompts
- `agent/parser.py`: Parse LLM output to action

### Training (Ready for Real Whisper)
- `training/config.py`: Learning config
- `training/model.py`: WhisperModel loader
- `training/trainer.py`: Training loop (fake now, real-ready)
- `training/eval.py`: Evaluation (fake now, real-ready)

### Data
- `data/loader.py`: Afrivoice categorical loader
  - Train: health + agriculture (80%)
  - Val: health + agriculture (20%)
  - Test (hidden): government (0% - held-out)

### Judge
- `judge/judge.py`: Hidden test evaluator
  - Loads checkpoint
  - Computes WER on government
  - Anti-cheating: model modification check

---

## Example Output

```
Using MockAgent
ACTION=freeze_less | REWARD=0.577
ACTION=lr_down | REWARD=-0.244
ACTION=lr_up | REWARD=0.190
ACTION=lr_down | REWARD=-0.082
ACTION=freeze_less | REWARD=-0.125
Episode finished
Trajectory saved to trajectory.json (5 steps)
```

---

## Environment Variables

Set in `.env`:
```
GROK_API_KEY=hf_...     # Groq API key (for real agent)
HF_TOKEN=hf_...         # HuggingFace token (for Afrivoice)
```

---

## Configuration

### RL Settings (`env/environment.py`)
- `max_steps`: 5 (actions per episode)
- `step_penalty`: 0.05

### Training (`training/config.py`)
- `lr`: 1e-5
- `batch_size`: 8
- `grad_accum`: 1
- `frozen_layers`: 0

### Dataset (`data/loader.py`)
- `MAX_DURATION_SEC`: 30.0
- Train/Val: 80/20 split on health+agriculture
- Test: government (hidden)

---

## Next: Real Whisper Integration

### 1. Load Data
```bash
uv run python data/loader.py
```
Downloads & splits Afrivoice (requires HF_TOKEN)

### 2. Update Trainer
Replace fake loss in `training/trainer.py`:
```python
# Real loop with actual Whisper training
for batch in train_loader:
    outputs = model(batch["audio"])
    loss = criterion(outputs, batch["text"])
    loss.backward()
```

### 3. Update Evaluator
Replace fake WER in `training/eval.py`:
```python
from jiwer import wer
# Real WER computation
```

### 4. Run Full Episode
```bash
uv run python run_episode.py
```

### 5. Evaluate
```bash
uv run python judge/judge.py
```

---

## Key Design Decisions

1. **Deterministic Splits**: Reproducible 80/20, government hidden
2. **Fake-to-Real**: MVP fake, swappable to real trainer/eval
3. **Groq API**: Free tier, compatible with budget
4. **JSON Trajectories**: Easy analysis & reproduction
5. **Anti-Cheating**: Judge verifies model modification

---

## Files

```
├── run_episode.py          # Main
├── STATUS.py               # Implementation report
├── test_whisper_baseline.py
│
├── env/
│   ├── environment.py, state.py, actions.py, reward.py
│
├── agent/
│   ├── llm_agent.py, mock_agent.py, prompt.py, parser.py
│
├── training/
│   ├── config.py, model.py, trainer.py, eval.py, eval_real.py
│
├── data/
│   ├── loader.py, split_indices.json (generated)
│
├── judge/
│   └── judge.py
│
├── pyproject.toml, requirements.txt, .env
└── trajectory.json (generated)
```

---

## Testing

```bash
uv run python run_episode.py --mock       # Mock agent
uv run python STATUS.py                   # Status
uv run python test_whisper_baseline.py    # Model load test
uv run python data/loader.py              # Data load (requires HF auth)
uv run python judge/judge.py              # Judge test
```

---

## Application Writing

Once real Whisper loop runs:

1. Run 3-5 episodes
2. Document:
   - Best WER achieved
   - Which actions helped
   - Domain shift effects
   - Failure modes
3. Write 500-word application essay

---

## Dependencies

```
torch, torchaudio, transformers  # Whisper
datasets                         # Afrivoice
groq                            # LLM agent
jiwer                           # WER metric
numpy<2                         # Compatibility
python-dotenv                   # .env loading
```

---

**Status:** MVP complete, ready for real Whisper fine-tuning integration

**Goal:** True end-to-end RL loop for STT optimization on low-resource languages

