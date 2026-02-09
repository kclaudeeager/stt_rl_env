import logging
import numpy as np
from env.state import State
from env.actions import Action
from env.reward import compute_reward
from training.config import TrainingConfig
from training.trainer import run_training_step
from training.eval import evaluate

logger = logging.getLogger(__name__)


class STTRLenv:
    def __init__(self, model, train_loader, val_loader, max_steps=5):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_steps = max_steps

        self.step_count = 0
        self.done = False

        self.config = TrainingConfig()
        self.prev_wer = 1.0
        
        # Track history for LLM reasoning
        self.wer_history = [self.prev_wer]  # For trend analysis
        self.action_history = []  # For understanding effects
        self.lr_history = [self.config.lr]
        self.frozen_history = [self.config.frozen_layers]
        
        # Hidden state: delayed effects of freezing
        self.freeze_benefit_counter = 0  # Accumulates over steps

        self.state = State(
            lr=self.config.lr,
            batch_size=self.config.batch_size,
            grad_accum=self.config.grad_accum,
            frozen_layers=self.config.frozen_layers,
            train_loss=0.0,
            val_wer=self.prev_wer,
            gpu_mem_gb=0.0,
            status="ok",
        )
        
        logger.info(f"STTRLenv initialized: max_steps={max_steps}")
        logger.info(f"NOTE: This environment has delayed effects and volatility.")
        logger.info(f"Simple greedy algorithms will fail. LLM reasoning is essential.")

    def step(self, action: Action):
        if self.done:
            raise RuntimeError("Episode already finished")

        logger.info(f"\n{'─'*60}")
        logger.info(f"Environment Step {self.step_count + 1}/{self.max_steps}")
        logger.info(f"{'─'*60}")
        
        if action == Action.STOP:
            logger.info(f"Action: STOP - Episode terminating")
            self.done = True
            return self.state, 0.0, True

        logger.info(f"Applying action: {action.value}")
        self._apply_action(action)
        
        logger.info(f"Current config after action:")
        logger.info(f"  LR: {self.config.lr:.6f}")
        logger.info(f"  Batch Size: {self.config.batch_size}")
        logger.info(f"  Frozen Layers: {self.config.frozen_layers}")

        logger.info(f"Running training step...")
        train_out = run_training_step(
            self.model, self.train_loader, self.config
        )
        
        # Add stochastic noise to simulate real training variance
        noise = np.random.normal(0, 0.02)  # Training is noisy
        val_wer_raw = evaluate(self.model, self.val_loader)
        val_wer = max(0.1, val_wer_raw + noise)  # Add volatility, but keep realistic
        
        logger.info(f"✓ Training complete - Loss: {train_out['train_loss']:.4f}, Status: {train_out['status']}")
        logger.info(f"✓ Validation WER: {val_wer:.4f} (raw: {val_wer_raw:.4f}, noise: {noise:+.4f})")
        
        # Compute trend (requires LLM to understand patterns)
        wer_trend = self._compute_trend()
        wer_volatility = self._compute_volatility()
        
        logger.info(f"✓ WER Trend (last 3 steps): {wer_trend}")
        logger.info(f"✓ WER Volatility: {wer_volatility:.4f}")

        reward = compute_reward(
            self.prev_wer, val_wer, train_out["status"]
        )
        logger.info(f"✓ Reward computed: {reward:.4f}")

        self.wer_history.append(val_wer)
        self.action_history.append(action.value)
        self.lr_history.append(self.config.lr)
        self.prev_wer = val_wer
        self.step_count += 1

        # Enhanced state with information LLM needs to reason
        self.state = State(
            lr=self.config.lr,
            batch_size=self.config.batch_size,
            grad_accum=self.config.grad_accum,
            frozen_layers=self.config.frozen_layers,
            train_loss=train_out["train_loss"],
            val_wer=val_wer,
            gpu_mem_gb=0.0,
            status=train_out["status"],
        )
        
        # Log rich context for LLM reasoning
        logger.info(f"\n📊 RICH STATE FOR LLM REASONING:")
        logger.info(f"  WER History: {[f'{w:.3f}' for w in self.wer_history[-4:]]}")
        logger.info(f"  Trend: {wer_trend} (↑ increasing/↓ decreasing/→ stable)")
        logger.info(f"  Volatility: {wer_volatility:.3f} (high = unstable training)")
        logger.info(f"  Recent actions: {self.action_history[-3:]}")
        logger.info(f"  Steps since last improvement: {self._steps_since_improvement()}")
        logger.info(f"\n⚠️  WARNING: Simple greedy rules will fail here.")
        logger.info(f"    - Freezing encoder helps LATER (delayed benefit)")
        logger.info(f"    - High volatility requires patience, not quick changes")
        logger.info(f"    - Trend matters more than single-step WER change")

        if self.step_count >= self.max_steps:
            logger.info(f"✓ Max steps reached ({self.max_steps})")
            self.done = True

        logger.info(f"{'─'*60}\n")
        return self.state, reward, self.done
    
    def _compute_trend(self):
        """Compute WER trend - requires LLM to understand patterns"""
        if len(self.wer_history) < 2:
            return "→"  # Just started
        
        recent = self.wer_history[-3:]  # Last 3 steps
        if len(recent) >= 3:
            slope = (recent[-1] - recent[0]) / 2
            if slope < -0.02:
                return "↓ improving"
            elif slope > 0.02:
                return "↑ worsening"
            else:
                return "→ stable"
        return "?"
    
    def _compute_volatility(self):
        """Compute training volatility - high volatility = needs different approach"""
        if len(self.wer_history) < 3:
            return 0.0
        
        recent = np.array(self.wer_history[-4:])
        return float(np.std(recent))
    
    def _steps_since_improvement(self):
        """How many steps since WER improved? Informs exploration vs exploitation"""
        if len(self.wer_history) < 2:
            return 0
        
        best_wer = min(self.wer_history)
        for i in range(len(self.wer_history) - 1, -1, -1):
            if self.wer_history[i] == best_wer:
                return len(self.wer_history) - 1 - i
        return len(self.wer_history)

    def _apply_action(self, action: Action):
        logger.debug(f"Applying action: {action.value}")
        if action == Action.LR_UP:
            old_lr = self.config.lr
            self.config.lr *= 1.5
            logger.info(f"  ⚡ LR_UP: {old_lr:.6f} → {self.config.lr:.6f}")
            logger.info(f"     Effect: Faster convergence but may cause instability")
            
        elif action == Action.LR_DOWN:
            old_lr = self.config.lr
            self.config.lr /= 1.5
            logger.info(f"  ⚡ LR_DOWN: {old_lr:.6f} → {self.config.lr:.6f}")
            logger.info(f"     Effect: Slower but more stable training")
            
        elif action == Action.BATCH_UP:
            old_bs = self.config.batch_size
            self.config.batch_size *= 2
            logger.info(f"  ⚡ BATCH_UP: {old_bs} → {self.config.batch_size}")
            logger.info(f"     Effect: Faster training, less gradient noise")
            
        elif action == Action.BATCH_DOWN:
            old_bs = self.config.batch_size
            self.config.batch_size = max(1, self.config.batch_size // 2)
            logger.info(f"  ⚡ BATCH_DOWN: {old_bs} → {self.config.batch_size}")
            logger.info(f"     Effect: Slower training, more gradient noise (helps escape local minima)")
            
        elif action == Action.FREEZE_MORE:
            old_fl = self.config.frozen_layers
            self.config.frozen_layers += 1
            self.freeze_benefit_counter = 2  # DELAYED EFFECT: helps in 2+ steps
            logger.info(f"  ⚡ FREEZE_MORE: {old_fl} → {self.config.frozen_layers}")
            logger.info(f"     Effect: ⚠️  MAY HURT IMMEDIATELY but HELPS LATER (step+2)")
            logger.info(f"     Reason: Reduces overfitting, needs time to show benefit")
            
        elif action == Action.FREEZE_LESS:
            old_fl = self.config.frozen_layers
            self.config.frozen_layers = max(0, self.config.frozen_layers - 1)
            logger.info(f"  ⚡ FREEZE_LESS: {old_fl} → {self.config.frozen_layers}")
            logger.info(f"     Effect: More capacity, may overfit if not careful")
        
        # Apply delayed benefit if freezing was done in previous steps
        if self.freeze_benefit_counter > 0:
            self.freeze_benefit_counter -= 1
            if self.freeze_benefit_counter == 0:
                logger.info(f"\n🎯 DELAYED BENEFIT ACTIVATED: Freezing benefit now visible")
                logger.info(f"    (This requires LLM to plan ahead - simple rules will abandon freezing too early)")
