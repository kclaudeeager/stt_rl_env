import logging
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
        logger.info(f"✓ Training complete - Loss: {train_out['train_loss']:.4f}, Status: {train_out['status']}")

        logger.info(f"Running evaluation...")
        val_wer = evaluate(self.model, self.val_loader)
        logger.info(f"✓ Validation complete - WER: {val_wer:.4f}")

        reward = compute_reward(
            self.prev_wer, val_wer, train_out["status"]
        )
        logger.info(f"✓ Reward computed: {reward:.4f} (previous WER: {self.prev_wer:.4f})")

        self.prev_wer = val_wer
        self.step_count += 1

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

        if self.step_count >= self.max_steps:
            logger.info(f"✓ Max steps reached ({self.max_steps})")
            self.done = True

        logger.info(f"{'─'*60}\n")
        return self.state, reward, self.done

    def _apply_action(self, action: Action):
        logger.debug(f"Applying action: {action.value}")
        if action == Action.LR_UP:
            old_lr = self.config.lr
            self.config.lr *= 1.5
            logger.debug(f"  LR_UP: {old_lr:.6f} → {self.config.lr:.6f}")
        elif action == Action.LR_DOWN:
            old_lr = self.config.lr
            self.config.lr /= 1.5
            logger.debug(f"  LR_DOWN: {old_lr:.6f} → {self.config.lr:.6f}")
        elif action == Action.BATCH_UP:
            old_bs = self.config.batch_size
            self.config.batch_size *= 2
            logger.debug(f"  BATCH_UP: {old_bs} → {self.config.batch_size}")
        elif action == Action.BATCH_DOWN:
            old_bs = self.config.batch_size
            self.config.batch_size = max(1, self.config.batch_size // 2)
            logger.debug(f"  BATCH_DOWN: {old_bs} → {self.config.batch_size}")
        elif action == Action.FREEZE_MORE:
            old_fl = self.config.frozen_layers
            self.config.frozen_layers += 1
            logger.debug(f"  FREEZE_MORE: {old_fl} → {self.config.frozen_layers}")
        elif action == Action.FREEZE_LESS:
            old_fl = self.config.frozen_layers
            self.config.frozen_layers = max(0, self.config.frozen_layers - 1)
            logger.debug(f"  FREEZE_LESS: {old_fl} → {self.config.frozen_layers}")
