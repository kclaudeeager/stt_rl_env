"""
Real Whisper training loop.
Supports both real model training and fake mode for testing.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
import os
from training.config import TrainingConfig


def run_training_step(model, train_loader, config: TrainingConfig, num_steps=200) -> dict:
    """
    Run a short training window.
    
    Args:
        model: Whisper model (can be None for fake training)
        train_loader: DataLoader
        config: TrainingConfig with lr, batch_size, etc.
        num_steps: Number of training steps to run
    
    Returns:
        dict with train_loss and status ("ok", "oom", "nan", "diverged")
    """
    
    # Fallback fake training when model or train_loader not provided
    if model is None or train_loader is None:
        import random
        train_loss = random.uniform(1.0, 3.0)

        if config.batch_size > 64:
            return {"train_loss": float("inf"), "status": "oom"}

        if random.random() < 0.02:
            return {"train_loss": float("nan"), "status": "nan"}

        return {"train_loss": train_loss, "status": "ok"}
    
    try:
        model.train()

        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=config.lr)

        total_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(train_loader):
            if step >= num_steps:
                break

            try:
                # Expect batch dict with 'input_values' or 'input_features' and 'labels' or 'input_ids'
                inputs = None
                labels = None

                if isinstance(batch, dict):
                    inputs = batch.get("input_values") or batch.get("input_features")
                    labels = batch.get("labels") or batch.get("input_ids")

                if inputs is None or labels is None:
                    # skip malformed batch
                    continue

                # Convert to tensors if necessary
                if not torch.is_tensor(inputs):
                    try:
                        inputs = torch.tensor(inputs)
                    except Exception:
                        continue
                if not torch.is_tensor(labels):
                    try:
                        labels = torch.tensor(labels)
                    except Exception:
                        continue

                device = next(model.parameters()).device
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(input_features=inputs, labels=labels)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

                if torch.isnan(loss):
                    return {"train_loss": float("nan"), "status": "nan"}

                optimizer.zero_grad()
                loss.backward()

                # gradient accumulation
                if config.grad_accum and config.grad_accum > 1:
                    if (step + 1) % config.grad_accum == 0:
                        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                        optimizer.step()
                else:
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                    optimizer.step()

                total_loss += float(loss.item())
                num_batches += 1

            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" in msg:
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    return {"train_loss": float("inf"), "status": "oom"}
                else:
                    return {"train_loss": float("inf"), "status": "diverged"}

        avg_loss = total_loss / max(num_batches, 1)

        # Save lightweight checkpoint
        try:
            ckpt_dir = "checkpoints"
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, "whisper_last.pt")
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, ckpt_path)
        except Exception:
            pass

        return {"train_loss": avg_loss, "status": "ok"}
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            return {"train_loss": float("inf"), "status": "oom"}
        else:
            return {"train_loss": float("inf"), "status": "diverged"}
