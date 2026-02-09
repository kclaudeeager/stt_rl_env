from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class State:
    # Hyperparameters
    lr: float
    batch_size: int
    grad_accum: int
    frozen_layers: int

    # Metrics
    train_loss: float
    val_wer: float

    # System
    gpu_mem_gb: float
    status: str  # "ok", "oom", "nan", "diverged"

    def to_dict(self) -> Dict:
        return asdict(self)
