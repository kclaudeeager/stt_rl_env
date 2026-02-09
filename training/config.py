class TrainingConfig:
    def __init__(self, lr=1e-5, batch_size=8, grad_accum=1, frozen_layers=0):
        self.lr = lr
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.frozen_layers = frozen_layers

    def copy(self):
        return TrainingConfig(
            lr=self.lr,
            batch_size=self.batch_size,
            grad_accum=self.grad_accum,
            frozen_layers=self.frozen_layers,
        )
