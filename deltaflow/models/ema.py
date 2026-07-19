"""Exponential moving average of model weights."""

import torch.nn as nn


class EMA:
    """Exponential moving average helper for a shadow copy of a model.

    Usage::

        ema = EMA(beta=0.995)
        ema_model = copy.deepcopy(model).eval().requires_grad_(False)
        # after every optimizer step:
        ema.update_model_average(ema_model, model)
    """

    def __init__(self, beta: float = 0.995):
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def update_model_average(self, ema_model: nn.Module, model: nn.Module) -> None:
        for ema_params, current_params in zip(ema_model.parameters(), model.parameters()):
            old, new = ema_params.data, current_params.data
            ema_params.data = self.update_average(old, new)
