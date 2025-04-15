from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    SupportsFloat,
    TypedDict,
    Union,
)


import torch
import torch.nn.functional as F

from torch import inf, Tensor

from tqdm import tqdm


from util.utils import get_device

device = get_device()

def run_training_loop(model, train_loader, num_epochs=3, lr=0.001,):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in tqdm(range(num_epochs)):
        for batch_idx, batch in enumerate(iter(train_loader)):
            batch = batch.to(device)
            forward_pass_outputs = model(batch)
            loss = F.mse_loss(forward_pass_outputs.ravel(), batch.y.ravel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



class EarlyStopping:
    """Stop training when a metric has stopped improving."""


    def __init__(
        self,
        mode: Literal["min", "max"] = "min",
        patience: int = 15,
        threshold: float = 1e-4,
        threshold_mode: Literal["rel", "abs"] = "rel",
        cooldown: int = 0,
    ):  # noqa: D107

        self.patience = patience

        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best: float
        self.num_bad_epochs: int
        self.mode_worse: float  # the worse value for the chosen mode
        self.last_epoch = 0
        self._init_is_better(
            mode=mode, threshold=threshold, threshold_mode=threshold_mode
        )
        self._reset()

    def _reset(self):
        """Reset num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics: SupportsFloat, epoch=None):  # type: ignore[override]
        """Perform a step."""
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            return True
        else:
            return False

    @property
    def in_cooldown(self):  # noqa: D102
        return self.cooldown_counter > 0

    def is_better(self, a, best):  # noqa: D102
        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1.0 - self.threshold
            return a < best * rel_epsilon

        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold

        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = self.threshold + 1.0
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")

        if mode == "min":
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode