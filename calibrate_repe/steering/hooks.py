"""Steering Hooks — piece-wise operator for conditional refusal suppression.

If s(x) <= tau (benign), subtract alpha * v_refuse to suppress refusal.
For harmful prompts (s(x) > tau), no intervention is applied.
"""
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..model import get_layers


class SteeringHooks:
    """Manager for piece-wise steering hooks."""

    def __init__(
        self,
        v_harm: Tensor,
        v_refuse: Tensor,
        tau: float = 0.5,
        alpha: float = 1.0,
    ):
        self.v_harm = v_harm.float()
        self.v_refuse = v_refuse.float()
        self.tau = tau
        self.alpha = alpha
        self.handles: List = []
        self._scores: Optional[Tensor] = None

    def register(self, model: nn.Module, l_lat: int, l_post: int):
        """Register read + apply hooks on the model."""
        layers = get_layers(model)

        def read_hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            h_last = h[:, -1, :].float()
            self._scores = torch.matmul(h_last, self.v_harm.to(h_last.device))

        def apply_hook(module, input, output):
            if self._scores is None:
                return None
            h = output[0].clone() if isinstance(output, tuple) else output.clone()
            rest = output[1:] if isinstance(output, tuple) else ()

            mask = (self._scores <= self.tau).float().view(-1, 1, 1)
            shift = self.alpha * self.v_refuse.to(h.device).view(1, 1, -1)
            h[:, -1:, :] = h[:, -1:, :] - mask * shift

            return (h,) + rest if rest else (h,)

        self.handles.append(layers[l_lat].register_forward_hook(read_hook))
        self.handles.append(layers[l_post].register_forward_hook(apply_hook))

    def remove(self):
        """Remove all hooks."""
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self._scores = None

    def get_last_scores(self) -> Optional[Tensor]:
        return self._scores

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove()


def create_steering_hooks(
    v_harm_path: str,
    v_refuse_path: str,
    tau: float = 0.5,
    alpha: float = 1.0,
) -> SteeringHooks:
    """Create SteeringHooks from saved vector files."""
    return SteeringHooks(
        v_harm=torch.load(v_harm_path, weights_only=True),
        v_refuse=torch.load(v_refuse_path, weights_only=True),
        tau=tau,
        alpha=alpha,
    )
