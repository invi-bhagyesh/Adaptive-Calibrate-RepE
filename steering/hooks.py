#!/usr/bin/env python3
"""
Steering Hooks Module

Implements the piece-wise operator for conditional refusal suppression.
Contains hooks for:
1. Reading harmfulness scores at t_inst (L_lat layer)
2. Applying piece-wise intervention at t_post-inst based on scores

The key idea: if s(x) <= tau (benign), subtract alpha * v_refuse to suppress refusal.
For harmful prompts (s(x) > tau), no intervention is applied.

Usage:
    from steering.hooks import SteeringHooks
    
    hooks = SteeringHooks(
        v_harm=torch.load("v_harm.pt"),
        v_refuse=torch.load("v_refuse.pt"),
        tau=0.5,
        alpha=1.0
    )
    hooks.register(model, l_lat=13, l_post=31)
    
    # Run generation
    output = model.generate(...)
    
    hooks.remove()
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Callable, Any
import functools
import threading


class ScoreStore:
    """
    Thread-safe storage for harmfulness scores during forward pass.
    
    Keys are module IDs to handle multiple layers/modules.
    Uses thread-local storage to handle concurrent inference.
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._scores: Dict[int, Tensor] = {}
        self._batch_mapping: Dict[int, Dict[int, int]] = {}  # For multi-batch tracking
    
    def store(self, module_id: int, scores: Tensor):
        """Store scores for a module."""
        with self._lock:
            self._scores[module_id] = scores.detach().cpu()
    
    def get(self, module_id: int) -> Optional[Tensor]:
        """Get and remove scores for a module."""
        with self._lock:
            return self._scores.pop(module_id, None)
    
    def peek(self, module_id: int) -> Optional[Tensor]:
        """Get scores without removing them."""
        with self._lock:
            return self._scores.get(module_id)
    
    def clear(self):
        """Clear all stored scores."""
        with self._lock:
            self._scores.clear()
            self._batch_mapping.clear()


# Global score store for cross-hook communication
scores_store = ScoreStore()


def read_harmfulness_hook(
    module: nn.Module,
    input: Tuple[Tensor, ...],
    output: Tuple[Tensor, ...],
    *,
    v_harm: Tensor,
    idx_token: int = -1,
    store: ScoreStore,
    ref_module_id: int,
):
    """
    Forward hook to read harmfulness scores at t_inst.
    
    Computes s(x) = h(x) · v_harm and stores for later use.
    
    Args:
        module: The layer module
        input: Layer input tensors
        output: Layer output tensors
        v_harm: Normalized harmfulness direction vector [hidden_dim]
        idx_token: Token position for t_inst (-1 for last)
        store: ScoreStore instance for saving scores
        ref_module_id: Reference ID for cross-hook communication
    """
    # Get hidden states from output
    if isinstance(output, tuple):
        hidden_states = output[0]
    else:
        hidden_states = output
    
    # hidden_states shape: [batch, seq_len, hidden_dim]
    # Extract at t_inst position
    h_tinst = hidden_states[:, idx_token, :].float().detach()  # [B, D]
    
    # Compute harmfulness scores: s(x) = h(x) · v_harm
    v_harm_device = v_harm.to(h_tinst.device)
    scores = torch.matmul(h_tinst, v_harm_device)  # [B]
    
    # Store scores for use by the piecewise operator hook
    store.store(ref_module_id, scores)


def apply_piecewise_operator_hook(
    module: nn.Module,
    input: Tuple[Tensor, ...],
    output: Tuple[Tensor, ...],
    *,
    v_refuse: Tensor,
    alpha: float,
    tau: float,
    store: ScoreStore,
    ref_module_id: int,
    t_post_index: int = -1,
) -> Optional[Tuple[Tensor, ...]]:
    """
    Forward hook to apply piece-wise refusal suppression at t_post-inst.
    
    If s(x) <= tau (benign prompt), subtract alpha * v_refuse to suppress refusal.
    If s(x) > tau (harmful prompt), leave unchanged.
    
    Args:
        module: The layer module
        input: Layer input tensors
        output: Layer output tensors
        v_refuse: Normalized refusal direction vector [hidden_dim]
        alpha: Steering coefficient
        tau: Threshold for harmfulness score
        store: ScoreStore instance
        ref_module_id: Reference ID matching the read hook
        t_post_index: Token position for t_post-inst (-1 for last)
        
    Returns:
        Modified output tuple, or None for no modification
    """
    # Get stored scores from the read hook
    scores = store.get(ref_module_id)
    if scores is None:
        return None
    
    # Get hidden states from output
    if isinstance(output, tuple):
        hidden_states = output[0].clone()
        rest_of_output = output[1:]
    else:
        hidden_states = output.clone()
        rest_of_output = ()
    
    # hidden_states shape: [batch, seq_len, hidden_dim]
    batch_size = hidden_states.shape[0]
    scores = scores.to(hidden_states.device)
    
    # Create mask for benign prompts (s(x) <= tau)
    # These are the prompts where we want to suppress refusal
    benign_mask = (scores <= tau).float()  # [B]
    
    # Expand mask for broadcasting: [B, 1, 1]
    mask = benign_mask.unsqueeze(-1).unsqueeze(-1)
    
    # Prepare the refusal vector shift
    v_refuse_device = v_refuse.to(hidden_states.device).view(1, 1, -1)  # [1, 1, D]
    shift = alpha * v_refuse_device  # [1, 1, D]
    
    # Apply intervention at t_post_index position only
    # We subtract the refusal vector for benign prompts
    if t_post_index == -1:
        # Modify only the last token
        hidden_states[:, -1:, :] = hidden_states[:, -1:, :] - mask * shift
    else:
        # Modify at specific position
        hidden_states[:, t_post_index:t_post_index+1, :] = (
            hidden_states[:, t_post_index:t_post_index+1, :] - mask * shift
        )
    
    # Reconstruct output tuple
    if rest_of_output:
        return (hidden_states,) + rest_of_output
    else:
        return (hidden_states,)


class SteeringHooks:
    """
    Manager for piece-wise steering hooks.
    
    Registers and manages the read and apply hooks for conditional
    refusal suppression based on harmfulness scores.
    
    Example:
        hooks = SteeringHooks(v_harm, v_refuse, tau=0.5, alpha=1.0)
        hooks.register(model, l_lat=13, l_post=31)
        
        # Run inference
        output = model.generate(...)
        
        # Get intervention info
        print(hooks.get_last_scores())
        
        hooks.remove()
    """
    
    def __init__(
        self,
        v_harm: Tensor,
        v_refuse: Tensor,
        tau: float = 0.5,
        alpha: float = 1.0,
        idx_token_inst: int = -1,
        idx_token_post: int = -1,
    ):
        """
        Initialize steering hooks.
        
        Args:
            v_harm: Harmfulness direction vector [hidden_dim]
            v_refuse: Refusal direction vector [hidden_dim]
            tau: Threshold for harmfulness score
            alpha: Steering coefficient for refusal suppression
            idx_token_inst: Token position for t_inst (-1 for last instruction token)
            idx_token_post: Token position for t_post-inst (-1 for last token)
        """
        self.v_harm = v_harm.float()
        self.v_refuse = v_refuse.float()
        self.tau = tau
        self.alpha = alpha
        self.idx_token_inst = idx_token_inst
        self.idx_token_post = idx_token_post
        
        self.store = ScoreStore()
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self._registered = False
        self._last_scores: Optional[Tensor] = None
        self._ref_module_id: Optional[int] = None
    
    def register(
        self,
        model: nn.Module,
        l_lat: int,
        l_post: int,
    ):
        """
        Register hooks on the model.
        
        Args:
            model: The language model
            l_lat: Layer index for reading harmfulness (L_lat)
            l_post: Layer index for applying intervention (L_post)
        """
        if self._registered:
            raise RuntimeError("Hooks already registered. Call remove() first.")
        
        # Get layer modules
        block_modules = self._get_block_modules(model)
        
        # Generate unique reference ID for this registration
        self._ref_module_id = id(block_modules[l_lat])
        
        # Create partial hooks with bound parameters
        read_hook = functools.partial(
            read_harmfulness_hook,
            v_harm=self.v_harm,
            idx_token=self.idx_token_inst,
            store=self.store,
            ref_module_id=self._ref_module_id,
        )
        
        apply_hook = functools.partial(
            apply_piecewise_operator_hook,
            v_refuse=self.v_refuse,
            alpha=self.alpha,
            tau=self.tau,
            store=self.store,
            ref_module_id=self._ref_module_id,
            t_post_index=self.idx_token_post,
        )
        
        # Register hooks
        self.handles.append(block_modules[l_lat].register_forward_hook(read_hook))
        self.handles.append(block_modules[l_post].register_forward_hook(apply_hook))
        
        self._registered = True
    
    def remove(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.store.clear()
        self._registered = False
        self._ref_module_id = None
    
    def get_last_scores(self) -> Optional[Tensor]:
        """Get the last computed harmfulness scores (if still available)."""
        if self._ref_module_id is not None:
            return self.store.peek(self._ref_module_id)
        return None
    
    def update_parameters(
        self,
        tau: Optional[float] = None,
        alpha: Optional[float] = None,
    ):
        """
        Update steering parameters.
        
        Note: This requires re-registering hooks. Call remove() then register() again.
        
        Args:
            tau: New threshold value
            alpha: New steering coefficient
        """
        if tau is not None:
            self.tau = tau
        if alpha is not None:
            self.alpha = alpha
        
        if self._registered:
            raise RuntimeError(
                "Cannot update parameters while hooks are registered. "
                "Call remove() first, update parameters, then register() again."
            )
    
    def _get_block_modules(self, model: nn.Module) -> List[nn.Module]:
        """Get transformer block modules from various architectures."""
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return list(model.model.layers)
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            return list(model.transformer.h)
        elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
            return list(model.gpt_neox.layers)
        else:
            raise ValueError("Unknown model architecture - cannot find transformer layers")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()
        return False


def create_steering_hooks(
    v_harm_path: str,
    v_refuse_path: str,
    tau: float = 0.5,
    alpha: float = 1.0,
) -> SteeringHooks:
    """
    Convenience function to create SteeringHooks from saved vectors.
    
    Args:
        v_harm_path: Path to v_harm.pt
        v_refuse_path: Path to v_refuse.pt
        tau: Threshold for harmfulness score
        alpha: Steering coefficient
        
    Returns:
        Configured SteeringHooks instance
    """
    v_harm = torch.load(v_harm_path)
    v_refuse = torch.load(v_refuse_path)
    
    return SteeringHooks(
        v_harm=v_harm,
        v_refuse=v_refuse,
        tau=tau,
        alpha=alpha,
    )


# Utility function for quick testing
def test_steering_hooks():
    """Test steering hooks with dummy data."""
    print("Testing SteeringHooks...")
    
    # Create dummy vectors
    hidden_dim = 4096
    v_harm = torch.randn(hidden_dim)
    v_harm = v_harm / torch.norm(v_harm)
    
    v_refuse = torch.randn(hidden_dim)
    v_refuse = v_refuse / torch.norm(v_refuse)
    
    hooks = SteeringHooks(
        v_harm=v_harm,
        v_refuse=v_refuse,
        tau=0.5,
        alpha=1.0,
    )
    
    print(f"Created SteeringHooks with tau={hooks.tau}, alpha={hooks.alpha}")
    print(f"v_harm shape: {hooks.v_harm.shape}")
    print(f"v_refuse shape: {hooks.v_refuse.shape}")
    print("Test passed!")


if __name__ == "__main__":
    test_steering_hooks()
