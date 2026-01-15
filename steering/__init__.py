"""
Steering Module - Piece-wise operator for conditional refusal suppression.

Implements hooks for:
1. Reading harmfulness scores at t_inst (L_lat layer)
2. Applying conditional intervention at t_post-inst if s(x) <= tau
"""
from .hooks import SteeringHooks, create_steering_hooks

__all__ = ["SteeringHooks", "create_steering_hooks"]
