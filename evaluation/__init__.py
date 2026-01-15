"""
Evaluation Module - Metrics, tau selection, and comparison.
"""
from .metrics import compute_metrics, compare_results, detect_refusal
from .choose_tau import find_optimal_tau, compute_metrics_at_tau

__all__ = [
    "compute_metrics",
    "compare_results",
    "detect_refusal",
    "find_optimal_tau",
    "compute_metrics_at_tau",
]
