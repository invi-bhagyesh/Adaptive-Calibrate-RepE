"""Evaluation — metrics, tau selection, and comparison."""

from .metrics import compute_metrics, compare_results, print_metrics
from .choose_tau import load_scores, find_optimal_tau, compute_metrics_at_tau, print_tau_analysis

__all__ = [
    "compute_metrics",
    "compare_results",
    "print_metrics",
    "load_scores",
    "find_optimal_tau",
    "compute_metrics_at_tau",
    "print_tau_analysis",
]
