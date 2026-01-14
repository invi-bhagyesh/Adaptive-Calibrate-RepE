# Evaluation module for metrics and threshold selection
from .metrics import (
    compute_metrics,
    compare_baseline_steered,
    detect_refusal,
    generate_report,
)
from .choose_tau import (
    find_optimal_thresholds,
    compute_metrics_at_threshold,
)

__all__ = [
    'compute_metrics',
    'compare_baseline_steered',
    'detect_refusal',
    'generate_report',
    'find_optimal_thresholds',
    'compute_metrics_at_threshold',
]
