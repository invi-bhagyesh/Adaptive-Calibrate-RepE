"""
Extraction Module - LAT Probe and Mean Difference Vectors

Extracts harmfulness vectors from model activations:
- LAT probe: linear readout for harmfulness at t_inst → v_harm.pt
- Mean diff: direction between harmful/harmless centroids
"""
from .lat_probe import (
    extract_activations,
    fit_lat_probe,
    score_prompts,
)
from .mean_diff import compute_mean_diff

__all__ = [
    "extract_activations",
    "fit_lat_probe", 
    "score_prompts",
    "compute_mean_diff",
]
