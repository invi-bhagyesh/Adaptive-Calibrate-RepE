# Extraction module for hidden state analysis and probe training
from .lat_probe import fit_lat_probe, score_prompts
from .refusal_vector import extract_activations_at_post_inst
from .mean_diff import extract_activations, compute_mean_difference

__all__ = [
    'fit_lat_probe',
    'score_prompts',
    'extract_activations_at_post_inst',
    'extract_activations',
    'compute_mean_difference',
]
