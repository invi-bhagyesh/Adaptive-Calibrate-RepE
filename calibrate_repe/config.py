from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Configuration for Adaptive Safety Calibration."""

    # Model
    model_path: str = "meta-llama/Llama-2-7b-chat-hf"
    model_type: str = "llama2"  # llama2, llama3, qwen, vicuna

    # Layers
    l_lat: int = 14      # Layer for harmfulness reading (L_lat)
    l_post: int = 31     # Layer for intervention (L_post)

    # Steering parameters
    tau: float = 0.5     # Harmfulness threshold
    alpha: float = 1.0   # Refusal suppression coefficient

    # Paths
    output_dir: str = "./outputs"
    data_dir: str = "./data"

    # Training
    ridge_lambda: float = 1e-3
    limit_prompts: int = None

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    @property
    def v_harm_path(self) -> str:
        return f"{self.output_dir}/v_harm.pt"

    @property
    def v_refuse_path(self) -> str:
        return f"{self.output_dir}/v_refuse.pt"

    @property
    def scores_path(self) -> str:
        return f"{self.output_dir}/lat_scores.jsonl"
