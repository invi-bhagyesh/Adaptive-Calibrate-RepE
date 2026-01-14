<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> LLMs Encode Harmfulness and Refusal Separately </h1>

**Content warning**: This repository contains text that is offensive, harmful, or otherwise inappropriate in nature.

This repository contains the official implementation for the paper **"LLMs Encode Harmfulness and Refusal Separately"**. Our research reveals that large language models (LLMs) encode harmfulness and refusal as distinct concepts in their latent representations.

- [Paper](https://arxiv.org/abs/2507.11878)
- [Website](https://chats-lab.github.io/LLMs_Encode_Harmfulness_Refusal_Separately/)
- [Blog](https://www.lesswrong.com/posts/gzNe2Grj2KksvzHWM/llms-encode-harmfulness-and-refusal-separately)

### Key Findings

- **Separate Encoding**: Hidden states at the last token of instruction tokens (`t_inst`) encode harmfulness, while the last token of post-instruction tokens (`t_post-inst`) encodes refusal behavior
- **Causal Evidence**: Steering along the harmfulness direction changes the model's internal perception of harmfulness, while the refusal direction only affects surface-level refusal characteristics
- **Jailbreak Analysis**: Some jailbreak methods work by suppressing refusal signals without reversing the model's internal harmfulness judgment
- **Latent Guard**: Internal harmfulness representations can serve as safeguards for detecting unsafe inputs

## Project Structure

```
├── notebooks/                      # 📓 Standalone experiment notebooks
│   ├── complete_experiment.ipynb   # Full pipeline: extraction → LAT → steering → eval
│   └── analysis.ipynb              # Visualization and analysis of results
│
├── scripts/                        # 🔧 Command-line tools
│   ├── run_inference.py            # Generate responses and label refusals
│   ├── run_inference.sh            # Shell wrapper
│   ├── run_calibration.sh          # Full calibration pipeline
│   ├── sweep_params.py             # Parameter sweep for τ and α
│   ├── sweep_params.sh             # Shell wrapper
│   └── tiny_check.py               # Quick sanity check for labeling
│
├── extraction/                     # 🔬 Feature extraction and probe training
│   ├── lat_probe.py                # LAT probe (--fit and --score modes)
│   ├── mean_diff.py                # Baseline mean difference vectors
│   └── refusal_vector.py           # Extract v_refuse at t_post-inst
│
├── steering/                       # 🎯 Intervention hooks
│   └── hooks.py                    # PiecewiseSteeringHooks implementation
│
├── evaluation/                     # 📊 Metrics and analysis
│   ├── choose_tau.py               # Threshold selection with ROC analysis
│   └── metrics.py                  # Safety/compliance rate computation
│
├── tests/                          # 🧪 Testing
│   ├── sanity_prompts.json         # Test prompts (4 harmless + 4 harmful)
│   └── run_sanity_test.py          # Baseline vs steered comparison
│
├── src/                            # 📦 Original implementation
│   ├── extract_hidden.py           # Hidden state extraction
│   ├── intervention.py             # Controlled generation with interventions
│   ├── inference.py                # Model inference on datasets
│   ├── eval.py                     # Evaluation utilities
│   ├── utils.py                    # Helper functions (formatInp, REFUSAL_PHRASE)
│   ├── template_inversion.py       # Reply inversion task templates
│   ├── classifier.ipynb            # Latent guard classifier notebook
│   └── run/*.pt                    # Pre-extracted directions
│
├── data/                           # 📁 Datasets
│   ├── advbench.json               # Harmful instructions
│   ├── xstest-harmless.json        # Harmless instructions (may get over-refused)
│   ├── sorry-badq.json             # Jailbroken harmful instructions
│   └── ...                         # Additional datasets
│
└── outputs/                        # 📤 Generated outputs (created at runtime)
    ├── v_harm.pt                   # Harmfulness direction vector
    ├── v_refuse.pt                 # Refusal direction vector
    ├── lat_scores.jsonl            # Per-example harmfulness scores
    └── experiment_results.json     # Full experiment results
```

## Quick Start

### Option 1: Notebook (Recommended)

The easiest way to run all experiments is through the standalone notebook:

```bash
# Install dependencies
pip install -r requirements.txt

# Open and run the complete experiment notebook
jupyter notebook notebooks/complete_experiment.ipynb
```

The notebook runs the full pipeline:

1. Load model and datasets
2. Extract hidden states at t_inst and t_post-inst
3. Train LAT probe for harmfulness detection
4. Compute refusal direction vector
5. Implement piece-wise steering intervention
6. Evaluate baseline vs steered model

### Option 2: Command Line

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full calibration pipeline
./scripts/run_calibration.sh "Qwen/Qwen2-7B-Instruct" "qwen"

# 3. Run parameter sweep to find optimal τ and α
./scripts/sweep_params.sh "Qwen/Qwen2-7B-Instruct" "qwen"

# 4. Analyze results
jupyter notebook notebooks/analysis.ipynb
```

### Using Pre-extracted Directions

The `src/run/` directory contains pre-extracted direction vectors:

- `qwen-dir-hf.pt` - Harmfulness direction for Qwen
- `llama2_harmful_dir.pt` - Harmfulness direction for LLaMA-2
- `llama3-dir-hf.pt` - Harmfulness direction for LLaMA-3

## Experiments

### Hidden State Analysis

Our analysis focuses on two key token positions:

- **`t_inst`**: Last token of the user's instruction (encodes harmfulness)
- **`t_post-inst`**: Last token of the entire input prompt (encodes refusal behavior)

```bash
sh run_diff_mean.sh
```

This will reproduce hidden states for two specified clusters (e.g., harmful prompts and harmless prompts) and the according direction from one cluster to the other.

### Intervention Experiments

Perform controlled interventions to modify model behavior:

```bash
# Run intervention with specific parameters
sh complete_intervene.sh
```

Key parameters:

- `--intervention_vector`: Path to the steering vectors
- `--reverse_intervention`: Whether to reverse the steering vector (1/0)
- `--use_inversion`: Whether to do reply inversion task (1/0)

### Latent Guard Implementation

One of our contributions is the **Latent Guard** - an intrinsic safeguard that uses the model's own internal harmfulness representations.
Implementations are in `classifier.ipynb`.

#### Compare with Baselines

```bash
# Run LlamaGuard 3 with Ollama
python run_llama_guard.py --input data/{test_prompts_name}.json --output results/llamaguard.txt
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{zhao2025llmsencodeharmfulnessrefusal,
      title={LLMs Encode Harmfulness and Refusal Separately},
      author={Jiachen Zhao and Jing Huang and Zhengxuan Wu and David Bau and Weiyan Shi},
      year={2025},
      eprint={2507.11878},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.11878},
}
```
