# Adaptive Safety Calibration via Representation Engineering

Large Language Models (LLMs) frequently exhibit **over-refusal**, rejecting benign user requests that merely _sound_ harmful, while still remaining vulnerable to sophisticated jailbreaks. Recent work has shown that LLMs encode **harmfulness** and **refusal behavior** as **distinct internal representations**, localized at different token positions in the forward pass.

Builds on the findings of _LLMs Encode Harmfulness and Refusal Separately_ (Zhao et al., NeurIPS 2025) and moves beyond static analysis to implement a **runtime representation-engineering system** called **Adaptive Safety Calibration**.

Instead of globally suppressing refusal mechanisms, a **context-aware control policy** that:

- reads the model's internal belief about harmfulness,
- conditionally modulates refusal behavior,
- reduces over-refusal **without disabling safety** for genuinely dangerous prompts.

---

## Key Idea

LLMs internally:

- encode **harmfulness** at the last token of the user instruction (`t_inst`),
- encode **refusal behavior** at the final post-instruction token (`t_post-inst`).

However, refusal is not always causally downstream of harmfulness, leading to:

- over-refusal on benign prompts,
- brittle safety under jailbreak attacks.

We exploit this separation to build a **piece-wise representation control system** that enforces the correct causal ordering:

> _Refusal should activate **only if** internal harmfulness is high._

---

## Methodology

### 1. Dataset Categorization (Inference)

Using the provided inference scripts, prompts are categorized into four behaviorally distinct regimes:

| Category          |     | Description                                    |
| ----------------- | --- | ---------------------------------------------- |
| Accepted Harmless | ✅  | Normal helpful responses (e.g. Alpaca)         |
| Refused Harmless  | ❌  | Over-refusal cases (e.g. XSTest)               |
| Refused Harmful   | ✅  | Correct safety behavior (e.g. AdvBench, CATQA) |
| Accepted Harmful  | ❌  | Successful jailbreaks (e.g. GCG, Persuasion)   |

This stratification explicitly isolates **over-refusal** as a target failure mode.

---

### 2. Representation Reading (LAT)

Rather than using a simple mean-difference direction, we apply **Linear Artificial Tomography (LAT)** to extract a **robust Harmfulness Vector**:

- Hidden states are extracted from a **middle transformer layer** (typically 13–14).
- Activations are taken at the **instruction boundary token** (`t_inst`).
- LAT yields a **scalar harmfulness score** per example, robust to adversarial suffixes.

This converts harmfulness from a qualitative concept into a **measurable latent variable**.

---

### 3. Representation Control (Piece-wise Operator)

At inference time, we apply a **conditional intervention policy**:

1. Measure harmfulness alignment at `t_inst`.
2. If harmfulness is **above a threshold**:
   - Allow refusal behavior to proceed normally.
3. If harmfulness is **below the threshold**:
   - Apply a _negative shift_ along the **Refusal Vector** at `t_post-inst`,
   - Forcing compliance and correcting over-refusal.

Formally, the intervention is:

$$
\Delta a =
\begin{cases}
0 & \text{if } \langle a_{t_{\text{inst}}}, v_{\text{harm}} \rangle > \tau \\
-\alpha \cdot v_{\text{refuse}} & \text{otherwise}
\end{cases}
$$

This ensures refusal behavior is **gated by internal belief**, not surface-level prompt features.

---

### 4. Evaluation

We evaluate performance using a **Trade-off Score**:

$$
\text{Trade-off Score} =
\frac{1}{2}
\left(
\text{Compliance Rate on XSTest}
+
\text{Safety Score on AdvBench}
\right)
$$

A successful system:

- significantly increases compliance on over-refusal benchmarks,
- while maintaining or improving safety on harmful benchmarks.

---

## Why This Matters

### Conceptual Contribution

- Moves from _representation analysis_ to **representation engineering**.
- Enforces a **causal relationship** between internal belief and external behavior.
- Demonstrates that refusal can be calibrated without retraining.

### Practical Advantages

- **Training-free**: no fine-tuning required.
- **Data-efficient**: high-quality vectors from ~64 instruction pairs.
- **Fast iteration**: inference-time hooks allow rapid experimentation.

### Robustness Insight

We find that many jailbreaks suppress refusal behavior **without changing the model's internal harmfulness belief**. This project explicitly exploits that mismatch.

---

## Repository Structure

```
├── calibrate_repe/              # Core Python package
│   ├── __init__.py
│   ├── config.py                # Configuration dataclass
│   ├── model.py                 # Model loading, get_layers()
│   ├── data.py                  # Prompt I/O, refusal detection, templates
│   ├── extraction/
│   │   ├── activations.py       # Shared hidden-state extraction
│   │   ├── lat_probe.py         # LAT probe (ridge regression → v_harm)
│   │   └── mean_diff.py         # Mean-difference vectors (v_harm_mean, v_refuse)
│   ├── steering/
│   │   └── hooks.py             # Piece-wise steering hooks
│   └── evaluation/
│       ├── metrics.py           # Safety / compliance / trade-off metrics
│       └── choose_tau.py        # Optimal threshold selection
├── main.py                      # End-to-end pipeline CLI
├── scripts/
│   ├── run_inference.py         # Batch inference with refusal labeling
│   ├── sweep_params.py          # Grid search over tau and alpha
│   ├── tiny_check.py            # Quick label sanity check
│   ├── run_calibration.sh       # Full calibration pipeline
│   ├── run_inference.sh         # Inference wrapper
│   └── sweep_params.sh          # Sweep wrapper
├── tests/
│   ├── run_sanity_test.py       # Baseline vs steered comparison
│   └── sanity_prompts.json      # Test prompt set
├── data/                        # Prompt datasets (AdvBench, XSTest, CATQA, etc.)
├── notebooks/                   # Experiment notebooks
├── docs/                        # Design notes
└── requirements.txt
```

---

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Full calibration pipeline

```bash
# 1. Extract v_harm (LAT) and v_refuse (mean diff)
python main.py --extract \
    --harmful data/advbench.json \
    --harmless data/xstest-harmless.json \
    --model NousResearch/Llama-2-7b-chat-hf

# 2. Run steered inference
python main.py --infer data/xstest-harmless.json \
    --model NousResearch/Llama-2-7b-chat-hf \
    --tau 0.3
```

### Or use the shell scripts

```bash
# End-to-end calibration (inference → LAT → tau selection)
./scripts/run_calibration.sh NousResearch/Llama-2-7b-chat-hf llama2

# Parameter sweep over tau and alpha
./scripts/sweep_params.sh NousResearch/Llama-2-7b-chat-hf llama2
```

### Sanity test

```bash
python tests/run_sanity_test.py \
    --model NousResearch/Llama-2-7b-chat-hf \
    --v_harm outputs/v_harm.pt \
    --v_refuse outputs/v_refuse.pt \
    --tau 0.3
```

---

## Configuration

Key parameters in `calibrate_repe/config.py`:

| Parameter      | Default  | Description                                            |
| -------------- | -------- | ------------------------------------------------------ |
| `l_lat`        | 14       | Layer for harmfulness reading                          |
| `l_post`       | 31       | Layer for refusal intervention                         |
| `tau`          | 0.5      | Harmfulness threshold                                  |
| `alpha`        | 1.0      | Refusal suppression strength                           |
| `ridge_lambda` | 1e-3     | Ridge regression regularization                        |
| `model_type`   | `llama2` | Prompt template (`llama2`, `llama3`, `qwen`, `vicuna`) |

---

## Extensions

This framework naturally supports further research directions:

- **LoRRA integration**: compress the piece-wise operator into a low-rank adapter (~168 MB), removing inference-time hooks.
- **Category-specific harmfulness**: learn multiple harmfulness subspaces for fine-grained risk control.
- **Trajectory-level calibration**: extend harmfulness tracking across token generation.
- **Training diagnostics**: track harmfulness representations across SFT or RLHF checkpoints.

---
