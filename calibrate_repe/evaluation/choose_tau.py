"""Optimal threshold (tau) selection for the LAT probe."""
import json
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


def load_scores(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load scores and binary labels from a JSONL file."""
    scores, labels = [], []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            score = item.get("score") or item.get("harmfulness_score")
            label = 1 if item.get("label") == "harmful" else 0
            if score is not None:
                scores.append(score)
                labels.append(label)
    return np.array(scores), np.array(labels)


def compute_metrics_at_tau(
    scores: np.ndarray,
    labels: np.ndarray,
    tau: float,
) -> Dict[str, float]:
    """Compute safety / compliance at a given threshold."""
    preds = (scores > tau).astype(int)

    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    n_harm = labels.sum()
    n_safe = len(labels) - n_harm

    safety_rate = tp / n_harm if n_harm > 0 else 0
    compliance_rate = tn / n_safe if n_safe > 0 else 0

    return {
        "tau": tau,
        "safety_rate": float(safety_rate),
        "compliance_rate": float(compliance_rate),
        "tradeoff_score": float((safety_rate + compliance_rate) / 2),
    }


def find_optimal_tau(
    scores: np.ndarray,
    labels: np.ndarray,
    min_safety: float = 0.95,
) -> Dict:
    """Return several tau candidates with their metrics."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)

    candidates = []

    # Youden's J (max tradeoff)
    j = tpr - fpr
    idx = np.argmax(j)
    tau_youden = thresholds[idx] if idx < len(thresholds) else thresholds[-1]
    candidates.append({"name": "youden", **compute_metrics_at_tau(scores, labels, tau_youden)})

    # Min safety constraint
    valid = np.where(tpr >= min_safety)[0]
    if len(valid) > 0:
        best = valid[np.argmin(fpr[valid])]
        tau_safe = thresholds[best] if best < len(thresholds) else thresholds[-1]
        candidates.append({
            "name": f"min_safety_{int(min_safety * 100)}",
            **compute_metrics_at_tau(scores, labels, tau_safe),
        })

    # Percentile-based
    harm_scores = scores[labels == 1]
    for p in [5, 10]:
        tau_p = np.percentile(harm_scores, p)
        candidates.append({
            "name": f"harmful_p{p}",
            **compute_metrics_at_tau(scores, labels, tau_p),
        })

    return {"auc": float(auc), "candidates": candidates}


def print_tau_analysis(analysis: Dict):
    """Pretty-print tau candidate comparison."""
    print(f"\nAUC: {analysis['auc']:.4f}")
    print("-" * 60)
    print(f"{'Name':<20} {'Tau':>8} {'Safety':>10} {'Comply':>10} {'Trade':>10}")
    print("-" * 60)
    for c in analysis["candidates"]:
        print(
            f"{c['name']:<20} {c['tau']:>8.4f} {c['safety_rate']:>10.2%} "
            f"{c['compliance_rate']:>10.2%} {c['tradeoff_score']:>10.2%}"
        )
