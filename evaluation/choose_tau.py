#!/usr/bin/env python3
"""
Choose Tau - Threshold Selection for LAT Probe

Analyzes LAT scores to choose the optimal threshold tau for the piece-wise operator.
Computes ROC curves, precision-recall curves, and suggests threshold candidates.

Usage:
    python evaluation/choose_tau.py --scores outputs/lat_scores.jsonl --output outputs/tau_analysis.json
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_scores(scores_file: str) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Load scores and labels from JSONL file.
    
    Args:
        scores_file: Path to lat_scores.jsonl
        
    Returns:
        Tuple of (scores, labels, raw_data) where labels are 1 for harmful
    """
    scores = []
    labels = []
    raw_data = []
    
    with open(scores_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            score = item.get('score') or item.get('harmfulness_score')
            label_str = item.get('label', '')
            label_binary = item.get('label_binary')
            
            if score is not None:
                scores.append(score)
                
                # Determine label
                if label_binary is not None:
                    labels.append(label_binary)
                elif label_str == 'harmful':
                    labels.append(1)
                elif label_str == 'harmless':
                    labels.append(0)
                else:
                    # Skip if no label
                    scores.pop()
                    continue
                
                raw_data.append(item)
    
    logger.info(f"Loaded {len(scores)} scored examples")
    return np.array(scores), np.array(labels), raw_data


def compute_metrics_at_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    tau: float
) -> Dict[str, float]:
    """
    Compute classification metrics at a given threshold.
    
    For piece-wise operator:
    - Predictions <= tau are treated as "benign" (intervention applied)
    - Predictions > tau are treated as "harmful" (no intervention)
    
    Args:
        scores: Harmfulness scores
        labels: True labels (1=harmful, 0=harmless)
        tau: Threshold
        
    Returns:
        Dictionary of metrics
    """
    predictions = (scores > tau).astype(int)  # Predict harmful if score > tau
    
    # True label counts
    n_harmful = labels.sum()
    n_harmless = len(labels) - n_harmful
    
    # Confusion matrix elements
    true_positives = ((predictions == 1) & (labels == 1)).sum()
    false_positives = ((predictions == 1) & (labels == 0)).sum()
    true_negatives = ((predictions == 0) & (labels == 0)).sum()
    false_negatives = ((predictions == 0) & (labels == 1)).sum()
    
    # Rates
    tpr = true_positives / n_harmful if n_harmful > 0 else 0  # Sensitivity / Recall for harmful
    fpr = false_positives / n_harmless if n_harmless > 0 else 0
    tnr = true_negatives / n_harmless if n_harmless > 0 else 0  # Specificity
    
    # For piece-wise operator interpretation:
    # - Safety rate = fraction of harmful prompts correctly identified (TPR)
    # - Compliance rate = fraction of harmless prompts allowed through (TNR)
    safety_rate = tpr  # Harmful detected, refusal preserved
    compliance_rate = tnr  # Harmless allowed through, refusal suppressed
    
    # Tradeoff score
    tradeoff_score = (safety_rate + compliance_rate) / 2
    
    # F1 score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = tpr
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'tau': tau,
        'safety_rate': float(safety_rate),  # TPR - harmful correctly blocked
        'compliance_rate': float(compliance_rate),  # TNR - harmless allowed
        'tradeoff_score': float(tradeoff_score),
        'true_positive_rate': float(tpr),
        'false_positive_rate': float(fpr),
        'true_negative_rate': float(tnr),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'true_positives': int(true_positives),
        'false_positives': int(false_positives),
        'true_negatives': int(true_negatives),
        'false_negatives': int(false_negatives),
    }


def find_optimal_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    min_safety_rate: float = 0.95,
) -> Dict[str, Any]:
    """
    Find optimal threshold candidates.
    
    Args:
        scores: Harmfulness scores
        labels: True labels
        min_safety_rate: Minimum required safety rate (TPR for harmful)
        
    Returns:
        Dictionary with threshold candidates and analysis
    """
    # Compute ROC curve
    fpr, tpr, thresholds_roc = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    
    # Compute precision-recall curve
    precision, recall, thresholds_pr = precision_recall_curve(labels, scores)
    
    # Candidate thresholds based on different criteria
    candidates = []
    
    # 1. Maximum F1 threshold
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    max_f1_idx = np.argmax(f1_scores)
    tau_max_f1 = thresholds_pr[max_f1_idx]
    candidates.append({
        'name': 'max_f1',
        'tau': float(tau_max_f1),
        **compute_metrics_at_threshold(scores, labels, tau_max_f1)
    })
    
    # 2. Threshold at minimum safety rate constraint
    valid_indices = np.where(tpr >= min_safety_rate)[0]
    if len(valid_indices) > 0:
        # Find the one with highest TNR (lowest FPR) among valid
        best_idx = valid_indices[np.argmin(fpr[valid_indices])]
        tau_safe = thresholds_roc[best_idx] if best_idx < len(thresholds_roc) else thresholds_roc[-1]
        candidates.append({
            'name': f'min_safety_{int(min_safety_rate*100)}pct',
            'tau': float(tau_safe),
            **compute_metrics_at_threshold(scores, labels, tau_safe)
        })
    
    # 3. Maximum tradeoff (Youden's J statistic)
    j_scores = tpr - fpr
    max_j_idx = np.argmax(j_scores)
    tau_youden = thresholds_roc[max_j_idx] if max_j_idx < len(thresholds_roc) else thresholds_roc[-1]
    candidates.append({
        'name': 'youden_optimal',
        'tau': float(tau_youden),
        **compute_metrics_at_threshold(scores, labels, tau_youden)
    })
    
    # 4. Percentile-based thresholds on harmful scores
    harmful_scores = scores[labels == 1]
    for percentile in [5, 10, 20]:
        tau_percentile = np.percentile(harmful_scores, percentile)
        candidates.append({
            'name': f'harmful_p{percentile}',
            'tau': float(tau_percentile),
            **compute_metrics_at_threshold(scores, labels, tau_percentile)
        })
    
    # 5. Percentile-based on harmless scores (conservative)
    harmless_scores = scores[labels == 0]
    for percentile in [80, 90, 95]:
        tau_percentile = np.percentile(harmless_scores, percentile)
        candidates.append({
            'name': f'harmless_p{percentile}',
            'tau': float(tau_percentile),
            **compute_metrics_at_threshold(scores, labels, tau_percentile)
        })
    
    return {
        'auc': float(auc),
        'candidates': candidates,
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds_roc.tolist(),
        },
        'score_stats': {
            'harmful_mean': float(harmful_scores.mean()),
            'harmful_std': float(harmful_scores.std()),
            'harmful_min': float(harmful_scores.min()),
            'harmful_max': float(harmful_scores.max()),
            'harmless_mean': float(harmless_scores.mean()),
            'harmless_std': float(harmless_scores.std()),
            'harmless_min': float(harmless_scores.min()),
            'harmless_max': float(harmless_scores.max()),
        }
    }


def print_analysis(analysis: Dict[str, Any]):
    """Pretty print the analysis results."""
    print("\n" + "="*70)
    print("TAU THRESHOLD ANALYSIS")
    print("="*70)
    
    print(f"\nAUC: {analysis['auc']:.4f}")
    
    print("\nScore Statistics:")
    stats = analysis['score_stats']
    print(f"  Harmful:  mean={stats['harmful_mean']:.4f}, std={stats['harmful_std']:.4f}, "
          f"range=[{stats['harmful_min']:.4f}, {stats['harmful_max']:.4f}]")
    print(f"  Harmless: mean={stats['harmless_mean']:.4f}, std={stats['harmless_std']:.4f}, "
          f"range=[{stats['harmless_min']:.4f}, {stats['harmless_max']:.4f}]")
    
    print("\n" + "-"*70)
    print("Threshold Candidates:")
    print("-"*70)
    print(f"{'Name':<25} {'Tau':>8} {'Safety':>8} {'Comply':>8} {'Trade':>8} {'F1':>8}")
    print("-"*70)
    
    for cand in analysis['candidates']:
        print(f"{cand['name']:<25} {cand['tau']:>8.4f} {cand['safety_rate']:>8.2%} "
              f"{cand['compliance_rate']:>8.2%} {cand['tradeoff_score']:>8.2%} {cand['f1_score']:>8.4f}")
    
    print("-"*70)
    
    # Find best by tradeoff
    best_tradeoff = max(analysis['candidates'], key=lambda x: x['tradeoff_score'])
    print(f"\nRecommended (max tradeoff): tau = {best_tradeoff['tau']:.4f}")
    print(f"  Safety Rate: {best_tradeoff['safety_rate']:.2%}")
    print(f"  Compliance Rate: {best_tradeoff['compliance_rate']:.2%}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Choose optimal threshold tau")
    parser.add_argument('--scores', type=str, required=True,
                       help="Path to lat_scores.jsonl")
    parser.add_argument('--output', type=str, default='outputs/tau_analysis.json',
                       help="Output JSON file path")
    parser.add_argument('--min_safety', type=float, default=0.95,
                       help="Minimum required safety rate")
    
    args = parser.parse_args()
    
    # Load scores
    scores, labels, _ = load_scores(args.scores)
    
    if len(scores) == 0:
        logger.error("No valid scores found!")
        return
    
    # Analyze
    analysis = find_optimal_thresholds(scores, labels, args.min_safety)
    
    # Print results
    print_analysis(analysis)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2)
    
    logger.info(f"Analysis saved to {args.output}")


if __name__ == "__main__":
    main()
