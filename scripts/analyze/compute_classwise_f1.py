#!/usr/bin/env python3
"""
Compute Class-wise F1 Scores from Saved Predictions

This script loads the preds_perK.npy and labels.npy files saved by
train_turnlevel_k_sweep_bayesian_v2.py and computes per-class F1 scores.

Usage:
    python compute_classwise_f1.py \
        --results_dir results/turnlevel_k_sweep_bayesian/4way_sentence-roberta_*/seed42
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import json

def load_experiment_data(results_dir):
    """Load predictions, labels, and K values"""
    results_dir = Path(results_dir)
    
    print(f"Loading data from: {results_dir}")
    
    # Load files
    preds = np.load(results_dir / "preds_perK.npy")  # (num_K, N, C)
    labels = np.load(results_dir / "labels.npy")     # (N,)
    Ks = np.load(results_dir / "Ks.npy")             # (num_K,)
    
    # Load metadata
    with open(results_dir / "metadata.json") as f:
        metadata = json.load(f)
    
    print(f"  ✓ preds_perK.npy: {preds.shape}")
    print(f"  ✓ labels.npy: {labels.shape}")
    print(f"  ✓ Ks.npy: {Ks.shape}")
    print(f"  ✓ Metadata: task={metadata['task']}, num_classes={metadata['num_classes']}")
    
    return preds, labels, Ks, metadata


def compute_classwise_metrics(preds, labels, Ks, metadata, treat_minus1_as_class=False):
    """
    Compute class-wise F1 scores for each K
    
    Args:
        preds: (num_K, N, C) - softmax probabilities
        labels: (N,) - ground truth labels
        Ks: (num_K,) - K values
        metadata: experiment metadata
        treat_minus1_as_class: whether to treat -1 as a separate class
    
    Returns:
        DataFrame with columns: K, overall metrics, + per-class F1 scores
    """
    num_K, N, C = preds.shape
    task = metadata['task']
    
    # Define class names
    class_names = {
        '4way': ['angry', 'happy', 'sad', 'neutral'],
        '6way': ['angry', 'happy', 'sad', 'neutral', 'excited', 'frustrated']
    }
    
    emotions = class_names.get(task, [f'class_{i}' for i in range(C)])
    
    results = []
    
    for k_idx, K in enumerate(Ks):
        print(f"\nProcessing K={K} ({k_idx+1}/{num_K})...")
        
        # Get predictions for this K
        probs_K = preds[k_idx]  # (N, C)
        ypred = np.argmax(probs_K, axis=1)
        
        # Handle -1 labels
        if treat_minus1_as_class:
            # Treat -1 as a valid class for evaluation
            mask = np.ones(len(labels), dtype=bool)
        else:
            # Skip -1 labels
            mask = labels >= 0
        
        ypred_eval = ypred[mask]
        ytrue_eval = labels[mask]
        
        # Overall metrics
        f1w = f1_score(ytrue_eval, ypred_eval, average="weighted", zero_division=0)
        f1m = f1_score(ytrue_eval, ypred_eval, average="macro", zero_division=0)
        
        # Per-class F1
        f1_per_class = f1_score(ytrue_eval, ypred_eval, average=None, zero_division=0)
        
        # Build result dict
        result = {
            "K": K,
            "f1_weighted": f1w,
            "f1_macro": f1m,
        }
        
        # Add per-class F1
        for i, emotion in enumerate(emotions):
            if i < len(f1_per_class):
                result[f"{emotion}_f1"] = f1_per_class[i]
            else:
                result[f"{emotion}_f1"] = np.nan
        
        results.append(result)
        
        # Print brief summary
        print(f"  Overall F1: {f1w:.4f}")
        for i, emotion in enumerate(emotions):
            if i < len(f1_per_class):
                print(f"    {emotion.capitalize()}: {f1_per_class[i]:.4f}")
    
    return pd.DataFrame(results)


def save_detailed_reports(preds, labels, Ks, metadata, output_dir):
    """
    Save detailed classification reports for each K
    """
    reports_dir = output_dir / "class_reports"
    reports_dir.mkdir(exist_ok=True)
    
    task = metadata['task']
    class_names = {
        '4way': ['angry', 'happy', 'sad', 'neutral'],
        '6way': ['angry', 'happy', 'sad', 'neutral', 'excited', 'frustrated']
    }
    
    emotions = class_names.get(task, None)
    
    for k_idx, K in enumerate(Ks):
        probs_K = preds[k_idx]
        ypred = np.argmax(probs_K, axis=1)
        
        # Skip -1 labels
        mask = labels >= 0
        ypred_eval = ypred[mask]
        ytrue_eval = labels[mask]
        
        # Classification report
        report = classification_report(
            ytrue_eval, ypred_eval,
            target_names=emotions,
            zero_division=0,
            output_dict=True
        )
        
        # Save report as JSON
        report_file = reports_dir / f"k_{int(K)}_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Confusion matrix
        cm = confusion_matrix(ytrue_eval, ypred_eval)
        np.save(reports_dir / f"k_{int(K)}_confusion.npy", cm)
    
    print(f"\n  ✓ Saved {len(Ks)} detailed reports to {reports_dir}")


def main():
    parser = argparse.ArgumentParser(description='Compute class-wise F1 from saved predictions')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Path to experiment results directory')
    parser.add_argument('--treat_minus1_as_class', action='store_true',
                       help='Treat -1 labels as a separate class')
    parser.add_argument('--save_detailed_reports', action='store_true',
                       help='Save detailed classification reports for each K')
    
    args = parser.parse_args()
    
    # Load data
    preds, labels, Ks, metadata = load_experiment_data(args.results_dir)
    
    # Compute class-wise metrics
    print("\n" + "="*80)
    print("COMPUTING CLASS-WISE F1 SCORES")
    print("="*80)
    
    df = compute_classwise_metrics(
        preds, labels, Ks, metadata,
        treat_minus1_as_class=args.treat_minus1_as_class
    )
    
    # Save results
    output_dir = Path(args.results_dir)
    output_file = output_dir / "k_sweep_classwise_results.csv"
    df.to_csv(output_file, index=False)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    print("\n" + "="*80)
    print(f"✓ Saved to: {output_file}")
    
    # Save detailed reports if requested
    if args.save_detailed_reports:
        print("\nSaving detailed classification reports...")
        save_detailed_reports(preds, labels, Ks, metadata, output_dir)
    
    print("\n✅ Complete!")
    print(f"\nNext step: Run emotion saturation analysis:")
    print(f"  python analyze_emotion_saturation.py \\")
    print(f"    --results_dir {args.results_dir} \\")
    print(f"    --task {metadata['task']}")


if __name__ == "__main__":
    main()
