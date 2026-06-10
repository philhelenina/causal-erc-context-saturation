#!/usr/bin/env python3
"""
Emotion-Specific Saturation Analysis
Analyzes K-sweep results for each emotion class separately

Usage:
    python analyze_emotion_saturation.py \
        --results_dir results/turnlevel_k_sweep_bayesian/4way_sentence-roberta_*/seed42 \
        --task 4way
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def find_saturation_point(scores, K_values, s=3, delta=0.001):
    """
    Find minimal saturating window K* (Algorithm 1 from paper)
    
    Args:
        scores: Array of F1 scores
        K_values: Array of K values
        s: Consecutive length for plateau detection
        delta: Tolerance for plateau (default: 0.1% = 0.001)
    
    Returns:
        K_star: Saturation point
    """
    if len(scores) < s + 1:
        return K_values[-1]
    
    # Compute deltas
    deltas = [scores[i] - scores[i-1] for i in range(1, len(scores))]
    
    # Find first plateau
    for j in range(len(deltas) - s + 1):
        if all(deltas[j+k] <= delta for k in range(s)):
            return K_values[j]
    
    return K_values[-1]


def load_classwise_results(results_dir, task):
    """
    Load class-wise F1 scores from K-sweep results
    
    Expected file structure:
    - k_sweep_results.csv: overall metrics
    - class_reports/k_{K}_report.json: per-class metrics (if exists)
    """
    results_dir = Path(results_dir)
    
    # Load overall results
    overall_csv = results_dir / "k_sweep_results.csv"
    if not overall_csv.exists():
        raise FileNotFoundError(f"Not found: {overall_csv}")
    
    df = pd.read_csv(overall_csv)
    
    # Try to load class-wise results
    class_reports_dir = results_dir / "class_reports"
    
    emotion_classes = {
        '4way': ['angry', 'happy', 'sad', 'neutral'],
        '6way': ['angry', 'happy', 'sad', 'neutral', 'excited', 'frustrated']
    }
    
    emotions = emotion_classes.get(task, ['angry', 'happy', 'sad', 'neutral'])
    
    # Check if class-wise F1 is already in CSV
    has_classwise = all(f"{e}_f1" in df.columns for e in emotions)
    
    if has_classwise:
        print("✓ Found class-wise F1 in CSV")
        return df, emotions
    
    # Try to load from class_reports directory
    if class_reports_dir.exists():
        print(f"✓ Found class_reports directory: {class_reports_dir}")
        
        classwise_data = {e: [] for e in emotions}
        
        for k in df['K']:
            report_file = class_reports_dir / f"k_{int(k)}_report.json"
            
            if report_file.exists():
                with open(report_file) as f:
                    report = json.load(f)
                
                for emotion in emotions:
                    if emotion in report:
                        classwise_data[emotion].append(report[emotion]['f1-score'])
                    else:
                        classwise_data[emotion].append(np.nan)
            else:
                for emotion in emotions:
                    classwise_data[emotion].append(np.nan)
        
        # Add to dataframe
        for emotion in emotions:
            df[f"{emotion}_f1"] = classwise_data[emotion]
        
        return df, emotions
    
    print("⚠ Class-wise F1 not found in CSV or class_reports")
    print("   Will need to re-extract from model checkpoints")
    return df, emotions


def plot_emotion_saturation(df, emotions, output_dir):
    """
    Plot emotion-specific K vs F1 curves
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, emotion in enumerate(emotions):
        ax = axes[idx]
        
        col = f"{emotion}_f1"
        if col not in df.columns:
            ax.text(0.5, 0.5, f'No data for {emotion}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Plot curve
        ax.plot(df['K'], df[col], marker='o', linewidth=2, markersize=4, label=emotion.capitalize())
        
        # Mark peak
        peak_idx = df[col].idxmax()
        peak_k = df.loc[peak_idx, 'K']
        peak_f1 = df.loc[peak_idx, col]
        ax.plot(peak_k, peak_f1, 'r*', markersize=15, label=f'Peak: K={peak_k}')
        
        # Mark baseline (K=0)
        baseline_f1 = df.loc[df['K'] == 0, col].values[0]
        ax.axhline(baseline_f1, color='gray', linestyle='--', alpha=0.5, label=f'Baseline: {baseline_f1:.3f}')
        
        ax.set_xlabel('K (Context Window)', fontsize=11)
        ax.set_ylabel('F1 Score', fontsize=11)
        ax.set_title(f'{emotion.capitalize()} - Context Saturation', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'emotion_saturation_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'emotion_saturation_curves.png'}")
    plt.close()


def plot_saturation_comparison(saturation_stats, output_dir, task):
    """
    Plot saturation point comparison across emotions (like Figure 4a from paper)
    """
    emotions = list(saturation_stats.keys())
    
    # Extract p90 values
    p90_values = [saturation_stats[e]['p90'] for e in emotions]
    K_star_values = [saturation_stats[e]['K_star'] for e in emotions]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: K* (absolute)
    bars1 = ax1.barh(emotions, K_star_values, color='steelblue', alpha=0.7)
    ax1.set_xlabel('K* (Saturation Point)', fontsize=12)
    ax1.set_title(f'{task} - Absolute Saturation Points', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width + 2, bar.get_y() + bar.get_height()/2, 
                f'K={int(width)}', va='center', fontsize=10)
    
    # Plot 2: p90 (normalized)
    bars2 = ax2.barh(emotions, p90_values, color='coral', alpha=0.7)
    ax2.set_xlabel('p90 (% of dialogue length)', fontsize=12)
    ax2.set_title(f'{task} - Normalized Saturation Points', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'saturation_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'saturation_comparison.png'}")
    plt.close()


def plot_all_emotions_combined(df, emotions, output_dir):
    """
    Plot all emotions on same graph for comparison
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for emotion in emotions:
        col = f"{emotion}_f1"
        if col in df.columns:
            ax.plot(df['K'], df[col], marker='o', linewidth=2, 
                   markersize=3, label=emotion.capitalize(), alpha=0.8)
    
    # Add overall F1
    ax.plot(df['K'], df['f1_weighted'], 'k--', linewidth=2.5, 
           label='Overall (Weighted)', alpha=0.9)
    
    ax.set_xlabel('K (Context Window)', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Emotion-Specific Context Requirements', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'all_emotions_combined.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'all_emotions_combined.png'}")
    plt.close()


def compute_saturation_stats(df, emotions, dialogue_length=91.7):
    """
    Compute saturation statistics for each emotion
    
    Args:
        dialogue_length: Mean dialogue length for p90 normalization
    """
    stats = {}
    
    for emotion in emotions:
        col = f"{emotion}_f1"
        if col not in df.columns:
            continue
        
        scores = df[col].values
        K_values = df['K'].values
        
        # Find saturation point
        K_star = find_saturation_point(scores, K_values, s=3, delta=0.001)
        
        # Compute p90 (normalized)
        p90 = (K_star / dialogue_length) * 100
        
        # Find peak
        peak_idx = df[col].idxmax()
        peak_K = df.loc[peak_idx, 'K']
        peak_F1 = df.loc[peak_idx, col]
        
        # Baseline (K=0)
        baseline_F1 = df.loc[df['K'] == 0, col].values[0]
        
        # K=100 performance
        k100_F1 = df.loc[df['K'] == 100, col].values[0] if 100 in df['K'].values else np.nan
        
        stats[emotion] = {
            'K_star': K_star,
            'p90': p90,
            'peak_K': peak_K,
            'peak_F1': peak_F1,
            'baseline_F1': baseline_F1,
            'k100_F1': k100_F1,
            'gain_at_peak': peak_F1 - baseline_F1,
            'loss_at_k100': baseline_F1 - k100_F1 if not np.isnan(k100_F1) else np.nan
        }
    
    return stats


def print_summary_table(stats, emotions):
    """
    Print formatted summary table
    """
    print("\n" + "="*100)
    print("EMOTION-SPECIFIC SATURATION ANALYSIS")
    print("="*100)
    print(f"{'Emotion':<12} {'K*':<8} {'p90':<10} {'Peak K':<10} {'Peak F1':<10} {'Baseline':<10} {'K=100':<10} {'Gain':<10} {'Loss':<10}")
    print("-"*100)
    
    for emotion in emotions:
        if emotion not in stats:
            continue
        s = stats[emotion]
        print(f"{emotion.capitalize():<12} "
              f"{s['K_star']:<8.0f} "
              f"{s['p90']:<10.2f} "
              f"{s['peak_K']:<10.0f} "
              f"{s['peak_F1']:<10.3f} "
              f"{s['baseline_F1']:<10.3f} "
              f"{s['k100_F1']:<10.3f} "
              f"{s['gain_at_peak']:<10.3f} "
              f"{s['loss_at_k100']:<10.3f}")
    
    print("="*100)


def main():
    parser = argparse.ArgumentParser(description='Analyze emotion-specific saturation patterns')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Path to results directory')
    parser.add_argument('--task', type=str, default='4way',
                       choices=['4way', '6way'],
                       help='Task type')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots')
    parser.add_argument('--dialogue_length', type=float, default=91.7,
                       help='Mean dialogue length for p90 normalization')
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        args.output_dir = Path(args.results_dir) / "emotion_analysis"
    else:
        args.output_dir = Path(args.output_dir)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Task: {args.task}")
    print()
    
    # Load data
    df, emotions = load_classwise_results(args.results_dir, args.task)
    
    # Compute saturation statistics
    stats = compute_saturation_stats(df, emotions, args.dialogue_length)
    
    # Print summary
    print_summary_table(stats, emotions)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_emotion_saturation(df, emotions, args.output_dir)
    plot_saturation_comparison(stats, emotions, args.output_dir, args.task)
    plot_all_emotions_combined(df, emotions, args.output_dir)
    
    # Save stats to JSON
    stats_file = args.output_dir / 'saturation_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved: {stats_file}")
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
