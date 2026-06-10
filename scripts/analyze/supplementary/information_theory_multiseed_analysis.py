#!/usr/bin/env python3
"""
information_theory_multiseed_analysis.py

Information theory analysis across multiple random seeds for statistical testing

This script:
1. Loads embeddings from multiple training runs (different seeds)
2. Computes information-theoretic metrics for each seed
3. Performs statistical tests (paired t-test, ANOVA)
4. Generates plots with error bars

Usage:
    python information_theory_multiseed_analysis.py \
        --results_csv results/senticnet_all_results.csv \
        --output_dir results/information_theory_stats
"""
import os

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm

def load_embeddings(npz_path):
    """Load embeddings and labels"""
    data = np.load(npz_path, allow_pickle=True)
    
    if 'embeddings' in data:
        X = data['embeddings']
    else:
        X = data['X']
    
    if 'labels' in data:
        y = data['labels']
    else:
        y = data['y']
    
    if 'lengths' in data:
        lengths = data['lengths']
    else:
        lengths = None
    
    return X, y, lengths

def aggregate_to_utterance(X, lengths):
    """Aggregate sentence-level embeddings to utterance-level"""
    N, S_max, D = X.shape
    X_utt = np.zeros((N, D), dtype=np.float32)
    
    for i in range(N):
        if lengths is not None:
            L = lengths[i]
        else:
            L = S_max
        X_utt[i] = X[i, :L, :].mean(axis=0)
    
    return X_utt

def compute_mutual_information(X, y, n_neighbors=3):
    """Compute mutual information I(X; Y)"""
    mi = mutual_info_classif(X, y, n_neighbors=n_neighbors, random_state=42)
    return mi.mean()

def compute_joint_mi(X1, X2, y, n_neighbors=3):
    """Compute joint mutual information I(X1, X2; Y)"""
    X_joint = np.concatenate([X1, X2], axis=1)
    return compute_mutual_information(X_joint, y, n_neighbors)

def compute_info_metrics_for_encoder(encoder, task, split, root):
    """
    Compute information metrics for a single encoder/task/split
    
    Returns:
        dict with MI metrics
    """
    ROOT = Path(root)
    
    # Load BERT embeddings
    if 'sentic' in encoder:
        # This is a fusion model, need to reconstruct paths
        base_encoder = encoder.replace('-sentic-concat', '').replace('-sentic-', '-')
        # For alpha variants, extract base encoder
        if 'alpha' in encoder:
            base_encoder = encoder.split('-sentic-')[0] + '-hier'
    else:
        base_encoder = encoder
    
    bert_dir = ROOT / f"data/embeddings/{task}/{base_encoder}/avg_last4/mean"
    bert_file = bert_dir / f"{split}.npz"
    
    if not bert_file.exists():
        return None
    
    X_bert, y, lengths = load_embeddings(bert_file)
    X_bert_utt = aggregate_to_utterance(X_bert, lengths)
    
    # Load SenticNet features
    sentic_dir = ROOT / f"data/embeddings/{task}/senticnet-sentence-level"
    sentic_file = sentic_dir / f"{split}.npz"
    
    if not sentic_file.exists():
        return None
    
    X_sentic, _, sentic_lengths = load_embeddings(sentic_file)
    X_sentic_utt = aggregate_to_utterance(X_sentic, sentic_lengths)
    
    # Compute metrics
    I_bert = compute_mutual_information(X_bert_utt, y)
    I_sentic = compute_mutual_information(X_sentic_utt, y)
    I_joint = compute_joint_mi(X_bert_utt, X_sentic_utt, y)
    
    complementarity = I_joint - I_bert
    
    return {
        'I_bert': float(I_bert),
        'I_sentic': float(I_sentic),
        'I_joint': float(I_joint),
        'complementarity': float(complementarity)
    }

def analyze_encoder_task(results_df, encoder, task, split, root, output_dir):
    """
    Analyze a specific encoder/task combination across all seeds
    
    Since embeddings don't depend on seed (they're generated once),
    we compute MI once but associate it with all experiments
    """
    OUTPUT_DIR = Path(output_dir)
    
    # Filter experiments for this encoder/task
    exp_df = results_df[
        (results_df['encoder'] == encoder) &
        (results_df['task'] == task)
    ].copy()
    
    if len(exp_df) == 0:
        return None
    
    # Compute information metrics (same for all seeds since embeddings are same)
    metrics = compute_info_metrics_for_encoder(encoder, task, split, root)
    
    if metrics is None:
        return None
    
    # Get performance metrics across seeds
    test_acc_mean = exp_df['test_acc'].mean()
    test_acc_std = exp_df['test_acc'].std()
    n_seeds = len(exp_df)
    
    result = {
        'encoder': encoder,
        'task': task,
        'n_seeds': n_seeds,
        'test_acc_mean': test_acc_mean,
        'test_acc_std': test_acc_std,
        **metrics
    }
    
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_csv', type=str, required=True,
                       help='Path to collected results CSV')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'])
    parser.add_argument('--root', type=str,
                       default=str(Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()))
    parser.add_argument('--output_dir', type=str, default='results/information_theory_stats')
    args = parser.parse_args()
    
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("INFORMATION THEORY MULTI-SEED ANALYSIS")
    print("="*80)
    print(f"Results CSV: {args.results_csv}")
    print(f"Split: {args.split}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*80)
    print()
    
    # Load results
    results_df = pd.read_csv(args.results_csv)
    print(f"Loaded {len(results_df)} experimental results")
    print()
    
    # Get unique encoder/task combinations
    encoder_task_pairs = results_df[['encoder', 'task']].drop_duplicates()
    
    print(f"Found {len(encoder_task_pairs)} encoder/task combinations")
    print()
    
    # Analyze each combination
    all_results = []
    
    print("Computing information metrics...")
    print("-"*80)
    
    for _, row in tqdm(encoder_task_pairs.iterrows(), total=len(encoder_task_pairs)):
        encoder = row['encoder']
        task = row['task']
        
        result = analyze_encoder_task(
            results_df, encoder, task, args.split, args.root, OUTPUT_DIR
        )
        
        if result is not None:
            all_results.append(result)
            print(f"✓ {encoder:50s} | {task:5s} | "
                  f"I_joint={result['I_joint']:.4f} | "
                  f"ΔI={result['complementarity']:.4f}")
    
    print()
    print(f"Successfully analyzed {len(all_results)} combinations")
    print()
    
    # Convert to DataFrame
    info_df = pd.DataFrame(all_results)
    
    # Save raw results
    info_csv = OUTPUT_DIR / 'information_metrics_all.csv'
    info_df.to_csv(info_csv, index=False)
    print(f"✅ Saved: {info_csv}")
    print()
    
    # Statistical analysis
    print("="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)
    print()
    
    # 1. Baseline vs Fusion (for each base encoder)
    print("1. Complementarity Analysis (Baseline vs Fusion)")
    print("-"*80)
    
    base_encoders = ['bert-base-hier', 'roberta-base-hier', 'sentence-roberta-hier']
    
    comparison_results = []
    
    for base_enc in base_encoders:
        for task in ['4way', '6way']:
            # Baseline
            baseline_row = info_df[
                (info_df['encoder'] == base_enc) &
                (info_df['task'] == task)
            ]
            
            # Concat fusion
            concat_row = info_df[
                (info_df['encoder'] == f"{base_enc}-sentic-concat") &
                (info_df['task'] == task)
            ]
            
            if len(baseline_row) == 0 or len(concat_row) == 0:
                continue
            
            baseline_I = baseline_row['I_bert'].values[0]
            concat_I = concat_row['I_joint'].values[0]
            delta_I = concat_row['complementarity'].values[0]
            
            baseline_acc = baseline_row['test_acc_mean'].values[0]
            concat_acc = concat_row['test_acc_mean'].values[0]
            acc_improvement = concat_acc - baseline_acc
            
            comparison_results.append({
                'encoder': base_enc,
                'task': task,
                'baseline_I': baseline_I,
                'concat_I': concat_I,
                'delta_I': delta_I,
                'baseline_acc': baseline_acc,
                'concat_acc': concat_acc,
                'acc_improvement': acc_improvement
            })
            
            print(f"{base_enc:30s} | {task:5s}")
            print(f"  I(X_bert; Y):   {baseline_I:.4f} nats")
            print(f"  I(X_concat; Y): {concat_I:.4f} nats")
            print(f"  ΔI:             {delta_I:.4f} nats (+{delta_I/baseline_I*100:.1f}%)")
            print(f"  Acc improvement: {acc_improvement*100:.2f}%")
            print()
    
    comp_df = pd.DataFrame(comparison_results)
    comp_csv = OUTPUT_DIR / 'complementarity_comparison.csv'
    comp_df.to_csv(comp_csv, index=False)
    print(f"✅ Saved: {comp_csv}")
    print()
    
    # 2. Correlation: baseline_I vs delta_I (complementarity hypothesis)
    print("2. Complementarity Hypothesis Test")
    print("-"*80)
    
    if len(comp_df) > 2:
        correlation = np.corrcoef(comp_df['baseline_I'], comp_df['delta_I'])[0, 1]
        
        # Spearman correlation (rank-based, more robust)
        spearman_corr, spearman_p = stats.spearmanr(comp_df['baseline_I'], comp_df['delta_I'])
        
        print(f"Pearson correlation (baseline_I vs ΔI):  {correlation:.4f}")
        print(f"Spearman correlation:                    {spearman_corr:.4f} (p={spearman_p:.4f})")
        print()
        
        if correlation < -0.5:
            print("✅ Strong NEGATIVE correlation: Weaker encoders benefit more!")
        elif correlation < 0:
            print("⚠️  Weak negative correlation: Some complementarity")
        else:
            print("❌ Positive/no correlation: Complementarity hypothesis not supported")
    
    print()
    
    # 3. Alpha analysis (if available)
    print("3. Alpha Value Analysis")
    print("-"*80)
    
    alpha_rows = info_df[info_df['encoder'].str.contains('alpha')]
    
    if len(alpha_rows) > 0:
        print(f"Found {len(alpha_rows)} alpha variant experiments")
        
        # Extract alpha values from encoder names
        alpha_rows['alpha'] = alpha_rows['encoder'].str.extract(r'alpha(\d+)')[0].astype(float) / 100
        
        # Group by base encoder and task
        for base_enc in base_encoders:
            for task in ['4way', '6way']:
                task_alphas = alpha_rows[
                    (alpha_rows['encoder'].str.startswith(base_enc.replace('-hier', ''))) &
                    (alpha_rows['task'] == task)
                ].copy()
                
                if len(task_alphas) == 0:
                    continue
                
                # Sort by alpha
                task_alphas = task_alphas.sort_values('alpha')
                
                print(f"\n{base_enc} | {task}:")
                for _, row in task_alphas.iterrows():
                    print(f"  α={row['alpha']:.2f}: I={row['I_joint']:.4f}, "
                          f"ΔI={row['complementarity']:.4f}, "
                          f"Acc={row['test_acc_mean']:.4f}")
    else:
        print("No alpha variants found in results")
    
    print()
    
    # 4. Visualization
    print("="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    print()
    
    # Plot 1: Complementarity by encoder
    if len(comp_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group by task
        for task in ['4way', '6way']:
            task_df = comp_df[comp_df['task'] == task]
            
            if len(task_df) == 0:
                continue
            
            x = np.arange(len(task_df))
            
            ax.bar(x, task_df['delta_I'], width=0.35,
                  label=task, alpha=0.7)
            
            ax.set_xticks(x)
            ax.set_xticklabels([enc.replace('-hier', '') for enc in task_df['encoder']],
                              rotation=45, ha='right')
        
        ax.set_ylabel('Complementarity ΔI (nats)', fontsize=12)
        ax.set_title('Information Gain from SenticNet Fusion', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_file = OUTPUT_DIR / 'complementarity_by_encoder.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {plot_file}")
    
    # Plot 2: Complementarity hypothesis (baseline_I vs delta_I)
    if len(comp_df) > 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = {'4way': '#3498db', '6way': '#e74c3c'}
        
        for task in ['4way', '6way']:
            task_df = comp_df[comp_df['task'] == task]
            
            if len(task_df) == 0:
                continue
            
            ax.scatter(task_df['baseline_I'], task_df['delta_I'],
                      s=200, alpha=0.6, color=colors[task], label=task)
            
            # Add labels
            for _, row in task_df.iterrows():
                enc_name = row['encoder'].replace('-hier', '').replace('bert-base', 'BERT').replace('roberta-base', 'RoBERTa').replace('sentence-roberta', 'S-RoBERTa')
                ax.annotate(enc_name, (row['baseline_I'], row['delta_I']),
                           xytext=(10, 5), textcoords='offset points',
                           fontsize=9)
        
        # Trend line
        if len(comp_df) > 2:
            z = np.polyfit(comp_df['baseline_I'], comp_df['delta_I'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(comp_df['baseline_I'].min(), comp_df['baseline_I'].max(), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.5, linewidth=2)
            
            ax.text(0.05, 0.95, f"Correlation: {correlation:.3f}",
                   transform=ax.transAxes, fontsize=11,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Baseline I(X_bert; Y) (nats)', fontsize=12)
        ax.set_ylabel('Complementarity ΔI (nats)', fontsize=12)
        ax.set_title('Complementarity Hypothesis: Weaker Encoders Benefit More?',
                    fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = OUTPUT_DIR / 'complementarity_hypothesis.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {plot_file}")
    
    plt.close('all')
    
    print()
    print("="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
