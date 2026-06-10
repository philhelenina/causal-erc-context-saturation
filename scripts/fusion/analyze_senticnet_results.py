#!/usr/bin/env python3
"""
analyze_senticnet_results.py

Statistical analysis comparing baseline vs SenticNet fusion
- Paired t-test (same encoder, seed)
- Effect size (Cohen's d)
- Alpha variants comparison
- Visualization
"""
import os

import pandas as pd
import numpy as np
from scipy import stats
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(str(Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()))
RESULTS_DIR = ROOT / 'results' / 'senticnet_experiments'

def load_results(results_csv):
    """Load results CSV with proper parsing"""
    
    if not Path(results_csv).exists():
        raise FileNotFoundError(f"Results file not found: {results_csv}")
    
    df = pd.read_csv(results_csv)
    
    print(f"Loaded {len(df)} results from: {results_csv}")
    print(f"Columns: {df.columns.tolist()}")
    print()
    
    # Add fusion type column
    df['fusion_type'] = 'unknown'
    
    # Baseline: no sentic in encoder name
    df.loc[~df['encoder'].str.contains('sentic'), 'fusion_type'] = 'baseline'
    
    # Concat: sentic-concat
    df.loc[df['encoder'].str.contains('sentic-concat'), 'fusion_type'] = 'concat'
    
    # Alpha: sentic-alpha
    df.loc[df['encoder'].str.contains('sentic-alpha'), 'fusion_type'] = 'alpha'
    
    # Extract base encoder name
    df['base_encoder'] = df['encoder'].str.replace('-sentic-concat', '').str.replace(r'-sentic-alpha\d+', '', regex=True)
    
    # Extract alpha value for alpha variants
    df['alpha_value'] = np.nan
    alpha_mask = df['encoder'].str.contains('alpha')
    if alpha_mask.any():
        df.loc[alpha_mask, 'alpha_value'] = df.loc[alpha_mask, 'encoder'].str.extract(r'alpha(\d+)')[0].astype(float) / 100
    
    print("Fusion type distribution:")
    print(df['fusion_type'].value_counts())
    print()
    
    return df

def paired_comparison(df, task, base_encoder, fusion_type='concat', alpha_value=None):
    """
    Paired t-test comparing baseline vs fusion
    
    Args:
        df: Results dataframe
        task: '4way' or '6way'
        base_encoder: base encoder name (e.g., 'bert-base-hier')
        fusion_type: 'concat' or 'alpha'
        alpha_value: alpha value if fusion_type='alpha'
    """
    
    # Baseline
    baseline = df[
        (df['task'] == task) &
        (df['base_encoder'] == base_encoder) &
        (df['fusion_type'] == 'baseline')
    ].copy()
    
    # Fusion
    if fusion_type == 'concat':
        fusion = df[
            (df['task'] == task) &
            (df['base_encoder'] == base_encoder) &
            (df['fusion_type'] == 'concat')
        ].copy()
        fusion_name = 'Concat'
    elif fusion_type == 'alpha':
        fusion = df[
            (df['task'] == task) &
            (df['base_encoder'] == base_encoder) &
            (df['fusion_type'] == 'alpha') &
            (df['alpha_value'] == alpha_value)
        ].copy()
        fusion_name = f'Alpha={alpha_value:.2f}'
    else:
        raise ValueError(f"Unknown fusion_type: {fusion_type}")
    
    if len(baseline) == 0 or len(fusion) == 0:
        print(f"[WARNING] No data for {task} / {base_encoder} / {fusion_name}")
        print(f"  Baseline: {len(baseline)}, Fusion: {len(fusion)}")
        return None
    
    # Match seeds
    common_seeds = sorted(set(baseline['seed']).intersection(set(fusion['seed'])))
    
    if len(common_seeds) == 0:
        print(f"[WARNING] No common seeds for {task} / {base_encoder} / {fusion_name}")
        return None
    
    baseline = baseline[baseline['seed'].isin(common_seeds)].sort_values('seed').reset_index(drop=True)
    fusion = fusion[fusion['seed'].isin(common_seeds)].sort_values('seed').reset_index(drop=True)
    
    # Extract scores (use test_acc or test_weighted_f1, whichever is available)
    if 'test_weighted_f1' in baseline.columns:
        score_col = 'test_weighted_f1'
    elif 'test_acc' in baseline.columns:
        score_col = 'test_acc'
    elif 'test_f1' in baseline.columns:
        score_col = 'test_f1'
    else:
        print(f"[ERROR] No test score column found!")
        return None
    
    baseline_scores = baseline[score_col].values
    fusion_scores = fusion[score_col].values
    
    # Paired differences
    differences = fusion_scores - baseline_scores
    
    # Statistics
    n = len(differences)
    diff_mean = differences.mean()
    diff_std = differences.std(ddof=1)
    diff_se = diff_std / np.sqrt(n)
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(fusion_scores, baseline_scores)
    
    # Cohen's d (paired)
    cohens_d = diff_mean / diff_std if diff_std > 0 else 0
    
    # Effect size interpretation
    if abs(cohens_d) >= 0.8:
        effect = "large"
    elif abs(cohens_d) >= 0.5:
        effect = "medium"
    elif abs(cohens_d) >= 0.2:
        effect = "small"
    else:
        effect = "negligible"
    
    # Significance
    if p_value < 0.001:
        sig = "***"
    elif p_value < 0.01:
        sig = "**"
    elif p_value < 0.05:
        sig = "*"
    else:
        sig = "ns"
    
    return {
        'task': task,
        'base_encoder': base_encoder,
        'fusion_type': fusion_type,
        'fusion_name': fusion_name,
        'alpha_value': alpha_value,
        'n_pairs': n,
        'seeds': common_seeds,
        'baseline_mean': baseline_scores.mean(),
        'baseline_std': baseline_scores.std(ddof=1),
        'fusion_mean': fusion_scores.mean(),
        'fusion_std': fusion_scores.std(ddof=1),
        'diff_mean': diff_mean,
        'diff_std': diff_std,
        'diff_se': diff_se,
        't_stat': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'effect_size': effect,
        'significance': sig,
        'baseline_scores': baseline_scores,
        'fusion_scores': fusion_scores,
        'differences': differences,
        'score_col': score_col
    }

def print_comparison(result):
    """Print formatted comparison results"""
    
    if result is None:
        return
    
    print("="*80)
    print(f"{result['task'].upper()} / {result['base_encoder']} / {result['fusion_name']}")
    print("="*80)
    print()
    
    print(f"Sample size: n = {result['n_pairs']} pairs")
    print(f"Seeds: {result['seeds']}")
    print(f"Metric: {result['score_col']}")
    print()
    
    print(f"Baseline:")
    print(f"  Mean: {result['baseline_mean']:.6f} ± {result['baseline_std']:.6f}")
    print()
    
    print(f"Fusion ({result['fusion_name']}):")
    print(f"  Mean: {result['fusion_mean']:.6f} ± {result['fusion_std']:.6f}")
    print()
    
    print(f"Improvement:")
    print(f"  Absolute: {result['diff_mean']:+.6f}")
    print(f"  Relative: {result['diff_mean']/result['baseline_mean']*100:+.2f}%")
    print()
    
    print(f"Paired t-test:")
    print(f"  t({result['n_pairs']-1}) = {result['t_stat']:.4f}")
    print(f"  p = {result['p_value']:.6f} {result['significance']}")
    print()
    
    print(f"Effect Size (Cohen's d):")
    print(f"  d = {result['cohens_d']:.4f} ({result['effect_size']})")
    print()
    
    print("CONCLUSION:")
    if result['p_value'] < 0.05:
        print(f"  ✅ {result['fusion_name']} significantly improves performance")
    else:
        print(f"  ❌ No significant improvement from {result['fusion_name']}")
    print()
    print("="*80)
    print()

def plot_results(all_results, output_dir):
    """Create visualization plots"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter concat results only for first plot
    concat_results = [r for r in all_results if r['fusion_type'] == 'concat']
    
    if len(concat_results) > 0:
        # Plot 1: Baseline vs Concat by encoder and task
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for ax, task in zip(axes, ['4way', '6way']):
            task_results = [r for r in concat_results if r['task'] == task]
            
            if len(task_results) == 0:
                continue
            
            encoders = [r['base_encoder'].replace('-hier', '') for r in task_results]
            x = np.arange(len(encoders))
            width = 0.35
            
            baseline_means = [r['baseline_mean'] for r in task_results]
            baseline_stds = [r['baseline_std'] for r in task_results]
            fusion_means = [r['fusion_mean'] for r in task_results]
            fusion_stds = [r['fusion_std'] for r in task_results]
            
            ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds,
                   label='Baseline', alpha=0.8, capsize=5, color='steelblue')
            ax.bar(x + width/2, fusion_means, width, yerr=fusion_stds,
                   label='+ SenticNet', alpha=0.8, capsize=5, color='coral')
            
            ax.set_xlabel('Encoder', fontsize=12)
            ax.set_ylabel('Test Score', fontsize=12)
            ax.set_title(f'{task.upper()} Classification', fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(encoders, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add significance markers
            for i, r in enumerate(task_results):
                if r['significance'] != 'ns':
                    y = max(r['baseline_mean'], r['fusion_mean']) + max(r['baseline_std'], r['fusion_std'])
                    ax.text(i, y, r['significance'], ha='center', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plot_file = output_dir / 'baseline_vs_concat.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {plot_file}")
        plt.close()
    
    # Plot 2: Alpha comparison (if alpha results exist)
    alpha_results = [r for r in all_results if r['fusion_type'] == 'alpha']
    
    if len(alpha_results) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        idx = 0
        for task in ['4way', '6way']:
            for encoder in sorted(set(r['base_encoder'] for r in alpha_results)):
                task_enc_results = [r for r in alpha_results 
                                   if r['task'] == task and r['base_encoder'] == encoder]
                
                if len(task_enc_results) == 0:
                    continue
                
                # Sort by alpha
                task_enc_results = sorted(task_enc_results, key=lambda x: x['alpha_value'])
                
                alphas = [r['alpha_value'] for r in task_enc_results]
                improvements = [r['diff_mean'] * 100 for r in task_enc_results]
                p_values = [r['p_value'] for r in task_enc_results]
                
                ax = axes[idx]
                ax.plot(alphas, improvements, 'o-', linewidth=2, markersize=8)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                
                # Mark significant points
                for alpha, imp, p in zip(alphas, improvements, p_values):
                    if p < 0.05:
                        ax.plot(alpha, imp, 'r*', markersize=15)
                
                ax.set_xlabel('Alpha (α)', fontsize=11)
                ax.set_ylabel('Improvement (%)', fontsize=11)
                ax.set_title(f'{task.upper()} / {encoder.replace("-hier", "")}', 
                           fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                idx += 1
        
        # Remove unused subplots
        for i in range(idx, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plot_file = output_dir / 'alpha_comparison.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {plot_file}")
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_csv', type=str, 
                       default=str(RESULTS_DIR / 'results.csv'),
                       help='Results CSV file')
    parser.add_argument('--output_dir', type=str,
                       default=str(RESULTS_DIR / 'analysis'),
                       help='Output directory')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("SENTICNET FUSION STATISTICAL ANALYSIS")
    print("="*80)
    print()
    
    # Load data
    df = load_results(args.results_csv)
    
    # Get unique combinations
    encoders = sorted(df['base_encoder'].unique())
    tasks = sorted(df['task'].unique())
    
    print(f"Found {len(encoders)} encoders: {encoders}")
    print(f"Found {len(tasks)} tasks: {tasks}")
    print()
    
    # All comparisons
    all_results = []
    
    # 1. Baseline vs Concat
    print("\n" + "="*80)
    print("BASELINE vs CONCAT FUSION")
    print("="*80)
    
    for task in tasks:
        for encoder in encoders:
            result = paired_comparison(df, task, encoder, fusion_type='concat')
            
            if result is not None:
                print_comparison(result)
                all_results.append(result)
    
    # 2. Baseline vs Alpha variants
    alpha_values = sorted(df[df['fusion_type'] == 'alpha']['alpha_value'].dropna().unique())
    
    if len(alpha_values) > 0:
        print("\n" + "="*80)
        print("BASELINE vs ALPHA VARIANTS")
        print("="*80)
        print(f"Alpha values: {alpha_values}")
        print()
        
        for task in tasks:
            for encoder in encoders:
                for alpha in alpha_values:
                    result = paired_comparison(df, task, encoder, 
                                              fusion_type='alpha', alpha_value=alpha)
                    
                    if result is not None:
                        print_comparison(result)
                        all_results.append(result)
    
    # Save summary
    if len(all_results) > 0:
        summary_data = []
        
        for r in all_results:
            summary_data.append({
                'task': r['task'],
                'base_encoder': r['base_encoder'],
                'fusion_type': r['fusion_name'],
                'alpha': r['alpha_value'] if r['alpha_value'] else '',
                'n': r['n_pairs'],
                'baseline_mean': f"{r['baseline_mean']:.6f}",
                'fusion_mean': f"{r['fusion_mean']:.6f}",
                'improvement_pct': f"{r['diff_mean']/r['baseline_mean']*100:+.2f}%",
                't_stat': f"{r['t_stat']:.4f}",
                'p_value': f"{r['p_value']:.6f}",
                'cohens_d': f"{r['cohens_d']:.4f}",
                'significance': r['significance']
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        summary_csv = output_dir / 'comparison_summary.csv'
        summary_df.to_csv(summary_csv, index=False)
        
        print("\n" + "="*80)
        print("SUMMARY TABLE")
        print("="*80)
        print()
        print(summary_df.to_string(index=False))
        print()
        
        print(f"✅ Summary saved to: {summary_csv}")
        
        # Create plots
        print("\nGenerating plots...")
        plot_results(all_results, output_dir)
    else:
        print("[ERROR] No valid comparisons found!")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Output directory: {output_dir}")

if __name__ == '__main__':
    main()