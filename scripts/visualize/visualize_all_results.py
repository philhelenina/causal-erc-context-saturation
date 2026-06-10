#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_all_results_enhanced.py
Enhanced visualizations with star markers for best configs and value labels
"""
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
HOME = Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()
RESULTS_BASE = HOME / "results" / "analysis"

def load_summary():
    """Load combined summary statistics"""
    summary_file = RESULTS_BASE / "summary_combined.csv"
    if not summary_file.exists():
        print(f"⚠️  Run aggregate_all_results.py first!")
        return None
    return pd.read_csv(summary_file)

def plot_aggregator_comparison_enhanced(df, output_dir):
    """Compare aggregators with VALUES and STARS"""
    
    hier_df = df[df['type'] == 'hierarchical']
    
    if hier_df.empty:
        print("⚠️  No hierarchical data for aggregator comparison")
        return
    
    # Group by task and aggregator
    agg_stats = hier_df.groupby(['task', 'aggregator']).agg({
        'weighted_f1_mean': ['mean', 'std'],
        'macro_f1_mean': ['mean', 'std']
    }).reset_index()
    
    agg_stats.columns = ['task', 'aggregator', 'wf1_mean', 'wf1_std', 'mf1_mean', 'mf1_std']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    task_colors = {'4way': 'steelblue', '6way': 'coral'}
    
    # Weighted F1
    ax = axes[0]
    for task in ['4way', '6way']:
        task_data = agg_stats[agg_stats['task'] == task].sort_values('wf1_mean', ascending=False)
        x = np.arange(len(task_data))
        width = 0.35
        offset = -width/2 if task == '4way' else width/2
        
        bars = ax.bar(
            x + offset, task_data['wf1_mean'], width,
            yerr=task_data['wf1_std'],
            alpha=0.7,
            label=task,
            color=task_colors[task],
            capsize=5
        )
        
        # Add value labels ONLY (no stars on histograms!)
        for i, (idx, row) in enumerate(task_data.iterrows()):
            ax.text(
                i + offset, row['wf1_mean'] + 0.01,
                f"{row['wf1_mean']:.4f}",
                ha='center', va='bottom', fontsize=10,
                fontweight='normal'
            )
    
    ax.set_xlabel('Aggregator', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weighted F1', fontsize=12, fontweight='bold')
    ax.set_title('Aggregator Comparison: Weighted F1 (Hierarchical)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(task_data['aggregator'], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Macro F1
    ax = axes[1]
    for task in ['4way', '6way']:
        task_data = agg_stats[agg_stats['task'] == task].sort_values('mf1_mean', ascending=False)
        x = np.arange(len(task_data))
        width = 0.35
        offset = -width/2 if task == '4way' else width/2
        
        bars = ax.bar(
            x + offset, task_data['mf1_mean'], width,
            yerr=task_data['mf1_std'],
            alpha=0.7,
            label=task,
            color=task_colors[task],
            capsize=5
        )
        
        # Add value labels ONLY (no stars!)
        for i, (idx, row) in enumerate(task_data.iterrows()):
            ax.text(
                i + offset, row['mf1_mean'] + 0.01,
                f"{row['mf1_mean']:.4f}",
                ha='center', va='bottom', fontsize=10,
                fontweight='normal'
            )
    
    ax.set_xlabel('Aggregator', fontsize=12, fontweight='bold')
    ax.set_ylabel('Macro F1', fontsize=12, fontweight='bold')
    ax.set_title('Aggregator Comparison: Macro F1 (Hierarchical)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(task_data['aggregator'], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig.savefig(output_dir / "aggregator_comparison_enhanced.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir / 'aggregator_comparison_enhanced.png'}")

def plot_encoder_comparison_enhanced(df, output_dir):
    """Compare encoders with VALUES and STARS"""
    
    flat_df = df[df['type'] == 'flat']
    
    if flat_df.empty:
        print("⚠️  No flat baseline data for encoder comparison")
        return
    
    # Group by task and encoder
    encoder_stats = flat_df.groupby(['task', 'encoder']).agg({
        'weighted_f1_mean': ['mean', 'std'],
        'macro_f1_mean': ['mean', 'std']
    }).reset_index()
    
    encoder_stats.columns = ['task', 'encoder', 'wf1_mean', 'wf1_std', 'mf1_mean', 'mf1_std']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    task_colors = {'4way': 'steelblue', '6way': 'coral'}
    
    # Weighted F1
    ax = axes[0]
    for task in ['4way', '6way']:
        task_data = encoder_stats[encoder_stats['task'] == task]
        x = np.arange(len(task_data))
        width = 0.35
        offset = -width/2 if task == '4way' else width/2
        
        bars = ax.bar(
            x + offset, task_data['wf1_mean'], width,
            yerr=task_data['wf1_std'],
            alpha=0.7,
            label=task,
            color=task_colors[task],
            capsize=5
        )
        
        # Add value labels ONLY
        for i, (idx, row) in enumerate(task_data.iterrows()):
            ax.text(
                i + offset, row['wf1_mean'] + 0.005,
                f"{row['wf1_mean']:.4f}",
                ha='center', va='bottom', fontsize=10,
                fontweight='normal'
            )
    
    ax.set_xlabel('Encoder', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weighted F1', fontsize=12, fontweight='bold')
    ax.set_title('Encoder Comparison: Weighted F1 (Flat Baseline)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(task_data['encoder'], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Macro F1
    ax = axes[1]
    for task in ['4way', '6way']:
        task_data = encoder_stats[encoder_stats['task'] == task]
        x = np.arange(len(task_data))
        width = 0.35
        offset = -width/2 if task == '4way' else width/2
        
        bars = ax.bar(
            x + offset, task_data['mf1_mean'], width,
            yerr=task_data['mf1_std'],
            alpha=0.7,
            label=task,
            color=task_colors[task],
            capsize=5
        )
        
        # Add value labels ONLY
        for i, (idx, row) in enumerate(task_data.iterrows()):
            ax.text(
                i + offset, row['mf1_mean'] + 0.005,
                f"{row['mf1_mean']:.4f}",
                ha='center', va='bottom', fontsize=10,
                fontweight='normal'
            )
    
    ax.set_xlabel('Encoder', fontsize=12, fontweight='bold')
    ax.set_ylabel('Macro F1', fontsize=12, fontweight='bold')
    ax.set_title('Encoder Comparison: Macro F1 (Flat Baseline)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(task_data['encoder'], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig.savefig(output_dir / "encoder_comparison_enhanced.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir / 'encoder_comparison_enhanced.png'}")

def plot_top_configs_enhanced(df, output_dir, top_k=10):
    """Bar plot with STARS on best configs"""
    
    for task in ['4way', '6way']:
        for exp_type in ['flat', 'hierarchical']:
            subset = df[(df['task'] == task) & (df['type'] == exp_type)]
            
            if subset.empty:
                continue
            
            # Sort and get top K
            top_df = subset.sort_values('weighted_f1_mean', ascending=False).head(top_k).copy()
            
            # Create config label
            if exp_type == 'flat':
                top_df['config'] = (
                    top_df['encoder'].str[:15] + '/' +
                    top_df['layer'].str[:6] + '/' +
                    top_df['pool'].str[:8] + '/' +
                    top_df['classifier']
                )
            else:  # hierarchical
                top_df['config'] = (
                    top_df['encoder'].str[:12] + '/' +
                    top_df['layer'].str[:6] + '/' +
                    top_df['pool'].str[:8] + '/' +
                    top_df['aggregator'] + '/' +
                    top_df['classifier']
                )
            
            fig, ax = plt.subplots(figsize=(14, 7))
            
            x = range(len(top_df))
            bars = ax.barh(
                x,
                top_df['weighted_f1_mean'],
                xerr=top_df['weighted_f1_std'],
                alpha=0.7,
                capsize=5,
                color='steelblue'
            )
            
            # Color the best one differently
            bars[0].set_color('gold')
            bars[0].set_alpha(1.0)
            
            ax.set_yticks(x)
            ax.set_yticklabels(top_df['config'], fontsize=10)
            ax.set_xlabel('Weighted F1', fontsize=12, fontweight='bold')
            ax.set_title(f'Top {top_k} Configurations ({task.upper()} - {exp_type.capitalize()})', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            ax.invert_yaxis()
            
            # Add values (no star emoji!)
            for i, (idx, row) in enumerate(top_df.iterrows()):
                ax.text(
                    row['weighted_f1_mean'] + 0.002,
                    i,
                    f"{row['weighted_f1_mean']:.4f}",
                    va='center',
                    fontsize=10,
                    fontweight='bold' if i == 0 else 'normal'
                )
            
            plt.tight_layout()
            fig.savefig(output_dir / f"top_configs_{task}_{exp_type}_enhanced.png", 
                       dpi=200, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: {output_dir / f'top_configs_{task}_{exp_type}_enhanced.png'}")

def plot_heatmap_flat_with_star(df, output_dir):
    """Heatmap for flat baseline with STAR on best value"""
    
    flat_df = df[df['type'] == 'flat']
    
    if flat_df.empty:
        print("⚠️  No flat data for heatmap")
        return
    
    for task in ['4way', '6way']:
        task_df = flat_df[flat_df['task'] == task]
        
        if task_df.empty:
            continue
        
        # Pivot: encoder/layer vs pool/classifier
        pivot = task_df.pivot_table(
            values='weighted_f1_mean',
            index=['encoder', 'layer'],
            columns=['pool', 'classifier'],
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Weighted F1'},
            ax=ax,
            vmin=pivot.min().min(),
            vmax=pivot.max().max(),
            annot_kws={'fontsize': 10}
        )
        
        # Add STAR on the maximum value
        max_val = pivot.max().max()
        max_pos = np.where(pivot.values == max_val)
        
        if len(max_pos[0]) > 0:
            row_idx = max_pos[0][0]
            col_idx = max_pos[1][0]
            
            # Add white star
            ax.text(
                col_idx + 0.5, row_idx + 0.2,
                '★',
                ha='center', va='center',
                fontsize=30, color='white',
                weight='bold',
                zorder=10
            )
        
        ax.set_title(f'Performance Heatmap: Flat Baseline ({task.upper()})', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Pool Method + Classifier', fontsize=13, fontweight='bold')
        ax.set_ylabel('Encoder + Layer', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        fig.savefig(output_dir / f"heatmap_flat_{task}_star.png", dpi=200, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir / f'heatmap_flat_{task}_star.png'}")

def plot_heatmap_hierarchical_with_star(df, output_dir):
    """Heatmap for hierarchical with STAR on best value"""
    
    hier_df = df[df['type'] == 'hierarchical']
    
    if hier_df.empty:
        print("⚠️  No hierarchical data for heatmap")
        return
    
    for task in ['4way', '6way']:
        task_df = hier_df[hier_df['task'] == task]
        
        if task_df.empty:
            continue
        
        # Pivot: aggregator/classifier vs pool/layer
        pivot = task_df.pivot_table(
            values='weighted_f1_mean',
            index=['aggregator', 'classifier'],
            columns=['pool', 'layer'],
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Weighted F1'},
            ax=ax,
            vmin=pivot.min().min(),
            vmax=pivot.max().max(),
            annot_kws={'fontsize': 10}
        )
        
        # Add STAR on the maximum value
        max_val = pivot.max().max()
        max_pos = np.where(pivot.values == max_val)
        
        if len(max_pos[0]) > 0:
            row_idx = max_pos[0][0]
            col_idx = max_pos[1][0]
            
            # Add white star
            ax.text(
                col_idx + 0.5, row_idx + 0.2,
                '★',
                ha='center', va='center',
                fontsize=30, color='white',
                weight='bold',
                zorder=10
            )
        
        ax.set_title(f'Performance Heatmap: Hierarchical ({task.upper()})', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Pool Method + Layer', fontsize=13, fontweight='bold')
        ax.set_ylabel('Aggregator + Classifier', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        fig.savefig(output_dir / f"heatmap_hierarchical_{task}_star.png", dpi=200, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir / f'heatmap_hierarchical_{task}_star.png'}")

def create_results_summary_table(df, output_dir):
    """Create a summary table showing best configs"""
    
    summary_rows = []
    
    for task in ['4way', '6way']:
        for exp_type in ['flat', 'hierarchical']:
            subset = df[(df['task'] == task) & (df['type'] == exp_type)]
            
            if subset.empty:
                continue
            
            best_row = subset.loc[subset['weighted_f1_mean'].idxmax()]
            
            config_str = f"{best_row['encoder']}/{best_row['layer']}/{best_row['pool']}"
            if exp_type == 'hierarchical':
                config_str += f"/{best_row['aggregator']}"
            config_str += f"/{best_row['classifier']}"
            
            summary_rows.append({
                'Task': task.upper(),
                'Type': exp_type.capitalize(),
                'Configuration': config_str,
                'Weighted F1': f"{best_row['weighted_f1_mean']:.4f} ± {best_row['weighted_f1_std']:.4f}",
                'Macro F1': f"{best_row['macro_f1_mean']:.4f} ± {best_row['macro_f1_std']:.4f}",
                'Accuracy': f"{best_row['acc_mean']:.4f} ± {best_row['acc_std']:.4f}"
            })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "best_configs_summary.csv", index=False)
    
    # Also create a nice formatted text file
    with open(output_dir / "BEST_RESULTS.txt", 'w') as f:
        f.write("="*100 + "\n")
        f.write("BEST CONFIGURATIONS SUMMARY\n")
        f.write("="*100 + "\n\n")
        
        for _, row in summary_df.iterrows():
            f.write(f"{row['Task']} - {row['Type']}:\n")
            f.write(f"  Configuration: {row['Configuration']}\n")
            f.write(f"  Weighted F1:   {row['Weighted F1']}\n")
            f.write(f"  Macro F1:      {row['Macro F1']}\n")
            f.write(f"  Accuracy:      {row['Accuracy']}\n")
            f.write("\n")
    
    print(f"\n✅ Saved: {output_dir / 'best_configs_summary.csv'}")
    print(f"✅ Saved: {output_dir / 'BEST_RESULTS.txt'}")
    
    return summary_df

def main():
    print("="*100)
    print("ENHANCED RESULTS VISUALIZATION (with stars and values)")
    print("="*100)
    print()
    
    # Load data
    df = load_summary()
    if df is None:
        return
    
    print(f"Loaded {len(df)} configurations")
    print(f"  Flat:         {len(df[df['type']=='flat'])}")
    print(f"  Hierarchical: {len(df[df['type']=='hierarchical'])}")
    print()
    
    # Create output directory
    output_dir = RESULTS_BASE / "figures_enhanced"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate enhanced plots
    print("Generating enhanced visualizations...")
    print()
    
    plot_encoder_comparison_enhanced(df, output_dir)
    plot_aggregator_comparison_enhanced(df, output_dir)
    plot_top_configs_enhanced(df, output_dir, top_k=10)
    
    print("\n📊 Creating heatmaps with stars...")
    plot_heatmap_flat_with_star(df, output_dir)
    plot_heatmap_hierarchical_with_star(df, output_dir)
    
    # Create summary table
    print("\n" + "="*100)
    print("Creating summary table...")
    print("="*100)
    summary_df = create_results_summary_table(df, output_dir)
    
    print("\n" + "="*100)
    print("📊 BEST CONFIGURATIONS:")
    print("="*100)
    print(summary_df.to_string(index=False))
    
    print()
    print("="*100)
    print("✅ ENHANCED VISUALIZATION COMPLETE")
    print("="*100)
    print(f"\nAll figures saved to: {output_dir}")
    print(f"\nGenerated plots:")
    print(f"  Histograms (values only, no stars):")
    print(f"    - encoder_comparison_enhanced.png")
    print(f"    - aggregator_comparison_enhanced.png")
    print(f"  Top configs (gold bar for #1):")
    print(f"    - top_configs_[task]_[type]_enhanced.png (8 files)")
    print(f"  Heatmaps (★ on best value):")
    print(f"    - heatmap_flat_[task]_star.png (2 files)")
    print(f"    - heatmap_hierarchical_[task]_star.png (2 files)")
    print(f"  Summary:")
    print(f"    - best_configs_summary.csv")
    print(f"    - BEST_RESULTS.txt")

if __name__ == "__main__":
    main()