#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_all_results_n10.py
Aggregate n=10 experiments from results_n10/ directory

NEW STRUCTURE (results_n10):
- Flat: results_n10/[task]/flat/[encoder]/[layer]/[pool]/[classifier]/seed_[X]/results.json
- Hier: results_n10/[task]/hierarchical/[encoder]/[layer]/[pool]/[aggregator]/[classifier]/seed_[X]/results.json
"""
import os

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# ============================================================================
# EDIT THIS PATH TO MATCH YOUR SETUP
# ============================================================================
BASE_DIR = Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()
RESULTS_BASE = BASE_DIR / "results_n10"

def collect_flat_results(task="4way"):
    """Collect flat baseline results from new structure"""
    
    results = []
    task_dir = RESULTS_BASE / task / "flat"
    
    if not task_dir.exists():
        print(f"⚠️  Warning: {task_dir} does not exist")
        return pd.DataFrame()
    
    # Find all results.json files
    result_files = list(task_dir.rglob("results.json"))
    
    print(f"  Found {len(result_files)} flat result files for {task}")
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Parse path: results_n10/4way/flat/[encoder]/[layer]/[pool]/[classifier]/seed_XX/results.json
            parts = result_file.parts
            
            try:
                # Find indices
                flat_idx = parts.index("flat")
                encoder = parts[flat_idx + 1]
                layer = parts[flat_idx + 2]
                pool = parts[flat_idx + 3]
                classifier = parts[flat_idx + 4]
                seed_str = parts[flat_idx + 5]
                seed = int(seed_str.split('_')[1]) if 'seed_' in seed_str else 42
            except (ValueError, IndexError) as e:
                print(f"  ⚠️  Could not parse path: {result_file}")
                continue
            
            # Extract metrics
            metrics = data.get('metrics', {})
            
            result = {
                'type': 'flat',
                'task': task,
                'encoder': encoder,
                'layer': layer,
                'pool': pool,
                'aggregator': None,
                'classifier': classifier,
                'seed': seed,
                'accuracy': metrics.get('accuracy', np.nan),
                'macro_f1': metrics.get('macro_f1', np.nan),
                'weighted_f1': metrics.get('weighted_f1', np.nan),
                'path': str(result_file)
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"  ⚠️  Error reading {result_file}: {e}")
            continue
    
    return pd.DataFrame(results)

def collect_hierarchical_results(task="4way"):
    """Collect hierarchical results from new structure"""
    
    results = []
    task_dir = RESULTS_BASE / task / "hierarchical"
    
    if not task_dir.exists():
        print(f"⚠️  Warning: {task_dir} does not exist")
        return pd.DataFrame()
    
    # Find all results.json files
    result_files = list(task_dir.rglob("results.json"))
    
    print(f"  Found {len(result_files)} hierarchical result files for {task}")
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Parse path: results_n10/4way/hierarchical/[encoder]/[layer]/[pool]/[aggregator]/[classifier]/seed_XX/results.json
            parts = result_file.parts
            
            try:
                # Find indices
                hier_idx = parts.index("hierarchical")
                encoder = parts[hier_idx + 1]
                layer = parts[hier_idx + 2]
                pool = parts[hier_idx + 3]
                aggregator = parts[hier_idx + 4]
                classifier = parts[hier_idx + 5]
                seed_str = parts[hier_idx + 6]
                seed = int(seed_str.split('_')[1]) if 'seed_' in seed_str else 42
            except (ValueError, IndexError) as e:
                print(f"  ⚠️  Could not parse path: {result_file}")
                continue
            
            # Extract metrics
            metrics = data.get('metrics', {})
            
            result = {
                'type': 'hierarchical',
                'task': task,
                'encoder': encoder,
                'layer': layer,
                'pool': pool,
                'aggregator': aggregator,
                'classifier': classifier,
                'seed': seed,
                'accuracy': metrics.get('accuracy', np.nan),
                'macro_f1': metrics.get('macro_f1', np.nan),
                'weighted_f1': metrics.get('weighted_f1', np.nan),
                'path': str(result_file)
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"  ⚠️  Error reading {result_file}: {e}")
            continue
    
    return pd.DataFrame(results)

def compute_summary_statistics(df, group_type='flat'):
    """Compute mean and std across seeds for each configuration"""
    
    if df.empty:
        return pd.DataFrame()
    
    # Group by configuration (exclude seed)
    if group_type == 'flat':
        group_cols = ['type', 'task', 'encoder', 'layer', 'pool', 'classifier']
    else:  # hierarchical
        group_cols = ['type', 'task', 'encoder', 'layer', 'pool', 'aggregator', 'classifier']
    
    summary = df.groupby(group_cols).agg({
        'accuracy': ['mean', 'std', 'count'],
        'macro_f1': ['mean', 'std'],
        'weighted_f1': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    if group_type == 'flat':
        summary.columns = [
            'type', 'task', 'encoder', 'layer', 'pool', 'classifier',
            'acc_mean', 'acc_std', 'n_seeds',
            'macro_f1_mean', 'macro_f1_std',
            'weighted_f1_mean', 'weighted_f1_std'
        ]
    else:
        summary.columns = [
            'type', 'task', 'encoder', 'layer', 'pool', 'aggregator', 'classifier',
            'acc_mean', 'acc_std', 'n_seeds',
            'macro_f1_mean', 'macro_f1_std',
            'weighted_f1_mean', 'weighted_f1_std'
        ]
    
    return summary

def find_best_configs(summary_df, top_k=5):
    """Find top K configs by weighted F1"""
    
    if summary_df.empty:
        return pd.DataFrame()
    
    best = summary_df.nlargest(top_k, 'weighted_f1_mean')
    return best

def print_best_configs(best_df, title="Best Configurations"):
    """Pretty print best configs"""
    
    if best_df.empty:
        print(f"  No results found")
        return
    
    print(f"\n{title}:")
    print(f"{'-'*100}")
    
    for idx, row in best_df.iterrows():
        print(f"#{idx+1}")
        print(f"  Encoder:     {row['encoder']}")
        print(f"  Layer:       {row['layer']}")
        print(f"  Pool:        {row['pool']}")
        
        if pd.notna(row.get('aggregator')):
            print(f"  Aggregator:  {row['aggregator']}")
        
        print(f"  Classifier:  {row['classifier']}")
        print(f"  Weighted F1: {row['weighted_f1_mean']:.4f} ± {row['weighted_f1_std']:.4f}")
        print(f"  Macro F1:    {row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f}")
        print(f"  Accuracy:    {row['acc_mean']:.4f} ± {row['acc_std']:.4f}")
        print(f"  N seeds:     {int(row['n_seeds'])}")
        print()

def main():
    """Main aggregation function"""
    
    print(f"\n{'='*100}")
    print("AGGREGATING ALL RESULTS (n=10 seeds)")
    print(f"{'='*100}\n")
    print(f"Results base: {RESULTS_BASE}")
    print(f"Expected: 1,200 total experiments")
    print(f"  - Flat: 720 (3 encoders × 2 layers × 3 pools × 2 classifiers × 2 tasks × 10 seeds)")
    print(f"  - Hierarchical: 480 (2 layers × 2 pools × 5 aggregators × 2 classifiers × 2 tasks × 10 seeds)")
    print(f"{'='*100}\n")
    
    all_results = []
    all_summaries = []
    
    for task in ["4way", "6way"]:
        print(f"\n{'='*100}")
        print(f"PROCESSING {task.upper()}")
        print(f"{'='*100}\n")
        
        # Collect results
        print(f"📁 Collecting results...")
        flat_df = collect_flat_results(task)
        hier_df = collect_hierarchical_results(task)
        
        # Combine
        task_df = pd.concat([flat_df, hier_df], ignore_index=True)
        
        if task_df.empty:
            print(f"⚠️  No results found for {task}")
            continue
        
        print(f"\n✅ Total collected: {len(task_df)} results")
        print(f"   Flat: {len(flat_df)}")
        print(f"   Hierarchical: {len(hier_df)}")
        print(f"   Seeds: {sorted(task_df['seed'].unique())}")
        print(f"   N unique seeds: {task_df['seed'].nunique()}")
        
        all_results.append(task_df)
        
        # Compute summaries separately for flat and hier
        print(f"\n📊 Computing summary statistics...")
        
        flat_summary = compute_summary_statistics(flat_df[flat_df['type']=='flat'], 'flat') if not flat_df.empty else pd.DataFrame()
        hier_summary = compute_summary_statistics(hier_df[hier_df['type']=='hierarchical'], 'hierarchical') if not hier_df.empty else pd.DataFrame()
        
        # Combine summaries
        if not flat_summary.empty and not hier_summary.empty:
            if 'aggregator' not in flat_summary.columns:
                flat_summary['aggregator'] = None
            col_order = ['type', 'task', 'encoder', 'layer', 'pool', 'aggregator', 'classifier',
                        'acc_mean', 'acc_std', 'n_seeds', 'macro_f1_mean', 'macro_f1_std',
                        'weighted_f1_mean', 'weighted_f1_std']
            flat_summary = flat_summary[col_order]
            hier_summary = hier_summary[col_order]
            task_summary = pd.concat([flat_summary, hier_summary], ignore_index=True)
        elif not flat_summary.empty:
            if 'aggregator' not in flat_summary.columns:
                flat_summary['aggregator'] = None
            task_summary = flat_summary
        elif not hier_summary.empty:
            task_summary = hier_summary
        else:
            task_summary = pd.DataFrame()
        
        all_summaries.append(task_summary)
        
        # Find best configs
        print(f"\n🏆 Best Configurations:")
        
        if not flat_summary.empty:
            best_flat = find_best_configs(flat_summary, top_k=5)
            print_best_configs(best_flat, f"Top 5 Flat Baseline ({task.upper()})")
        
        if not hier_summary.empty:
            best_hier = find_best_configs(hier_summary, top_k=5)
            print_best_configs(best_hier, f"Top 5 Hierarchical ({task.upper()})")
        
        # Save task-specific results
        out_dir = RESULTS_BASE / task / "analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Raw results
        task_df.to_csv(out_dir / "all_results.csv", index=False)
        print(f"\n✅ Saved: {out_dir / 'all_results.csv'}")
        
        # Summary statistics
        if not task_summary.empty:
            task_summary.to_csv(out_dir / "summary_statistics.csv", index=False)
            print(f"✅ Saved: {out_dir / 'summary_statistics.csv'}")
        
        # Best configs
        if not flat_summary.empty:
            best_flat = find_best_configs(flat_summary, top_k=10)
            best_flat.to_csv(out_dir / "best_flat.csv", index=False)
            print(f"✅ Saved: {out_dir / 'best_flat.csv'}")
        
        if not hier_summary.empty:
            best_hier = find_best_configs(hier_summary, top_k=10)
            best_hier.to_csv(out_dir / "best_hierarchical.csv", index=False)
            print(f"✅ Saved: {out_dir / 'best_hierarchical.csv'}")
    
    # Combined analysis
    if all_results:
        print(f"\n{'='*100}")
        print("COMBINED ANALYSIS (ALL EXPERIMENTS)")
        print(f"{'='*100}\n")
        
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_summary = pd.concat(all_summaries, ignore_index=True)
        
        # Save combined
        out_dir = RESULTS_BASE / "analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        combined_df.to_csv(out_dir / "all_results_combined.csv", index=False)
        print(f"✅ Saved: {out_dir / 'all_results_combined.csv'}")
        
        combined_summary.to_csv(out_dir / "summary_combined.csv", index=False)
        print(f"✅ Saved: {out_dir / 'summary_combined.csv'}")
        
        # Overall best by type and task
        print(f"\n🏆 OVERALL BEST CONFIGURATIONS:")
        print(f"{'='*100}\n")
        
        for task in ["4way", "6way"]:
            for exp_type in ["flat", "hierarchical"]:
                subset = combined_summary[
                    (combined_summary['task'] == task) & 
                    (combined_summary['type'] == exp_type)
                ]
                
                if not subset.empty:
                    best_row = subset.loc[subset['weighted_f1_mean'].idxmax()]
                    
                    print(f"{task.upper()} - {exp_type.upper()}:")
                    print(f"  Encoder:     {best_row['encoder']}")
                    print(f"  Layer:       {best_row['layer']}")
                    print(f"  Pool:        {best_row['pool']}")
                    
                    if exp_type == 'hierarchical' and pd.notna(best_row['aggregator']):
                        print(f"  Aggregator:  {best_row['aggregator']}")
                    
                    print(f"  Classifier:  {best_row['classifier']}")
                    print(f"  Weighted F1: {best_row['weighted_f1_mean']:.4f} ± {best_row['weighted_f1_std']:.4f}")
                    print(f"  Macro F1:    {best_row['macro_f1_mean']:.4f} ± {best_row['macro_f1_std']:.4f}")
                    print(f"  Accuracy:    {best_row['acc_mean']:.4f} ± {best_row['acc_std']:.4f}")
                    print()
        
        # Comparison: Flat vs Hierarchical
        print(f"\n📊 FLAT vs HIERARCHICAL COMPARISON:")
        print(f"{'='*100}\n")
        
        type_comparison = combined_summary.groupby(['task', 'type']).agg({
            'weighted_f1_mean': ['mean', 'max'],
            'macro_f1_mean': ['mean', 'max'],
            'acc_mean': ['mean', 'max']
        }).round(4)
        
        print(type_comparison)
        type_comparison.to_csv(out_dir / "flat_vs_hierarchical.csv")
        print(f"\n✅ Saved: {out_dir / 'flat_vs_hierarchical.csv'}")
        
        # Encoder comparison (flat only for fair comparison)
        print(f"\n📊 ENCODER COMPARISON (Flat Baseline):")
        print(f"{'='*100}\n")
        
        flat_only = combined_summary[combined_summary['type'] == 'flat']
        if not flat_only.empty:
            encoder_comparison = flat_only.groupby(['task', 'encoder']).agg({
                'weighted_f1_mean': 'mean',
                'macro_f1_mean': 'mean',
                'acc_mean': 'mean'
            }).round(4)
            
            print(encoder_comparison)
            encoder_comparison.to_csv(out_dir / "encoder_comparison.csv")
            print(f"\n✅ Saved: {out_dir / 'encoder_comparison.csv'}")
        
        # Aggregator comparison (hierarchical only)
        print(f"\n📊 AGGREGATOR COMPARISON (Hierarchical):")
        print(f"{'='*100}\n")
        
        hier_only = combined_summary[combined_summary['type'] == 'hierarchical']
        if not hier_only.empty:
            agg_comparison = hier_only.groupby(['task', 'aggregator']).agg({
                'weighted_f1_mean': 'mean',
                'macro_f1_mean': 'mean',
                'acc_mean': 'mean'
            }).round(4)
            
            print(agg_comparison)
            agg_comparison.to_csv(out_dir / "aggregator_comparison.csv")
            print(f"\n✅ Saved: {out_dir / 'aggregator_comparison.csv'}")
        
        # Classifier comparison (MLP vs LSTM, across both types)
        print(f"\n📊 CLASSIFIER COMPARISON (MLP vs LSTM):")
        print(f"{'='*100}\n")
        
        clf_comparison = combined_summary.groupby(['task', 'type', 'classifier']).agg({
            'weighted_f1_mean': 'mean',
            'macro_f1_mean': 'mean',
            'acc_mean': 'mean'
        }).round(4)
        
        print(clf_comparison)
        clf_comparison.to_csv(out_dir / "classifier_comparison.csv")
        print(f"\n✅ Saved: {out_dir / 'classifier_comparison.csv'}")
        
        # Seed variance analysis
        print(f"\n📊 SEED VARIANCE ANALYSIS:")
        print(f"{'='*100}\n")
        
        variance_analysis = combined_summary.groupby(['task', 'type']).agg({
            'acc_std': 'mean',
            'macro_f1_std': 'mean',
            'weighted_f1_std': 'mean'
        }).round(4)
        
        print("Average std across seeds:")
        print(variance_analysis)
        variance_analysis.to_csv(out_dir / "seed_variance.csv")
        print(f"\n✅ Saved: {out_dir / 'seed_variance.csv'}")
    
    print(f"\n{'='*100}")
    print("✅ AGGREGATION COMPLETE")
    print(f"{'='*100}")
    print(f"\nAll results saved to: {RESULTS_BASE / 'analysis'}")
    print(f"\nKey files:")
    print(f"  - all_results_combined.csv: Raw data (all seeds)")
    print(f"  - summary_combined.csv: Mean ± std by configuration")
    print(f"  - flat_vs_hierarchical.csv: Flat vs Hierarchical comparison")
    print(f"  - encoder_comparison.csv: Average by encoder (flat only)")
    print(f"  - aggregator_comparison.csv: Average by aggregator (hier only)")
    print(f"  - classifier_comparison.csv: MLP vs LSTM")
    print(f"  - seed_variance.csv: Variance across seeds")
    print(f"  - 4way/best_flat.csv: Top 10 flat for 4way")
    print(f"  - 4way/best_hierarchical.csv: Top 10 hierarchical for 4way")
    print(f"  - 6way/best_flat.csv: Top 10 flat for 6way")
    print(f"  - 6way/best_hierarchical.csv: Top 10 hierarchical for 6way")

if __name__ == "__main__":
    main()
