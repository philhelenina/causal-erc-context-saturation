#!/usr/bin/env python3
"""
collect_senticnet_results.py

Collect all experimental results into a single CSV file

Recursively searches for results.json files and merges them
"""

import json
import pandas as pd
from pathlib import Path
import argparse

def collect_results(results_dir):
    """
    Recursively collect all JSON result files
    
    Expected filename format:
      {encoder}_{task}_seed{seed}.json
      e.g., bert-base-hier_4way_seed42.json
      e.g., bert-base-hier-sentic-alpha005_4way_seed43.json
    
    Returns:
        DataFrame with all results
    """
    results_dir = Path(results_dir)
    
    all_results = []
    
    # Find all JSON files (not just results.json)
    result_files = list(results_dir.rglob('*.json'))
    
    # Filter out non-result files (like fusion_meta.json, config.json, etc)
    result_files = [f for f in result_files if 'seed' in f.name]
    
    print(f"Found {len(result_files)} result JSON files")
    print()
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Parse filename: {encoder}_{task}_seed{seed}.json
            filename = result_file.stem  # Remove .json
            
            # Split by underscore, but be careful with encoder names that have underscores/hyphens
            # e.g., "bert-base-hier-sentic-alpha005_4way_seed42"
            
            # Find task (4way or 6way)
            if '_4way_' in filename:
                task = '4way'
                parts = filename.split('_4way_')
                encoder = parts[0]
                seed_part = parts[1]
            elif '_6way_' in filename:
                task = '6way'
                parts = filename.split('_6way_')
                encoder = parts[0]
                seed_part = parts[1]
            else:
                print(f"[WARNING] Cannot parse task from: {filename}")
                continue
            
            # Extract seed
            if seed_part.startswith('seed'):
                seed = int(seed_part.replace('seed', ''))
            else:
                print(f"[WARNING] Cannot parse seed from: {seed_part}")
                continue
            
            # Create result entry
            result = {
                'encoder': encoder,
                'task': task,
                'seed': seed,
                'filename': result_file.name,
                'path': str(result_file.parent),
                **data  # Add all metrics from JSON
            }
            
            all_results.append(result)
            
        except Exception as e:
            print(f"[WARNING] Failed to load {result_file}: {e}")
            continue
    
    if len(all_results) == 0:
        print("[ERROR] No results found!")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Sort by encoder, task, seed
    df = df.sort_values(['encoder', 'task', 'seed']).reset_index(drop=True)
    
    return df

def print_summary(df):
    """Print summary statistics"""
    
    print("="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print()
    
    print(f"Total results: {len(df)}")
    print()
    
    print("By encoder:")
    encoder_counts = df['encoder'].value_counts().sort_index()
    for enc, count in encoder_counts.items():
        print(f"  {enc:50s}: {count:3d} results")
    print()
    
    print("By task:")
    task_counts = df['task'].value_counts().sort_index()
    for task, count in task_counts.items():
        print(f"  {task:5s}: {count:3d} results")
    print()
    
    print("By seed:")
    seed_counts = df['seed'].value_counts().sort_index()
    print(f"  Seeds: {sorted(seed_counts.index.tolist())}")
    print(f"  Counts: {seed_counts.tolist()}")
    print()
    
    # Check for missing experiments
    expected_seeds = set(range(42, 52))
    encoders = sorted(df['encoder'].unique())
    tasks = sorted(df['task'].unique())
    
    print("Completeness check:")
    print("-"*80)
    
    for encoder in encoders:
        for task in tasks:
            subset = df[(df['encoder'] == encoder) & (df['task'] == task)]
            found_seeds = set(subset['seed'].unique())
            missing_seeds = expected_seeds - found_seeds
            
            status = "✅" if len(missing_seeds) == 0 else "⚠️ "
            missing_str = f"missing {sorted(missing_seeds)}" if missing_seeds else "complete"
            
            print(f"  {status} {encoder:50s} / {task:5s}: {len(found_seeds):2d}/10 seeds ({missing_str})")
    
    print()
    print("="*80)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Results directory to search')
    parser.add_argument('--output', type=str, default='results.csv',
                       help='Output CSV file')
    args = parser.parse_args()
    
    print("="*80)
    print("COLLECT SENTICNET RESULTS")
    print("="*80)
    print(f"Search directory: {args.results_dir}")
    print(f"Output file: {args.output}")
    print("="*80)
    print()
    
    # Collect results
    df = collect_results(args.results_dir)
    
    if df is None:
        return
    
    # Print summary
    print_summary(df)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Results saved to: {output_path}")
    print(f"   Total: {len(df)} results")
    print()

if __name__ == '__main__':
    main()