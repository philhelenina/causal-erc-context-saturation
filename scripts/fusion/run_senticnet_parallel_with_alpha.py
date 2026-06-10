#!/usr/bin/env python3
"""
run_senticnet_parallel.py (with alpha variants)

Parallel SenticNet experiments across 4 A100 GPUs

Usage:
    python run_senticnet_parallel.py --dry_run           # Test configuration
    python run_senticnet_parallel.py                     # Run baseline + concat only
    python run_senticnet_parallel.py --include_alpha    # Include alpha variants
"""
import os
import argparse
import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime
import time

# Experiment configuration
ENCODERS = [
    'bert-base-hier',
    'roberta-base-hier',
    'sentence-roberta-hier'
]

TASKS = ['4way', '6way']
SEEDS = list(range(42, 52))  # 42-51
LAYER = 'avg_last4'
POOL = 'mean'
CLASSIFIER = 'lstm'

# Alpha values for gated fusion (only used if --include_alpha)
ALPHAS = [0.05, 0.10, 0.20, 0.50, 1.00]

# Paths
ROOT = Path(str(Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()))
RESULTS_DIR = ROOT / 'results' / 'senticnet_experiments'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def create_experiment_configs(include_alpha=False):
    """
    Create all experiment configurations
    
    Args:
        include_alpha: If True, include alpha variants
    
    Returns:
        List of dicts with experiment parameters
    """
    configs = []
    
    for encoder in ENCODERS:
        for task in TASKS:
            for seed in SEEDS:
                # 1. Baseline (encoder only)
                configs.append({
                    'encoder': encoder,
                    'task': task,
                    'seed': seed,
                    'fusion': 'baseline',
                    'layer': LAYER,
                    'pool': POOL,
                    'classifier': CLASSIFIER
                })
                
                # 2. Concat fusion (encoder + SenticNet)
                configs.append({
                    'encoder': f"{encoder}-sentic-concat",
                    'task': task,
                    'seed': seed,
                    'fusion': 'concat',
                    'layer': LAYER,
                    'pool': POOL,
                    'classifier': CLASSIFIER
                })
                
                # 3. Alpha variants (if enabled)
                if include_alpha:
                    for alpha in ALPHAS:
                        alpha_tag = f"alpha{int(alpha * 100):03d}"
                        configs.append({
                            'encoder': f"{encoder}-sentic-{alpha_tag}",
                            'task': task,
                            'seed': seed,
                            'fusion': f'gated_{alpha_tag}',
                            'alpha': alpha,
                            'layer': LAYER,
                            'pool': POOL,
                            'classifier': CLASSIFIER
                        })
    
    return configs

def distribute_jobs(configs, n_gpus=4):
    """
    Distribute jobs across GPUs
    
    Strategy: Round-robin by encoder and task to balance workload
    """
    # Sort by (encoder, task, fusion, seed) for deterministic distribution
    configs = sorted(configs, key=lambda x: (x['encoder'], x['task'], x['fusion'], x['seed']))
    
    # Distribute
    gpu_jobs = [[] for _ in range(n_gpus)]
    for i, config in enumerate(configs):
        gpu_id = i % n_gpus
        gpu_jobs[gpu_id].append(config)
    
    return gpu_jobs

def run_single_experiment(config, gpu_id, dry_run=False):
    """
    Run a single experiment on specified GPU
    
    Returns:
        Command string that would be executed
    """
    # Get absolute path of train script (same directory as this script)
    script_dir = Path(__file__).parent.absolute()
    train_script = script_dir / 'train_npz_hier_classifier.py'
    
    # Construct command - use sys.executable for correct Python path
    # Note: CUDA_VISIBLE_DEVICES handles GPU selection, no --gpu needed
    cmd = [
        sys.executable, str(train_script),  # Use absolute path
        '--task', config['task'],
        '--encoder', config['encoder'],
        '--layer', config['layer'],
        '--pool', config['pool'],
        '--classifier', config['classifier'],
        '--seed', str(config['seed']),
        '--save_dir', str(RESULTS_DIR)
    ]
    
    cmd_str = ' '.join(cmd)
    
    if dry_run:
        fusion_info = config['fusion']
        if 'alpha' in config:
            fusion_info += f" (α={config['alpha']:.2f})"
        print(f"[GPU {gpu_id}] Would run: {config['encoder']:50s} | {config['task']:5s} | seed={config['seed']} | {fusion_info}")
        return cmd_str
    
    # Set environment - preserve existing env and add CUDA_VISIBLE_DEVICES
    import os
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Run
    fusion_info = config['fusion']
    if 'alpha' in config:
        fusion_info += f" (α={config['alpha']:.2f})"
    
    print(f"[GPU {gpu_id}] Running: {config['encoder']:50s} | {config['task']:5s} | seed={config['seed']} | {fusion_info}")
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            cwd=ROOT
        )
        
        if result.returncode == 0:
            print(f"[GPU {gpu_id}] ✓ Success: {config['encoder']} | {config['task']} | seed={config['seed']}")
            return True
        else:
            print(f"[GPU {gpu_id}] ✗ Failed: {config['encoder']} | {config['task']} | seed={config['seed']}")
            print(f"[GPU {gpu_id}] Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"[GPU {gpu_id}] ✗ Exception: {e}")
        return False

def run_gpu_worker(gpu_id, jobs, dry_run=False):
    """
    Worker function for a single GPU
    Run all jobs assigned to this GPU sequentially
    """
    print(f"\n{'='*80}")
    print(f"GPU {gpu_id} Worker: {len(jobs)} jobs assigned")
    print(f"{'='*80}\n")
    
    if dry_run:
        for i, job in enumerate(jobs, 1):
            fusion_info = job['fusion']
            if 'alpha' in job:
                fusion_info += f" (α={job['alpha']:.2f})"
            print(f"[GPU {gpu_id}] Job {i}/{len(jobs)}: {job['encoder']:50s} | {job['task']:5s} | seed={job['seed']} | {fusion_info}")
        return
    
    start_time = time.time()
    success_count = 0
    
    for i, job in enumerate(jobs, 1):
        print(f"\n[GPU {gpu_id}] Job {i}/{len(jobs)}")
        success = run_single_experiment(job, gpu_id, dry_run=False)
        if success:
            success_count += 1
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"GPU {gpu_id} Summary:")
    print(f"  Total jobs: {len(jobs)}")
    print(f"  Success: {success_count}")
    print(f"  Failed: {len(jobs) - success_count}")
    print(f"  Time: {elapsed/3600:.2f} hours")
    print(f"{'='*80}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', action='store_true',
                       help='Print configuration without running')
    parser.add_argument('--n_gpus', type=int, default=4,
                       help='Number of GPUs to use')
    parser.add_argument('--include_alpha', action='store_true',
                       help='Include alpha variants (default: baseline + concat only)')
    args = parser.parse_args()
    
    print("="*80)
    print("SENTICNET PARALLEL EXPERIMENTS")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPUs: {args.n_gpus}")
    print(f"Dry run: {args.dry_run}")
    print(f"Include alpha: {args.include_alpha}")
    print()
    
    # Create configurations
    configs = create_experiment_configs(include_alpha=args.include_alpha)
    
    print(f"Total experiments: {len(configs)}")
    print(f"  Encoders: {ENCODERS}")
    print(f"  Tasks: {TASKS}")
    print(f"  Seeds: {SEEDS}")
    
    if args.include_alpha:
        print(f"  Fusions: baseline, concat, alpha variants")
        print(f"  Alphas: {ALPHAS}")
        print(f"  = 3 encoders × 2 tasks × 10 seeds × (2 + {len(ALPHAS)} alphas) = {len(configs)} runs")
    else:
        print(f"  Fusions: baseline, concat")
        print(f"  = 3 encoders × 2 tasks × 10 seeds × 2 = {len(configs)} runs")
    print()
    
    # Distribute jobs
    gpu_jobs = distribute_jobs(configs, n_gpus=args.n_gpus)
    
    print("Job distribution:")
    for gpu_id, jobs in enumerate(gpu_jobs):
        print(f"  GPU {gpu_id}: {len(jobs)} jobs")
        
        # Show breakdown
        fusion_counts = {}
        for job in jobs:
            fusion = job['fusion']
            if 'alpha' in job:
                fusion = f"alpha_{job['alpha']:.2f}"
            fusion_counts[fusion] = fusion_counts.get(fusion, 0) + 1
        print(f"    Breakdown: {fusion_counts}")
    
    print("="*80)
    print()
    
    if args.dry_run:
        print("DRY RUN - Configuration preview:\n")
        for gpu_id, jobs in enumerate(gpu_jobs):
            run_gpu_worker(gpu_id, jobs, dry_run=True)
        print("\nDry run complete. Remove --dry_run to execute.")
        return
    
    # Launch parallel workers using multiprocessing
    import multiprocessing as mp
    
    print("Launching parallel workers...")
    print("Note: Each GPU will run its jobs sequentially")
    print("Monitor with: watch -n 1 nvidia-smi")
    print()
    
    processes = []
    for gpu_id, jobs in enumerate(gpu_jobs):
        if len(jobs) == 0:
            continue
        
        p = mp.Process(target=run_gpu_worker, args=(gpu_id, jobs, False))
        p.start()
        processes.append(p)
        time.sleep(2)  # Stagger starts slightly
    
    # Wait for all to complete
    for p in processes:
        p.join()
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {RESULTS_DIR}")
    print("="*80)

if __name__ == '__main__':
    main()
