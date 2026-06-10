#!/usr/bin/env python3
"""
run_senticnet_parallel_clear_gpu.py

명확한 GPU 분배로 SenticNet 실험 병렬 실행

GPU 분배 전략:
  GPU 0: BERT (all tasks, all configs, all seeds)
  GPU 1: RoBERTa (all tasks, all configs, all seeds)
  GPU 2: Sentence-RoBERTa 4way (all configs, all seeds)
  GPU 3: Sentence-RoBERTa 6way (all configs, all seeds)

Usage:
    python run_senticnet_parallel_clear_gpu.py --dry_run    # 설정 확인
    python run_senticnet_parallel_clear_gpu.py              # 실행
"""
import os
import argparse
import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime
import time
import multiprocessing as mp

# Experiment configuration
ENCODERS = [
    'bert-base-hier',
    'roberta-base-hier',
    'sentence-roberta-hier'
]

TASKS = ['4way', '6way']
SEEDS = list(range(42, 52))  # 42-51 (10 seeds)
LAYER = 'avg_last4'
POOL = 'mean'
CLASSIFIER = 'lstm'

# Alpha values for gated fusion
ALPHAS = [0.05, 0.10, 0.20, 0.50, 1.00]

# Paths
ROOT = Path(str(Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()))
RESULTS_DIR = ROOT / 'results' / 'senticnet_experiments'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def create_experiment_configs():
    """
    Create all experiment configurations
    
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
                
                # 2. Concat fusion
                configs.append({
                    'encoder': f"{encoder}-sentic-concat",
                    'task': task,
                    'seed': seed,
                    'fusion': 'concat',
                    'layer': LAYER,
                    'pool': POOL,
                    'classifier': CLASSIFIER
                })
                
                # 3. Alpha variants
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

def distribute_jobs_by_gpu():
    """
    명확한 GPU별 작업 분배
    
    GPU 0: BERT (all tasks, all configs)
    GPU 1: RoBERTa (all tasks, all configs)
    GPU 2: Sentence-RoBERTa 4way
    GPU 3: Sentence-RoBERTa 6way
    
    Returns:
        List of 4 lists, each containing jobs for one GPU
    """
    all_configs = create_experiment_configs()
    
    gpu_jobs = [[], [], [], []]
    
    for config in all_configs:
        base_encoder = config['encoder'].split('-sentic-')[0]
        task = config['task']
        
        # GPU 분배
        if base_encoder == 'bert-base-hier':
            gpu_id = 0
        elif base_encoder == 'roberta-base-hier':
            gpu_id = 1
        elif base_encoder == 'sentence-roberta-hier':
            if task == '4way':
                gpu_id = 2
            else:  # 6way
                gpu_id = 3
        else:
            # Should not happen
            gpu_id = 0
        
        gpu_jobs[gpu_id].append(config)
    
    return gpu_jobs

def run_single_experiment(config, gpu_id, dry_run=False):
    """
    Run a single experiment on specified GPU
    
    Returns:
        True if success, False if failed
    """
    # Use absolute path to train script
    train_script = ROOT / 'scripts' / 'fusion' / 'train_npz_hier_classifier.py'
    
    # Construct command
    cmd = [
        sys.executable, str(train_script),
        '--task', config['task'],
        '--encoder', config['encoder'],
        '--layer', config['layer'],
        '--pool', config['pool'],
        '--classifier', config['classifier'],
        '--seed', str(config['seed']),
        '--save_dir', str(RESULTS_DIR)
    ]
    
    if dry_run:
        fusion_info = config['fusion']
        if 'alpha' in config:
            fusion_info += f" (α={config['alpha']:.2f})"
        print(f"[GPU {gpu_id}] {config['encoder']:50s} | {config['task']:5s} | seed={config['seed']:2d} | {fusion_info}")
        return True
    
    # Set environment
    import os
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Run
    fusion_info = config['fusion']
    if 'alpha' in config:
        fusion_info += f" (α={config['alpha']:.2f})"
    
    print(f"[GPU {gpu_id}] START: {config['encoder']:50s} | {config['task']:5s} | seed={config['seed']:2d} | {fusion_info}")
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            cwd=ROOT
        )
        
        if result.returncode == 0:
            print(f"[GPU {gpu_id}] ✓ DONE:  {config['encoder']:50s} | {config['task']:5s} | seed={config['seed']:2d}")
            return True
        else:
            print(f"[GPU {gpu_id}] ✗ FAIL:  {config['encoder']:50s} | {config['task']:5s} | seed={config['seed']:2d}")
            print(f"[GPU {gpu_id}]   Error: {result.stderr[:200]}")
            return False
            
    except Exception as e:
        print(f"[GPU {gpu_id}] ✗ ERROR: {e}")
        return False

def run_gpu_worker(gpu_id, jobs, dry_run=False):
    """
    Worker function for a single GPU
    Run all jobs assigned to this GPU sequentially
    """
    print(f"\n{'='*80}")
    print(f"GPU {gpu_id} WORKER STARTED")
    print(f"{'='*80}")
    
    # Show assignment summary
    encoder_counts = {}
    task_counts = {}
    fusion_counts = {}
    
    for job in jobs:
        base_enc = job['encoder'].split('-sentic-')[0]
        encoder_counts[base_enc] = encoder_counts.get(base_enc, 0) + 1
        task_counts[job['task']] = task_counts.get(job['task'], 0) + 1
        
        fusion = job['fusion']
        if 'alpha' in job:
            fusion = f"alpha"
        fusion_counts[fusion] = fusion_counts.get(fusion, 0) + 1
    
    print(f"Assignment:")
    print(f"  Total jobs: {len(jobs)}")
    print(f"  Encoders: {encoder_counts}")
    print(f"  Tasks: {task_counts}")
    print(f"  Fusions: {fusion_counts}")
    print(f"{'='*80}\n")
    
    if dry_run:
        print(f"[GPU {gpu_id}] DRY RUN - Jobs preview:\n")
        for i, job in enumerate(jobs, 1):
            print(f"[GPU {gpu_id}] Job {i:3d}/{len(jobs)}: ", end='')
            run_single_experiment(job, gpu_id, dry_run=True)
        return
    
    start_time = time.time()
    success_count = 0
    fail_count = 0
    
    for i, job in enumerate(jobs, 1):
        print(f"\n[GPU {gpu_id}] ═══ Job {i:3d}/{len(jobs)} ═══")
        success = run_single_experiment(job, gpu_id, dry_run=False)
        if success:
            success_count += 1
        else:
            fail_count += 1
        
        # Progress report every 10 jobs
        if i % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = avg_time * (len(jobs) - i)
            print(f"\n[GPU {gpu_id}] Progress: {i}/{len(jobs)} ({i/len(jobs)*100:.1f}%)")
            print(f"[GPU {gpu_id}] Success: {success_count}, Failed: {fail_count}")
            print(f"[GPU {gpu_id}] Elapsed: {elapsed/3600:.2f}h, ETA: {remaining/3600:.2f}h")
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"GPU {gpu_id} WORKER COMPLETED")
    print(f"{'='*80}")
    print(f"Total jobs:   {len(jobs)}")
    print(f"Success:      {success_count} ({success_count/len(jobs)*100:.1f}%)")
    print(f"Failed:       {fail_count} ({fail_count/len(jobs)*100:.1f}%)")
    print(f"Total time:   {elapsed/3600:.2f} hours")
    print(f"Avg per job:  {elapsed/len(jobs):.1f} seconds")
    print(f"{'='*80}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', action='store_true',
                       help='Print configuration without running')
    args = parser.parse_args()
    
    print("="*80)
    print("SENTICNET PARALLEL EXPERIMENTS - CLEAR GPU ASSIGNMENT")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    print("Configuration:")
    print(f"  Encoders: {ENCODERS}")
    print(f"  Tasks: {TASKS}")
    print(f"  Seeds: {len(SEEDS)} seeds ({SEEDS[0]}-{SEEDS[-1]})")
    print(f"  Alphas: {ALPHAS}")
    print(f"  Configs per encoder/task/seed: 7 (baseline + concat + 5 alphas)")
    print()
    
    total = len(ENCODERS) * len(TASKS) * len(SEEDS) * 7
    print(f"Total experiments: {total}")
    print(f"  = 3 encoders × 2 tasks × 10 seeds × 7 configs")
    print()
    
    # Distribute jobs
    gpu_jobs = distribute_jobs_by_gpu()
    
    print("="*80)
    print("GPU ASSIGNMENT (명확한 분배)")
    print("="*80)
    print()
    
    gpu_descriptions = [
        "BERT (all tasks, all configs)",
        "RoBERTa (all tasks, all configs)",
        "Sentence-RoBERTa 4way (all configs)",
        "Sentence-RoBERTa 6way (all configs)"
    ]
    
    for gpu_id, (jobs, desc) in enumerate(zip(gpu_jobs, gpu_descriptions)):
        print(f"GPU {gpu_id}: {desc}")
        print(f"  Jobs: {len(jobs)}")
        
        # Count by fusion type
        fusion_counts = {}
        for job in jobs:
            if 'baseline' in job['fusion']:
                fusion = 'baseline'
            elif 'concat' in job['fusion']:
                fusion = 'concat'
            else:
                fusion = 'alpha'
            fusion_counts[fusion] = fusion_counts.get(fusion, 0) + 1
        
        print(f"  Breakdown: {fusion_counts}")
        print()
    
    print("="*80)
    print()
    
    if args.dry_run:
        print("DRY RUN - Full job listing:\n")
        for gpu_id, jobs in enumerate(gpu_jobs):
            run_gpu_worker(gpu_id, jobs, dry_run=True)
        print("\n" + "="*80)
        print("Dry run complete.")
        print("Remove --dry_run to execute.")
        print("="*80)
        return
    
    # Confirmation
    print("Ready to launch 4 GPU workers.")
    print("Each GPU will run its jobs SEQUENTIALLY.")
    print()
    print("Monitor with:")
    print("  watch -n 1 nvidia-smi")
    print("  watch -n 5 'ls results/senticnet_experiments -R | grep results.json | wc -l'")
    print()
    
    input("Press Enter to start, or Ctrl+C to cancel...")
    print()
    
    # Launch parallel workers
    print("Launching GPU workers...")
    print()
    
    processes = []
    for gpu_id, jobs in enumerate(gpu_jobs):
        if len(jobs) == 0:
            print(f"[MAIN] GPU {gpu_id}: No jobs assigned, skipping")
            continue
        
        print(f"[MAIN] Launching GPU {gpu_id} worker ({len(jobs)} jobs)...")
        p = mp.Process(target=run_gpu_worker, args=(gpu_id, jobs, False))
        p.start()
        processes.append(p)
        time.sleep(3)  # Stagger starts
    
    print()
    print(f"[MAIN] All {len(processes)} workers launched!")
    print("[MAIN] Waiting for completion...")
    print()
    
    # Wait for all to complete
    for i, p in enumerate(processes):
        p.join()
        print(f"[MAIN] Worker {i} finished")
    
    print()
    print("="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results: {RESULTS_DIR}")
    print("="*80)
    print()
    print("Next steps:")
    print("  1. Collect results:")
    print("     python collect_senticnet_results.py \\")
    print("         --results_dir results/senticnet_experiments \\")
    print("         --output results/senticnet_all_results.csv")
    print()
    print("  2. Run analysis:")
    print("     python analyze_senticnet_results.py")
    print()
    print("  3. Information theory analysis:")
    print("     python information_theory_multiseed_analysis.py \\")
    print("         --results_csv results/senticnet_all_results.csv \\")
    print("         --output_dir results/information_theory_stats")
    print()

if __name__ == '__main__':
    main()