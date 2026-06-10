#!/usr/bin/env python3
"""
Multi-seed Results Extraction and Aggregation
- Extract class-wise F1 from all seeds
- Aggregate with mean ± std
- Create comprehensive visualizations
"""
import os

import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, classification_report
import sys

print("="*80)
print("MULTI-SEED RESULTS EXTRACTION & AGGREGATION")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================

HOME = Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()
RESULTS_ROOT = HOME / "results" / "turnlevel_k_sweep_bayesian"
SEEDS = list(range(42, 52))  # 42-51

# Find result directories
print("\nSearching for result directories...")
result_dirs_4way = list(RESULTS_ROOT.glob("4way_sentence-roberta*"))
result_dirs_6way = list(RESULTS_ROOT.glob("6way_sentence-roberta*"))

if not result_dirs_4way:
    print("❌ No 4-way results found!")
    sys.exit(1)
if not result_dirs_6way:
    print("❌ No 6-way results found!")
    sys.exit(1)

result_dir_4way = result_dirs_4way[0]
result_dir_6way = result_dirs_6way[0]

print(f"\n✓ Found 4-way: {result_dir_4way.name}")
print(f"✓ Found 6-way: {result_dir_6way.name}")

# ============================================================================
# Step 1: Compute Class-wise F1 for each seed
# ============================================================================

def compute_classwise_f1(seed_dir, task_name):
    """Compute class-wise F1 for a single seed"""
    
    # Load predictions and labels
    preds_file = seed_dir / "preds_perK.npy"
    labels_file = seed_dir / "labels.npy"
    Ks_file = seed_dir / "Ks.npy"
    
    if not all([preds_file.exists(), labels_file.exists(), Ks_file.exists()]):
        return None
    
    preds_perK = np.load(preds_file)  # (N_K, N_samples, N_classes)
    labels = np.load(labels_file)      # (N_samples,)
    Ks = np.load(Ks_file)              # (N_K,)
    
    # Get number of classes
    n_classes = preds_perK.shape[2]
    
    # Define emotion names
    if n_classes == 4:
        emotion_names = ['angry', 'happy', 'sad', 'neutral']
    elif n_classes == 6:
        emotion_names = ['angry', 'happy', 'sad', 'neutral', 'excited', 'frustrated']
    else:
        emotion_names = [f'class_{i}' for i in range(n_classes)]
    
    results = []
    
    for k_idx, K in enumerate(Ks):
        # Get predictions for this K
        preds = np.argmax(preds_perK[k_idx], axis=1)
        
        # Filter out -1 labels
        mask = labels >= 0
        preds_filtered = preds[mask]
        labels_filtered = labels[mask]
        
        if len(labels_filtered) == 0:
            continue
        
        # Compute overall metrics
        f1_weighted = f1_score(labels_filtered, preds_filtered, average='weighted', zero_division=0)
        f1_macro = f1_score(labels_filtered, preds_filtered, average='macro', zero_division=0)
        
        # Compute per-class F1
        f1_per_class = f1_score(labels_filtered, preds_filtered, average=None, zero_division=0)
        
        # Build result row
        row = {
            'K': K,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro
        }
        
        for i, emotion in enumerate(emotion_names):
            if i < len(f1_per_class):
                row[f'{emotion}_f1'] = f1_per_class[i]
            else:
                row[f'{emotion}_f1'] = 0.0
        
        results.append(row)
    
    return pd.DataFrame(results)


print("\n" + "="*80)
print("STEP 1: Computing Class-wise F1 for each seed")
print("="*80)

# 4-way
print("\n[4-way]")
all_results_4way = []
for seed in SEEDS:
    seed_dir = result_dir_4way / f"seed{seed}"
    if not seed_dir.exists():
        print(f"  ⚠ seed{seed} not found")
        continue
    
    df = compute_classwise_f1(seed_dir, "4way")
    if df is not None:
        df['seed'] = seed
        all_results_4way.append(df)
        print(f"  ✓ seed{seed}: {len(df)} K values")
        
        # Save individual result
        output_file = seed_dir / "k_sweep_classwise_results.csv"
        df.drop(columns=['seed']).to_csv(output_file, index=False)
        print(f"    Saved: {output_file}")

# 6-way
print("\n[6-way]")
all_results_6way = []
for seed in SEEDS:
    seed_dir = result_dir_6way / f"seed{seed}"
    if not seed_dir.exists():
        print(f"  ⚠ seed{seed} not found")
        continue
    
    df = compute_classwise_f1(seed_dir, "6way")
    if df is not None:
        df['seed'] = seed
        all_results_6way.append(df)
        print(f"  ✓ seed{seed}: {len(df)} K values")
        
        # Save individual result
        output_file = seed_dir / "k_sweep_classwise_results.csv"
        df.drop(columns=['seed']).to_csv(output_file, index=False)
        print(f"    Saved: {output_file}")

if not all_results_4way:
    print("\n❌ No 4-way results to aggregate!")
    sys.exit(1)
if not all_results_6way:
    print("\n❌ No 6-way results to aggregate!")
    sys.exit(1)

# ============================================================================
# Step 2: Aggregate results (mean ± std)
# ============================================================================

print("\n" + "="*80)
print("STEP 2: Aggregating results (mean ± std)")
print("="*80)

# Combine all seeds
df_all_4way = pd.concat(all_results_4way, ignore_index=True)
df_all_6way = pd.concat(all_results_6way, ignore_index=True)

# Group by K and compute statistics
def aggregate_by_K(df):
    grouped = df.groupby('K')
    
    # Get metric columns
    metric_cols = [col for col in df.columns if col not in ['K', 'seed']]
    
    # Compute mean and std
    mean_df = grouped[metric_cols].mean().reset_index()
    std_df = grouped[metric_cols].std().reset_index()
    count_df = grouped[metric_cols].count().reset_index()
    
    return mean_df, std_df, count_df

mean_4way, std_4way, count_4way = aggregate_by_K(df_all_4way)
mean_6way, std_6way, count_6way = aggregate_by_K(df_all_6way)

# Save aggregated results
output_dir = Path("multiseed_aggregated")
output_dir.mkdir(exist_ok=True)

mean_4way.to_csv(output_dir / "4way_mean.csv", index=False)
std_4way.to_csv(output_dir / "4way_std.csv", index=False)
count_4way.to_csv(output_dir / "4way_count.csv", index=False)
df_all_4way.to_csv(output_dir / "4way_all_seeds.csv", index=False)

mean_6way.to_csv(output_dir / "6way_mean.csv", index=False)
std_6way.to_csv(output_dir / "6way_std.csv", index=False)
count_6way.to_csv(output_dir / "6way_count.csv", index=False)
df_all_6way.to_csv(output_dir / "6way_all_seeds.csv", index=False)

print(f"\n✓ Saved aggregated results to: {output_dir}/")

# ============================================================================
# Step 3: Print summary statistics
# ============================================================================

print("\n" + "="*80)
print("STEP 3: Summary Statistics")
print("="*80)

def print_summary(mean_df, std_df, count_df, task_name):
    print(f"\n[{task_name}]")
    print(f"\n{'K':<8} {'F1 Weighted':<25} {'F1 Macro':<25} {'n':<5}")
    print("-"*70)
    
    for _, row in mean_df.iterrows():
        K = int(row['K'])
        f1w_mean = row['f1_weighted']
        f1w_std = std_df.loc[std_df['K'] == K, 'f1_weighted'].values[0]
        f1m_mean = row['f1_macro']
        f1m_std = std_df.loc[std_df['K'] == K, 'f1_macro'].values[0]
        n = int(count_df.loc[count_df['K'] == K, 'f1_weighted'].values[0])
        
        print(f"{K:<8} {f1w_mean:.4f} ± {f1w_std:.4f}      {f1m_mean:.4f} ± {f1m_std:.4f}      {n}")
    
    # Find peak
    peak_idx = mean_df['f1_weighted'].idxmax()
    peak_K = int(mean_df.loc[peak_idx, 'K'])
    peak_f1w = mean_df.loc[peak_idx, 'f1_weighted']
    peak_std = std_df.loc[peak_idx, 'f1_weighted']
    
    print(f"\n→ Peak: K={peak_K}, F1={peak_f1w:.4f} ± {peak_std:.4f}")

print_summary(mean_4way, std_4way, count_4way, "4-way")
print_summary(mean_6way, std_6way, count_6way, "6-way")

# ============================================================================
# Step 4: Visualizations
# ============================================================================

print("\n" + "="*80)
print("STEP 4: Creating visualizations")
print("="*80)

# Plot 1: Overall metrics (4-way and 6-way)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 4-way
ax = axes[0]
K_vals = mean_4way['K'].values
f1w_mean = mean_4way['f1_weighted'].values
f1w_std = std_4way['f1_weighted'].values

ax.plot(K_vals, f1w_mean, 'o-', linewidth=2, markersize=6, label='F1 Weighted', color='blue')
ax.fill_between(K_vals, f1w_mean - f1w_std, f1w_mean + f1w_std, alpha=0.2, color='blue')

peak_idx = np.argmax(f1w_mean)
ax.axvline(K_vals[peak_idx], color='red', linestyle='--', alpha=0.3)
ax.set_xlabel('K (Context Window)', fontsize=12)
ax.set_ylabel('F1 Weighted', fontsize=12)
ax.set_title(f'4-way: Overall Performance (n={len(SEEDS)} seeds)', fontweight='bold', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# 6-way
ax = axes[1]
K_vals = mean_6way['K'].values
f1w_mean = mean_6way['f1_weighted'].values
f1w_std = std_6way['f1_weighted'].values

ax.plot(K_vals, f1w_mean, 's-', linewidth=2, markersize=6, label='F1 Weighted', color='green')
ax.fill_between(K_vals, f1w_mean - f1w_std, f1w_mean + f1w_std, alpha=0.2, color='green')

peak_idx = np.argmax(f1w_mean)
ax.axvline(K_vals[peak_idx], color='red', linestyle='--', alpha=0.3)
ax.set_xlabel('K (Context Window)', fontsize=12)
ax.set_ylabel('F1 Weighted', fontsize=12)
ax.set_title(f'6-way: Overall Performance (n={len(SEEDS)} seeds)', fontweight='bold', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "overall_multiseed.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir}/overall_multiseed.png")

# Plot 2: Per-emotion (4-way)
emotions_4way = ['angry', 'happy', 'sad', 'neutral']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, emotion in enumerate(emotions_4way):
    ax = axes[i // 2, i % 2]
    col = f'{emotion}_f1'
    
    K_vals = mean_4way['K'].values
    f1_mean = mean_4way[col].values
    f1_std = std_4way[col].values
    
    ax.plot(K_vals, f1_mean, 'o-', linewidth=2, markersize=5)
    ax.fill_between(K_vals, f1_mean - f1_std, f1_mean + f1_std, alpha=0.2)
    
    peak_idx = np.argmax(f1_mean)
    ax.axvline(K_vals[peak_idx], color='red', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('K (Context Window)')
    ax.set_ylabel('F1 Score')
    ax.set_title(f'4-way: {emotion.capitalize()} (n={len(SEEDS)})', fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "emotions_4way_multiseed.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir}/emotions_4way_multiseed.png")

# Plot 3: Per-emotion (6-way)
emotions_6way = ['angry', 'happy', 'sad', 'neutral', 'excited', 'frustrated']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for i, emotion in enumerate(emotions_6way):
    ax = axes[i // 3, i % 3]
    col = f'{emotion}_f1'
    
    K_vals = mean_6way['K'].values
    f1_mean = mean_6way[col].values
    f1_std = std_6way[col].values
    
    ax.plot(K_vals, f1_mean, 's-', linewidth=2, markersize=5)
    ax.fill_between(K_vals, f1_mean - f1_std, f1_mean + f1_std, alpha=0.2)
    
    peak_idx = np.argmax(f1_mean)
    ax.axvline(K_vals[peak_idx], color='red', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('K (Context Window)')
    ax.set_ylabel('F1 Score')
    ax.set_title(f'6-way: {emotion.capitalize()} (n={len(SEEDS)})', fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "emotions_6way_multiseed.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir}/emotions_6way_multiseed.png")

# ============================================================================
# Step 5: Final report
# ============================================================================

print("\n" + "="*80)
print("FINAL REPORT")
print("="*80)

# Peak statistics
peak_4way_idx = mean_4way['f1_weighted'].idxmax()
peak_4way_K = int(mean_4way.loc[peak_4way_idx, 'K'])
peak_4way_f1 = mean_4way.loc[peak_4way_idx, 'f1_weighted']
peak_4way_std = std_4way.loc[peak_4way_idx, 'f1_weighted']

peak_6way_idx = mean_6way['f1_weighted'].idxmax()
peak_6way_K = int(mean_6way.loc[peak_6way_idx, 'K'])
peak_6way_f1 = mean_6way.loc[peak_6way_idx, 'f1_weighted']
peak_6way_std = std_6way.loc[peak_6way_idx, 'f1_weighted']

print(f"\n4-way Peak Performance:")
print(f"  K* = {peak_4way_K}")
print(f"  F1 = {peak_4way_f1:.4f} ± {peak_4way_std:.4f}")
print(f"  95% CI = [{peak_4way_f1 - 1.96*peak_4way_std:.4f}, {peak_4way_f1 + 1.96*peak_4way_std:.4f}]")

print(f"\n6-way Peak Performance:")
print(f"  K* = {peak_6way_K}")
print(f"  F1 = {peak_6way_f1:.4f} ± {peak_6way_std:.4f}")
print(f"  95% CI = [{peak_6way_f1 - 1.96*peak_6way_std:.4f}, {peak_6way_f1 + 1.96*peak_6way_std:.4f}]")

print(f"\nSOTA Comparison:")
print(f"  4-way Previous SOTA: 81.4%")
print(f"  4-way Ours:          {peak_4way_f1*100:.2f}% ± {peak_4way_std*100:.2f}%")
if peak_4way_f1 > 0.814:
    print(f"  → ✓ BEATS SOTA by {(peak_4way_f1 - 0.814)*100:.2f}%")
else:
    print(f"  → Comparable to SOTA")

print(f"\n  6-way Previous SOTA: 64.4%")
print(f"  6-way Ours:          {peak_6way_f1*100:.2f}% ± {peak_6way_std*100:.2f}%")
if peak_6way_f1 > 0.644:
    print(f"  → ✓ BEATS SOTA by {(peak_6way_f1 - 0.644)*100:.2f}%")
else:
    print(f"  → Comparable to SOTA")

# Save final report
report = {
    '4way': {
        'peak_K': peak_4way_K,
        'peak_F1_mean': float(peak_4way_f1),
        'peak_F1_std': float(peak_4way_std),
        'n_seeds': len(SEEDS)
    },
    '6way': {
        'peak_K': peak_6way_K,
        'peak_F1_mean': float(peak_6way_f1),
        'peak_F1_std': float(peak_6way_std),
        'n_seeds': len(SEEDS)
    }
}

with open(output_dir / "final_report.json", 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n✓ Final report saved: {output_dir}/final_report.json")

print("\n" + "="*80)
print("ALL DONE! ✓")
print("="*80)
print(f"\nAll results saved in: {output_dir}/")
print(f"  - *_mean.csv (mean values)")
print(f"  - *_std.csv (standard deviations)")
print(f"  - *_all_seeds.csv (raw data)")
print(f"  - *.png (visualizations)")
print(f"  - final_report.json (summary)")
print("="*80)