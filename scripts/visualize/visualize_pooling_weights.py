#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_pooling_weights.py
- Visualize weight distributions for different pooling methods
- Compare linear vs exponential decay approaches
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
T = 10  # Number of tokens
OUTPUT_DIR = Path("./figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# Token positions
positions = np.arange(1, T+1)

# ============================================================================
# POOLING WEIGHT CALCULATIONS
# ============================================================================

# 1. Uniform mean pooling
w_mean = np.ones(T) / T

# 2. Linear positional weighting
w_pos = positions / positions.sum()  # Right bias: 1, 2, 3, ..., T
w_pos_rev = (T - positions + 1) / (T - positions + 1).sum()  # Left bias: T, T-1, ..., 1

# 3. Exponential decay variants
tau_fast = 2.0
tau_med = 5.0
tau_slow = 10.0

w_exp_fast = np.exp(-(positions - 1) / tau_fast)
w_exp_fast = w_exp_fast / w_exp_fast.sum()

w_exp_med = np.exp(-(positions - 1) / tau_med)
w_exp_med = w_exp_med / w_exp_med.sum()

w_exp_slow = np.exp(-(positions - 1) / tau_slow)
w_exp_slow = w_exp_slow / w_exp_slow.sum()

# ============================================================================
# VISUALIZATION 1: Side-by-Side Comparison
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Left plot: Linear methods
ax1 = axes[0]
ax1.plot(positions, w_mean, 'o-', label='mean (uniform)', 
         linewidth=2.5, markersize=8, color='steelblue')
ax1.plot(positions, w_pos, 's-', label='wmean_pos (right bias)', 
         linewidth=2.5, markersize=8, color='orange')
ax1.plot(positions, w_pos_rev, '^-', label='wmean_pos_rev (left bias)', 
         linewidth=2.5, markersize=8, color='red')
ax1.set_xlabel('Token Position', fontsize=12, fontweight='bold')
ax1.set_ylabel('Weight', fontsize=12, fontweight='bold')
ax1.set_title('Linear Pooling Methods', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='best')
ax1.grid(alpha=0.3)
ax1.set_ylim([0, 0.20])

# Right plot: Exponential decay variants
ax2 = axes[1]
ax2.plot(positions, w_mean, 'o-', label='mean (uniform)', 
         linewidth=2.5, markersize=8, alpha=0.5, color='gray')
ax2.plot(positions, w_exp_fast, 'd-', label=f'wmean_exp_fast (τ={tau_fast})', 
         linewidth=2.5, markersize=8, color='red')
ax2.plot(positions, w_exp_med, 'v-', label=f'wmean_exp_med (τ={tau_med})', 
         linewidth=2.5, markersize=8, color='orange')
ax2.plot(positions, w_exp_slow, 'p-', label=f'wmean_exp_slow (τ={tau_slow})', 
         linewidth=2.5, markersize=8, color='green')
ax2.set_xlabel('Token Position', fontsize=12, fontweight='bold')
ax2.set_ylabel('Weight', fontsize=12, fontweight='bold')
ax2.set_title('Exponential Decay Pooling Methods', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11, loc='best')
ax2.grid(alpha=0.3)
ax2.set_ylim([0, 0.20])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'pooling_comparison_exponential.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {OUTPUT_DIR / 'pooling_comparison_exponential.png'}")
plt.close()

# ============================================================================
# VISUALIZATION 2: Unified Comparison
# ============================================================================

plt.figure(figsize=(12, 6))
plt.plot(positions, w_mean, 'o-', label='mean (uniform)', 
         linewidth=2.5, markersize=8, alpha=0.7, color='steelblue')
plt.plot(positions, w_pos_rev, '^-', label='wmean_pos_rev (linear left bias)', 
         linewidth=2.5, markersize=8, alpha=0.7, color='orange')
plt.plot(positions, w_exp_fast, 'd-', label=f'wmean_exp_fast (τ={tau_fast}) - High arousal', 
         linewidth=2.5, markersize=8, color='red')
plt.plot(positions, w_exp_med, 'v-', label=f'wmean_exp_med (τ={tau_med}) - Balanced', 
         linewidth=2.5, markersize=8, color='orange')
plt.plot(positions, w_exp_slow, 'p-', label=f'wmean_exp_slow (τ={tau_slow}) - Low arousal', 
         linewidth=2.5, markersize=8, color='green')

plt.xlabel('Token Position', fontsize=13, fontweight='bold')
plt.ylabel('Weight', fontsize=13, fontweight='bold')
plt.title('Pooling Weight Distribution Comparison (T=10)', fontsize=15, fontweight='bold')
plt.legend(fontsize=11, loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'pooling_comparison_all.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {OUTPUT_DIR / 'pooling_comparison_all.png'}")
plt.close()

# ============================================================================
# VISUALIZATION 3: Tau Sensitivity Analysis
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 6))

# Test multiple tau values
tau_values = [1.0, 2.0, 5.0, 10.0, 20.0]
colors = ['darkred', 'red', 'orange', 'green', 'darkgreen']

for tau, color in zip(tau_values, colors):
    w = np.exp(-(positions - 1) / tau)
    w = w / w.sum()
    ax.plot(positions, w, 'o-', label=f'τ={tau}', 
            linewidth=2.5, markersize=8, color=color)

ax.plot(positions, w_mean, 's--', label='mean (uniform)', 
        linewidth=2, markersize=6, alpha=0.5, color='gray')

ax.set_xlabel('Token Position', fontsize=13, fontweight='bold')
ax.set_ylabel('Weight', fontsize=13, fontweight='bold')
ax.set_title('Exponential Decay Sensitivity to τ (T=10)', fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'pooling_tau_sensitivity.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {OUTPUT_DIR / 'pooling_tau_sensitivity.png'}")
plt.close()

# ============================================================================
# NUMERICAL ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("WEIGHT DISTRIBUTION TABLE (T=10)")
print("="*80)
print(f"{'Position':<10} {'mean':<10} {'pos_rev':<10} {'exp_fast':<12} {'exp_med':<12} {'exp_slow':<12}")
print("-"*80)
for i, pos in enumerate(positions):
    print(f"{pos:<10} {w_mean[i]:<10.4f} {w_pos_rev[i]:<10.4f} "
          f"{w_exp_fast[i]:<12.4f} {w_exp_med[i]:<12.4f} {w_exp_slow[i]:<12.4f}")

print("\n" + "="*80)
print("KEY STATISTICS")
print("="*80)
print(f"{'Method':<20} {'First Token':<15} {'Last Token':<15} {'Ratio (1st/Last)':<20}")
print("-"*80)

methods = [
    ('mean', w_mean),
    ('wmean_pos_rev', w_pos_rev),
    ('wmean_exp_fast', w_exp_fast),
    ('wmean_exp_med', w_exp_med),
    ('wmean_exp_slow', w_exp_slow)
]

for name, weights in methods:
    ratio = weights[0] / weights[-1] if weights[-1] > 0 else float('inf')
    print(f"{name:<20} {weights[0]:<15.4f} {weights[-1]:<15.4f} {ratio:<20.2f}x")

# ============================================================================
# CONCENTRATION ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("CONCENTRATION ANALYSIS (Top-K Token Weight)")
print("="*80)
print(f"{'Method':<20} {'Top-1':<12} {'Top-3':<12} {'Top-5':<12}")
print("-"*80)

for name, weights in methods:
    top1 = weights[0]
    top3 = weights[:3].sum()
    top5 = weights[:5].sum()
    print(f"{name:<20} {top1:<12.2%} {top3:<12.2%} {top5:<12.2%}")

# ============================================================================
# MATHEMATICAL FORMULAS
# ============================================================================

print("\n" + "="*80)
print("MATHEMATICAL FORMULAS")
print("="*80)
print("1. Mean (Uniform):")
print("   w_t = 1/T")
print()
print("2. Linear Positional (Left Bias):")
print("   w_t = (T - t + 1) / Σ(T - t + 1)")
print()
print("3. Exponential Decay:")
print("   w_t = exp(-(t-1)/τ) / Σ exp(-(t-1)/τ)")
print("   - τ small → fast decay (first token emphasis)")
print("   - τ large → slow decay (more uniform)")
print()
print("="*80)
print("✅ All visualizations saved to:", OUTPUT_DIR)
print("="*80)
