#!/usr/bin/env python3
"""
Manually compute mean from hierarchical and compare with flat
"""
import os

import numpy as np
from pathlib import Path

flat_path = (Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve() / Path("data/embeddings/4way/sentence-roberta/avg_last4/mean/train.npz"))
hier_path = (Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve() / Path("data/embeddings/4way/sentence-roberta-hier/avg_last4/mean/train.npz"))

# Load flat
arr_flat = np.load(flat_path, allow_pickle=True)
X_flat = arr_flat["embeddings"]

# Load hierarchical
arr_hier = np.load(hier_path, allow_pickle=True)
X_hier = arr_hier["embeddings"]
lengths = arr_hier["lengths"]

# Compute mean manually
X_hier_mean = []
for i in range(len(X_hier)):
    L = int(lengths[i])
    X_hier_mean.append(X_hier[i, :L, :].mean(axis=0))
X_hier_mean = np.stack(X_hier_mean, axis=0)

print("="*80)
print("COMPARISON: Flat vs Hierarchical (manual mean)")
print("="*80)

print(f"\nFlat shape:            {X_flat.shape}")
print(f"Hierarchical shape:    {X_hier.shape}")
print(f"Hierarchical mean:     {X_hier_mean.shape}")

print(f"\nLengths stats:")
print(f"  Min: {lengths.min()}")
print(f"  Max: {lengths.max()}")
print(f"  Mean: {lengths.mean():.2f}")
print(f"  Zeros: {(lengths == 0).sum()}")
print(f"  > 12: {(lengths > 12).sum()}")

print(f"\nEmbedding comparison (first 5 samples):")
for i in range(5):
    diff = np.abs(X_flat[i] - X_hier_mean[i]).mean()
    print(f"  Sample {i}: mean absolute diff = {diff:.6f}")

# Overall difference
overall_diff = np.abs(X_flat - X_hier_mean).mean()
print(f"\nOverall mean absolute difference: {overall_diff:.6f}")

# Check if they're identical
if overall_diff < 1e-5:
    print("\n✓ IDENTICAL! Flat and hierarchical mean are the same!")
else:
    print(f"\n✗ DIFFERENT! Mean difference: {overall_diff}")
    
    # Check if order is different
    print(f"\nChecking if it's just a reordering issue...")
    
    # Find closest match for first flat sample
    dists = np.linalg.norm(X_hier_mean - X_flat[0], axis=1)
    closest_idx = dists.argmin()
    closest_dist = dists[closest_idx]
    
    print(f"  Flat[0] closest match in hier_mean: index {closest_idx}, dist={closest_dist:.6f}")
    
    if closest_dist < 1e-5:
        print("  → Looks like a reordering issue!")
    else:
        print("  → Not a reordering issue, embeddings are actually different!")
