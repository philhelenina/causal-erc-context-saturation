#!/usr/bin/env python3
"""
information_theory_analysis.py

Information-theoretic analysis of BERT + SenticNet fusion

Key metrics:
1. Mutual Information: I(X; Y)
2. Conditional Mutual Information: I(X; Y | Z)
3. Redundancy vs Synergy
4. Information Bottleneck

Usage:
    python information_theory_analysis.py \
        --task 4way \
        --encoder sentence-roberta-hier
"""
import os

import argparse
import numpy as np
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

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
    """
    Aggregate sentence-level embeddings to utterance-level
    
    Args:
        X: (N, S_max, D) embeddings
        lengths: (N,) number of sentences per utterance
    
    Returns:
        (N, D) utterance-level embeddings
    """
    N, S_max, D = X.shape
    X_utt = np.zeros((N, D), dtype=np.float32)
    
    for i in range(N):
        if lengths is not None:
            L = lengths[i]
        else:
            L = S_max
        
        # Mean pooling over sentences
        X_utt[i] = X[i, :L, :].mean(axis=0)
    
    return X_utt

def compute_mutual_information(X, y, n_neighbors=3):
    """
    Compute mutual information I(X; Y)
    
    Args:
        X: (N, D) features
        y: (N,) labels
        n_neighbors: k for KNN-based MI estimation
    
    Returns:
        float: Mutual information in nats (divide by log(2) for bits)
    """
    # Use scikit-learn's MI estimator
    mi = mutual_info_classif(X, y, n_neighbors=n_neighbors, random_state=42)
    
    # Return average MI across features
    return mi.mean()

def compute_joint_mi(X1, X2, y, n_neighbors=3):
    """
    Compute joint mutual information I(X1, X2; Y)
    
    Args:
        X1: (N, D1) first feature set
        X2: (N, D2) second feature set
        y: (N,) labels
    
    Returns:
        float: Joint mutual information
    """
    # Concatenate features
    X_joint = np.concatenate([X1, X2], axis=1)
    
    return compute_mutual_information(X_joint, y, n_neighbors)

def compute_conditional_mi(X1, X2, y, n_neighbors=3):
    """
    Compute conditional mutual information I(X1; Y | X2)
    
    Approximation: I(X1; Y | X2) ≈ I(X1, X2; Y) - I(X2; Y)
    
    Args:
        X1: (N, D1) first feature set
        X2: (N, D2) second feature set (conditioning)
        y: (N,) labels
    
    Returns:
        float: Conditional mutual information
    """
    joint_mi = compute_joint_mi(X1, X2, y, n_neighbors)
    x2_mi = compute_mutual_information(X2, y, n_neighbors)
    
    return joint_mi - x2_mi

def compute_redundancy_synergy(X_bert, X_sentic, y, n_neighbors=3):
    """
    Decompose information into redundancy and synergy
    
    Redundancy: Information shared by both
    Unique (BERT): Information unique to BERT
    Unique (SenticNet): Information unique to SenticNet
    Synergy: Information only available when both are used
    
    Returns:
        dict with redundancy, unique_bert, unique_sentic, synergy
    """
    # Individual MI
    I_bert = compute_mutual_information(X_bert, y, n_neighbors)
    I_sentic = compute_mutual_information(X_sentic, y, n_neighbors)
    
    # Joint MI
    I_joint = compute_joint_mi(X_bert, X_sentic, y, n_neighbors)
    
    # Conditional MI
    I_bert_given_sentic = compute_conditional_mi(X_bert, X_sentic, y, n_neighbors)
    I_sentic_given_bert = compute_conditional_mi(X_sentic, X_bert, y, n_neighbors)
    
    # Redundancy (approximation)
    redundancy = min(I_bert, I_sentic)
    
    # Unique information
    unique_bert = max(0, I_bert - redundancy)
    unique_sentic = max(0, I_sentic - redundancy)
    
    # Synergy (information gained only when both are used)
    synergy = max(0, I_joint - I_bert - I_sentic + redundancy)
    
    return {
        'I_bert': I_bert,
        'I_sentic': I_sentic,
        'I_joint': I_joint,
        'redundancy': redundancy,
        'unique_bert': unique_bert,
        'unique_sentic': unique_sentic,
        'synergy': synergy,
        'complementarity': I_joint - I_bert  # How much SenticNet adds
    }

def compute_entropy_metrics(y):
    """
    Compute entropy metrics
    
    Returns:
        dict with H(Y), perplexity
    """
    # Label entropy
    y_counts = np.bincount(y)
    y_probs = y_counts / y_counts.sum()
    H_y = entropy(y_probs, base=2)  # bits
    
    perplexity = 2 ** H_y
    
    return {
        'H_y': H_y,
        'perplexity': perplexity,
        'num_classes': len(y_probs)
    }

def dimensionality_reduction_analysis(X_bert, X_sentic, y):
    """
    Analyze how much information is retained after dimensionality reduction
    
    This helps understand the intrinsic dimensionality of the representations
    """
    results = {}
    
    # PCA on BERT
    pca_bert = PCA(n_components=min(50, X_bert.shape[1]))
    X_bert_pca = pca_bert.fit_transform(X_bert)
    
    # Cumulative explained variance
    cumvar_bert = np.cumsum(pca_bert.explained_variance_ratio_)
    
    # Find dimensionality for 95% variance
    d_95_bert = np.argmax(cumvar_bert >= 0.95) + 1
    
    results['bert'] = {
        'intrinsic_dim_95': d_95_bert,
        'explained_var_50d': cumvar_bert[49] if len(cumvar_bert) > 49 else cumvar_bert[-1]
    }
    
    # PCA on SenticNet
    if X_sentic.shape[1] > 1:
        pca_sentic = PCA(n_components=min(4, X_sentic.shape[1]))
        X_sentic_pca = pca_sentic.fit_transform(X_sentic)
        
        cumvar_sentic = np.cumsum(pca_sentic.explained_variance_ratio_)
        
        results['sentic'] = {
            'explained_var_all': cumvar_sentic[-1],
            'n_components': len(cumvar_sentic)
        }
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['4way', '6way'])
    parser.add_argument('--encoder', type=str, required=True,
                       choices=['bert-base-hier', 'roberta-base-hier', 'sentence-roberta-hier'])
    parser.add_argument('--layer', type=str, default='avg_last4')
    parser.add_argument('--pool', type=str, default='mean')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--root', type=str, 
                       default=str(Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()))
    parser.add_argument('--output_dir', type=str, default='results/information_theory')
    args = parser.parse_args()
    
    ROOT = Path(args.root)
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("INFORMATION THEORY ANALYSIS")
    print("="*80)
    print(f"Task: {args.task}")
    print(f"Encoder: {args.encoder}")
    print(f"Split: {args.split}")
    print("="*80)
    print()
    
    # Load BERT embeddings
    bert_dir = ROOT / f"data/embeddings/{args.task}/{args.encoder}/{args.layer}/{args.pool}"
    bert_file = bert_dir / f"{args.split}.npz"
    
    X_bert, y, lengths = load_embeddings(bert_file)
    print(f"Loaded BERT embeddings: {X_bert.shape}")
    
    # Aggregate to utterance level
    X_bert_utt = aggregate_to_utterance(X_bert, lengths)
    print(f"Aggregated to utterance: {X_bert_utt.shape}")
    
    # Load SenticNet features
    sentic_dir = ROOT / f"data/embeddings/{args.task}/senticnet-sentence-level"
    sentic_file = sentic_dir / f"{args.split}.npz"
    
    X_sentic, _, sentic_lengths = load_embeddings(sentic_file)
    print(f"Loaded SenticNet features: {X_sentic.shape}")
    
    # Aggregate to utterance level
    X_sentic_utt = aggregate_to_utterance(X_sentic, sentic_lengths)
    print(f"Aggregated to utterance: {X_sentic_utt.shape}")
    
    print()
    print("="*80)
    print("COMPUTING INFORMATION METRICS")
    print("="*80)
    print()
    
    # 1. Entropy metrics
    print("1. Entropy Analysis")
    print("-"*80)
    entropy_metrics = compute_entropy_metrics(y)
    print(f"Label Entropy H(Y): {entropy_metrics['H_y']:.4f} bits")
    print(f"Perplexity: {entropy_metrics['perplexity']:.2f}")
    print(f"Num classes: {entropy_metrics['num_classes']}")
    print()
    
    # 2. Mutual Information
    print("2. Mutual Information Analysis")
    print("-"*80)
    
    I_bert = compute_mutual_information(X_bert_utt, y)
    I_sentic = compute_mutual_information(X_sentic_utt, y)
    I_joint = compute_joint_mi(X_bert_utt, X_sentic_utt, y)
    
    print(f"I(X_bert; Y):         {I_bert:.4f} nats ({I_bert/np.log(2):.4f} bits)")
    print(f"I(X_sentic; Y):       {I_sentic:.4f} nats ({I_sentic/np.log(2):.4f} bits)")
    print(f"I(X_bert, X_sentic; Y): {I_joint:.4f} nats ({I_joint/np.log(2):.4f} bits)")
    print()
    
    # Complementarity
    delta_I = I_joint - I_bert
    print(f"ΔI (Complementarity): {delta_I:.4f} nats ({delta_I/np.log(2):.4f} bits)")
    print(f"  → SenticNet adds {delta_I/I_bert*100:.1f}% more information")
    print()
    
    # 3. Redundancy & Synergy
    print("3. Redundancy-Synergy Decomposition")
    print("-"*80)
    
    decomp = compute_redundancy_synergy(X_bert_utt, X_sentic_utt, y)
    
    print(f"Redundancy:       {decomp['redundancy']:.4f} nats ({decomp['redundancy']/np.log(2):.4f} bits)")
    print(f"Unique (BERT):    {decomp['unique_bert']:.4f} nats ({decomp['unique_bert']/np.log(2):.4f} bits)")
    print(f"Unique (SenticNet): {decomp['unique_sentic']:.4f} nats ({decomp['unique_sentic']/np.log(2):.4f} bits)")
    print(f"Synergy:          {decomp['synergy']:.4f} nats ({decomp['synergy']/np.log(2):.4f} bits)")
    print()
    
    # Interpretation
    total_info = decomp['redundancy'] + decomp['unique_bert'] + decomp['unique_sentic'] + decomp['synergy']
    
    print("Information Breakdown (%):")
    print(f"  Redundancy:       {decomp['redundancy']/total_info*100:.1f}%")
    print(f"  Unique (BERT):    {decomp['unique_bert']/total_info*100:.1f}%")
    print(f"  Unique (SenticNet): {decomp['unique_sentic']/total_info*100:.1f}%")
    print(f"  Synergy:          {decomp['synergy']/total_info*100:.1f}%")
    print()
    
    # 4. Dimensionality analysis
    print("4. Intrinsic Dimensionality Analysis")
    print("-"*80)
    
    dim_results = dimensionality_reduction_analysis(X_bert_utt, X_sentic_utt, y)
    
    print(f"BERT intrinsic dim (95% var): {dim_results['bert']['intrinsic_dim_95']}")
    print(f"BERT explained var (50 PCs):  {dim_results['bert']['explained_var_50d']*100:.1f}%")
    print()
    
    # 5. Save results
    print("="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    results = {
        'task': args.task,
        'encoder': args.encoder,
        'split': args.split,
        'entropy': entropy_metrics,
        'mutual_information': {
            'I_bert': float(I_bert),
            'I_sentic': float(I_sentic),
            'I_joint': float(I_joint),
            'complementarity': float(delta_I)
        },
        'decomposition': {k: float(v) for k, v in decomp.items()},
        'dimensionality': dim_results
    }
    
    import json
    output_file = OUTPUT_DIR / f'{args.encoder}_{args.task}_{args.split}_info_theory.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Saved: {output_file}")
    print()
    
    # Visualization
    print("Generating visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Information decomposition
    ax = axes[0]
    components = ['Redundancy', 'Unique\n(BERT)', 'Unique\n(SenticNet)', 'Synergy']
    values = [
        decomp['redundancy'],
        decomp['unique_bert'],
        decomp['unique_sentic'],
        decomp['synergy']
    ]
    
    colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
    ax.bar(components, values, color=colors, alpha=0.8)
    ax.set_ylabel('Information (nats)', fontsize=12)
    ax.set_title('Information Decomposition', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: MI comparison
    ax = axes[1]
    methods = ['BERT\nonly', 'SenticNet\nonly', 'Joint\n(BERT+SenticNet)']
    mi_values = [I_bert, I_sentic, I_joint]
    
    bars = ax.bar(methods, mi_values, color=['#3498db', '#f39c12', '#2ecc71'], alpha=0.8)
    ax.set_ylabel('Mutual Information (nats)', fontsize=12)
    ax.set_title('Mutual Information I(X; Y)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add complementarity annotation
    ax.annotate('', xy=(2, I_joint), xytext=(0, I_bert),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(1, (I_bert + I_joint)/2, f'ΔI = {delta_I:.3f}',
            ha='center', fontsize=11, color='red', fontweight='bold')
    
    plt.tight_layout()
    plot_file = OUTPUT_DIR / f'{args.encoder}_{args.task}_{args.split}_info_theory.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved plot: {plot_file}")
    
    print()
    print("="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()
