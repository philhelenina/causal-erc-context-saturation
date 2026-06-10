#!/usr/bin/env python3
"""
aggregate_component_ablation_results.py

Aggregate results from component ablation study
Generate Table 5 for TMLR paper
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

HOME = Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()
RESULTS_DIR = HOME / "results/component_ablation"

CONFIGS = ['baseline', 'no_position', 'no_lexical', 'no_context']
CONFIG_NAMES = {
    'baseline': 'Full Model (Baseline)',
    'no_position': '- Position Pooling',
    'no_lexical': '- Lexical Fusion',
    'no_context': '- Context Window',
}
SEEDS = [42, 43, 44, 45, 46]


def load_single_result(config_name, seed):
    """Load results from a single experiment"""
    result_file = RESULTS_DIR / config_name / f"seed{seed}" / "results.json"
    
    if not result_file.exists():
        print(f"⚠️  Missing: {result_file}")
        return None
    
    with open(result_file, 'r') as f:
        return json.load(f)


def aggregate_config_results(config_name):
    """Aggregate results across all seeds for one config"""
    accuracies = []
    f1_macros = []
    f1_weighteds = []
    
    for seed in SEEDS:
        result = load_single_result(config_name, seed)
        if result:
            accuracies.append(result['accuracy'])
            f1_macros.append(result['f1_macro'])
            f1_weighteds.append(result['f1_weighted'])
    
    if not accuracies:
        return None
    
    return {
        'accuracy_mean': np.mean(accuracies),
        'accuracy_std': np.std(accuracies, ddof=1),
        'f1_macro_mean': np.mean(f1_macros),
        'f1_macro_std': np.std(f1_macros, ddof=1),
        'f1_weighted_mean': np.mean(f1_weighteds),
        'f1_weighted_std': np.std(f1_weighteds, ddof=1),
        'n_seeds': len(accuracies),
        'raw_f1_weighted': f1_weighteds,  # For t-test
    }


def compute_statistical_significance(baseline_scores, ablation_scores):
    """Compute paired t-test between baseline and ablation"""
    if len(baseline_scores) != len(ablation_scores):
        return None, None
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(baseline_scores, ablation_scores)
    
    # Effect size (Cohen's d for paired samples)
    diff = np.array(baseline_scores) - np.array(ablation_scores)
    cohens_d = np.mean(diff) / np.std(diff, ddof=1)
    
    return p_value, cohens_d


def generate_table5():
    """Generate Table 5: Component Ablation"""
    print("\n" + "=" * 90)
    print("Table 5: Component Ablation Study (Test Set)")
    print("=" * 90 + "\n")
    
    # Aggregate all configs
    all_results = {}
    for config_name in CONFIGS:
        agg = aggregate_config_results(config_name)
        if agg:
            all_results[config_name] = agg
        else:
            print(f"❌ No results for {config_name}")
            return None
    
    # Get baseline
    baseline = all_results.get('baseline')
    if not baseline:
        print("❌ Baseline results not found!")
        return None
    
    baseline_f1 = baseline['f1_weighted_mean']
    baseline_scores = baseline['raw_f1_weighted']
    
    # Build table
    rows = []
    
    for config_name in CONFIGS:
        agg = all_results[config_name]
        
        f1_mean = agg['f1_weighted_mean']
        f1_std = agg['f1_weighted_std']
        acc_mean = agg['accuracy_mean']
        acc_std = agg['accuracy_std']
        
        # Compute delta from baseline
        delta_f1 = f1_mean - baseline_f1
        delta_pct = (delta_f1 / baseline_f1 * 100) if baseline_f1 > 0 else 0.0
        
        # Statistical significance
        if config_name != 'baseline':
            p_value, cohens_d = compute_statistical_significance(
                baseline_scores,
                agg['raw_f1_weighted']
            )
        else:
            p_value, cohens_d = None, None
        
        # Significance marker
        if p_value is not None:
            if p_value < 0.001:
                sig_marker = '***'
            elif p_value < 0.01:
                sig_marker = '**'
            elif p_value < 0.05:
                sig_marker = '*'
            else:
                sig_marker = ''
        else:
            sig_marker = ''
        
        row = {
            'Component': CONFIG_NAMES[config_name],
            'Config': config_name,
            'Accuracy (%)': f'{acc_mean*100:.2f} ± {acc_std*100:.2f}',
            'F1 Weighted (%)': f'{f1_mean*100:.2f} ± {f1_std*100:.2f}',
            'ΔF1 (%)': f'{delta_f1*100:+.2f}',
            'Relative (%)': f'{delta_pct:+.1f}%',
            'p-value': f'{p_value:.4f}' if p_value else '-',
            'Significance': sig_marker,
            'N': agg['n_seeds'],
        }
        
        rows.append(row)
        
        # Print
        comp = CONFIG_NAMES[config_name]
        print(f"{comp:30s} | F1: {f1_mean*100:5.2f}±{f1_std*100:4.2f}% | "
              f"ΔF1: {delta_f1*100:+6.2f}% ({delta_pct:+5.1f}%) {sig_marker:3s}")
    
    print("\n" + "=" * 90)
    print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05")
    print("=" * 90 + "\n")
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Save CSV
    csv_file = RESULTS_DIR / "table5_component_ablation.csv"
    df.to_csv(csv_file, index=False)
    print(f"✅ Saved: {csv_file}\n")
    
    return df


def generate_latex_table(df):
    """Generate LaTeX code for Table 5"""
    print("\n" + "=" * 90)
    print("LaTeX Code for Table 5")
    print("=" * 90 + "\n")
    
    latex = r"""\begin{table}[t]
\centering
\caption{Component Ablation Study on IEMOCAP 4-way Classification. 
         Each component is removed independently while keeping others fixed. 
         Results are averaged over 5 random seeds.
         Significance levels: ***$p<0.001$, **$p<0.01$, *$p<0.05$ (paired t-test vs baseline).}
\label{tab:component_ablation}
\begin{tabular}{lccccc}
\toprule
\textbf{Component} & \textbf{Accuracy (\%)} & \textbf{F1 Weighted (\%)} & \textbf{$\Delta$F1 (\%)} & \textbf{Relative (\%)} & \textbf{Sig.} \\
\midrule
"""
    
    for _, row in df.iterrows():
        comp = row['Component'].replace('- ', r'$-$ ')  # LaTeX minus
        acc = row['Accuracy (%)']
        f1 = row['F1 Weighted (%)']
        delta = row['ΔF1 (%)']
        rel = row['Relative (%)']
        sig = row['Significance']
        
        latex += f"{comp:30s} & {acc:15s} & {f1:15s} & {delta:8s} & {rel:8s} & {sig:3s} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    print(latex)
    
    # Save to file
    tex_file = RESULTS_DIR / "table5_latex.tex"
    with open(tex_file, 'w') as f:
        f.write(latex)
    
    print(f"✅ Saved: {tex_file}\n")


def generate_summary_statistics():
    """Generate detailed summary statistics"""
    print("\n" + "=" * 90)
    print("Detailed Summary Statistics")
    print("=" * 90 + "\n")
    
    baseline = aggregate_config_results('baseline')
    if not baseline:
        return
    
    baseline_f1 = baseline['f1_weighted_mean']
    baseline_scores = baseline['raw_f1_weighted']
    
    print(f"Baseline F1: {baseline_f1*100:.2f}% ± {baseline['f1_weighted_std']*100:.2f}%\n")
    
    for config_name in ['no_position', 'no_lexical', 'no_context']:
        agg = aggregate_config_results(config_name)
        if not agg:
            continue
        
        f1_mean = agg['f1_weighted_mean']
        f1_std = agg['f1_weighted_std']
        
        delta = f1_mean - baseline_f1
        delta_pct = (delta / baseline_f1 * 100)
        
        p_value, cohens_d = compute_statistical_significance(
            baseline_scores,
            agg['raw_f1_weighted']
        )
        
        print(f"{CONFIG_NAMES[config_name]:30s}")
        print(f"  F1: {f1_mean*100:.2f}% ± {f1_std*100:.2f}%")
        print(f"  ΔF1: {delta*100:+.2f}% ({delta_pct:+.1f}%)")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Cohen's d: {cohens_d:.3f}")
        print(f"  Interpretation: ", end='')
        
        if abs(cohens_d) < 0.2:
            print("negligible effect")
        elif abs(cohens_d) < 0.5:
            print("small effect")
        elif abs(cohens_d) < 0.8:
            print("medium effect")
        else:
            print("large effect")
        
        print()


def main():
    """Main execution"""
    print("\n" + "=" * 90)
    print("Component Ablation Study - Results Aggregation")
    print("=" * 90)
    
    # Check results directory
    if not RESULTS_DIR.exists():
        print(f"\n❌ Results directory not found: {RESULTS_DIR}")
        print("Run train_component_ablation.py first!")
        return
    
    # Check all results exist
    missing_count = 0
    for config_name in CONFIGS:
        for seed in SEEDS:
            result_file = RESULTS_DIR / config_name / f"seed{seed}" / "results.json"
            if not result_file.exists():
                missing_count += 1
    
    if missing_count > 0:
        print(f"\n⚠️  Warning: {missing_count} result files missing")
        print("Some experiments may have failed. Continuing with available results...\n")
    
    # Generate Table 5
    df = generate_table5()
    
    if df is not None:
        # Generate LaTeX
        generate_latex_table(df)
        
        # Generate detailed statistics
        generate_summary_statistics()
    
    print("\n" + "=" * 90)
    print("✅ Results aggregation complete!")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()