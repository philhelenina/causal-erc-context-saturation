#!/usr/bin/env python3
"""
analyze_sentences_iemocap.py
- hierarchical embedding과 동일한 문장 분리 로직 사용
- 발화당 문장 수 통계 분석
"""
import os

import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def simple_sent_split(text: str):
    """hierarchical embedding과 동일한 문장 분리 함수"""
    # 매우 보수적인 문장 분리기(., !, ? + 공백)
    parts = re.split(r'(?<=[\.!\?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def analyze_dataset(csv_path, text_col=None):
    """데이터셋 분석"""
    print(f"\n📂 Analyzing: {csv_path.name}")
    
    df = pd.read_csv(csv_path)
    
    # Text column 찾기
    if text_col is None:
        candidates = ["text", "utterance", "utt", "sentence", "transcript"]
        for col in candidates:
            if col in df.columns:
                text_col = col
                break
        if text_col is None:
            # object type column 찾기
            for col in df.columns:
                if df[col].dtype == object:
                    text_col = col
                    break
    
    if text_col is None:
        print(f"  ❌ No text column found in {df.columns.tolist()}")
        return None
    
    print(f"  ✓ Using column: '{text_col}'")
    print(f"  ✓ Total utterances: {len(df)}")
    
    # 문장 수 분석
    sentence_counts = []
    examples = []
    
    texts = df[text_col].astype(str).fillna("").tolist()
    
    for i, txt in enumerate(texts):
        if not txt or txt == 'nan':
            sents = []
        else:
            sents = simple_sent_split(txt)
            if not sents:  # 분리 실패시 원본을 하나의 문장으로
                sents = [txt.strip()]
        
        num_sents = len(sents)
        sentence_counts.append(num_sents)
        
        # 예제 수집
        if num_sents >= 3 and len(examples) < 5:
            examples.append({
                'index': i,
                'text': txt[:200] + '...' if len(txt) > 200 else txt,
                'sentences': sents,
                'count': num_sents
            })
    
    return {
        'sentence_counts': sentence_counts,
        'examples': examples,
        'text_col': text_col
    }

def print_statistics(results, split_name=""):
    """통계 출력"""
    counts = np.array(results['sentence_counts'])
    
    print(f"\n{'='*60}")
    if split_name:
        print(f"📊 {split_name.upper()} STATISTICS")
    else:
        print(f"📊 STATISTICS")
    print(f"{'='*60}")
    
    print(f"\nBasic Stats:")
    print(f"  Total utterances:     {len(counts)}")
    print(f"  Mean sentences:       {np.mean(counts):.2f}")
    print(f"  Median:              {np.median(counts):.0f}")
    print(f"  Std:                 {np.std(counts):.2f}")
    print(f"  Min:                 {np.min(counts)}")
    print(f"  Max (S_max):         {np.max(counts)}")
    
    # Distribution
    print(f"\nDistribution:")
    value_counts = Counter(counts)
    total = len(counts)
    
    for num_sent in sorted(value_counts.keys())[:10]:
        count = value_counts[num_sent]
        pct = count / total * 100
        bar = '█' * int(pct / 2)
        print(f"  {num_sent:2d} sentence(s): {count:5d} ({pct:5.1f}%) {bar}")
    
    # Summary
    print(f"\nSummary:")
    print(f"  0 sentences (empty):     {(counts == 0).sum():5d} ({(counts == 0).mean() * 100:5.1f}%)")
    print(f"  1 sentence:              {(counts == 1).sum():5d} ({(counts == 1).mean() * 100:5.1f}%)")
    print(f"  2 sentences:             {(counts == 2).sum():5d} ({(counts == 2).mean() * 100:5.1f}%)")
    print(f"  3 sentences:             {(counts == 3).sum():5d} ({(counts == 3).mean() * 100:5.1f}%)")
    print(f"  4+ sentences:            {(counts >= 4).sum():5d} ({(counts >= 4).mean() * 100:5.1f}%)")
    
    # For hierarchical model
    print(f"\n🏗️ Hierarchical Model Implications:")
    print(f"  Single-sentence (no aggregation): {(counts == 1).mean() * 100:.1f}%")
    print(f"  Multi-sentence (needs aggregation): {(counts > 1).mean() * 100:.1f}%")
    print(f"  Empty utterances: {(counts == 0).sum()}")
    print(f"  Variable m ∈ [0, {np.max(counts)}]")
    
    # Examples
    if results['examples']:
        print(f"\n📝 Examples of Multi-Sentence Utterances:")
        for ex in results['examples'][:3]:
            print(f"\n  Example (index {ex['index']}, {ex['count']} sentences):")
            print(f"    Original: {ex['text']}")
            print(f"    Split into:")
            for j, sent in enumerate(ex['sentences'][:5], 1):
                print(f"      [{j}] {sent}")

def create_visualization(all_results):
    """시각화 생성"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('IEMOCAP Sentence Distribution (using simple_sent_split)', fontsize=14, fontweight='bold')
    
    # 1. Overall histogram
    ax = axes[0, 0]
    all_counts = []
    for split, result in all_results.items():
        all_counts.extend(result['sentence_counts'])
    
    all_counts = np.array(all_counts)
    max_val = min(10, all_counts.max())
    ax.hist(all_counts[all_counts <= max_val], bins=range(0, max_val+2), 
            edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel('Number of Sentences')
    ax.set_ylabel('Frequency')
    ax.set_title('Overall Distribution')
    ax.axvline(all_counts.mean(), color='red', linestyle='--', 
              label=f'Mean: {all_counts.mean():.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Distribution by split (boxplot)
    ax = axes[0, 1]
    split_data = []
    split_labels = []
    for split in ['train', 'val', 'test']:
        if split in all_results:
            split_data.append(all_results[split]['sentence_counts'])
            split_labels.append(f"{split}\n(n={len(all_results[split]['sentence_counts'])})")
    
    if split_data:
        bp = ax.boxplot(split_data, labels=split_labels)
        ax.set_ylabel('Number of Sentences')
        ax.set_title('Distribution by Split')
        ax.grid(True, alpha=0.3)
    
    # 3. Pie chart
    ax = axes[0, 2]
    sizes = [
        (all_counts == 0).sum(),
        (all_counts == 1).sum(),
        (all_counts == 2).sum(),
        (all_counts == 3).sum(),
        (all_counts >= 4).sum()
    ]
    labels = ['0 (empty)', '1 sent', '2 sent', '3 sent', '4+ sent']
    colors = plt.cm.Set3(range(len(labels)))
    
    # Filter out zero counts
    non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
    if non_zero:
        sizes, labels, colors = zip(*non_zero)
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Sentence Count Distribution')
    
    # 4. CDF
    ax = axes[1, 0]
    sorted_counts = np.sort(all_counts)
    cdf = np.arange(1, len(sorted_counts)+1) / len(sorted_counts)
    ax.plot(sorted_counts, cdf, linewidth=2)
    ax.set_xlabel('Number of Sentences')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution Function')
    ax.grid(True, alpha=0.3)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Median')
    ax.axhline(0.9, color='orange', linestyle='--', alpha=0.5, label='90th percentile')
    ax.legend()
    
    # 5. Bar chart for common cases
    ax = axes[1, 1]
    value_counts = Counter(all_counts)
    common_vals = sorted(value_counts.keys())[:6]
    counts = [value_counts[v] for v in common_vals]
    ax.bar(common_vals, counts, color='coral', edgecolor='black')
    ax.set_xlabel('Number of Sentences')
    ax.set_ylabel('Count')
    ax.set_title('Most Common Sentence Counts')
    ax.set_xticks(common_vals)
    ax.grid(True, alpha=0.3)
    
    # 6. Summary text
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = f"""
SUMMARY STATISTICS
==================

Total utterances: {len(all_counts):,}
Mean sentences/utterance: {np.mean(all_counts):.2f}
Median: {np.median(all_counts):.0f}
Std dev: {np.std(all_counts):.2f}
Max (S_max): {np.max(all_counts)}

Distribution:
  Empty:        {(all_counts == 0).mean() * 100:5.1f}%
  1 sentence:   {(all_counts == 1).mean() * 100:5.1f}%
  2 sentences:  {(all_counts == 2).mean() * 100:5.1f}%
  3+ sentences: {(all_counts >= 3).mean() * 100:5.1f}%

Hierarchical Model:
  No aggregation: {(all_counts == 1).mean() * 100:.1f}%
  Needs aggregation: {(all_counts > 1).mean() * 100:.1f}%

📌 For paper:
"IEMOCAP utterances contain 
0-{np.max(all_counts)} sentences (mean={np.mean(all_counts):.1f}),
with {(all_counts == 1).mean() * 100:.0f}% single-sentence
and {(all_counts > 1).mean() * 100:.0f}% multi-sentence."
    """
    
    ax.text(0.05, 0.5, summary_text, fontsize=9, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    
    plt.tight_layout()
    return fig

def main():
    print("="*70)
    print("IEMOCAP SENTENCE STATISTICS (Hierarchical Embedding Logic)")
    print("="*70)
    
    # Paths
    HOME = Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()
    DATA_DIR = HOME / "data" / "iemocap_4way_data"
    
    # Alternative paths to check
    if not DATA_DIR.exists():
        print(f"⚠️ Default path not found: {DATA_DIR}")
        DATA_DIR = Path(".")  # Current directory
    
    # Analyze each split
    all_results = {}
    
    for split in ["train", "val", "test"]:
        csv_path = DATA_DIR / f"{split}_4way_unified.csv"
        
        if csv_path.exists():
            result = analyze_dataset(csv_path)
            if result:
                all_results[split] = result
                print_statistics(result, split)
        else:
            print(f"\n⚠️ File not found: {csv_path}")
    
    # Combined statistics
    if all_results:
        combined = {
            'sentence_counts': [],
            'examples': []
        }
        
        for result in all_results.values():
            combined['sentence_counts'].extend(result['sentence_counts'])
            combined['examples'].extend(result['examples'][:2])
        
        print_statistics(combined, "COMBINED (ALL SPLITS)")
        
        # Visualization
        print("\n📊 Creating visualization...")
        fig = create_visualization(all_results)
        output_path = Path("iemocap_sentence_distribution.png")
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        
        # Final summary for paper
        counts = np.array(combined['sentence_counts'])
        print("\n" + "="*70)
        print("📝 COPY-PASTE FOR PAPER:")
        print("="*70)
        print(f"""
\\textbf{{Sentence Distribution:}}
IEMOCAP utterances contain 0-{np.max(counts)} sentences per utterance 
using conservative punctuation-based segmentation (mean={np.mean(counts):.2f}±{np.std(counts):.2f}). 
Single-sentence utterances comprise {(counts == 1).mean() * 100:.1f}\\%, 
while {(counts > 1).mean() * 100:.1f}\\% require sentence aggregation 
in our hierarchical architecture. The maximum $m$ (sentences per utterance) 
is {np.max(counts)}, with {(counts >= 4).mean() * 100:.1f}\\% of utterances 
containing 4 or more sentences.
        """)

if __name__ == "__main__":
    main()
