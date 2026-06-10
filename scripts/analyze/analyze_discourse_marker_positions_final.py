#!/usr/bin/env python3
"""
analyze_discourse_markers_all.py
IEMOCAP 전체 데이터 통합 + 화용론적 담화표지 분류
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats

# 학술적 담화표지 분류
DISCOURSE_MARKERS_ACADEMIC = {
    # Schiffrin (1987) - Core markers
    'schiffrin_core': ['oh', 'well', 'you know', 'i mean', 'now', 'then', 'so', 'because', 'and', 'but', 'or'],
    
    # Fraser (1999, 2009) - Pragmatic markers
    'fraser_contrastive': ['but', 'however', 'although', 'nonetheless', 'nevertheless', 'still', 'yet', 'though'],
    'fraser_elaborative': ['and', 'moreover', 'furthermore', 'besides', 'additionally', 'also', 'too'],
    'fraser_inferential': ['so', 'therefore', 'thus', 'consequently', 'hence', 'accordingly', 'then'],
    'fraser_temporal': ['then', 'meanwhile', 'subsequently', 'afterwards', 'finally', 'next'],
    
    # Traugott (2010) - Subjectivity markers
    'subjective_epistemic': ['i think', 'i guess', 'i believe', 'maybe', 'perhaps', 'probably', 'possibly'],
    'subjective_attitudinal': ['unfortunately', 'happily', 'sadly', 'frankly', 'honestly', 'personally'],
    
    # Verhagen (2005) - Intersubjective markers  
    'intersubjective': ['you know', 'you see', 'right', 'okay', 'i mean', 'lets say', 'you understand'],
    
    # Beeching & Detges (2014) - Peripheral markers
    'left_peripheral': ['well', 'so', 'but', 'and', 'oh', 'now', 'look', 'listen'],
    'right_peripheral': ['though', 'right', 'you know', 'i think', 'or something', 'or whatever', 'and stuff'],
    
    # Aijmer (2013) - Pragmatic particles
    'pragmatic_particles': ['like', 'just', 'really', 'quite', 'pretty', 'sort of', 'kind of', 'actually', 'basically'],
    
    # Stance markers (Biber & Finegan 1989)
    'stance_certainty': ['definitely', 'certainly', 'obviously', 'clearly', 'surely', 'undoubtedly'],
    'stance_doubt': ['maybe', 'perhaps', 'possibly', 'probably', 'allegedly', 'supposedly']
}

def load_all_data(task='4way'):
    """전체 데이터 로드"""
    all_dfs = []
    
    for split in ['train', 'val', 'test']:
        csv_path = Path(f'data/iemocap_{task}_data/{split}_{task}_with_minus_one.csv')
        
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['split'] = split
            all_dfs.append(df)
            print(f"  Loaded {split}: {len(df)} utterances")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"Total {task}: {len(combined)} utterances\n")
    return combined

def extract_discourse_markers_academic(df, task='4way'):
    """학술적 분류로 담화표지 추출"""
    
    # 모든 마커를 flat list로 + 카테고리 매핑
    all_markers = []
    marker_to_categories = {}
    
    for category, markers in DISCOURSE_MARKERS_ACADEMIC.items():
        for marker in markers:
            all_markers.append(marker)
            if marker not in marker_to_categories:
                marker_to_categories[marker] = []
            marker_to_categories[marker].append(category)
    
    all_markers = list(set(all_markers))  # 중복 제거
    
    results = []
    
    for idx, row in df.iterrows():
        text = str(row.get('text', row.get('utterance', ''))).lower()
        tokens = text.split()
        n_tokens = len(tokens)
        
        if n_tokens == 0:
            continue
        
        # 감정 레이블 처리
        if 'label' in row:
            if isinstance(row['label'], str):
                emotion_map = {'ang': 0, 'hap': 1, 'sad': 2, 'neu': 3, 'exc': 4, 'fru': 5}
                emotion = emotion_map.get(row['label'], -1)
            else:
                emotion = int(row['label']) if not pd.isna(row['label']) else -1
        else:
            emotion = -1
        
        # 담화표지 찾기
        for marker in all_markers:
            marker_tokens = marker.split()
            
            if len(marker_tokens) == 1:
                # 단일 토큰
                for i, token in enumerate(tokens):
                    if token == marker or token.startswith(marker):
                        rel_pos = i / max(n_tokens - 1, 1)
                        results.append({
                            'marker': marker,
                            'categories': marker_to_categories[marker],
                            'position': rel_pos,
                            'emotion': emotion,
                            'n_tokens': n_tokens,
                            'task': task
                        })
            else:
                # 다중 토큰 (예: "you know")
                for i in range(len(tokens) - len(marker_tokens) + 1):
                    if tokens[i:i+len(marker_tokens)] == marker_tokens:
                        rel_pos = i / max(n_tokens - 1, 1)
                        results.append({
                            'marker': marker,
                            'categories': marker_to_categories[marker],
                            'position': rel_pos,
                            'emotion': emotion,
                            'n_tokens': n_tokens,
                            'task': task
                        })
    
    return pd.DataFrame(results)

def analyze_by_category(df_markers):
    """카테고리별 분석"""
    
    category_stats = {}
    
    for category in DISCOURSE_MARKERS_ACADEMIC.keys():
        # 해당 카테고리 마커들만 필터
        cat_markers = df_markers[df_markers['categories'].apply(
            lambda x: category in x if isinstance(x, list) else False
        )]
        
        if len(cat_markers) > 0:
            category_stats[category] = {
                'count': len(cat_markers),
                'mean_position': cat_markers['position'].mean(),
                'left_periphery': (cat_markers['position'] < 0.1).mean(),
                'right_periphery': (cat_markers['position'] > 0.9).mean(),
                'lr_ratio': (cat_markers['position'] < 0.1).mean() / max((cat_markers['position'] > 0.9).mean(), 0.001)
            }
    
    return category_stats

def create_comprehensive_figure(df_4way, df_6way):
    """종합 시각화"""
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Category distribution comparison
    ax1 = fig.add_subplot(gs[0, :])
    
    cat_4way = analyze_by_category(df_4way)
    cat_6way = analyze_by_category(df_6way)
    
    categories = list(cat_4way.keys())
    x = np.arange(len(categories))
    width = 0.35
    
    counts_4way = [cat_4way[c]['count'] for c in categories]
    counts_6way = [cat_6way[c]['count'] if c in cat_6way else 0 for c in categories]
    
    ax1.bar(x - width/2, counts_4way, width, label='4-way', color='steelblue')
    ax1.bar(x + width/2, counts_6way, width, label='6-way', color='coral')
    
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Discourse Marker Categories Distribution')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace('_', ' ').title() for c in categories], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Emotion-specific L/R ratios
    ax2 = fig.add_subplot(gs[1, 0])
    
    emotion_labels = {0: 'Angry', 1: 'Happy', 2: 'Sad', 3: 'Neutral', 4: 'Excited', 5: 'Frustrated'}
    
    lr_stats = []
    for emotion in sorted(df_4way['emotion'].unique()):
        if emotion == -1:
            continue
        emo_data = df_4way[df_4way['emotion'] == emotion]
        left = (emo_data['position'] < 0.1).mean()
        right = (emo_data['position'] > 0.9).mean()
        lr = left / max(right, 0.001)
        lr_stats.append((emotion_labels.get(emotion, str(emotion)), lr))
    
    lr_stats.sort(key=lambda x: x[1], reverse=True)
    emotions, ratios = zip(*lr_stats)
    
    ax2.barh(range(len(emotions)), ratios, color='skyblue')
    ax2.set_yticks(range(len(emotions)))
    ax2.set_yticklabels(emotions)
    ax2.set_xlabel('L/R Ratio')
    ax2.set_title('4-way: Left/Right Periphery Ratio by Emotion')
    ax2.grid(True, alpha=0.3)
    
    # 3. Subjective vs Intersubjective
    ax3 = fig.add_subplot(gs[1, 1])
    
    subj = df_4way[df_4way['categories'].apply(
        lambda x: any('subjective' in c for c in x) if isinstance(x, list) else False
    )]
    inter = df_4way[df_4way['categories'].apply(
        lambda x: 'intersubjective' in x if isinstance(x, list) else False
    )]
    
    data = pd.DataFrame({
        'Type': ['Subjective'] * len(subj) + ['Intersubjective'] * len(inter),
        'Position': list(subj['position']) + list(inter['position'])
    })
    
    if len(data) > 0:
        sns.violinplot(data=data, x='Type', y='Position', ax=ax3)
        ax3.set_title('Subjective vs Intersubjective Positioning')
        ax3.grid(True, alpha=0.3)
    
    # 4. Position heatmap by emotion and category
    ax4 = fig.add_subplot(gs[1, 2])
    
    # Create emotion x category mean position matrix
    emotions = [0, 1, 2, 3]
    categories_main = ['schiffrin_core', 'fraser_contrastive', 'subjective_epistemic', 'intersubjective']
    
    heatmap_data = []
    for emotion in emotions:
        row = []
        for cat in categories_main:
            cat_emo_data = df_4way[(df_4way['emotion'] == emotion) & 
                                   (df_4way['categories'].apply(lambda x: cat in x if isinstance(x, list) else False))]
            if len(cat_emo_data) > 0:
                row.append(cat_emo_data['position'].mean())
            else:
                row.append(0.5)
        heatmap_data.append(row)
    
    im = ax4.imshow(heatmap_data, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1)
    ax4.set_xticks(range(len(categories_main)))
    ax4.set_xticklabels([c.replace('_', '\n') for c in categories_main], fontsize=9)
    ax4.set_yticks(range(len(emotions)))
    ax4.set_yticklabels([emotion_labels.get(e, str(e)) for e in emotions])
    ax4.set_title('Mean Position by Emotion & Category')
    plt.colorbar(im, ax=ax4)
    
    # 5. Summary statistics table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    summary = f"""
    COMPREHENSIVE DISCOURSE MARKER ANALYSIS
    =====================================
    
    4-WAY RESULTS:
    Total markers: {len(df_4way)}
    Mean position: {df_4way['position'].mean():.3f}
    Left periphery (<0.1): {(df_4way['position'] < 0.1).mean():.1%}
    Right periphery (>0.9): {(df_4way['position'] > 0.9).mean():.1%}
    L/R Ratio: {(df_4way['position'] < 0.1).mean() / max((df_4way['position'] > 0.9).mean(), 0.001):.2f}
    
    6-WAY RESULTS:
    Total markers: {len(df_6way)}
    Mean position: {df_6way['position'].mean():.3f}
    Left periphery (<0.1): {(df_6way['position'] < 0.1).mean():.1%}
    Right periphery (>0.9): {(df_6way['position'] > 0.9).mean():.1%}
    L/R Ratio: {(df_6way['position'] < 0.1).mean() / max((df_6way['position'] > 0.9).mean(), 0.001):.2f}
    """
    
    ax5.text(0.1, 0.5, summary, fontsize=10, family='monospace',
             verticalalignment='center')
    
    plt.suptitle('IEMOCAP Discourse Markers: Pragmatic Analysis (Full Dataset)', fontsize=14, fontweight='bold')
    
    return fig

def main():
    
    results = {}
    
    # 4-way 분석
    print("Loading 4-way data...")
    df_4way_all = load_all_data('4way')
    df_4way_markers = extract_discourse_markers_academic(df_4way_all, '4way')
    
    # 6-way 분석
    print("Loading 6-way data...")
    df_6way_all = load_all_data('6way')
    df_6way_markers = extract_discourse_markers_academic(df_6way_all, '6way')
    
    # 카테고리별 분석
    cat_stats_4way = analyze_by_category(df_4way_markers)
    cat_stats_6way = analyze_by_category(df_6way_markers)
    
    # 시각화
    fig = create_comprehensive_figure(df_4way_markers, df_6way_markers)
    out_dir = Path('results/discourse_markers')
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / 'comprehensive_pragmatic_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 결과 저장
    results = {
        '4way': {
            'total_markers': len(df_4way_markers),
            'mean_position': float(df_4way_markers['position'].mean()),
            'left_periphery': float((df_4way_markers['position'] < 0.1).mean()),
            'right_periphery': float((df_4way_markers['position'] > 0.9).mean()),
            'category_stats': cat_stats_4way
        },
        '6way': {
            'total_markers': len(df_6way_markers),
            'mean_position': float(df_6way_markers['position'].mean()),
            'left_periphery': float((df_6way_markers['position'] < 0.1).mean()),
            'right_periphery': float((df_6way_markers['position'] > 0.9).mean()),
            'category_stats': cat_stats_6way
        }
    }
    
    with open(out_dir / 'comprehensive_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 감정별 요약 출력
    print("\n" + "="*60)
    print("EMOTION-SPECIFIC L/R RATIOS (4-way)")
    print("="*60)
    
    emotion_labels = {0: 'Angry', 1: 'Happy', 2: 'Sad', 3: 'Neutral'}
    
    for emotion in sorted(df_4way_markers['emotion'].unique()):
        if emotion == -1:
            continue
        emo_data = df_4way_markers[df_4way_markers['emotion'] == emotion]
        left = (emo_data['position'] < 0.1).mean()
        right = (emo_data['position'] > 0.9).mean()
        lr = left / max(right, 0.001)
        print(f"{emotion_labels.get(emotion, emotion):10s}: L={left:.3f}, R={right:.3f}, L/R={lr:.2f}")
    
    print("\n" + "="*60)
    print("CATEGORY-WISE ANALYSIS")
    print("="*60)
    
    for cat, stats in cat_stats_4way.items():
        print(f"\n{cat.replace('_', ' ').title()}:")
        print(f"  Count: {stats['count']}")
        print(f"  Mean position: {stats['mean_position']:.3f}")
        print(f"  L/R ratio: {stats['lr_ratio']:.2f}")
    
    print("\n✓ Analysis complete!")

if __name__ == "__main__":
    main()