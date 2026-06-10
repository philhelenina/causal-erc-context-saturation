# analyze_marker_correlation.py
# SenticCrystal: Discourse marker–emotion correlation analysis (for 4WAY/6WAY)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# === Paths ===
BASE = Path('results/discourse_markers')
OUT = BASE / 'correlation'
OUT.mkdir(parents=True, exist_ok=True)

# === Load both datasets ===
paths = {
    '4WAY': BASE / '4way' / 'marker_position_mean.csv',
    '6WAY': BASE / '6way' / 'marker_position_mean.csv',
}

for task, path in paths.items():
    df = pd.read_csv(path)
    df = df.set_index('emotion')
    corr = df.T.corr(method='spearman')

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, vmin=-1, vmax=1)
    plt.title(f'{task} | Spearman Correlation between Marker Positions and Emotions')
    plt.tight_layout()
    plt.savefig(OUT / f'{task.lower()}_marker_emotion_corr.png', dpi=300)
    plt.close()

    corr.to_csv(OUT / f'{task.lower()}_marker_emotion_corr.csv')
    print(f'✅ Saved correlation heatmap and CSV for {task}')