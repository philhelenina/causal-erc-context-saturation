# fig_comparison_discourse_markers_v2.py
# SenticCrystal: Figure 6 composite — Discourse Marker comparison (4WAY vs 6WAY)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.gridspec import GridSpec

# --- Paths ---
DATA_4 = Path('results/discourse_markers/4way/marker_position_mean.csv')
DATA_6 = Path('results/discourse_markers/6way/marker_position_mean.csv')
HEAT_4 = Path('results/discourse_markers/4way/marker_heatmap.png')
HEAT_6 = Path('results/discourse_markers/6way/marker_heatmap.png')
OUT = Path('results/discourse_markers/comparison')
OUT.mkdir(parents=True, exist_ok=True)

# --- Load Data ---
df4 = pd.read_csv(DATA_4)
df4['task'] = '4WAY'
df6 = pd.read_csv(DATA_6)
df6['task'] = '6WAY'

# Melt wide → long
df4_m = df4.melt(id_vars=['emotion', 'task'], var_name='marker', value_name='position_mean')
df6_m = df6.melt(id_vars=['emotion', 'task'], var_name='marker', value_name='position_mean')

df = pd.concat([df4_m, df6_m], ignore_index=True)
marker_order = sorted(df['marker'].dropna().unique(), key=lambda x: x.lower())

# --- Composite Figure Setup ---
fig = plt.figure(figsize=(16, 6))
gs = GridSpec(1, 3, width_ratios=[1, 2, 1.2], wspace=0.3)

# === Left Panel: Correlation Heatmaps ===
ax0 = fig.add_subplot(gs[0])
try:
    heat4 = plt.imread(HEAT_4)
    heat6 = plt.imread(HEAT_6)
    ax0.imshow(heat4)
    ax0.set_title('4WAY Correlation', fontsize=11)
    ax0.axis('off')
    inset_ax = fig.add_axes([0.16, 0.15, 0.23, 0.25])  # inset heatmap
    inset_ax.imshow(heat6)
    inset_ax.set_title('6WAY', fontsize=9)
    inset_ax.axis('off')
except Exception as e:
    ax0.text(0.5, 0.5, f'Missing heatmaps\n({e})', ha='center', va='center')
    ax0.axis('off')

# === Middle Panel: Marker Position Comparison ===
ax1 = fig.add_subplot(gs[1])
sns.barplot(data=df, x='marker', y='position_mean', hue='task', order=marker_order, palette='Set2', ax=ax1)
ax1.set_title('Average Marker Position by Task', fontsize=11)
ax1.set_ylabel('Mean Position (0–1 normalized)')
ax1.set_xlabel('Discourse Marker')
ax1.tick_params(axis='x', rotation=45)
ax1.legend(title='Task', loc='upper right', fontsize=8)

# === Right Panel: Summary Table ===
summary = pd.DataFrame({
    'Task': ['4WAY', '6WAY'],
    'Emotion Grouping': ['Polarity-based', 'Arousal-based'],
    'Dominant Markers': ['well, so, but, oh', 'because, actually, if, oh'],
    'Position Pattern': ['Centralized (mid-sentence)', 'Polarized (early/late)'],
    'Pragmatic Role': ['Transition / Contrast', 'Justification / Activation modulation']
})

ax2 = fig.add_subplot(gs[2])
ax2.axis('off')
ax2.set_title('Summary (Paper Table Extract)', fontsize=11)
cell_text = summary.values.tolist()
col_labels = summary.columns.tolist()
ax2.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center', colLoc='center', fontsize=9)

# --- Save Figure ---
plt.tight_layout()
plt.savefig(OUT/'fig6_composite_discourse_markers.png', dpi=300)
plt.close()

# --- Save Table CSV ---
summary.to_csv(OUT/'comparison_summary.csv', index=False)

print(f"✅ Saved composite figure to: {OUT}/fig6_composite_discourse_markers.png")
print(f"✅ Saved summary table to: {OUT}/comparison_summary.csv")
