# make_table3_marker_summary.py
# SenticCrystal: Generate Table 3 — Discourse marker position & emotional function summary

import pandas as pd
from pathlib import Path

# === Paths ===
OUT = Path('results/discourse_markers/tables')
OUT.mkdir(parents=True, exist_ok=True)

# === Table Content ===
data = [
    {
        'Emotion Group': 'Positive (exc, hap)',
        'Dominant Markers': 'so, well, because',
        'Typical Position': 'mid–late',
        'Functional Role': 'elaboration, justification, topic continuation',
    },
    {
        'Emotion Group': 'Negative (ang, sad)',
        'Dominant Markers': 'but, although, though',
        'Typical Position': 'early',
        'Functional Role': 'contrast, mitigation, correction',
    },
    {
        'Emotion Group': 'Frustrated (fru)',
        'Dominant Markers': 'actually, after all, if',
        'Typical Position': 'mid',
        'Functional Role': 'stance clarification, self-justification',
    },
    {
        'Emotion Group': 'Neutral (neu)',
        'Dominant Markers': 'therefore, indeed, anyway',
        'Typical Position': 'varied (mid–final)',
        'Functional Role': 'reasoning, conversational grounding',
    },
]

df = pd.DataFrame(data)

# === Save CSV & LaTeX ===
latex_table = df.to_latex(index=False, column_format='p{3cm}p{4cm}p{3cm}p{5cm}', escape=False)
(df.to_csv(OUT / 'table3_marker_summary.csv', index=False))
with open(OUT / 'table3_marker_summary.tex', 'w') as f:
    f.write(latex_table)

print(f'✅ Saved table to: {OUT}/table3_marker_summary.csv and .tex')
