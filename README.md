# Causal Emotion Recognition in Conversation

Code for the TMLR 2026 paper:

**Causal Emotion Recognition in Conversation: Context Saturation and Discourse-Marker Evidence**  
Cheonkam Jeong (cheonkamjeong@gmail.com) and Adeline Nyamathi

This repository contains code for reproducing the main text-only ERC experiments and linguistic analyses:

1. flat and hierarchical text embedding generation,
2. turn-level causal context K-sweeps,
3. context saturation and K-star analyses,
4. SenticNet fusion/ablation analyses,
5. discourse-marker position analyses,
6. multi-seed result aggregation and figure generation.

Raw IEMOCAP and MELD data are **not redistributed**. Users should obtain datasets from the official sources and place processed files under `data/` as described in `data/README.md`.

## Quick start

```bash
conda env create -f environment.yml
conda activate senticcrystal

# or
pip install -r requirements.txt
pip install -e .
```

Run commands from the repository root. You can also set the root explicitly:

```bash
export SENTICCRYSTAL_ROOT=/path/to/causal-erc-context-saturation
```

## Repository layout

```text
causal-erc-context-saturation/
├── src/senticcrystal/          # reusable model/util modules
├── scripts/
│   ├── generate/               # embedding generation
│   ├── train/                  # utterance-level and hierarchical classifiers
│   ├── train/turn/             # causal context K-sweep scripts
│   ├── fusion/                 # SenticNet fusion scripts
│   ├── analyze/                # main analysis scripts
│   ├── analyze/supplementary/  # optional/supplementary analyses
│   ├── optimization/           # development-time Optuna scripts
│   ├── ablation/               # component ablation scripts
│   ├── run/                    # multi-GPU launch examples
│   └── utils/                  # small data/NPZ utilities
├── data/                       # not tracked; see data/README.md
├── results/                    # not tracked; see results/README.md
├── docs/                       # experiment guides
├── paper/                      # citation metadata
└── legacy/                     # legacy/experimental modules
```

## Main workflows

### 1. Generate embeddings

```bash
python scripts/generate/generate_sroberta_npz_4way.py
python scripts/generate/generate_sroberta_hier_npz_4way.py
```

For MELD validation:

```bash
python scripts/generate/generate_sroberta_npz_meld.py
python scripts/generate/generate_sroberta_hier_npz_meld.py
```

### 2. Run causal context K-sweep

```bash
python scripts/train/turn/train_turnlevel_k_sweep_main.py \
  --task 4way \
  --model_tag sentence-roberta \
  --layer avg_last4 \
  --pool mean \
  --seed 42 \
  --k_min 0 --k_max 200 --k_step 10
```

Saved-prediction version for downstream K-star/information-flow analyses:

```bash
python scripts/train/turn/train_turnlevel_k_sweep_savepreds.py \
  --task 4way \
  --model_tag sentence-roberta \
  --layer avg_last4 \
  --pool mean
```

### 3. Analyze context saturation

```bash
python scripts/analyze/analyze_context_thresholds.py \
  --task 4way \
  --model_tag sentence-roberta \
  --layer avg_last4 \
  --pool mean

python scripts/analyze/analyze_context_thresholds_hard.py \
  --task 4way \
  --model_tag sentence-roberta \
  --layer avg_last4 \
  --pool mean \
  --stable_n 3 --hard_only
```

### 4. Analyze discourse markers

```bash
python scripts/analyze/merge_data_for_dm_analysis.py
python scripts/analyze/analyze_discourse_marker_positions_final.py
```

### 5. Multi-seed launch examples

```bash
bash scripts/run/run_all_n10_flat.sh
bash scripts/run/run_all_n10_hier.sh
```

These launchers are examples from a multi-GPU environment. Check GPU IDs and paths before running.

## Notes on reproducibility and paths

The public version avoids hard-coded user paths. Scripts default to the current working directory as the project root. To run from another directory, set:

```bash
export SENTICCRYSTAL_ROOT=/absolute/path/to/repo
```

Generated data, logs, embeddings, checkpoints, and raw dataset files are ignored by git.

## Citation

```bibtex
@article{jeong2026causalerc,
  title   = {Causal Emotion Recognition in Conversation: Context Saturation and Discourse-Marker Evidence},
  author  = {Jeong, Cheonkam and Nyamathi, Adeline},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=zCFQiJT7XN}
}
```

## License

MIT License. See `LICENSE`.
