# Extracting Search Trees from LLM Reasoning Traces Reveals Myopic Planning

This repository contains code and data for the paper *Extracting Search Trees from LLM Reasoning Traces Reveals Myopic Planning*. We study how 27 LLMs perform game-tree search when playing four-in-a-row, by extracting explicit move trees from their reasoning traces and fitting parametric cognitive models to their move choices.

## Overview

**Core question:** Do LLMs perform tree search through chain-of-thought reasoning. If so, how do they use search to inform their decisions?

**Approach:** We collect reasoning traces from 27 LLMs playing four-in-a-row (a 4×9 board game), extract the search trees from their reasoning traces, and fit computational models to predict their move choices. We compare a full-tree (minimax backpropagation) variant against a myopic (depth-1 heuristic) variant, and run causal intervention experiments to test which parts of the trace drive moves.

**Main findings:**
- The myopic model fits significantly better than full-tree minimax
- Extracted search trees carry predictive information about move decisions

## Repository Structure

```
.
├── extract_search_trees.ipynb        # Step 1: extract trees from reasoning traces via GPT-5
├── preprocess_dataframe.ipynb        # Step 2: augment dataframe with outcomes and metadata
│
├── run_full.py / run_full.sh         # Step 3a: full-tree model fits (SLURM array)
├── run_myopic.sh                     # Step 3b: myopic baseline fits
├── run_notree.sh                     # Step 3c: no-tree baseline fits
├── run_gamma.sh                      # Step 3d: gamma-discounted fits
│
├── run_model_recovery.py             # Step 4: model recovery validation
├── run_model_recovery.sh             # SLURM launcher for step 4
│
├── intervention_label.py             # Step 5a: label reasoning paragraphs via Claude API
├── intervention_edit.py              # Step 5b: apply surgical edits to traces
├── intervention_infer.py             # Step 5c: run local model with edited prefills
├── run_intervention.sh               # SLURM launcher for step 5c
├── run_download.sh                   # Download Qwen3-80B weights for intervention
│
├── features.py                       # Heuristic feature computation (5 board features)
├── preprocess.py                     # FEN parsing, board encoding, tree parsing
├── minimax.py                        # Tree annotation and minimax/negamax evaluation
├── fit.py                            # MLE fitting (L-BFGS-B, multi-restart)
│
├── four_in_a_row/                    # Board game utilities and tree rendering
│
├── analysis_tree_search_effort.ipynb    # Figure: tree size vs. win rate
├── analysis_myopic_vs_fulltree.ipynb    # Figure: myopic vs. full-tree accuracy
├── analysis_feature_weights.ipynb       # Figure: fitted weights vs. win rate
├── analysis_fit_result_comparison.ipynb # Figure: model comparison across variants
├── analysis_model_recovery.ipynb        # Figure: model recovery results
├── analysis_intervention.ipynb          # Figure: intervention causal effects
│
├── game_trees_df_annotated.pkl              # Extracted and annotated search trees (all 27 models)
├── game_trees_df_annotated_preprocessed.pkl # Enriched with outcomes and metadata
│
├── results/                                  # Pre-computed fit results (JSON) for all models × variants
│   ├── fit_results_full_<model>.json         # Full-tree fit
│   ├── fit_results_full_<model>_myopic.json  # Myopic baseline
│   ├── fit_results_full_<model>_notree.json  # No-tree baseline
│   ├── fit_results_full_<model>_gamma.json   # Gamma-discounted fit
│   ├── recovery/                             # Model recovery results per model
│   ├── intervention_labels_250.jsonl         # Labeled reasoning paragraphs (250 turns)
│   ├── intervention/                         # intervention edits + results
│
├── pyproject.toml                    # Dependencies (managed via uv)
└── uv.lock                           # Pinned dependency lockfile
```

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already.

Sync the environment (installs Python 3.11 and all pinned dependencies):

```bash
uv sync
```

You also need the Graphviz system binary for tree rendering. On macOS:

```bash
brew install graphviz
```

For other systems, see the [Graphviz download page](https://graphviz.org/download/).

## Data Download

The raw game logs (~1.1 GB compressed, ~3.1 GB uncompressed) are hosted anonymously on OSF.

```bash
wget -O aws-logs.tar.gz "https://osf.io/download/69fba9c72c2505747af2009d/?view_only=e2e531ce7af64cd5a0a9a889bd60bfc0"
tar -xzf aws-logs.tar.gz
rm aws-logs.tar.gz
```

This creates an `aws-logs/` directory:

```
aws-logs/
├── four_in_a_row-fen/       # FEN notation variant (used in the paper)
└── four_in_a_row-standard/  # Standard notation variant
```

Each variant contains **351 matchup folders** for all pairings of 27 models (27 × 26 / 2 = 351). Each matchup has **4 game folders** — 2 games where model A moves first and 2 where model B moves first:

```
four_in_a_row-fen/
└── <model-a>_vs_<model-b>/
    ├── batch-<id>--<timestamp>/
    ├── batch-.../
    ├── batch-.../
    └── batch-.../
```

Each game folder contains **4 files**:

| File | Description |
|------|-------------|
| `game_log.json` | Game event stream, metadata, and outcome |
| `game.log` | Debugging log |
| `<player-0-model>(0)-log.jsonl` | Full logs for player 0 (includes reasoning traces) |
| `<player-1-model>(1)-log.jsonl` | Full logs for player 1 (includes reasoning traces) |

## Reproducing Results

The pre-computed outputs needed to run all analysis notebooks are already included in this repository (`game_trees_df_annotated_preprocessed.pkl`, `results/`). You can go directly to [Analysis Notebooks](#analysis-notebooks) to reproduce all figures.

To reproduce results from scratch, follow the steps below.

### Step 1 — Extract Search Trees

`extract_search_trees.ipynb` parses the raw game logs and uses GPT-5 (via the OpenAI batch API and DSPy) to extract explicit move trees from each model's reasoning trace. Update the `logs_dir` variable at the top of the notebook to point to your local `aws-logs/four_in_a_row-fen/` directory and set your OpenAI API key.

**Output:** `game_trees_df.pkl` — a DataFrame with one row per game turn (18,888 rows), containing the extracted tree, move, model name, and search effort metrics.

### Step 2 — Preprocess

`preprocess_dataframe.ipynb` cleans the raw trees, annotates each turn with game outcome (win/loss from the model's perspective), token counts, and model metadata.

**Input:** `game_trees_df.pkl` + raw game logs  
**Output:** `game_trees_df_annotated_preprocessed.pkl`

### Step 3 — Fit Models

We fit four model variants to each model's move choices. Each variant is a SLURM array job (one job per model, 27 models total):

| Script | Variant | Description |
|--------|---------|-------------|
| `run_full.sh` | Full-tree | Minimax backpropagation over the full extracted tree |
| `run_myopic.sh` | Myopic | Depth-1 heuristic evaluation of root candidates only |
| `run_notree.sh` | No-tree | Replace LLM candidates with all legal moves |
| `run_gamma.sh` | Gamma | Fit depth-discounting parameter γ as a free variable |

```bash
sbatch run_full.sh
sbatch run_myopic.sh
sbatch run_notree.sh
sbatch run_gamma.sh
```

Each job writes a JSON result to `results/fit_results_full_<model>[_variant].json`.

**Pre-computed results** for all 27 models × 4 variants are already in `results/`.

#### Model parameters

Each variant fits up to 7 parameters via maximum-likelihood (L-BFGS-B, 20 random restarts):

| Parameter | Description |
|-----------|-------------|
| `w[0]` | Weight on centre-proximity feature |
| `w[1]` | Weight on connected-2 (two adjacent pieces) |
| `w[2]` | Weight on unconnected-2 (two non-adjacent pieces) |
| `w[3]` | Weight on three-in-a-row |
| `w[4]` | Weight on four-in-a-row (terminal) |
| `C` | Active scaling constant (own-piece salience boost) |
| `γ` | Minimax continuation discount (gamma variant only; fixed to 1 in full, 0 in myopic) |

### Step 4 — Model Recovery

Validates that the fitting procedure identifies the true generating process by fitting both variants to synthetic data sampled from each:

```bash
sbatch run_model_recovery.sh
# After all array jobs complete, aggregate:
python run_model_recovery.py --all
```

**Output:** `results/recovery/recovery_<model>.json` per model, aggregated into `results/model_recovery_all.json`.

### Step 5 — Intervention Experiment

The intervention experiment requires the [Qwen3-Next-80B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking) model weights and a CUDA-capable GPU (4× GPUs recommended). Install GPU dependencies first:

```bash
uv sync --extra gpu
```

**5a. Download model weights:**

```bash
LOCAL_DIR=/path/to/Qwen3-Next-80B-A3B-Thinking
sbatch run_download.sh  # or run manually with LOCAL_DIR set
```

**5b. Label reasoning paragraphs** (requires an Anthropic API key):

```bash
python intervention_label.py \
    --trees_pkl game_trees_df_annotated.pkl \
    --logs_dir aws-logs/four_in_a_row-fen/ \
    --logs_pattern '*qwen*qwen3*next*80b*' \
    --model_name qwen3-next-80b-a3b-thinking \
    --output results/intervention_labels.jsonl
```

**5c. Generate edited traces:**

```bash
python intervention_edit.py \
    --labels results/intervention_labels_250.jsonl \
    --output_dir results/intervention
```

**5d. Run inference with edited prefills:**

```bash
MODEL=/path/to/Qwen3-Next-80B-A3B-Thinking sbatch run_intervention.sh
```

**Pre-computed** labels, edits, and inference results are already in `results/intervention_labels_250.jsonl` and `results/intervention/`.

## Analysis Notebooks

All figures in the paper can be reproduced by running the analysis notebooks. No fitting step is required — pre-computed results are included.

| Notebook | Figure | Description |
|----------|--------|-------------|
| `analysis_tree_search_effort.ipynb` | Fig. 1 | Tree size vs. win rate across 27 models |
| `analysis_myopic_vs_fulltree.ipynb` | Fig. 2 | Myopic vs. full-tree prediction accuracy |
| `analysis_feature_weights.ipynb` | Fig. 3 | Fitted feature weights vs. model win rate |
| `analysis_fit_result_comparison.ipynb` | Fig. S1 | NLL comparison across all four variants |
| `analysis_model_recovery.ipynb` | Fig. S2 | Model recovery results for 13 open-source models |
| `analysis_intervention.ipynb` | Fig. 4 | Causal intervention: which trace segments drive moves |

## Board Representation

The game is played on a 4×9 board. Board states are encoded in FEN notation (inspired by chess): rows separated by `/`, pieces denoted `W`/`B`, digits represent consecutive empty squares. Internally, boards are 4×9 NumPy arrays with `+1` = current player, `-1` = opponent, `0` = empty.

Features are computed over *viable windows* — contiguous 4-square windows (horizontal, vertical, or diagonal) that are not blocked by an opponent piece.

## Citation

If you use this code or data, please cite our paper (citation to be added upon publication).
