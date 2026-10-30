# Extracting Search Trees from LLM Reasoning Traces Reveals Myopic Planning

Code and data for the NeurIPS 2026 paper by Sixing Chen, Ji-An Li, Saner Cakir, Sinan Akcali, Kayla Lee and
Marcelo G. Mattar.

LLMs played a round-robin four-in-a-row tournament (27 models, 1404 games). We extract the search tree from each
reasoning trace with an LLM judge, fit cognitive models that predict each move from its tree, and prune reasoning
traces to test which parts of the search causally drive the move.

## Setup

```bash
uv sync                    # Python 3.11 environment with the `fiar` package and all dependencies
brew install graphviz      # or see https://graphviz.org/download/ (needed for Figure 1D)
```

All commands below are run from the repository root (prefix them with `uv run` or activate `.venv`).

## Reproducing the figures

Everything needed for the figures is in this repository: the per-turn dataframe `data/game_trees.pkl` and the
precomputed results in `results/`. Run

```bash
jupyter nbconvert --to notebook --execute paper_figures.ipynb   # or open it and run all cells
```

`paper_figures.ipynb` writes every data panel to `figures/` and prints the statistics quoted in the text.

## Data

`data/game_trees.pkl` has one row per turn of the 27 models (18,888 turns). Its main columns are:

| Column | Content |
|---|---|
| `game_path`, `turn_idx` | Game log (relative to `data/logs/four_in_a_row-fen/`) and event index |
| `model_names` | Model, as named in the paper |
| `board_state` | FEN and player to move, as shown to the model (missing on the empty opening board) |
| `trees` | Extracted search tree: a list of depth-1 nodes `['r,c', child, ...]` |
| `trees_annotated` | Same tree with illegal moves and terminal (win/loss/tie) nodes tagged |
| `extracted_move`, `move_success` | The move played and whether it was legal |
| `max_ply`, `n_root_moves`, `total_nodes` | Depth, breadth and size of the tree |
| `outcome_perspective` | Game result for this model (1 win, 0.5 draw, 0 loss) |

The raw game logs, including the full reasoning traces (about 2.7 GB), are on Hugging Face at
[SixingChen01/fiar-reasoning](https://huggingface.co/datasets/SixingChen01/fiar-reasoning). They are needed only
to rerun the extraction or the intervention:

```bash
hf download SixingChen01/fiar-reasoning --repo-type dataset --local-dir data/logs --include "four_in_a_row-fen/*"
```

Each game folder `four_in_a_row-fen/<model-a>_vs_<model-b>/<game-id>/` holds `game_log.json` (events, reasoning
traces and outcome), `game.log`, and one `<provider>-<model>(<player>)-log.jsonl` per player with the raw API
requests and responses. The dataset also contains `four_in_a_row-standard/`, a tournament with a different board
notation that is not analysed in the paper.

## Repository layout

```
fiar/                   Python package
  board.py              FEN and search-tree parsing
  features.py           heuristic value functions: pattern features (main text) and window-count features (D.4)
  minimax.py            value backup over a tree (full-tree, myopic, discount)
  fit.py                datasets, likelihood, fitting, model variants
  models.py             model names
  render.py             search-tree rendering (Figure 1D)
pipeline/
  extract_search_trees.py   GPT-5 tree extraction and dataframe assembly (Section 2, C.2)
  prompts/                  the extraction prompt (Appendix C.2)
  fit_cognitive_models.py   fit one model variant to one LLM (Section 4, D.4)
  model_recovery.py         model recovery (D.2)
  slurm/                    array-job launchers for the two scripts above
intervention/           causal intervention (Section 5, C.5): label.py, edit.py, infer_local.py, infer_vllm.py
validation/
  tree_extraction/      judge vs human search trees (C.3)
  paragraph_labels/     paragraph labeler vs human labels (C.5.2)
results/
  fits/pattern/         <model>_<variant>.json for variants full, myopic, notree, gamma
  fits/windowcount/     the same with the window-count features (D.4)
  recovery/             <model>.json
  intervention/<model>/ labels.jsonl, edits_<strategy>.jsonl, results_<strategy>.csv
paper_figures.ipynb     all figures and in-text statistics
```

## Rerunning the analyses

### Search-tree extraction (Section 2.2)

```bash
python pipeline/extract_search_trees.py submit    # needs OPENAI_API_KEY
python pipeline/extract_search_trees.py fetch     # repeat until all responses are completed
python pipeline/extract_search_trees.py build     # -> data/game_trees_rebuilt.pkl
```

The judge samples at temperature 1, so a rerun gives similar but not identical trees. Applied to the original judge
outputs, `build` reproduces every column of `data/game_trees.pkl` that the analyses use (`trees_annotated` is not
rebuilt).

### Cognitive models (Section 4, Appendix D.4)

```bash
python pipeline/fit_cognitive_models.py --model deepseek-r1 --variant myopic
VARIANT=myopic sbatch pipeline/slurm/fit.sbatch          # all 13 fitted models
VARIANT=gamma BASIS=windowcount sbatch pipeline/slurm/fit.sbatch
```

Variants are `full`, `myopic`, `notree` and `gamma` (discount). Each fit uses L-BFGS-B with 20 random starts. With
the window-count basis, C is fixed to 1 and the discount fit is also started from the myopic solution, so run the
`myopic` fits first.

### Model recovery (Appendix D.2)

```bash
sbatch pipeline/slurm/recovery.sbatch     # needs results/fits/pattern/<model>_{full,myopic}.json
```

### Causal intervention (Section 5)

The intervention was run on Qwen3-Next-80B-A3B-Thinking, Qwen3-235B-A22B-Thinking-2507 and Kimi-K2-Thinking, on
the first 250 eligible turns of each. Each step reads and writes `results/intervention/<model>/`.

```bash
# 1. Label paragraphs with Claude Opus 4.7 (needs ANTHROPIC_API_KEY)
python intervention/label.py --model_name qwen3-next-80b-a3b-thinking
# 2. Build the seven edited traces
python intervention/edit.py --model qwen3-next-80b-a3b-thinking
# 3. Rerun the model on the edited traces
MODEL_DIR=/path/to/Qwen3-Next-80B-A3B-Thinking sbatch intervention/run_local.sbatch

# Qwen3-235B and Kimi-K2 are served with vLLM on 8 GPUs, e.g.
vllm serve moonshotai/Kimi-K2-Thinking --tensor-parallel-size 8 --max-model-len 16384 --trust-remote-code
MODEL=moonshotai/Kimi-K2-Thinking DIR=results/intervention/kimi-k2-thinking bash intervention/run_vllm.sh
```

Step 3 needs `uv sync --extra gpu`. The strategies are `fd` (remove the final decision), `fd_branch_comp` (also
remove the chosen branch), `bc_minus_d0only` (add back its depth-1 paragraphs) and `bc_minus_d0_and_deep1plus`
(also add back depth-2 paragraphs). Each `_ctrl` variant applies the same edit to the largest unchosen branch.

### Validation of the LLM judges (Appendix C.3, C.5.2)

```bash
python validation/tree_extraction/score.py
python validation/paragraph_labels/score.py
```

## Citation

```bibtex
@inproceedings{chen2026extracting,
  title     = {Extracting Search Trees from {LLM} Reasoning Traces Reveals Myopic Planning},
  author    = {Chen, Sixing and Li, Ji-An and Cakir, Saner and Akcali, Sinan and Lee, Kayla and Mattar, Marcelo G.},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
