# Extracting Search Trees from LLM Reasoning Traces Reveals Myopic Planning

Code and data for the NeurIPS 2026 paper by Sixing Chen, Ji-An Li, Saner Cakir, Sinan Akcali, Kayla Lee and
Marcelo G. Mattar.

LLMs played a round-robin four-in-a-row tournament (27 models, 1404 games). We extract the search tree from each
reasoning trace with an LLM judge, fit cognitive models that predict each move from its tree, and prune reasoning
traces to test which parts of the search causally drive the move.

## Quick start

```bash
uv sync                    # Python 3.11 environment with the `fiar` package and all dependencies
brew install graphviz      # or see https://graphviz.org/download/ (needed for Figure 1D)
jupyter nbconvert --to notebook --execute paper_figures.ipynb   # or open it and run all cells
```

`paper_figures.ipynb` writes every data panel to `figures/` and prints the statistics quoted in the text. It needs
only what is already in this repository: the per-turn dataframe `data/game_trees.pkl` and the precomputed results
in `results/`.

All commands in this README are run from the repository root (prefix them with `uv run` or activate `.venv`).

## Repository layout

```
paper_figures.ipynb     all figures and in-text statistics
data/game_trees.pkl     one row per turn, with the extracted search tree
fiar/                   Python package
  board.py              FEN and search-tree parsing
  features.py           heuristic value functions (pattern and window-count features)
  minimax.py            value backup over a tree (full-tree, myopic, discount)
  fit.py                datasets, likelihood, fitting, model variants
  models.py             model names
  render.py             search-tree rendering (Figure 1D)
pipeline/
  extract_search_trees.py   GPT-5 tree extraction and dataframe assembly (Section 2)
  prompts/                  the extraction prompt
  fit_cognitive_models.py   fit one model variant to one LLM (Section 4)
  model_recovery.py         model recovery (appendix)
  slurm/                    array-job launchers for the two scripts above
intervention/           causal intervention (Section 5)
  label.py              label each paragraph of a reasoning trace with Claude Opus 4.7
  edit.py               build the edited (pruned) traces
  infer_local.py        rerun the model on the edited traces, loaded locally with vLLM
  infer_vllm.py         the same against a vLLM server
  run_local.sbatch      launcher for infer_local.py
  run_vllm.sh           launcher for infer_vllm.py
results/
  fits/pattern/         <model>_<variant>.json for variants full, myopic, notree, gamma
  fits/windowcount/     the same with the window-count features (appendix)
  recovery/             <model>.json (appendix)
  intervention/<model>/ labels.jsonl, edits_<strategy>.jsonl, results_<strategy>.csv
```

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

## Rerunning the analyses

The three main analyses can be rerun from the raw logs. The appendix analyses (model recovery, window-count
features) are not covered here; their usage is in the docstrings of the scripts in `pipeline/`.

**Search-tree extraction (Section 2).** Needs `OPENAI_API_KEY`. The judge samples at temperature 1, so a rerun
gives similar but not identical trees.

```bash
python pipeline/extract_search_trees.py submit
python pipeline/extract_search_trees.py fetch     # repeat until all responses are completed
python pipeline/extract_search_trees.py build     # -> data/game_trees_rebuilt.pkl
```

**Cognitive models (Section 4).** Variants are `full`, `myopic`, `notree` and `gamma` (discount).

```bash
python pipeline/fit_cognitive_models.py --model deepseek-r1 --variant myopic   # one LLM
VARIANT=myopic sbatch pipeline/slurm/fit.sbatch                                # all 13 fitted LLMs
```

**Causal intervention (Section 5).** Run on Qwen3-Next-80B-A3B-Thinking, Qwen3-235B-A22B-Thinking-2507 and
Kimi-K2-Thinking. Each step reads and writes `results/intervention/<model>/`; step 3 needs `uv sync --extra gpu`.

```bash
# 1. Label paragraphs with Claude Opus 4.7 (needs ANTHROPIC_API_KEY)
python intervention/label.py --model_name qwen3-next-80b-a3b-thinking
# 2. Build the edited traces
python intervention/edit.py --model qwen3-next-80b-a3b-thinking
# 3. Rerun the model on the edited traces
MODEL_DIR=/path/to/Qwen3-Next-80B-A3B-Thinking sbatch intervention/run_local.sbatch
```

The two larger models are served with vLLM instead; see the header of `intervention/run_vllm.sh`.

## Citation

```bibtex
@inproceedings{chen2026extracting,
  title     = {Extracting Search Trees from {LLM} Reasoning Traces Reveals Myopic Planning},
  author    = {Chen, Sixing and Li, Ji-An and Cakir, Saner and Akcali, Sinan and Lee, Kayla and Mattar, Marcelo G.},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
