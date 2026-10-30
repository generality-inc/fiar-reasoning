# Validation of the tree extractor (Appendix C.3)

Compares the search trees extracted by the GPT-5 judge (`extract_search_trees.ipynb`)
with trees transcribed by a human annotator, on 140 reasoning traces: 10 randomly drawn
from each of the 14 trace-accessible models, from the traces analyzed in the paper.

```bash
python validation/tree_extraction/score.py
```

Prints the C.3.1 numbers (Sim_T, strict per-trace agreement on max ply / breadth / edges /
node set / whole tree, size of the mismatches, per-depth recall and precision, node counts
by depth) and the C.3.2 cross-check against a regex coordinate parser.

## Files

- `score.py` — computes every reported number.
- `trees.py` — tree helpers and the constrained unordered tree edit distance (Zhang, 1996)
  behind Sim_T = 1 − TED / max(|V₁|, |V₂|).
- `data/traces.jsonl` — the 140 traces: `key`, `model`, `player`, `game_path` (under
  `data/logs/four_in_a_row-fen/`), board `fen`, `final_move`, the judge's tree
  `judge_tree` (identical to `trees` in `data/game_trees.pkl`), and the `trace` text
  (reasoning followed by the answer).
- `data/human_trees.json` — the human tree for each `key`.

Trees are nested lists, `[coord, child, child, ...]`, with coordinates written `"r,c"`.
A top-level node is depth 1 (the model's own move), its children depth 2 (opponent
replies), and so on. Per-depth counts are distinct `(coord, depth)` pairs per trace,
summed over traces.
