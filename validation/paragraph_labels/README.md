# Validation of the paragraph labeler (Appendix C.5.2)

The causal intervention labels every paragraph of a reasoning trace with Claude Opus 4.7
(`intervention/label.py`): a paragraph `type` (PREAMBLE, BRANCH_START, BRANCH_ANALYSIS,
BRANCH_CONCLUSION, COMPARISON, FINAL_DECISION, META), the depth-1 candidate move it
belongs to (`branch_root`), and the moves it simulates (`mentions`, each a coordinate and
a depth). This folder compares those labels with human labels on 150 paragraphs (50 per
model), drawn at random from paragraphs that contain an explicit coordinate, within the
labeled turns used by the intervention.

```bash
python validation/paragraph_labels/score.py
```

Prints agreement on final-decision paragraphs and on simulated-move mentions and their
depth. It also checks that the stored judge labels are identical to
`results/intervention/<model>/labels.jsonl`.

## Files

- `score.py` — computes the reported numbers.
- `data/annotations.jsonl` — one record per paragraph: `model`, `turn`
  (`turn_idx_global` in the label files), `para` (paragraph index), `text`, the judge's
  label `judge`, the human label `human`, and the annotator's `notes`.

Depths in the data are 0-based as in `intervention/label.py` (0 = model's own move);
`score.py` reports them 1-based, as in the paper. The annotator labeled each paragraph
with the judge's label visible.
