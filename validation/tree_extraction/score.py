"""Score the GPT-5 tree extractor (judge) against the 140 human-annotated trees.

Reproduces Appendix C.3:
  C.3.1  Sim_T, strict per-trace agreement, mismatch sizes, per-depth recall /
         precision and node counts.
  C.3.2  coordinate recall of the judge and of a regex coordinate parser.

Usage (from the repo root):
    python validation/tree_extraction/score.py
"""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

import trees as T

DATA = Path(__file__).resolve().parent / "data"

# Explicit coordinate forms the extraction prompt recognizes: "(r,c)", "row r col c", "m r c".
COORD_PATTERNS = [
    r"\(\s*([0-3])\s*,\s*([0-8])\s*\)",
    r"\brow\s*([0-3])\s*,?\s*col(?:umn)?\s*([0-8])\b",
    r"\bm\s*([0-3])\s+([0-8])\b",
]


def parse_coords(text):
    """Set of distinct 'r,c' coordinates written explicitly in the trace."""
    return {f"{m.group(1)},{m.group(2)}"
            for pat in COORD_PATTERNS for m in re.finditer(pat, text, re.IGNORECASE)}


def load():
    human = json.loads((DATA / "human_trees.json").read_text())
    rows = [json.loads(l) for l in (DATA / "traces.jsonl").read_text().splitlines()]
    for r in rows:
        r["human_tree"] = human[r["key"]]["trees"]
    return rows


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    T._selftest()
    rows = load()
    n = len(rows)
    models = Counter(r["model"] for r in rows)
    print(f"Sample: {n} traces, {len(models)} models x {set(models.values())} traces each\n")

    # ── C.3.1 Sim_T ──
    sims = [T.sim_t(r["human_tree"], r["judge_tree"], labeled=True) for r in rows]
    sims_s = [T.sim_t(r["human_tree"], r["judge_tree"], labeled=False) for r in rows]
    print("== C.3.1 Tree similarity (constrained unordered TED, Zhang 1996) ==")
    print(f"  Sim_T (labeled)         {np.mean(sims):.3f}")
    print(f"  Sim_T (structure only)  {np.mean(sims_s):.3f}\n")

    # ── strict per-trace agreement ──
    checks = [
        ("max ply", lambda h, e: T.max_ply(h) == T.max_ply(e)),
        ("breadth", lambda h, e: T.breadth(h) == T.breadth(e)),
        ("edges", lambda h, e: T.edge_set(h) == T.edge_set(e)),
        ("node set", lambda h, e: T.coord_set(h) == T.coord_set(e)),
        ("whole tree", lambda h, e: T.canonical(h) == T.canonical(e)),
    ]
    print(f"== Strict per-trace agreement (n={n}) ==")
    for name, fn in checks:
        k = sum(fn(r["human_tree"], r["judge_tree"]) for r in rows)
        print(f"  {name:<11} {pct(k / n):>6}  ({k}/{n})")

    # ── size of whole-tree mismatches ──
    mism = []
    for r in rows:
        d = T.ted(r["human_tree"], r["judge_tree"])
        if d > 0:
            mism.append((d, max(T.size(r["human_tree"]), T.size(r["judge_tree"])), r["key"]))
    mism.sort(reverse=True)
    print(f"\n  Mismatched traces: {len(mism)};  median edit distance = "
          f"{np.median([m[0] for m in mism]):g} node(s)")
    print("  Largest disagreements (edit distance, tree size in nodes):")
    for d, s, k in mism[:5]:
        print(f"    TED={d:<3} size={s:<3} {k}")

    # ── per-depth recall / precision on (coord, depth) nodes ──
    tp, nh, ne = Counter(), Counter(), Counter()
    for r in rows:
        H, E = T.coord_depth_set(r["human_tree"]), T.coord_depth_set(r["judge_tree"])
        for _, d in H:
            nh[d] += 1
        for _, d in E:
            ne[d] += 1
        for _, d in H & E:
            tp[d] += 1
    print("\n== Per-depth node recall / precision ((coord, depth) match; depth 1 = model's own move) ==")
    print(f"  {'depth':>5} {'human':>6} {'judge':>6} {'recall':>7} {'precision':>9}")
    for d in sorted(set(nh) | set(ne)):
        rec = tp[d] / nh[d] if nh[d] else float("nan")
        pre = tp[d] / ne[d] if ne[d] else float("nan")
        print(f"  {d:>5} {nh[d]:>6} {ne[d]:>6} {rec:>7.3f} {pre:>9.3f}")
    deep = lambda c: sum(v for d, v in c.items() if d >= 3)
    print(f"  {'3+':>5} {deep(nh):>6} {deep(ne):>6} {deep(tp) / deep(nh):>7.3f} {deep(tp) / deep(ne):>9.3f}")
    hd = np.array([T.max_ply(r["human_tree"]) for r in rows])
    ed = np.array([T.max_ply(r["judge_tree"]) for r in rows])
    print(f"  Judge tree shallower than human in {(ed < hd).sum()}/{n} traces "
          f"(deeper in {(ed > hd).sum()})")

    # ── C.3.2 coordinate recall: judge vs regex parser ──
    hc_tot = j_hit = p_hit = 0
    n_parsed, n_human = [], []
    for r in rows:
        hc = T.coord_set(r["human_tree"])
        pc = parse_coords(r["trace"])
        hc_tot += len(hc)
        j_hit += len(hc & T.coord_set(r["judge_tree"]))
        p_hit += len(hc & pc)
        n_parsed.append(len(pc))
        n_human.append(len(hc))
    print("\n== C.3.2 Recall of human-marked coordinates ==")
    print(f"  judge   {j_hit / hc_tot:.3f}  ({j_hit}/{hc_tot})")
    print(f"  parser  {p_hit / hc_tot:.3f}  ({p_hit}/{hc_tot})")
    print(f"  parser extracts {np.mean(n_parsed):.1f} distinct coordinates/trace vs "
          f"{np.mean(n_human):.1f} human-annotated moves/trace")


if __name__ == "__main__":
    main()
