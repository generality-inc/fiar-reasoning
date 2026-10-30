"""Search-tree helpers and the order-invariant tree edit distance used for Sim_T.

Tree format (judge and human): a forest of nested lists,
    node = [coord, child_node, child_node, ...],  leaf = [coord],  tree = [node, ...]
where coord is "r,c". Top-level nodes are depth 1 (the model's own candidate
moves), their children depth 2 (opponent replies), and so on.

The distance is the constrained unordered tree edit distance of Zhang (1996):
siblings are unordered and disjoint subtrees map to disjoint subtrees. Insert,
delete and relabel each cost 1. Sim_T = clamp(1 - TED / max(|V1|, |V2|), 0, 1).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def _valid(node):
    return isinstance(node, list) and node and isinstance(node[0], str)


def children(node):
    return [c for c in node[1:] if _valid(c)]


def walk(tree, depth=1):
    """Yield (coord, depth) for every node in the forest."""
    for node in tree:
        if not _valid(node):
            continue
        yield node[0], depth
        for ch in children(node):
            yield from walk([ch], depth + 1)


def size(tree):
    return sum(1 for _ in walk(tree))


def coord_set(tree):
    return {c for c, _ in walk(tree)}


def coord_depth_set(tree):
    return set(walk(tree))


def edge_set(tree):
    edges = set()

    def rec(node):
        for ch in children(node):
            edges.add((node[0], ch[0]))
            rec(ch)

    for node in tree:
        if _valid(node):
            rec(node)
    return edges


def breadth(tree):
    """Number of depth-1 candidate moves."""
    return sum(1 for n in tree if _valid(n))


def max_ply(tree):
    return max((d for _, d in walk(tree)), default=0)


def canonical(tree):
    """Order-invariant canonical form (siblings sorted recursively)."""
    def key(node):
        return (node[0], tuple(sorted(key(c) for c in children(node))))
    return tuple(sorted(key(n) for n in tree if _valid(n)))


# ── constrained unordered tree edit distance (Zhang 1996) ─────────────────────
def _to_forest(tree):
    def conv(node):
        return (node[0], tuple(conv(c) for c in children(node)))
    return tuple(conv(n) for n in tree if _valid(n))


def _sz(t):
    return 1 + sum(_sz(c) for c in t[1])


def ted(tree1, tree2, labeled=True):
    """Constrained unordered TED between two forests (relabel cost 1 if labeled, else 0)."""
    BIG = 10 ** 7
    tmemo, fmemo = {}, {}

    def d_tree(a, b):
        if (a, b) in tmemo:
            return tmemo[a, b]
        rel = 0 if (not labeled or a[0] == b[0]) else 1
        best = rel + d_forest(a[1], b[1])
        if a[1]:   # delete root of a, keep one child subtree
            tot = sum(_sz(x) for x in a[1])
            best = min(best, 1 + min(d_tree(x, b) + tot - _sz(x) for x in a[1]))
        else:
            best = min(best, 1 + _sz(b))
        if b[1]:   # insert root of b
            tot = sum(_sz(y) for y in b[1])
            best = min(best, 1 + min(d_tree(a, y) + tot - _sz(y) for y in b[1]))
        else:
            best = min(best, 1 + _sz(a))
        tmemo[a, b] = best
        return best

    def d_forest(A, B):
        if (A, B) in fmemo:
            return fmemo[A, B]
        if not A or not B:
            res = sum(_sz(x) for x in A) + sum(_sz(y) for y in B)
        else:
            m, n = len(A), len(B)
            C = np.full((m + n, m + n), BIG, dtype=float)
            for i, a in enumerate(A):
                for j, b in enumerate(B):
                    C[i, j] = d_tree(a, b)
                C[i, n + i] = _sz(a)
            for j, b in enumerate(B):
                C[m + j, j] = _sz(b)
                C[m + j, n:] = 0.0
            r, c = linear_sum_assignment(C)
            res = int(C[r, c].sum())
        fmemo[A, B] = res
        return res

    return d_forest(_to_forest(tree1), _to_forest(tree2))


def sim_t(tree1, tree2, labeled=True):
    denom = max(size(tree1), size(tree2))
    if denom == 0:
        return 1.0
    return min(1.0, max(0.0, 1.0 - ted(tree1, tree2, labeled) / denom))


def _selftest():
    cases = [
        ([["0,4"], ["1,3", ["2,5"]]], [["0,4"], ["1,3", ["2,5"]]], True, 0),
        ([["a"], ["b"]], [["b"], ["a"]], True, 0),          # sibling order ignored
        ([["a"]], [["b"]], True, 1),
        ([["a"]], [["b"]], False, 0),
        ([["a"], ["b"]], [["a"]], True, 1),
        ([["0,4"], ["1,3", ["2,5"]]], [["3,7"], ["1,3", ["0,8"]]], True, 2),
    ]
    for t1, t2, lab, exp in cases:
        assert ted(t1, t2, lab) == exp, (t1, t2, lab)
