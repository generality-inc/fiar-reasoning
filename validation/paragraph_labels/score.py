"""Score the intervention's paragraph labeler (Claude Opus 4.7) against human labels.

Reproduces Appendix C.5.2 on the 150 annotated paragraphs in data/annotations.jsonl:
final-decision agreement, and agreement on the simulated-move mentions and their depth.
Depths are stored 0-based as in intervention/label.py (0 = model's own move) and are
reported here 1-based as in the paper (1 = model's own move).

Usage (from the repo root):
    python validation/paragraph_labels/score.py

It first checks that the judge labels stored here are identical to the labels the
intervention used (results/intervention/<model>/labels.jsonl), where available.
"""
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FD = "FINAL_DECISION"


def load():
    return [json.loads(l) for l in (HERE / "data" / "annotations.jsonl").read_text().splitlines()]


def check_labels(rows):
    """Compare stored judge labels with results/intervention/<model>/labels.jsonl."""
    for model in sorted({r["model"] for r in rows}):
        path = REPO / "results" / "intervention" / model / "labels.jsonl"
        sub = [r for r in rows if r["model"] == model]
        if not path.exists():
            print(f"  {model:<28} {len(sub)} paragraphs: no {path.relative_to(REPO)} (not checked)")
            continue
        recs = {}
        for line in path.open():
            rec = json.loads(line)
            recs[rec["turn_idx_global"]] = rec
        ok = 0
        for r in sub:
            rec = recs.get(r["turn"])
            if rec is None:
                continue
            lab = {l["para"]: l for l in rec["labels"]}.get(r["para"])
            p = rec["paragraphs"][r["para"]]
            text = p["text"] if isinstance(p, dict) else p
            ok += bool(lab) and text == r["text"] and r["judge"] == {
                "type": lab["type"], "branch_root": lab.get("branch_root"),
                "mentions": lab.get("mentions")}
        print(f"  {model:<28} {ok}/{len(sub)} paragraphs identical to {path.relative_to(REPO)}")


def match(src, ref):
    """Mentions in `src` whose coord is also in `ref` (same paragraph); and how many share a depth."""
    n = found = same_depth = 0
    for s, r in zip(src, ref):
        ref_depths = {}
        for m in r or []:
            ref_depths.setdefault(m["coord"], set()).add(m["depth"])
        for m in s or []:
            n += 1
            if m["coord"] in ref_depths:
                found += 1
                same_depth += m["depth"] in ref_depths[m["coord"]]
    return n, found, same_depth


def main():
    rows = load()
    print(f"Sample: {len(rows)} paragraphs  {dict(Counter(r['model'] for r in rows))}")
    check_labels(rows)

    H = [r["human"] for r in rows]
    J = [r["judge"] for r in rows]
    n = len(rows)

    fd = sum((h["type"] == FD) == (j["type"] == FD) for h, j in zip(H, J))
    print("\n== Final-decision paragraphs ==")
    print(f"  agreement (FINAL_DECISION vs not): {fd}/{n}   "
          f"[human {sum(h['type'] == FD for h in H)}, judge {sum(j['type'] == FD for j in J)} FD paragraphs]")
    print(f"  (all 7 types agree: {sum(h['type'] == j['type'] for h, j in zip(H, J))}/{n}; "
          f"branch_root agrees: {sum(h['branch_root'] == j['branch_root'] for h, j in zip(H, J))}/{n})")

    hm = [h["mentions"] for h in H]
    jm = [j["mentions"] for j in J]
    nj, fj, dj = match(jm, hm)
    nh, fh, dh = match(hm, jm)
    print("\n== Simulated-move mentions (coord match within paragraph) ==")
    print(f"  judge mentions confirmed by human:  {fj}/{nj} = {fj / nj:.3f};  same depth {dj}/{fj} = {dj / fj:.3f}")
    print(f"  human mentions recovered by judge:  {fh}/{nh} = {fh / nh:.3f};  same depth {dh}/{fh} = {dh / fh:.3f}")

    conf = Counter()
    for h, j in zip(hm, jm):
        jd = {m["coord"]: m["depth"] for m in j or []}
        for m in h or []:
            if m["coord"] in jd:
                conf[m["depth"] + 1, jd[m["coord"]] + 1] += 1
    print("  depth (human -> judge, paper convention), matched mentions: "
          + ", ".join(f"{a}->{b}: {c}" for (a, b), c in sorted(conf.items())))


if __name__ == "__main__":
    main()
