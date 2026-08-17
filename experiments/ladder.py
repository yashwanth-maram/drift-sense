#!/usr/bin/env python3
"""Ablation ladder.

Runs each rung of the localisation pipeline against the SAME seeded
samples, so rungs are compared pairwise rather than across independent
draws. At n=20 an unpaired binomial standard error is about 10 points,
which cannot resolve a 5-point difference; paired comparison only counts
the samples where two rungs disagree and is far more efficient.

    python experiments/ladder.py --n 30 --levels medium severe
    python experiments/ladder.py --n 40 --rungs 0 1 5 6 --kind finfet

Reports accuracy at several tolerances, not just 5 px. A miss at 6 px is a
one-cell lattice slip; a miss at 300 px is a lost match. Both score zero
at 5 px and need completely different fixes.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from driftsense.generator.sample import generate_sample, build_params
from driftsense.localize.pipeline import localize, MAX_RUNG

TOLERANCES = (1.0, 2.0, 5.0, 10.0, 25.0)


def build_eval_set(n, kind, level, seed):
    """Deterministic sample set. Regenerated identically for every rung."""
    p = build_params(noise_level=level)
    out = []
    for i in range(n):
        s = generate_sample(i, kind, base_seed=seed, params=p)
        out.append(s)
    return out


def run_rung(samples, rung):
    errs, confs, times = [], [], []
    for s in samples:
        t0 = time.time()
        r = localize(s.reference, s.search, rung=rung)
        times.append(time.time() - t0)
        errs.append(float(np.hypot(r.x - s.gt_x, r.y - s.gt_y)))
        confs.append(r.confidence)
    return np.array(errs), np.array(confs), np.array(times)


def fmt_row(label, errs, times):
    acc = "  ".join(f"{(errs <= t).mean() * 100:5.1f}" for t in TOLERANCES)
    return (f"  {label:<8} {acc}   med {np.median(errs):6.2f}"
            f"   {times.mean() * 1000:6.0f} ms")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--kind", default="dram", choices=["dram", "finfet"])
    ap.add_argument("--levels", nargs="+",
                    default=["low", "medium", "high", "severe"])
    ap.add_argument("--rungs", nargs="+", type=int,
                    default=list(range(MAX_RUNG + 1)))
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args(argv)

    print("=" * 72)
    print(f"ABLATION LADDER   kind={a.kind}  n={a.n}  seed={a.seed}")
    print("=" * 72)

    store = {}
    for level in a.levels:
        print(f"\n  building {a.n} samples at '{level}' ...", flush=True)
        samples = build_eval_set(a.n, a.kind, level, a.seed)

        head = "  ".join(f"{t:>5.0f}" for t in TOLERANCES)
        print(f"\n  {level.upper()}")
        print(f"  {'rung':<8} {head}   {'median':>10}   {'time':>9}")
        print("  " + "-" * 60)

        for rung in a.rungs:
            errs, confs, times = run_rung(samples, rung)
            store[(level, rung)] = errs
            print(fmt_row(str(rung), errs, times), flush=True)

    # paired deltas between consecutive rungs
    print("\n" + "=" * 72)
    print("PAIRED DELTAS at 5 px  (rung N vs rung N-1, same samples)")
    print("=" * 72)
    for level in a.levels:
        parts = []
        for j in range(1, len(a.rungs)):
            lo, hi = a.rungs[j - 1], a.rungs[j]
            if (level, lo) not in store or (level, hi) not in store:
                continue
            e0, e1 = store[(level, lo)], store[(level, hi)]
            gained = int(((e0 > 5) & (e1 <= 5)).sum())
            lost = int(((e0 <= 5) & (e1 > 5)).sum())
            net = (gained - lost) / len(e0) * 100
            parts.append(f"{lo}->{hi}: {net:+5.1f}pp (+{gained}/-{lost})")
        print(f"  {level:<8} " + "   ".join(parts))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
