#!/usr/bin/env python3
"""Every metric the submission needs, measured from the shipped localize.py.

Timing is taken from the production inference path, not from an
experiments script, because the rubric counts computation time as part of
the 50% inference score.

Reports:

    accuracy at 1, 2, 5 and 10 px      the 5 px figure is the headline;
                                       the others separate a sub-pixel hit
                                       from a one-cell lattice slip
    median, mean and p95 error
    precision, recall and AP           by sweeping the confidence as an
                                       acceptance threshold, which is what
                                       the reference harness does
    computation time                   mean, median, p99
    accuracy by template condition     both axes strip-pinned, one, neither
    error histogram                    expected bimodal: a tight cluster
                                       near zero plus discrete spikes at
                                       lattice multiples

    python experiments/metrics.py --data ../am_eval
    python experiments/metrics.py --data ../am_eval --baseline
    python experiments/metrics.py --generate --levels low medium high severe extreme
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from localize import localize, SCALE
from driftsense.localize import _strip_centres

TOLERANCES = (1.0, 2.0, 5.0, 10.0)
T = 100


def plain_baseline(ref, srch):
    """Ordinary template matching, for the comparison the deck needs."""
    t0 = time.time()
    size = max(int(round(min(srch.shape) / SCALE)), 8)
    t = cv2.resize(ref, (size, size), interpolation=cv2.INTER_AREA)
    s = cv2.matchTemplate(srch, t, cv2.TM_CCOEFF_NORMED)
    k = int(np.argmax(s))
    y, x = np.unravel_index(k, s.shape)
    return (x + size / 2.0, y + size / 2.0, float(s[y, x]),
            (time.time() - t0) * 1000.0)


def condition(ref):
    """How many axes the reference pins via a peripheral strip.

    Validated as a failure predictor: 92.6% accuracy when both axes are
    pinned, 8.3% when neither is, known before matching runs.
    """
    t = cv2.resize(ref, (T, T), interpolation=cv2.INTER_AREA)
    return (len(_strip_centres(t, 1, 0.45, 6)) > 0) \
        + (len(_strip_centres(t, 0, 0.45, 6)) > 0)


def average_precision(conf, correct):
    """AP from a confidence sweep.

    Every pair has exactly one true match to find, so the positive count is
    N at every threshold: recall = TP/N, precision = TP/(TP+FP) among
    accepted predictions.
    """
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, bool)
    n = len(conf)
    if n == 0:
        return 0.0, [], []
    order = np.argsort(-conf)
    c = correct[order]
    tp = np.cumsum(c)
    fp = np.cumsum(~c)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - prev_r)
        prev_r = r
    return float(ap), precision.tolist(), recall.tolist()


def load(data, n):
    for root, _, files in os.walk(data):
        if "manifest.csv" in files:
            rows = list(csv.DictReader(open(os.path.join(root, "manifest.csv"))))
            out = []
            for r in rows[:n]:
                i = int(r["id"])
                ref = cv2.imread(os.path.join(root, "reference", f"{i:05d}.png"),
                                 cv2.IMREAD_GRAYSCALE)
                s = cv2.imread(os.path.join(root, "search", f"{i:05d}.png"),
                               cv2.IMREAD_GRAYSCALE)
                if ref is not None and s is not None:
                    out.append((i, ref, s, float(r["gt_x"]), float(r["gt_y"])))
            return out
    raise SystemExit(f"no manifest.csv under {data}")


def generate(n, kind, seed, level):
    from driftsense.generator.sample import generate_sample, build_params
    p = build_params(noise_level=level)
    return [(i,) + (lambda s: (s.reference, s.search, s.gt_x, s.gt_y))(
        generate_sample(i, kind, base_seed=seed, params=p))
        for i in range(n)]


def evaluate(pairs, label, tol=5.0, baseline=False):
    errs, confs, times, conds = [], [], [], []
    berrs, btimes = [], []

    for _, ref, srch, gx, gy in pairs:
        r = localize(ref, srch)
        errs.append(float(np.hypot(r.x - gx, r.y - gy)))
        confs.append(r.confidence)
        times.append(r.ms)
        conds.append(condition(ref))
        if baseline:
            bx, by, _, bms = plain_baseline(ref, srch)
            berrs.append(float(np.hypot(bx - gx, by - gy)))
            btimes.append(bms)

    e = np.asarray(errs)
    c = np.asarray(confs)
    t = np.asarray(times)
    cond = np.asarray(conds)
    n = len(e)

    print("\n" + "=" * 72)
    print(f"  {label}   n = {n}")
    print("=" * 72)

    print("\n  ACCURACY")
    for tl in TOLERANCES:
        star = "   <- headline" if tl == 5.0 else ""
        print(f"    within {tl:>4.0f} px : {float((e <= tl).mean()) * 100:6.1f}%{star}")
    if baseline:
        be = np.asarray(berrs)
        print(f"\n    plain template matching, 5 px :"
              f" {float((be <= tol).mean()) * 100:.1f}%")
        print(f"    improvement                   :"
              f" {(float((e <= tol).mean()) - float((be <= tol).mean())) * 100:+.1f} pp")

    print("\n  ERROR")
    print(f"    median : {np.median(e):7.2f} px")
    print(f"    mean   : {e.mean():7.2f} px")
    print(f"    p95    : {np.percentile(e, 95):7.2f} px")

    print("\n  COMPUTATION TIME   (from localize.py)")
    print(f"    mean   : {t.mean():7.0f} ms/pair")
    print(f"    median : {np.median(t):7.0f} ms")
    print(f"    p99    : {np.percentile(t, 99):7.0f} ms")
    if baseline:
        print(f"    plain  : {np.mean(btimes):7.0f} ms"
              f"   ({t.mean() / max(np.mean(btimes), 1e-9):.0f}x)")

    ap, _, _ = average_precision(c, e <= tol)
    print("\n  CONFIDENCE")
    print(f"    average precision : {ap:.4f}")
    print(f"    median on hits    : {np.median(c[e <= tol]):.4f}"
          if (e <= tol).any() else "")
    print(f"    median on misses  : {np.median(c[e > tol]):.4f}"
          if (e > tol).any() else "    no misses")
    for q in (0.5, 0.8):
        k = int(n * q)
        if k >= 5:
            keep = np.argsort(-c)[:k]
            print(f"    top {q * 100:.0f}% by confidence : "
                  f"{float((e[keep] <= tol).mean()) * 100:.1f}% accurate")

    print("\n  BY TEMPLATE CONDITION")
    for cv_, nm in ((2, "both axes pinned"), (1, "one axis pinned"),
                    (0, "neither pinned")):
        m = cond == cv_
        if m.sum():
            print(f"    {nm:<20} n={int(m.sum()):>4}"
                  f"   {float((e[m] <= tol).mean()) * 100:6.1f}%")

    print("\n  ERROR DISTRIBUTION")
    edges = [0, 1, 2, 5, 10, 20, 50, 100, 1e9]
    names = ["<1", "1-2", "2-5", "5-10", "10-20", "20-50", "50-100", ">100"]
    for lo, hi, nm in zip(edges[:-1], edges[1:], names):
        k = int(((e >= lo) & (e < hi)).sum())
        if k:
            print(f"    {nm:>8} px : {k:>4}  {'#' * min(int(k * 40 / n), 40)}")

    return {"label": label, "n": n,
            "accuracy": {str(tl): float((e <= tl).mean()) for tl in TOLERANCES},
            "median_error": float(np.median(e)),
            "mean_ms": float(t.mean()), "ap": ap,
            "by_condition": {str(k): float((e[cond == k] <= tol).mean())
                             for k in (0, 1, 2) if (cond == k).any()}}


def main(argv=None):
    ap_ = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--data")
    ap_.add_argument("--generate", action="store_true")
    ap_.add_argument("--n", type=int, default=100)
    ap_.add_argument("--kind", default="dram", choices=["dram", "finfet"])
    ap_.add_argument("--seed", type=int, default=3131)
    ap_.add_argument("--levels", nargs="+", default=["medium"])
    ap_.add_argument("--tol", type=float, default=5.0)
    ap_.add_argument("--baseline", action="store_true",
                     help="also measure plain template matching")
    ap_.add_argument("--json", help="write the summary to this path")
    a = ap_.parse_args(argv)

    print("=" * 72)
    print("DRIFT-SENSE METRICS")
    print("=" * 72)
    print("\n  All figures from the shipped localize.py, including timing.")

    results = []
    if a.data:
        results.append(evaluate(load(a.data, a.n),
                                f"{a.data}", a.tol, a.baseline))
    if a.generate or not a.data:
        for lv in a.levels:
            results.append(evaluate(generate(a.n, a.kind, a.seed, lv),
                                    f"generated {a.kind}, {lv}",
                                    a.tol, a.baseline))

    if len(results) > 1:
        print("\n" + "=" * 72)
        print("  SUMMARY")
        print("=" * 72)
        print(f"  {'set':<32}{'@5px':>8}{'median':>9}{'ms':>8}{'AP':>8}")
        for r in results:
            print(f"  {r['label'][:31]:<32}{r['accuracy']['5.0'] * 100:>7.1f}%"
                  f"{r['median_error']:>9.2f}{r['mean_ms']:>8.0f}{r['ap']:>8.3f}")

    if a.json:
        with open(a.json, "w") as fh:
            json.dump(results, fh, indent=1)
        print(f"\n  written to {a.json}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
