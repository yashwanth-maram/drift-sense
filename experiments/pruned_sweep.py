#!/usr/bin/env python3
"""Multi-resampling matched filter: is it real, and can it be made cheap?

The exhaustive offset sweep reaches 90.0% on DRAM at n=60, against 66.7%
plain. But the diagnostic showed the selected offset matches the physical
one in 3 of 60, every correlation with a physical parameter is near zero,
and the offset is FURTHER from the physical value when the sweep succeeds
(3.0) than when it fails (2.0).

So the sweep is not estimating a transformation. The remaining explanation
is the max itself: the true position has an exact match available at SOME
offset, while an impostor is approximate at EVERY offset, so maximising
over a resampling family favours the truth systematically. Under that
reading the selected offset is a byproduct with no meaning, which is
exactly what was measured.

Two things follow, and this script tests both.

    MECHANISM   compare max-over-offsets at the true position against
                max-over-offsets at the best impostor. If the gap widens
                relative to plain ZNCC, the explanation holds.

    COST        the sweep costs about 2 s per pair because it correlates
                the whole image once per offset. If the mechanism is the
                max rather than the choice, positions can be pruned first:
                plain ZNCC, keep the top peaks, then sweep offsets only
                there. Same operator, a fraction of the work.

Works on generated samples or on a manifest directory, so the same code
benchmarks our generator and Applied Materials'.

    python experiments/pruned_sweep.py --n 60 --seed 3131
    python experiments/pruned_sweep.py --data ../am_eval --n 100
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from driftsense.subpixel_template import template_at_offset
from driftsense.localize import _strip_centres

T = 100
SCALE = 10
TOL = 5.0


def plain_template(ref):
    return cv2.resize(ref, (T, T), interpolation=cv2.INTER_AREA)


def top_peaks(surface, k=30, sep=6):
    r = max(int(sep), 2)
    dil = cv2.dilate(surface, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
    ys, xs = np.nonzero(surface >= dil - 1e-6)
    if len(ys) == 0:
        i = int(np.argmax(surface))
        y, x = np.unravel_index(i, surface.shape)
        return [(int(y), int(x))]
    v = surface[ys, xs]
    o = np.argsort(v)[::-1][:k]
    return [(int(ys[i]), int(xs[i])) for i in o]


def zncc_patch(a, b):
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return -1.0
    return float((a * b).sum() / (na * nb))


def run_plain(ref, srch):
    t = plain_template(ref)
    s = cv2.matchTemplate(srch, t, cv2.TM_CCOEFF_NORMED)
    k = int(np.argmax(s))
    y, x = np.unravel_index(k, s.shape)
    return x + T / 2.0, y + T / 2.0, s


def run_full_sweep(ref, srch, step=1):
    best = None
    for py in range(0, SCALE, step):
        for px in range(0, SCALE, step):
            t = template_at_offset(ref, float(px), float(py), T, SCALE)
            s = cv2.matchTemplate(srch, t, cv2.TM_CCOEFF_NORMED)
            v = float(s.max())
            if best is None or v > best[0]:
                k = int(np.argmax(s))
                y, x = np.unravel_index(k, s.shape)
                best = (v, x + T / 2.0, y + T / 2.0)
    return best[1], best[2]


def run_pruned(ref, srch, k=30, step=1, surface=None):
    """Prune positions with plain ZNCC, then take the max over offsets
    only at the survivors."""
    if surface is None:
        _, _, surface = run_plain(ref, srch)
    peaks = top_peaks(surface, k)
    templates = [template_at_offset(ref, float(px), float(py), T, SCALE)
                 for py in range(0, SCALE, step)
                 for px in range(0, SCALE, step)]
    best = (-2.0, 0.0, 0.0)
    for (py, px) in peaks:
        patch = srch[py:py + T, px:px + T]
        if patch.shape != (T, T):
            continue
        m = max(zncc_patch(t, patch) for t in templates)
        if m > best[0]:
            best = (m, px + T / 2.0, py + T / 2.0)
    return best[1], best[2]


def mechanism_test(ref, srch, gx, gy, step=2, tol=TOL):
    """max-over-offsets at the true position vs at the best impostor."""
    tx = int(np.clip(round(gx - T / 2), 0, srch.shape[1] - T))
    ty = int(np.clip(round(gy - T / 2), 0, srch.shape[0] - T))
    true_patch = srch[ty:ty + T, tx:tx + T]

    _, _, surf = run_plain(ref, srch)
    peaks = [(py, px) for py, px in top_peaks(surf, 40)
             if np.hypot(px + T / 2 - gx, py + T / 2 - gy) > tol]
    if not peaks:
        return None

    templates = [template_at_offset(ref, float(px), float(py), T, SCALE)
                 for py in range(0, SCALE, step)
                 for px in range(0, SCALE, step)]
    m_true = max(zncc_patch(t, true_patch) for t in templates)
    m_imp = -2.0
    for py, px in peaks[:12]:
        patch = srch[py:py + T, px:px + T]
        if patch.shape != (T, T):
            continue
        m_imp = max(m_imp, max(zncc_patch(t, patch) for t in templates))

    plain_t = plain_template(ref)
    p_true = zncc_patch(plain_t, true_patch)
    p_imp = max(zncc_patch(plain_t, srch[py:py + T, px:px + T])
                for py, px in peaks[:12]
                if srch[py:py + T, px:px + T].shape == (T, T))
    return m_true - m_imp, p_true - p_imp


def load_pairs(data, n):
    for r, _, files in os.walk(data):
        if "manifest.csv" in files:
            rows = list(csv.DictReader(open(os.path.join(r, "manifest.csv"))))
            out = []
            for row in rows[:n]:
                i = int(row["id"])
                ref = cv2.imread(os.path.join(r, "reference", f"{i:05d}.png"),
                                 cv2.IMREAD_GRAYSCALE)
                s = cv2.imread(os.path.join(r, "search", f"{i:05d}.png"),
                               cv2.IMREAD_GRAYSCALE)
                if ref is not None and s is not None:
                    out.append((i, ref, s, float(row["gt_x"]),
                                float(row["gt_y"])))
            return out
    raise SystemExit(f"no manifest.csv under {data}")


def gen_pairs(n, kind, seed, level):
    from driftsense.generator.sample import generate_sample, build_params
    p = build_params(noise_level=level)
    out = []
    for i in range(n):
        s = generate_sample(i, kind, base_seed=seed, params=p)
        out.append((i, s.reference, s.search, s.gt_x, s.gt_y))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--kind", default="dram", choices=["dram", "finfet"])
    ap.add_argument("--seed", type=int, default=3131)
    ap.add_argument("--level", default="medium")
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--no-full", action="store_true",
                    help="skip the full sweep, which is the slow one")
    a = ap.parse_args(argv)

    pairs = load_pairs(a.data, a.n) if a.data else \
        gen_pairs(a.n, a.kind, a.seed, a.level)
    src = a.data if a.data else f"generated {a.kind} seed {a.seed}"

    print("=" * 78)
    print(f"PRUNED SWEEP   {src}   n={len(pairs)}  topK={a.topk}"
          f"  step={a.step}")
    print("=" * 78)
    print(f"\n  {'id':>4} {'plain':>10} {'full':>10} {'pruned':>10}"
          f" {'cond':>6}")
    print("  " + "-" * 44)

    ep, ef, epr, cond = [], [], [], []
    tp = tf = tpr = 0.0
    gaps_m, gaps_p = [], []

    for idx, ref, srch, gx, gy in pairs:
        tmpl = plain_template(ref)
        c = (len(_strip_centres(tmpl, 1, 0.45, 6)) > 0) \
            + (len(_strip_centres(tmpl, 0, 0.45, 6)) > 0)
        cond.append(int(c))

        t0 = time.time()
        x, y, surf = run_plain(ref, srch)
        tp += time.time() - t0
        e0 = float(np.hypot(x - gx, y - gy))
        ep.append(e0)

        if a.no_full:
            e1 = float("nan")
        else:
            t0 = time.time()
            x, y = run_full_sweep(ref, srch, a.step)
            tf += time.time() - t0
            e1 = float(np.hypot(x - gx, y - gy))
        ef.append(e1)

        t0 = time.time()
        x, y = run_pruned(ref, srch, a.topk, a.step, surf)
        tpr += time.time() - t0
        e2 = float(np.hypot(x - gx, y - gy))
        epr.append(e2)

        mt = mechanism_test(ref, srch, gx, gy)
        if mt is not None:
            gaps_m.append(mt[0])
            gaps_p.append(mt[1])

        f = "nan" if np.isnan(e1) else f"{e1:10.1f}"
        print(f"  {idx:>4} {e0:>10.1f} {f:>10} {e2:>10.1f} {c:>6}")

    ep, ef, epr = np.asarray(ep), np.asarray(ef), np.asarray(epr)
    cond = np.asarray(cond)
    n = len(ep)
    acc = lambda e: float(np.nanmean(e <= a.tol)) * 100

    print(f"\n  {'method':<16}{'accuracy':>10}{'median err':>12}{'ms/pair':>10}")
    print("  " + "-" * 50)
    print(f"  {'plain ZNCC':<16}{acc(ep):>9.1f}%{np.median(ep):>12.2f}"
          f"{tp / n * 1000:>10.0f}")
    if not a.no_full:
        print(f"  {'full sweep':<16}{acc(ef):>9.1f}%{np.nanmedian(ef):>12.2f}"
              f"{tf / n * 1000:>10.0f}")
    print(f"  {'pruned sweep':<16}{acc(epr):>9.1f}%{np.median(epr):>12.2f}"
          f"{tpr / n * 1000:>10.0f}")

    print(f"\n  by template condition")
    print(f"    {'condition':<20}{'n':>5}{'plain':>9}{'pruned':>9}")
    for c, nm in ((2, "both axes pinned"), (1, "one axis pinned"),
                  (0, "neither pinned")):
        m = cond == c
        if m.sum():
            print(f"    {nm:<20}{int(m.sum()):>5}"
                  f"{float((ep[m] <= a.tol).mean()) * 100:>8.1f}%"
                  f"{float((epr[m] <= a.tol).mean()) * 100:>8.1f}%")

    if gaps_m:
        gm, gp = np.asarray(gaps_m), np.asarray(gaps_p)
        print(f"\n  MECHANISM  true minus best impostor, over {len(gm)} samples")
        print(f"    plain ZNCC            median {np.median(gp):+.4f}"
              f"   positive in {int((gp > 0).sum())}/{len(gp)}")
        print(f"    max over resamplings  median {np.median(gm):+.4f}"
              f"   positive in {int((gm > 0).sum())}/{len(gm)}")
        print(f"    change                       {np.median(gm - gp):+.4f}")

    fixed = int(((ep > a.tol) & (epr <= a.tol)).sum())
    broke = int(((ep <= a.tol) & (epr > a.tol)).sum())
    print(f"\n  pruned fixes {fixed}, breaks {broke},"
          f" net {(fixed - broke) / n * 100:+.1f} pp")

    print("\n" + "=" * 78)
    d = acc(epr) - acc(ep)
    if d > 8:
        print(f"  Pruned sweep gains {d:+.1f} points at"
              f" {tpr / max(tp, 1e-9):.0f}x the cost of plain matching.")
        print("  If the mechanism rows show the margin widening, the")
        print("  max-over-resamplings explanation holds and there is nothing")
        print("  to learn -- no model, no weights, no loading step.")
    elif d > 2:
        print(f"  Pruned sweep gains {d:+.1f} points. Less than the full")
        print("  sweep, so pruning is discarding positions the full version")
        print("  recovers. Raise topK and re-measure.")
    else:
        print("  Pruning removes the gain. The full sweep's advantage")
        print("  depends on positions that plain ZNCC ranks low, which means")
        print("  the max must be taken over the whole surface and the cost")
        print("  cannot be avoided this way.")
    print("=" * 78)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
