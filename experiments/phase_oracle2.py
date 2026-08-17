#!/usr/bin/env python3
"""Sub-pixel phase, with the sign error corrected.

An earlier version of this test reported that phase does not matter. That
result is void: it shifted the reference by +fx when alignment requires
-fx, so the "corrected phase" template was, for many values, worse aligned
than doing nothing.

The geometry. The crop origin x0 is an integer in canvas pixels. The
search image samples the canvas in blocks of ten, so search pixel j covers
canvas [10j, 10j+10). Template pixel m averages reference pixels
[10m, 10m+10), which is canvas [x0+10m, x0+10m+10). Those grids coincide
only when x0 is a multiple of ten -- one time in ten. Otherwise the
template is a fractionally shifted rendering, and to align it the crop
must move by -(x0 mod 10) canvas pixels.

Why this is worth re-testing when shear was not. Shear displaces a row by
s*y/999, which depends only on y. An impostor at x_true + n*pitch sits at
the same row and carries the identical offset, so the handicap is
symmetric across exactly the candidates that compete and cannot change
their ordering. Sub-pixel phase is different: each candidate sits at its
own fractional offset, so the effect is candidate-dependent and CAN alter
relative scores.

That makes phase the remaining untested registration variable capable of
changing candidate ordering. This does not predict that correcting it will
fix the failures.

Reported per failure:
    score at the true position, uncorrected and phase-corrected
    the margin against the best competing peak, both ways
    whether the truth's rank improves -- the quantity that decides whether
    any selection rule could recover it

    python experiments/phase_oracle2.py --data ../am_eval --n 100
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

T = 100
SCALE = 10
TOL = 5.0


def template_at_phase(reference, px, py):
    """Downsample after shifting the crop by (px, py) canvas pixels.

    A positive px moves the sampling window right, so to compensate a crop
    origin of x0 = 10k + fx the shift must be -fx.
    """
    if px == 0 and py == 0:
        return cv2.resize(reference, (T, T), interpolation=cv2.INTER_AREA)
    M = np.float32([[1, 0, px], [0, 1, py]])
    shifted = cv2.warpAffine(reference, M,
                             (reference.shape[1], reference.shape[0]),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
    return cv2.resize(shifted, (T, T), interpolation=cv2.INTER_AREA)


def required_shift(gt):
    """Canvas-pixel shift that aligns the template grid to the search grid.

    gt = x0/10 + 50, so x0 = (gt - 50) * 10 and the misalignment is
    x0 mod 10. The correction is its negative.
    """
    x0 = int(round((gt - T / 2) * SCALE))
    return -(x0 % SCALE)


def peak_rank(surface, y, x, sep=6, cap=3000):
    """How many distinct peaks outscore the value at (y, x)."""
    h, w = surface.shape
    y = int(np.clip(y, 0, h - 1))
    x = int(np.clip(x, 0, w - 1))
    v = float(surface[y, x])
    flat = surface.ravel()
    idx = np.argpartition(flat, -min(cap, flat.size - 1))[-cap:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    taken, n = [], 0
    for k in idx:
        if flat[k] <= v:
            break
        ky, kx = divmod(int(k), w)
        if all(abs(ky - ty) > sep or abs(kx - tx) > sep for ty, tx in taken):
            taken.append((ky, kx))
            n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--all", action="store_true",
                    help="report every sample, not only failures")
    a = ap.parse_args(argv)

    root = None
    for r, _, files in os.walk(a.data):
        if "manifest.csv" in files:
            root = r
            rows = list(csv.DictReader(open(os.path.join(r, "manifest.csv"))))
            break
    if root is None:
        raise SystemExit(f"no manifest.csv under {a.data}")

    print("=" * 78)
    print("PHASE ORACLE  (sign corrected)")
    print("=" * 78)
    print("\n  Using the known crop origin to align the template grid. This is")
    print("  the best any phase-aware method could do, so it bounds what a")
    print("  sweep would be worth.\n")
    print(f"  {'id':>4} {'shift':>7} {'z plain':>9} {'z aligned':>10}"
          f" {'gain':>8} {'rank0':>7} {'rank1':>7} {'fixed':>7}")
    print("  " + "-" * 66)

    gains, r0s, r1s = [], [], []
    fixed = 0
    n_fail = 0

    for row in rows[:a.n]:
        i = int(row["id"])
        ref = cv2.imread(os.path.join(root, "reference", f"{i:05d}.png"),
                         cv2.IMREAD_GRAYSCALE)
        srch = cv2.imread(os.path.join(root, "search", f"{i:05d}.png"),
                          cv2.IMREAD_GRAYSCALE)
        if ref is None or srch is None:
            continue
        gx, gy = float(row["gt_x"]), float(row["gt_y"])

        t0 = template_at_phase(ref, 0, 0)
        s0 = cv2.matchTemplate(srch, t0, cv2.TM_CCOEFF_NORMED)
        k = int(np.argmax(s0))
        py, px = np.unravel_index(k, s0.shape)
        err0 = float(np.hypot(px + T / 2 - gx, py + T / 2 - gy))
        if err0 <= a.tol and not a.all:
            continue
        n_fail += 1

        fx, fy = required_shift(gx), required_shift(gy)
        t1 = template_at_phase(ref, fx, fy)
        s1 = cv2.matchTemplate(srch, t1, cv2.TM_CCOEFF_NORMED)

        ty = int(np.clip(round(gy - T / 2), 0, s0.shape[0] - 1))
        tx = int(np.clip(round(gx - T / 2), 0, s0.shape[1] - 1))
        z0 = float(s0[ty, tx])
        z1 = float(s1[ty, tx])
        r0 = peak_rank(s0, ty, tx)
        r1 = peak_rank(s1, ty, tx)

        k1 = int(np.argmax(s1))
        p1y, p1x = np.unravel_index(k1, s1.shape)
        err1 = float(np.hypot(p1x + T / 2 - gx, p1y + T / 2 - gy))
        ok = err1 <= a.tol
        fixed += int(ok and err0 > a.tol)

        gains.append(z1 - z0)
        r0s.append(r0)
        r1s.append(r1)
        print(f"  {i:>4} {f'{fx},{fy}':>7} {z0:>9.4f} {z1:>10.4f}"
              f" {z1 - z0:>+8.4f} {r0:>7} {r1:>7} {'YES' if ok else '':>7}")

    if not gains:
        print("\n  nothing to analyse")
        return 0

    g = np.asarray(gains)
    r0a, r1a = np.asarray(r0s), np.asarray(r1s)
    n = len(g)

    print(f"\n  cases analysed             : {n}")
    print(f"  median score gain          : {np.median(g):+.4f}")
    print(f"  cases where score improved : {int((g > 0).sum())}/{n}")
    print(f"\n  median rank of truth, plain   : {np.median(r0a):.1f}")
    print(f"  median rank of truth, aligned : {np.median(r1a):.1f}")
    print(f"  rank improved              : {int((r1a < r0a).sum())}/{n}")
    print(f"  truth reaches rank 0       : {int((r1a == 0).sum())}/{n}"
          f"  (was {int((r0a == 0).sum())})")
    print(f"\n  failures fixed             : {fixed}/{n_fail}")

    print("\n" + "=" * 78)
    if fixed / max(n_fail, 1) > 0.3:
        print("  Phase alignment recovers a substantial share of the")
        print("  failures. Build a phase-aware template stage -- and note it")
        print("  needs ten candidate templates per axis at inference, since")
        print("  the crop origin is unknown, so validate that the sweep does")
        print("  not reintroduce the cost a scale sweep was measured to have.")
    elif np.median(r1a) < np.median(r0a):
        print("  Alignment improves the score and the ranking but does not")
        print("  usually put the truth first. Worth combining with candidate")
        print("  restriction rather than used alone.")
    else:
        print("  Alignment lifts the score at the true position without")
        print("  changing the ordering: competing peaks gain comparably. The")
        print("  handicap is more symmetric across candidates than expected,")
        print("  and this closes the registration line.")
    print("=" * 78)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
