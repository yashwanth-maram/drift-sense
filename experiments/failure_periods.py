#!/usr/bin/env python3
"""Are the failures on an indistinguishable period, or a distinguishable one?

Two candidate spacings matter, and they are not the same thing.

  LATTICE PITCH        the spacing between adjacent lines, roughly 5-16 px
                       in the search image. Adjacent lines are drawn at
                       exact nanometre positions but sampled onto a 10 nm
                       grid, so consecutive periods land at DIFFERENT
                       sub-pixel offsets and are therefore distinguishable.

  SUPER-PERIOD         the distance after which the sub-pixel sampling
                       pattern repeats. Measured by scanning shifts and
                       finding where a patch best matches itself: 16-43 px
                       in practice. Positions separated by a super-period
                       are identical up to noise and CANNOT be told apart.

So the failures divide cleanly:

  error is a multiple of the SUPER-PERIOD  -> indistinguishable. No method
                                              can recover it. The current
                                              accuracy is at the ceiling.

  error is a multiple of the LATTICE PITCH -> distinguishable. The
                                              information is present and we
                                              are failing at something
                                              solvable.

This runs on failures that already exist, so no new generation is needed.

    python experiments/failure_periods.py --data ../am_eval --n 100
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from driftsense.localize import localize

TOL = 5.0
PATCH = 64


def noise_sigma(img):
    a = img.astype(np.float32)
    d = a[:, :-2] - 2.0 * a[:, 1:-1] + a[:, 2:]
    g = np.abs(a[:, 2:] - a[:, :-2])
    flat = d[g <= np.quantile(g, 0.25)]
    if len(flat) < 100:
        return float(np.std(d) / np.sqrt(6.0))
    return float(np.median(np.abs(flat)) / (0.6745 * np.sqrt(6.0)))


def measure_periods(img, cx, cy, axis, max_shift=70):
    """Self-match profile around (cx, cy) along one axis.

    Returns (lattice_pitch, super_period, ratio_at_super) where the pitch
    is the first local minimum and the super-period is the global one.
    """
    h, w = img.shape
    cx = int(np.clip(cx, PATCH, w - PATCH - max_shift - 1))
    cy = int(np.clip(cy, PATCH, h - PATCH - max_shift - 1))
    base = img[cy:cy + PATCH, cx:cx + PATCH].astype(np.float64)

    rm = []
    for d in range(2, max_shift):
        if axis == 1:
            other = img[cy:cy + PATCH, cx + d:cx + d + PATCH]
        else:
            other = img[cy + d:cy + d + PATCH, cx:cx + PATCH]
        rm.append((d, float(np.sqrt(((base - other.astype(np.float64)) ** 2).mean()))))

    if not rm:
        return None
    ds = np.array([r[0] for r in rm])
    vs = np.array([r[1] for r in rm])

    gi = int(np.argmin(vs))
    super_p = int(ds[gi])

    pitch = None
    for i in range(1, len(vs) - 1):
        if vs[i] < vs[i - 1] and vs[i] <= vs[i + 1]:
            pitch = int(ds[i])
            break

    expected = noise_sigma(img) * np.sqrt(2.0)
    return pitch, super_p, float(vs[gi] / max(expected, 1e-6))


def classify(err, pitch, super_p, tol=0.22):
    """Which spacing does this error land on?"""
    def near_multiple(v, p):
        if not p or p < 2:
            return False, 0
        k = round(v / p)
        if k < 1:
            return False, 0
        return abs(v - k * p) <= tol * p, int(k)

    on_super, ks = near_multiple(err, super_p)
    on_pitch, kp = near_multiple(err, pitch)
    if on_super:
        return "SUPER-PERIOD", ks
    if on_pitch:
        return "lattice pitch", kp
    return "neither", 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--rung", type=int, default=1)
    ap.add_argument("--tol", type=float, default=TOL)
    a = ap.parse_args(argv)

    root = None
    for r, _, files in os.walk(a.data):
        if "manifest.csv" in files:
            root = r
            rows = list(csv.DictReader(open(os.path.join(r, "manifest.csv"))))
            break
    if root is None:
        raise SystemExit(f"no manifest.csv under {a.data}")

    print("=" * 76)
    print("FAILURE PERIOD ANALYSIS")
    print("=" * 76)
    print("\n  For each failure, the dominant error axis is compared against")
    print("  the two spacings measured from the search image itself.\n")
    print(f"  {'id':>4} {'axis':>5} {'error':>8} {'pitch':>7} {'super':>7}"
          f" {'ratio':>7}   classification")
    print("  " + "-" * 66)

    counts = {"SUPER-PERIOD": 0, "lattice pitch": 0, "neither": 0}
    ratios = []

    for row in rows[:a.n]:
        i = int(row["id"])
        ref = cv2.imread(os.path.join(root, "reference", f"{i:05d}.png"),
                         cv2.IMREAD_GRAYSCALE)
        srch = cv2.imread(os.path.join(root, "search", f"{i:05d}.png"),
                          cv2.IMREAD_GRAYSCALE)
        if ref is None or srch is None:
            continue

        gx, gy = float(row["gt_x"]), float(row["gt_y"])
        res = localize(ref, srch, rung=a.rung)
        dx, dy = res.x - gx, res.y - gy
        if np.hypot(dx, dy) <= a.tol:
            continue

        axis = 1 if abs(dx) >= abs(dy) else 0
        err = abs(dx) if axis == 1 else abs(dy)
        m = measure_periods(srch, gx, gy, axis)
        if m is None:
            continue
        pitch, super_p, ratio = m
        kind, k = classify(err, pitch, super_p)
        counts[kind] += 1
        ratios.append(ratio)

        ax = "x" if axis == 1 else "y"
        note = f"{kind}" + (f" x{k}" if k else "")
        print(f"  {i:>4} {ax:>5} {err:>8.1f} {str(pitch):>7} {super_p:>7}"
              f" {ratio:>7.2f}   {note}")

    n = sum(counts.values())
    if not n:
        print("\n  no failures")
        return 0

    print(f"\n  failures analysed        : {n}")
    print(f"  on the SUPER-PERIOD      : {counts['SUPER-PERIOD']:>3}"
          f"  ({counts['SUPER-PERIOD'] / n * 100:.0f}%)   indistinguishable")
    print(f"  on the lattice pitch     : {counts['lattice pitch']:>3}"
          f"  ({counts['lattice pitch'] / n * 100:.0f}%)   distinguishable")
    print(f"  on neither               : {counts['neither']:>3}"
          f"  ({counts['neither'] / n * 100:.0f}%)")
    print(f"  median self-match ratio  : {np.median(ratios):.2f}"
          f"   (1.0 = identical to noise)")

    print("\n" + "=" * 76)
    frac_super = counts["SUPER-PERIOD"] / n
    frac_solvable = (counts["lattice pitch"] + counts["neither"]) / n
    if frac_super > 0.6:
        print("  VERDICT: most failures sit on the super-period. Those")
        print("  candidates are identical up to noise and cannot be")
        print("  recovered by any method. Report accuracy against this")
        print("  ceiling and make the confidence score the deliverable.")
    elif frac_solvable > 0.6:
        print("  VERDICT: most failures are on distinguishable spacings.")
        print("  The information is present and we are failing at something")
        print("  solvable. Periodic subtraction and phase-locked")
        print("  demodulation are worth building after all.")
    else:
        print("  VERDICT: mixed. Split the reporting -- quote the solvable")
        print("  subset separately from the indistinguishable one, and")
        print("  attack only the former.")
    print("=" * 76)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
