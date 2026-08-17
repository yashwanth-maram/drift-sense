#!/usr/bin/env python3
"""Does the template's sampling phase matter?

The crop origin is an integer in CANVAS pixels, and the search image
samples the canvas in blocks of ten. So the template's pixel grid lines up
with the search grid only when the origin is a multiple of ten -- one time
in ten. Otherwise the template is a fractionally shifted rendering of the
same content, which should depress the score at the TRUE position
specifically, since that is the only place an exact match was available.

The ground truth shows the phase directly: gt_x = x0/10 + 50, so the
fractional part of gt_x is the sub-pixel offset. Values like 411.9, 774.1
and 545.3 are phases 9, 1 and 3.

This measures the effect as an ORACLE, using the known phase, before
anything is built. A phase sweep costs up to 100 template renderings per
pair, and a scale sweep has already been measured to cost 15 points, so a
sweep is only worth building if the effect is large.

Reports:
    zncc at the true position with phase 0     -- what we do now
    zncc at the true position with true phase  -- the best possible
    the margin against the best competing peak, both ways

    python experiments/phase_oracle.py --n 30
    python experiments/phase_oracle.py --data ../am_eval --n 100
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

SCALE = 10
T = 100


def template_at_phase(reference, px, py):
    """Downsample the reference after shifting it by (px, py) canvas pixels.

    A positive shift moves the sampling window, changing which canvas
    pixels fall into each template pixel.
    """
    M = np.float32([[1, 0, -px], [0, 1, -py]])
    shifted = cv2.warpAffine(reference, M, (reference.shape[1], reference.shape[0]),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
    return cv2.resize(shifted, (T, T), interpolation=cv2.INTER_AREA)


def true_phase(gt_x, gt_y):
    """Recover the sampling phase from the ground truth coordinate."""
    fx = int(round((gt_x - T / 2) * SCALE)) % SCALE
    fy = int(round((gt_y - T / 2) * SCALE)) % SCALE
    return fx, fy


def score_at(surface, gt_x, gt_y, half=T // 2, win=2):
    y = int(np.clip(round(gt_y - half), 0, surface.shape[0] - 1))
    x = int(np.clip(round(gt_x - half), 0, surface.shape[1] - 1))
    y0, y1 = max(y - win, 0), min(y + win + 1, surface.shape[0])
    x0, x1 = max(x - win, 0), min(x + win + 1, surface.shape[1])
    return float(surface[y0:y1, x0:x1].max()), (y, x)


def best_competitor(surface, ty, tx, exclude=25):
    s = surface.copy()
    y0, y1 = max(ty - exclude, 0), min(ty + exclude + 1, s.shape[0])
    x0, x1 = max(tx - exclude, 0), min(tx + exclude + 1, s.shape[1])
    s[y0:y1, x0:x1] = -2.0
    return float(s.max())


def run(pairs):
    print(f"  {'id':>4} {'phase':>7} {'zncc p0':>9} {'zncc true':>10}"
          f" {'gain':>8} {'margin p0':>10} {'margin true':>12}")
    print("  " + "-" * 68)

    g0, gt, m0, mt = [], [], [], []
    flips = 0
    for idx, ref, srch, gx, gy in pairs:
        fx, fy = true_phase(gx, gy)

        t0 = template_at_phase(ref, 0, 0)
        tt = template_at_phase(ref, fx, fy)
        s0 = cv2.matchTemplate(srch, t0, cv2.TM_CCOEFF_NORMED)
        st = cv2.matchTemplate(srch, tt, cv2.TM_CCOEFF_NORMED)

        v0, (ty, tx) = score_at(s0, gx, gy)
        vt, _ = score_at(st, gx, gy)
        c0 = best_competitor(s0, ty, tx)
        ct = best_competitor(st, ty, tx)

        g0.append(v0); gt.append(vt)
        m0.append(v0 - c0); mt.append(vt - ct)
        if (v0 - c0) < 0 <= (vt - ct):
            flips += 1

        print(f"  {idx:>4} {f'({fx},{fy})':>7} {v0:>9.4f} {vt:>10.4f}"
              f" {vt - v0:>+8.4f} {v0 - c0:>+10.4f} {vt - ct:>+12.4f}")

    n = len(g0)
    g0, gt = np.array(g0), np.array(gt)
    m0, mt = np.array(m0), np.array(mt)
    print(f"\n  samples                        : {n}")
    print(f"  median zncc, phase 0           : {np.median(g0):.4f}")
    print(f"  median zncc, true phase        : {np.median(gt):.4f}")
    print(f"  median gain                    : {np.median(gt - g0):+.4f}")
    print(f"\n  true position wins, phase 0    : {int((m0 > 0).sum())}/{n}"
          f"  ({(m0 > 0).mean() * 100:.0f}%)")
    print(f"  true position wins, true phase : {int((mt > 0).sum())}/{n}"
          f"  ({(mt > 0).mean() * 100:.0f}%)")
    print(f"  failures rescued by phase      : {flips}")

    print()
    if (mt > 0).mean() - (m0 > 0).mean() > 0.05:
        print("  VERDICT: phase matters. A sweep is worth building -- but note")
        print("  it renders up to 100 templates per pair, and a scale sweep")
        print("  has already been measured to cost 15 points, so it must be")
        print("  validated on held-out data before shipping.")
    else:
        print("  VERDICT: phase does not move the ranking. Even with the")
        print("  correct phase handed to it, the true position does not win")
        print("  materially more often. Drop this line.")


def from_dataset(data_dir, n):
    root = None
    for r, _, files in os.walk(data_dir):
        if "manifest.csv" in files:
            root = r
            rows = list(csv.DictReader(open(os.path.join(r, "manifest.csv"))))
            break
    if root is None:
        raise SystemExit(f"no manifest.csv under {data_dir}")

    out = []
    for row in rows[:n]:
        i = int(row["id"])
        rp = os.path.join(root, "reference", f"{i:05d}.png")
        sp = os.path.join(root, "search", f"{i:05d}.png")
        ref = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
        srch = cv2.imread(sp, cv2.IMREAD_GRAYSCALE)
        if ref is None or srch is None:
            continue
        out.append((i, ref, srch, float(row["gt_x"]), float(row["gt_y"])))
    return out


def from_generator(n, kind, seed, level):
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
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--kind", default="dram", choices=["dram", "finfet"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--level", default="medium")
    a = ap.parse_args(argv)

    print("=" * 72)
    print("SUB-PIXEL PHASE ORACLE")
    print("=" * 72)
    print("\n  Using the KNOWN phase from ground truth. This is the best any")
    print("  phase-aware method could do, so it bounds what a sweep is worth.\n")

    pairs = from_dataset(a.data, a.n) if a.data else \
        from_generator(a.n, a.kind, a.seed, a.level)
    run(pairs)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
