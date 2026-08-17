#!/usr/bin/env python3
"""Are the failing cases solvable in principle?

Part B of measure_ceiling showed that removing all noise and drift changes
nothing: 0 of 10 failures were fixed. But that only shows THIS matcher
does not improve with clean data. It does not show that no matcher could.

This test settles it, by working on the CANVAS -- the ground-truth layout
before any imaging is applied. For each failure it extracts the true
region and the region the matcher chose, straight from the canvas, and
compares them pixel by pixel.

    identical   -> the two locations are the same object. Nothing could
                   distinguish them. The failure is a property of the data
                   and the current accuracy is at the ceiling.

    different   -> signal exists and ZNCC is not extracting it. The
                   failure is a property of the matcher and is worth
                   attacking.

Only our generator can run this, because it needs the canvas itself rather
than an image of it.

    python experiments/canvas_ambiguity.py --n 30
    python experiments/canvas_ambiguity.py --n 30 --kind finfet
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from driftsense.generator.patterns.zones import generate_zone_canvas
from driftsense.generator import sem
from driftsense.generator.sample import (
    generate_sample, build_params, FINE_CANVAS_PX, SCALE)
from driftsense.localize import localize

TOL = 5.0
HALF = 50


def canvas_for(index, kind, base_seed, params):
    """Rebuild the exact canvas a sample was cut from.

    Reproduces generate_sample's geometry stream: the same SeedSequence
    spawn, so the same mats, presets and layout.
    """
    ss = np.random.SeedSequence([int(base_seed), int(index)])
    geom_rng = np.random.default_rng(ss.spawn(3)[0])
    z = generate_zone_canvas(
        FINE_CANVAS_PX, kind, geom_rng,
        mat_nm=params["mat_size_nm"], strip_nm=params["strip_width_nm"],
        ler_sigma_nm=params["ler_sigma_nm"],
        linewidth_bias_nm=params["linewidth_bias_nm"],
        corner_rounding_px=params["corner_rounding_px"],
        defect_rate=params["defect_rate"])
    c = sem.edge_brighten(z["canvas"], params["edge_brighten_strength"])
    c = sem.psf_blur(c, params["beam_spot_size_nm"])
    return sem.area_downsample(c, SCALE)


def region(img, cx, cy, half=HALF):
    h, w = img.shape
    x = int(np.clip(round(cx - half), 0, w - 2 * half))
    y = int(np.clip(round(cy - half), 0, h - 2 * half))
    return img[y:y + 2 * half, x:x + 2 * half]


def compare(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    rmse = float(np.sqrt(((a - b) ** 2).mean()))
    maxd = float(np.abs(a - b).max())
    za, zb = a - a.mean(), b - b.mean()
    na, nb = np.linalg.norm(za), np.linalg.norm(zb)
    corr = float((za * zb).sum() / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0
    return rmse, maxd, corr


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--kind", default="dram", choices=["dram", "finfet"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--rung", type=int, default=1)
    ap.add_argument("--level", default="medium")
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--save", default=None,
                    help="directory to write side-by-side canvas crops")
    a = ap.parse_args(argv)

    params = build_params(noise_level=a.level)
    if a.save:
        os.makedirs(a.save, exist_ok=True)

    print("=" * 72)
    print(f"CANVAS AMBIGUITY   kind={a.kind}  n={a.n}  seed={a.seed}")
    print("=" * 72)
    print("\n  Comparing the two candidate regions on the noise-free canvas,")
    print("  before any imaging. Identical regions cannot be told apart by")
    print("  any algorithm.\n")
    print(f"  {'id':>4} {'error':>9} {'rmse':>8} {'maxdiff':>8} {'corr':>8}"
          f"   verdict")
    print("  " + "-" * 62)

    rows = []
    for i in range(a.n):
        s = generate_sample(i, a.kind, base_seed=a.seed, params=params)
        r = localize(s.reference, s.search, rung=a.rung)
        err = float(np.hypot(r.x - s.gt_x, r.y - s.gt_y))
        if err <= a.tol:
            continue

        canvas = canvas_for(i, a.kind, a.seed, params)
        rt = region(canvas, s.gt_x, s.gt_y)
        rp = region(canvas, r.x, r.y)
        rmse, maxd, corr = compare(rt, rp)

        if rmse < 1.0:
            v = "IDENTICAL - unsolvable"
        elif rmse < 6.0:
            v = "near-identical"
        else:
            v = "DIFFERENT - signal exists"
        rows.append((i, err, rmse, maxd, corr, v))
        print(f"  {i:>4} {err:>9.1f} {rmse:>8.2f} {maxd:>8.0f} {corr:>8.3f}"
              f"   {v}")

        if a.save:
            gap = np.full((2 * HALF, 8), 255, np.uint8)
            strip = np.hstack([rt, gap, rp,
                               gap, np.abs(rt.astype(np.int16)
                                           - rp.astype(np.int16))
                               .clip(0, 255).astype(np.uint8)])
            cv2.imwrite(os.path.join(a.save, f"canvas_{i:05d}.png"),
                        cv2.resize(strip, None, fx=3, fy=3,
                                   interpolation=cv2.INTER_NEAREST))

    if not rows:
        print("\n  no failures at this setting")
        return 0

    arr = np.array([[r[2], r[4]] for r in rows])
    ident = sum(1 for r in rows if r[2] < 1.0)
    near = sum(1 for r in rows if 1.0 <= r[2] < 6.0)
    diff = sum(1 for r in rows if r[2] >= 6.0)

    print(f"\n  failures analysed        : {len(rows)}")
    print(f"  median rmse              : {np.median(arr[:, 0]):.2f} grey levels")
    print(f"  median correlation       : {np.median(arr[:, 1]):.3f}")
    print(f"\n  identical (rmse < 1)     : {ident}   unsolvable")
    print(f"  near-identical (< 6)     : {near}")
    print(f"  different (>= 6)         : {diff}   signal exists")

    print("\n" + "=" * 72)
    if diff > len(rows) / 2:
        print("  VERDICT: most failures have DIFFERENT canvas content.")
        print("  The information is there and ZNCC is not using it. The")
        print("  current accuracy is not a ceiling and residual matching")
        print("  is worth building.")
    elif ident + near > len(rows) / 2:
        print("  VERDICT: most failures are on near-identical content.")
        print("  No matcher can separate them. Report accuracy against this")
        print("  ceiling rather than against 100%, and make the confidence")
        print("  score the deliverable.")
    else:
        print("  VERDICT: mixed. Split the reporting -- quote the solvable")
        print("  subset separately from the ambiguous one.")
    print("=" * 72)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())