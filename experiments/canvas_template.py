#!/usr/bin/env python3
"""Is the template itself the problem?

The arithmetic that motivates this. For ZNCC over N pixels at true
correlation rho, the sampling standard deviation is (1-rho^2)/sqrt(N).
With rho = 0.878 and N = 10,000 that is 0.0023. The measured median gap
between the true position and the winning impostor is -0.1021.

    0.1021 / 0.0023  =  44 sigma

So the failures are not noise pushing an impostor past a near-tie. The
template deterministically matches the wrong site better, by a margin
forty-four standard deviations wide. If the template were a faithful
rendering of the true site, nothing could beat it by more than about
3 sigma, or 0.007.

That reframes twelve experiments. Weighting, verification, residuals and
integer inference all assumed near-ties resolved by noise. There are no
near-ties.

What could make the template unfaithful? The two imaging paths differ in
the order of their operations:

    reference : canvas -> blur -> CROP -> Poisson(dose 2000) -> our /10
    search    : canvas -> blur -> /10  -> Poisson(dose 200)

Poisson noise lands before our downsample on one path and after on the
other. Means survive that, but the clip to [0, 255] does not -- it is
non-linear, and it bites differently at sigma 5.6 (dose 2000, mu 250)
than at sigma 17.9 (dose 200). Edge-brightening pushes contacts to about
244, right against the ceiling.

The test builds the template three ways and compares:

    reference-derived   what we ship
    canvas-derived      the true site taken straight from the canvas,
                        area-averaged by 10, with no reference imaging
                        chain at all
    search-derived      the true site cut straight out of the search
                        image, which is the upper bound: same pixels,
                        same noise, same everything

If the canvas template wins where the reference template loses, the fault
is in template construction and not in matching. Requires our generator,
since it needs the canvas.

    python experiments/canvas_template.py --n 40
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
from driftsense.generator.sample import (generate_sample, build_params,
                                         FINE_CANVAS_PX, SCALE)

T = 100
TOL = 5.0


def canvas_for(index, kind, base_seed, params):
    """Rebuild the exact canvas, area-averaged to search resolution."""
    ss = np.random.SeedSequence([int(base_seed), int(index)])
    geom = np.random.default_rng(ss.spawn(3)[0])
    z = generate_zone_canvas(
        FINE_CANVAS_PX, kind, geom,
        mat_nm=params["mat_size_nm"], strip_nm=params["strip_width_nm"],
        ler_sigma_nm=params["ler_sigma_nm"],
        linewidth_bias_nm=params["linewidth_bias_nm"],
        corner_rounding_px=params["corner_rounding_px"],
        defect_rate=params["defect_rate"])
    c = sem.edge_brighten(z["canvas"], params["edge_brighten_strength"])
    c = sem.psf_blur(c, params["beam_spot_size_nm"])
    return sem.area_downsample(c, SCALE)


def locate(tmpl, search):
    s = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
    k = int(np.argmax(s))
    y, x = np.unravel_index(k, s.shape)
    return x + T / 2.0, y + T / 2.0, float(s[y, x]), s


def score_at(surf, gx, gy):
    y = int(np.clip(round(gy - T / 2), 0, surf.shape[0] - 1))
    x = int(np.clip(round(gx - T / 2), 0, surf.shape[1] - 1))
    return float(surf[y, x])


def crop(img, cx, cy, half=T // 2):
    h, w = img.shape
    x = int(np.clip(round(cx - half), 0, w - 2 * half))
    y = int(np.clip(round(cy - half), 0, h - 2 * half))
    return img[y:y + 2 * half, x:x + 2 * half]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--kind", default="dram", choices=["dram", "finfet"])
    ap.add_argument("--seed", type=int, default=3131)
    ap.add_argument("--level", default="medium")
    ap.add_argument("--tol", type=float, default=TOL)
    a = ap.parse_args(argv)

    p = build_params(noise_level=a.level)

    print("=" * 82)
    print(f"CANVAS TEMPLATE TEST   kind={a.kind}  n={a.n}  level={a.level}")
    print("=" * 82)
    print("\n  ref    = template from the reference image, what we ship")
    print("  canvas = true site from the canvas, no reference imaging")
    print("  search = true site cut from the search image, the upper bound\n")
    print(f"  {'id':>4} {'ref err':>9} {'can err':>9} {'srch err':>9}"
          f" {'z ref@gt':>9} {'z can@gt':>9} {'z srch@gt':>10}")
    print("  " + "-" * 64)

    er, ec, es = [], [], []
    zr, zc, zs = [], [], []

    for i in range(a.n):
        s = generate_sample(i, a.kind, base_seed=a.seed, params=p)
        canvas = canvas_for(i, a.kind, a.seed, p)

        t_ref = cv2.resize(s.reference, (T, T), interpolation=cv2.INTER_AREA)
        t_can = crop(canvas, s.gt_x, s.gt_y)
        t_srch = crop(s.search, s.gt_x, s.gt_y)
        if t_can.shape != (T, T) or t_srch.shape != (T, T):
            continue

        rx, ry, _, sr = locate(t_ref, s.search)
        cx, cy, _, sc = locate(t_can, s.search)
        sx, sy, _, ss = locate(t_srch, s.search)

        e1 = float(np.hypot(rx - s.gt_x, ry - s.gt_y))
        e2 = float(np.hypot(cx - s.gt_x, cy - s.gt_y))
        e3 = float(np.hypot(sx - s.gt_x, sy - s.gt_y))
        er.append(e1); ec.append(e2); es.append(e3)
        zr.append(score_at(sr, s.gt_x, s.gt_y))
        zc.append(score_at(sc, s.gt_x, s.gt_y))
        zs.append(score_at(ss, s.gt_x, s.gt_y))

        mark = "  <-- canvas fixes it" if e1 > a.tol >= e2 else ""
        print(f"  {i:>4} {e1:>9.1f} {e2:>9.1f} {e3:>9.1f}"
              f" {zr[-1]:>9.4f} {zc[-1]:>9.4f} {zs[-1]:>10.4f}{mark}")

    if not er:
        print("\n  nothing measured")
        return 1

    er, ec, es = map(np.asarray, (er, ec, es))
    zr, zc, zs = map(np.asarray, (zr, zc, zs))
    n = len(er)

    print(f"\n  samples                : {n}")
    print(f"\n  {'template':<12}{'accuracy':>10}{'median err':>12}"
          f"{'median z at gt':>16}")
    for nm, e, z in (("reference", er, zr), ("canvas", ec, zc),
                     ("search", es, zs)):
        print(f"  {nm:<12}{float((e <= a.tol).mean()) * 100:>9.1f}%"
              f"{np.median(e):>12.2f}{np.median(z):>16.4f}")

    fixed = int(((er > a.tol) & (ec <= a.tol)).sum())
    broke = int(((er <= a.tol) & (ec > a.tol)).sum())
    print(f"\n  canvas template fixes  : {fixed}")
    print(f"  canvas template breaks : {broke}")
    print(f"  median score lift      : {np.median(zc - zr):+.4f}")

    print("\n" + "=" * 82)
    d = float((ec <= a.tol).mean() - (er <= a.tol).mean())
    if d > 0.10:
        print(f"  The reference imaging chain is the fault, worth"
              f" {d * 100:+.0f} points.")
        print("  Matching was never the problem. The fix is in template")
        print("  construction: undo whatever the reference path does that")
        print("  the search path does not. Note the canvas is not available")
        print("  at inference, so the next step is to identify WHICH step")
        print("  differs and invert it.")
    elif float((es <= a.tol).mean()) - (er <= a.tol).mean() > 0.10:
        print("  The canvas template does not help, but a template cut from")
        print("  the search image does. So the loss is not in the reference")
        print("  imaging chain -- it is that the two captures are genuinely")
        print("  different exposures of the same site, and no template")
        print("  construction can close that.")
    else:
        print("  Even a template taken straight from the search image at the")
        print("  true position does not win. The true site is not the best")
        print("  match to its own pixels, which means the ground-truth")
        print("  coordinate or the search image itself needs re-examining.")
    print("=" * 82)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
