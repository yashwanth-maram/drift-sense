#!/usr/bin/env python3
"""What is the sweep-selected offset actually measuring?

The exhaustive offset sweep reaches 91.7% on DRAM and 96.7% on FinFET,
against a plain baseline of 66.7% and 80.0%. It also EXCEEDS the physical
oracle, which uses the offset computed from the known crop origin:
75.0% and 93.3%. An offset chosen by matching score beats the
geometrically correct one, and by 16.7 points on DRAM.

So the sweep is probably not recovering the sampling-grid phase. The
leading hypothesis is that it absorbs search-side geometry -- principally
raster shear, which displaces row y by s*y/999 and can only be absorbed
modulo the sampling period. That predicts, specifically:

    phi_sweep - phi_oracle  ~  -s * y_target / 999   (mod 10)

Pre-registered before running, so a story cannot be fitted to whatever
appears. If the residual after removing that term stays large, shear is
not the explanation and the alternatives below apply.

The distinction decides the next step. If the offset depends on the search
image, then learning R -> (phi_x, phi_y) from the reference alone is
predicting an unpredictable quantity, and any estimator must see both
images. If it depends on shear specifically, no learning is needed at all:
we already measure shear from the search image to about 0.2 px, so the
offset is computable analytically from a first-pass position estimate.

    python experiments/sweep_diagnostic.py --n 60 --seed 3131
    python experiments/sweep_diagnostic.py --n 40 --kind finfet --seed 7777
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from driftsense.generator.sample import generate_sample, build_params, SCALE
from driftsense.subpixel_template import template_at_offset, estimate_offset

T = 100
TOL = 5.0


def circ_diff(a, b, period=SCALE):
    """Signed difference on a circle, in [-period/2, period/2)."""
    d = (np.asarray(a) - np.asarray(b)) % period
    return np.where(d >= period / 2, d - period, d)


def sweep_offset(reference, search, step=1):
    """Offset whose template produces the highest peak anywhere.

    Note this uses no ground truth: the criterion is the global maximum of
    the correlation surface, which is available at inference.
    """
    best = None
    for py in range(0, SCALE, step):
        for px in range(0, SCALE, step):
            t = template_at_offset(reference, float(px), float(py), T, SCALE)
            s = cv2.matchTemplate(search, t, cv2.TM_CCOEFF_NORMED)
            v = float(s.max())
            if best is None or v > best[0]:
                k = int(np.argmax(s))
                yy, xx = np.unravel_index(k, s.shape)
                best = (v, px, py, xx + T / 2.0, yy + T / 2.0)
    return best


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 4 or a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--kind", default="dram", choices=["dram", "finfet"])
    ap.add_argument("--seed", type=int, default=3131)
    ap.add_argument("--level", default="medium")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--tol", type=float, default=TOL)
    a = ap.parse_args(argv)

    p = build_params(noise_level=a.level)
    shear = float(p["shear_amplitude_px"])
    jitter = float(p["drift_jitter_px"])

    print("=" * 84)
    print(f"SWEEP DIAGNOSTIC   kind={a.kind}  n={a.n}  shear={shear}"
          f"  jitter={jitter}")
    print("=" * 84)
    print("\n  Pre-registered prediction:")
    print("      phi_sweep - phi_oracle  ~  -s * y / 999   (mod 10)\n")
    print(f"  {'id':>4} {'gt_y':>7} {'phiS':>6} {'phiO':>6} {'phiE':>6}"
          f" {'diff':>7} {'pred':>7} {'resid':>7} {'err':>8}")
    print("  " + "-" * 68)

    rows = []
    for i in range(a.n):
        s = generate_sample(i, a.kind, base_seed=a.seed, params=p)
        b = sweep_offset(s.reference, s.search, a.step)
        if b is None:
            continue
        _, sx, sy, px_, py_ = b
        err = float(np.hypot(px_ - s.gt_x, py_ - s.gt_y))

        x0 = int(round((s.gt_x - T / 2) * SCALE))
        y0 = int(round((s.gt_y - T / 2) * SCALE))
        ox, oy = (-x0) % SCALE, (-y0) % SCALE
        ex = estimate_offset(s.reference, SCALE, 1)
        ey = estimate_offset(s.reference, SCALE, 0)

        dx = float(circ_diff(sx, ox))
        pred = float(circ_diff(-shear * s.gt_y / 999.0, 0.0))
        rows.append({"i": i, "gy": s.gt_y, "gx": s.gt_x,
                     "sx": sx, "sy": sy, "ox": ox, "oy": oy,
                     "ex": ex, "ey": ey, "dx": dx, "pred": pred,
                     "resid": float(circ_diff(dx, pred)), "err": err})
        r = rows[-1]
        print(f"  {i:>4} {s.gt_y:>7.1f} {sx:>6} {ox:>6} {ex:>6.1f}"
              f" {dx:>+7.2f} {pred:>+7.2f} {r['resid']:>+7.2f} {err:>8.1f}")

    if not rows:
        print("\n  nothing measured")
        return 1

    dxs = np.array([r["dx"] for r in rows])
    preds = np.array([r["pred"] for r in rows])
    resid = np.array([r["resid"] for r in rows])
    gy = np.array([r["gy"] for r in rows])
    gx = np.array([r["gx"] for r in rows])
    errs = np.array([r["err"] for r in rows])
    sw = np.array([r["sx"] for r in rows], float)
    orc = np.array([r["ox"] for r in rows], float)
    est = np.array([r["ex"] for r in rows], float)

    print(f"\n  samples                       : {len(rows)}")
    print(f"  sweep accuracy at {a.tol:.0f} px        :"
          f" {float((errs <= a.tol).mean()) * 100:.1f}%")
    print(f"  sweep offset == oracle offset : "
          f"{int((np.abs(dxs) < 0.5).sum())}/{len(rows)}")

    print("\n  AGREEMENT")
    print(f"    |phi_sweep - phi_oracle|  median {np.median(np.abs(dxs)):.2f}"
          f"   (0 = sweep recovers the physical phase)")
    print(f"    |phi_est   - phi_oracle|  median "
          f"{np.median(np.abs(circ_diff(est, orc))):.2f}")

    print("\n  THE PRE-REGISTERED SHEAR PREDICTION")
    print(f"    predicted spread          : {preds.min():+.2f} to {preds.max():+.2f} px")
    print(f"    observed spread           : {dxs.min():+.2f} to {dxs.max():+.2f} px")
    print(f"    corr(diff, prediction)    : {corr(dxs, preds):+.3f}")
    print(f"    residual sd after removal : {resid.std():.2f}"
          f"   vs raw sd {dxs.std():.2f}")

    print("\n  ALTERNATIVES")
    print(f"    corr(diff, gt_y)          : {corr(dxs, gy):+.3f}")
    print(f"    corr(diff, gt_x)          : {corr(dxs, gx):+.3f}")
    print(f"    corr(diff, oracle phi)    : {corr(dxs, orc):+.3f}")
    print(f"    corr(sweep phi, oracle)   : {corr(sw, orc):+.3f}")
    print(f"    corr(sweep phi, gt_x)     : {corr(sw, gx):+.3f}")

    hit = errs <= a.tol
    if hit.any() and (~hit).any():
        print(f"\n    |diff| when sweep succeeds : "
              f"{np.median(np.abs(dxs[hit])):.2f}")
        print(f"    |diff| when sweep fails    : "
              f"{np.median(np.abs(dxs[~hit])):.2f}")

    print("\n" + "=" * 84)
    c_pred = corr(dxs, preds)
    agree = float((np.abs(dxs) < 0.5).mean())
    if agree > 0.6:
        print("  The sweep recovers the physical sampling phase after all.")
        print("  Then the estimator is the only weak link, and an analytical")
        print("  phase fit over several harmonics is the next thing to try.")
    elif not np.isnan(c_pred) and abs(c_pred) > 0.5:
        print("  CASE 1: the offset tracks the shear prediction. The sweep is")
        print("  absorbing search-side geometry, not sampling phase. No")
        print("  learned estimator is needed -- shear is already measurable")
        print("  from the search image to about 0.2 px, so the offset follows")
        print("  analytically from a first-pass position estimate.")
    else:
        print("  CASE 2 or 3: the offset is neither the physical phase nor")
        print("  the shear prediction. It is a search-conditioned template")
        print("  registration parameter, so it cannot be predicted from the")
        print("  reference alone and any estimator must see both images.")
        print("  Check the correlations above for what it does track.")
    print("=" * 84)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
