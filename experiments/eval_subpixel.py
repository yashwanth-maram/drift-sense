#!/usr/bin/env python3
"""Does exact sub-pixel template alignment recover the canvas advantage?

Measured at n=60 on both families: a clean canvas template cropped after
downsampling (grid-aligned) beats one cropped before (66.7 -> 75.0 on
DRAM, 80.0 -> 95.0 on FinFET), while a clean template and a noisy one
score identically. Grid alignment is the mechanism.

The canvas is not available at inference, so this tests three ways of
getting the alignment from the reference image alone:

    plain      INTER_AREA, what we ship
    estimated  one offset per axis, read from the reference's own lattice
               phase before matching
    sweep      all ten offsets per axis, best score wins
    oracle     offset computed from the known crop origin -- the bound
    E          clean canvas, grid-aligned -- the target

The sweep is reported but viewed with suspicion. A hundred candidate
templates is the same structure that cost fifteen points when forty-one
candidate scales were compared against five: every extra candidate is
another chance for a wrong position to find a flattering alignment. The
estimated variant avoids that, since the offset is a property of the
reference and not of any candidate.

    python experiments/eval_subpixel.py --n 60
    python experiments/eval_subpixel.py --n 60 --kind finfet
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from driftsense.generator.sample import (generate_sample, build_params,
                                         FINE_CANVAS_PX, SCALE)
from driftsense.generator import sem
from driftsense.generator.patterns.zones import generate_zone_canvas
from driftsense.subpixel_template import (template_at_offset, build_template,estimate_offset)

T = 100
TOL = 5.0


def grid_aligned(index, kind, seed, params, gx, gy):
    ss = np.random.SeedSequence([int(seed), int(index)])
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
    small = sem.area_downsample(c, SCALE)
    x = int(np.clip(round(gx - T / 2), 0, small.shape[1] - T))
    y = int(np.clip(round(gy - T / 2), 0, small.shape[0] - T))
    return small[y:y + T, x:x + T]


def locate(tmpl, search, gx, gy):
    s = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
    k = int(np.argmax(s))
    py, px = np.unravel_index(k, s.shape)
    err = float(np.hypot(px + T / 2 - gx, py + T / 2 - gy))
    ty = int(np.clip(round(gy - T / 2), 0, s.shape[0] - 1))
    tx = int(np.clip(round(gx - T / 2), 0, s.shape[1] - 1))
    zt = float(s[ty, tx])
    m = s.copy()
    y0, y1 = max(ty - 20, 0), min(ty + 21, s.shape[0])
    x0, x1 = max(tx - 20, 0), min(tx + 21, s.shape[1])
    m[y0:y1, x0:x1] = -2.0
    return err, zt, float(m.max())


def sweep_locate(reference, search, gx, gy, scale=SCALE, step=2):
    """Best over all fractional offsets."""
    best = None
    for py in range(0, scale, step):
        for px in range(0, scale, step):
            t = template_at_offset(reference, float(px), float(py), T, scale)
            s = cv2.matchTemplate(search, t, cv2.TM_CCOEFF_NORMED)
            v = float(s.max())
            if best is None or v > best[0]:
                k = int(np.argmax(s))
                yy, xx = np.unravel_index(k, s.shape)
                best = (v, xx, yy, s)
    _, xx, yy, s = best
    err = float(np.hypot(xx + T / 2 - gx, yy + T / 2 - gy))
    ty = int(np.clip(round(gy - T / 2), 0, s.shape[0] - 1))
    tx = int(np.clip(round(gx - T / 2), 0, s.shape[1] - 1))
    return err, float(s[ty, tx]), 0.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--kind", default="dram", choices=["dram", "finfet"])
    ap.add_argument("--seed", type=int, default=3131)
    ap.add_argument("--level", default="medium")
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--no-sweep", action="store_true")
    a = ap.parse_args(argv)

    p = build_params(noise_level=a.level)
    names = ["plain", "estimated", "oracle", "E gridalign"]
    if not a.no_sweep:
        names.insert(2, "sweep")
    res = {n: {"e": [], "zt": [], "zi": []} for n in names}
    times = {n: 0.0 for n in names}

    print("=" * 76)
    print(f"SUB-PIXEL TEMPLATE   kind={a.kind}  n={a.n}  level={a.level}")
    print("=" * 76)
    print("\n  plain      INTER_AREA, what we ship")
    print("  estimated  offset read from the reference's own lattice phase")
    if not a.no_sweep:
        print("  sweep      all fractional offsets, best score")
    print("  oracle     offset from the known crop origin -- the bound")
    print("  E          clean canvas, grid-aligned -- the target\n")
    print(f"  {'id':>4}" + "".join(f"{n[:9]:>11}" for n in names))
    print("  " + "-" * (4 + 11 * len(names)))

    for i in range(a.n):
        s = generate_sample(i, a.kind, base_seed=a.seed, params=p)
        row = {}

        t0 = time.time()
        e, zt, zi = locate(build_template(s.reference, "plain"),
                           s.search, s.gt_x, s.gt_y)
        times["plain"] += time.time() - t0
        res["plain"]["e"].append(e); res["plain"]["zt"].append(zt)
        res["plain"]["zi"].append(zi); row["plain"] = e

        t0 = time.time()
        e, zt, zi = locate(build_template(s.reference, "estimated"),
                           s.search, s.gt_x, s.gt_y)
        times["estimated"] += time.time() - t0
        res["estimated"]["e"].append(e); res["estimated"]["zt"].append(zt)
        res["estimated"]["zi"].append(zi); row["estimated"] = e

        if not a.no_sweep:
            t0 = time.time()
            e, zt, zi = sweep_locate(s.reference, s.search, s.gt_x, s.gt_y)
            times["sweep"] += time.time() - t0
            res["sweep"]["e"].append(e); res["sweep"]["zt"].append(zt)
            res["sweep"]["zi"].append(zi); row["sweep"] = e

        # oracle: the crop origin is known, so the exact offset is known
        x0 = int(round((s.gt_x - T / 2) * SCALE))
        y0 = int(round((s.gt_y - T / 2) * SCALE))
        t = template_at_offset(s.reference, float((-x0) % SCALE),
                               float((-y0) % SCALE), T, SCALE)
        e, zt, zi = locate(t, s.search, s.gt_x, s.gt_y)
        res["oracle"]["e"].append(e); res["oracle"]["zt"].append(zt)
        res["oracle"]["zi"].append(zi); row["oracle"] = e

        ge = grid_aligned(i, a.kind, a.seed, p, s.gt_x, s.gt_y)
        e, zt, zi = locate(ge, s.search, s.gt_x, s.gt_y)
        res["E gridalign"]["e"].append(e); res["E gridalign"]["zt"].append(zt)
        res["E gridalign"]["zi"].append(zi); row["E gridalign"] = e

        print(f"  {i:>4}" + "".join(f"{row[n]:>11.1f}" for n in names))

    print(f"\n  {'variant':<14}{'accuracy':>10}{'median err':>12}"
          f"{'z true':>9}{'z imp':>9}{'margin':>9}{'ms/pair':>10}")
    print("  " + "-" * 74)
    for n in names:
        e = np.asarray(res[n]["e"])
        if not e.size:
            continue
        zt, zi = np.median(res[n]["zt"]), np.median(res[n]["zi"])
        ms = times[n] / max(a.n, 1) * 1000 if times[n] else 0.0
        print(f"  {n:<14}{float((e <= a.tol).mean()) * 100:>9.1f}%"
              f"{np.median(e):>12.2f}{zt:>9.4f}{zi:>9.4f}{zt - zi:>+9.4f}"
              + (f"{ms:>10.0f}" if ms else f"{'':>10}"))

    ep = np.asarray(res["plain"]["e"])
    ee = np.asarray(res["estimated"]["e"])
    eg = np.asarray(res["E gridalign"]["e"])
    acc = lambda x: float((x <= a.tol).mean())
    fixed = int(((ep > a.tol) & (ee <= a.tol)).sum())
    broke = int(((ep <= a.tol) & (ee > a.tol)).sum())
    print(f"\n  estimated fixes {fixed}, breaks {broke},"
          f" net {(fixed - broke) / len(ep) * 100:+.1f} pp")

    gap = acc(eg) - acc(ep)
    got = acc(ee) - acc(ep)
    print("\n" + "=" * 76)
    if gap <= 0.02:
        print("  No grid-alignment advantage at this seed, so there is")
        print("  nothing to recover. Check that the canvas result")
        print("  reproduces here before reading anything into the rest.")
    elif got > 0.6 * gap:
        print(f"  Estimating the offset from the reference alone recovers"
              f" {got / gap * 100:.0f}%")
        print("  of the grid-alignment advantage, using nothing unavailable")
        print("  at inference. Benchmark on Applied Materials' set next.")
    elif got > 0.2 * gap:
        print(f"  Partial: {got / gap * 100:.0f}% of the advantage recovered."
              f" The offset")
        print("  estimator is the weak link, not the mechanism. Compare")
        print("  against the oracle row to see how much better it could get.")
    else:
        print("  The mechanism is real but the offset cannot be estimated")
        print("  from the reference alone. Compare the oracle row: if the")
        print("  oracle reaches E, the estimator needs work; if it does not,")
        print("  exact resampling is not the whole story.")
    print("=" * 76)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
