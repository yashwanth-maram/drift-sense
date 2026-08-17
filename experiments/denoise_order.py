#!/usr/bin/env python3
"""Does the ORDER of noise and reduction matter?

The canvas template beats the reference-derived one by +8.3 points on
DRAM and +15 on FinFET, fixing 18 cases across two 60-sample runs and
breaking 4. But the median score at the TRUE position barely moves:
+0.0006 and +0.0117.

So the canvas template does not win by scoring higher where it should. It
wins by scoring LOWER where it should not. That points at a specific
hypothesis:

    reference construction creates candidate-dependent artifacts

rather than simply depressing the true correlation.

A candidate mechanism is the order of operations. We area-average a NOISY
reference by 10. The search image was area-averaged BEFORE its noise was
applied. Averaging a hundred noisy pixels leaves residual structure that
is correlated across the template in a way the search's noise is not, and
correlated nuisance is exactly what produces spurious high scores at wrong
sites.

Four controlled variants isolate it:

    A  current      high-res noise, no denoise, downsample
    B  denoised     high-res noise, denoise, downsample     <-- the test
    C  clean        no high-res noise, downsample           <-- oracle
    D  search-like  no high-res noise, downsample, then low-res noise

A -> B is the question. If B approaches C, the mechanism is identified and
it has an inference-time implementation, since B needs nothing but the
reference image.

Simple filters first, deliberately. A strong denoiser could improve
matching by changing the image in ways unrelated to the hypothesis, which
would be a new confound rather than an answer.

Two quantities are reported, because the canvas result says they differ:
the score at the true position, and the score of the best wrong candidate.

    python experiments/denoise_order.py --n 60
    python experiments/denoise_order.py --n 60 --kind finfet
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from driftsense.generator.sample import (generate_sample, build_params,
                                         REFERENCE_PX, SCALE)
from driftsense.generator import sem

T = 100
TOL = 5.0


def downsample(img):
    return cv2.resize(img, (T, T), interpolation=cv2.INTER_AREA)


def variant_A(ref_img):
    """What we ship: downsample the noisy reference."""
    return downsample(ref_img)


def variant_B(ref_img, mode="gauss", strength=1.0):
    """Denoise at the reference's own resolution, THEN downsample.

    At 1 nm/px the lattice is 40-64 px wide and heavily oversampled, so
    smoothing at a scale of a few pixels costs almost no real structure
    while removing the high-resolution capture noise.
    """
    if mode == "gauss":
        k = int(2 * round(strength) + 1)
        d = cv2.GaussianBlur(ref_img, (k, k), strength)
    elif mode == "bilateral":
        d = cv2.bilateralFilter(ref_img, 5, 40 * strength, 3 * strength)
    elif mode == "median":
        d = cv2.medianBlur(ref_img, int(2 * round(strength) + 1))
    else:
        d = ref_img
    return downsample(d)


def clean_reference(index, kind, seed, params, gt_x, gt_y):
    """Variants C and D: the reference crop with NO capture noise.

    Rebuilds the geometry stream, re-renders the canvas, applies the
    shared beam only, and cuts the same crop.
    """
    from driftsense.generator.patterns.zones import generate_zone_canvas
    from driftsense.generator.sample import FINE_CANVAS_PX

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
    x0 = int(round((gt_x - T / 2) * SCALE))
    y0 = int(round((gt_y - T / 2) * SCALE))
    x0 = int(np.clip(x0, 0, c.shape[1] - REFERENCE_PX))
    y0 = int(np.clip(y0, 0, c.shape[0] - REFERENCE_PX))
    return c[y0:y0 + REFERENCE_PX, x0:x0 + REFERENCE_PX]


def grid_aligned(index, kind, seed, params, gx, gy):
    """Clean canvas downsampled on the SEARCH grid, then cropped."""
    from driftsense.generator.patterns.zones import generate_zone_canvas
    from driftsense.generator.sample import FINE_CANVAS_PX

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


def evaluate(tmpl, search, gx, gy, tol):
    s = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
    k = int(np.argmax(s))
    py, px = np.unravel_index(k, s.shape)
    err = float(np.hypot(px + T / 2 - gx, py + T / 2 - gy))

    ty = int(np.clip(round(gy - T / 2), 0, s.shape[0] - 1))
    tx = int(np.clip(round(gx - T / 2), 0, s.shape[1] - 1))
    z_true = float(s[ty, tx])

    m = s.copy()
    y0, y1 = max(ty - 20, 0), min(ty + 21, s.shape[0])
    x0, x1 = max(tx - 20, 0), min(tx + 21, s.shape[1])
    m[y0:y1, x0:x1] = -2.0
    z_imp = float(m.max())
    return err, z_true, z_imp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--kind", default="dram", choices=["dram", "finfet"])
    ap.add_argument("--seed", type=int, default=3131)
    ap.add_argument("--level", default="medium")
    ap.add_argument("--mode", default="gauss",
                    choices=["gauss", "bilateral", "median"])
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=TOL)
    a = ap.parse_args(argv)

    p = build_params(noise_level=a.level)
    names = ["A current", "B denoised", "C cropfirst", "E gridalign"]
    res = {n: {"err": [], "zt": [], "zi": []} for n in names}

    print("=" * 78)
    print(f"DENOISE ORDER   kind={a.kind}  n={a.n}  filter={a.mode}"
          f"  strength={a.strength}")
    print("=" * 78)
    print("\n  A current      noisy reference, downsample")
    print("  B denoised     noisy reference, denoise, downsample")
    print("  C clean        no reference noise, downsample")
    print("  D search-like  no reference noise, downsample, then noise\n")
    print(f"  {'id':>4}" + "".join(f"{n.split()[0]:>9}" for n in names)
          + f"{'B fixes':>9}")
    print("  " + "-" * 52)

    rng = np.random.default_rng(0)
    for i in range(a.n):
        s = generate_sample(i, a.kind, base_seed=a.seed, params=p)
        clean = clean_reference(i, a.kind, a.seed, p, s.gt_x, s.gt_y)

        tmpls = {
            "A current": variant_A(s.reference),
            "B denoised": variant_B(s.reference, a.mode, a.strength),
            # C: crop the clean canvas, THEN downsample -- the template
            # grid is offset from the search grid by x0 mod 10
            "C cropfirst": downsample(clean),
            # E: downsample the WHOLE canvas, THEN crop -- the template
            # lands exactly on the search's pixel grid. This is what
            # canvas_for did, and the difference C vs E isolates grid
            # alignment from clean pixels.
            "E gridalign": grid_aligned(i, a.kind, a.seed, p,
                                        s.gt_x, s.gt_y),
        }
        errs = {}
        for n, t in tmpls.items():
            if t.shape != (T, T):
                continue
            e, zt, zi = evaluate(t, s.search, s.gt_x, s.gt_y, a.tol)
            res[n]["err"].append(e)
            res[n]["zt"].append(zt)
            res[n]["zi"].append(zi)
            errs[n] = e
        if len(errs) < 4:
            continue
        mark = "  YES" if errs["A current"] > a.tol >= errs["B denoised"] else ""
        print(f"  {i:>4}" + "".join(f"{errs[n]:>9.1f}" for n in names) + mark)

    print(f"\n  {'variant':<16}{'accuracy':>10}{'median err':>12}"
          f"{'z true':>9}{'z impostor':>12}{'margin':>9}")
    print("  " + "-" * 68)
    for n in names:
        if not res[n]["err"]:
            continue
        e = np.asarray(res[n]["err"])
        zt = np.median(res[n]["zt"])
        zi = np.median(res[n]["zi"])
        print(f"  {n:<16}{float((e <= a.tol).mean()) * 100:>9.1f}%"
              f"{np.median(e):>12.2f}{zt:>9.4f}{zi:>12.4f}{zt - zi:>+9.4f}")

    eA = np.asarray(res["A current"]["err"])
    eB = np.asarray(res["B denoised"]["err"])
    eC = np.asarray(res["E gridalign"]["err"])
    accA, accB, accC = [float((x <= a.tol).mean()) for x in (eA, eB, eC)]
    fixed = int(((eA > a.tol) & (eB <= a.tol)).sum())
    broke = int(((eA <= a.tol) & (eB > a.tol)).sum())

    print(f"\n  B fixes {fixed}, breaks {broke}, net {(fixed - broke) / len(eA) * 100:+.1f} pp")
    print(f"  impostor score  A {np.median(res['A current']['zi']):.4f}"
          f"  ->  C {np.median(res['C cropfirst']['zi']):.4f}"
          f"  ->  E {np.median(res['E gridalign']['zi']):.4f}")

    print("\n" + "=" * 78)
    gap = accC - accA
    got = accB - accA
    if gap > 0.02 and got > 0.5 * gap:
        print(f"  Denoising before downsampling recovers {got / gap * 100:.0f}%"
              f" of the clean-template advantage.")
        print("  The mechanism is identified and it needs only the reference")
        print("  image, so it works at inference. Tune the filter, then")
        print("  benchmark on Applied Materials' set.")
    elif gap > 0.02:
        print(f"  The clean template is worth {gap * 100:+.1f} pp but denoising")
        print(f"  recovers only {got * 100:+.1f}. The advantage comes from some")
        print("  other difference between the reference and search paths --")
        print("  compare the remaining ones (jitter, vignette, barrel).")
    else:
        print("  The clean template holds no advantage at this setting, so")
        print("  there is nothing here to recover. Check whether the earlier")
        print("  canvas gain reproduces at this seed and noise level.")
    print("=" * 78)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
