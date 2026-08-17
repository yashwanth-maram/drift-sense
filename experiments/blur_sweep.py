#!/usr/bin/env python3
"""Does low-passing both images recover accuracy under heavy noise?

The shipped method degrades sharply with noise:

    low 100.0%   medium 87.5%   high 60.0%   severe 35.0%   extreme 17.5%

and the collapse is concentrated in strip-free references -- 12.5% at
high, 0.0% at severe, against 90%+ for references that a peripheral strip
pins on at least one axis.

There is a structural reason to expect this. The resampling sweep works
because the true site is the one place where an EXACT match exists at some
sampling offset, while wrong sites are approximate at every offset. Heavy
noise destroys exactness, so the mechanism loses the thing it exploits.

That suggests the opposite treatment from everything tried so far. If fine
detail is corrupted, matching on deliberately blurred images should retain
the coarse structure that still carries location information, at the cost
of precision that heavy noise has already taken away.

This contradicts an earlier oracle result -- removing noise entirely fixed
0 of 10 failures -- but that was measured at MEDIUM, where noise is
demonstrably not the bottleneck. At high and severe, accuracy tracks noise
level monotonically, so the regime is different and the oracle does not
carry over.

Both images are blurred identically, so the comparison stays fair. Blur is
applied AFTER the template is built, so the resampling mechanism is
untouched.

    python experiments/blur_sweep.py --n 40 --level high
    python experiments/blur_sweep.py --n 40 --level severe --kind finfet
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

from localize import template_at_offset, _subpixel, SCALE, OFFSET_STEP
from driftsense.localize import _strip_centres

T = 100
TOL = 5.0
SIGMAS = (0.0, 0.6, 1.0, 1.5, 2.5, 4.0)


def blur(img, sigma):
    if sigma <= 0:
        return img
    k = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (k, k), sigma)


def locate_blurred(reference, search, sigma, step=OFFSET_STEP, scale=SCALE):
    """The shipped operator, with both images low-passed identically.

    The blur is applied to the template after construction, so the
    resampling family itself is unchanged.
    """
    h, w = search.shape
    size = max(int(round(min(h, w) / scale)), 8)
    resample = max(reference.shape[0] / float(size), 1.0)
    phase_max = max(int(round(resample)), 1)
    s_blur = blur(search, sigma)

    best = None
    for py in range(0, phase_max, step):
        for px in range(0, phase_max, step):
            try:
                t = template_at_offset(reference, float(px), float(py),
                                       size, resample)
                t = blur(t, sigma)
                s = cv2.matchTemplate(s_blur, t, cv2.TM_CCOEFF_NORMED)
            except Exception:
                continue
            v = float(s.max())
            if best is None or v > best[0]:
                k = int(np.argmax(s))
                yy, xx = np.unravel_index(k, s.shape)
                best = (v, int(xx), int(yy), s)
    if best is None:
        return None
    score, px_, py_, surf = best
    dy, dx = _subpixel(surf, py_, px_)
    m = size * 0.5
    return (float(np.clip(px_ + dx + size / 2.0, m, w - m)),
            float(np.clip(py_ + dy + size / 2.0, m, h - m)),
            score)


def condition(ref):
    t = cv2.resize(ref, (T, T), interpolation=cv2.INTER_AREA)
    return (len(_strip_centres(t, 1, 0.45, 6)) > 0) \
        + (len(_strip_centres(t, 0, 0.45, 6)) > 0)


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
    out = []
    for i in range(n):
        s = generate_sample(i, kind, base_seed=seed, params=p)
        out.append((i, s.reference, s.search, s.gt_x, s.gt_y))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--kind", default="dram", choices=["dram", "finfet"])
    ap.add_argument("--seed", type=int, default=3131)
    ap.add_argument("--level", default="high")
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--sigmas", nargs="+", type=float, default=list(SIGMAS))
    a = ap.parse_args(argv)

    pairs = load(a.data, a.n) if a.data else \
        generate(a.n, a.kind, a.seed, a.level)
    src = a.data if a.data else f"generated {a.kind}, {a.level}"

    print("=" * 74)
    print(f"BLUR SWEEP   {src}   n={len(pairs)}")
    print("=" * 74)
    print("\n  Both images low-passed identically before matching. sigma 0")
    print("  is the shipped method unchanged.\n")

    conds = np.array([condition(r[1]) for r in pairs])
    results = {}

    for sg in a.sigmas:
        errs, times = [], []
        t0 = time.time()
        for _, ref, srch, gx, gy in pairs:
            r = locate_blurred(ref, srch, sg)
            if r is None:
                errs.append(1e9)
                continue
            errs.append(float(np.hypot(r[0] - gx, r[1] - gy)))
        times.append((time.time() - t0) / max(len(pairs), 1) * 1000)
        e = np.asarray(errs)
        results[sg] = e
        acc = float((e <= a.tol).mean()) * 100
        line = f"  sigma {sg:4.1f} :  {acc:6.1f}%   median {np.median(e):8.2f} px"
        for c, nm in ((2, "both"), (1, "one"), (0, "none")):
            m = conds == c
            if m.sum():
                line += f"   {nm} {float((e[m] <= a.tol).mean()) * 100:5.1f}%"
        print(line, flush=True)

    base = results[a.sigmas[0]]
    best_sg = max(results, key=lambda s: float((results[s] <= a.tol).mean()))
    b0 = float((base <= a.tol).mean()) * 100
    b1 = float((results[best_sg] <= a.tol).mean()) * 100

    print(f"\n  baseline (sigma {a.sigmas[0]:.1f}) : {b0:.1f}%")
    print(f"  best     (sigma {best_sg:.1f}) : {b1:.1f}%")
    fixed = int(((base > a.tol) & (results[best_sg] <= a.tol)).sum())
    broke = int(((base <= a.tol) & (results[best_sg] > a.tol)).sum())
    print(f"  fixed {fixed}, broke {broke}, net {b1 - b0:+.1f} pp")

    print("\n" + "=" * 74)
    n = len(pairs)
    se = float(np.sqrt(max(fixed + broke, 1)) / n) * 100
    if b1 - b0 > 2 * se:
        print(f"  Blurring helps by {b1 - b0:+.1f} points at sigma {best_sg:.1f},")
        print(f"  more than two standard errors ({se:.1f}). Worth adding as a")
        print("  fallback when the confidence score is low -- but validate on")
        print("  the reference generator's data before shipping, since six")
        print("  earlier gains measured on synthetic data did not transfer.")
    elif b1 - b0 > 0:
        print(f"  Any gain ({b1 - b0:+.1f} pp) is within noise at this sample")
        print("  size. Re-run at larger n before drawing a conclusion.")
    else:
        print("  Blurring does not help. The loss under heavy noise is not")
        print("  about fine detail being corrupted, and this line closes.")
    print("=" * 74)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
