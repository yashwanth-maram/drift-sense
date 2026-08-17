#!/usr/bin/env python3
"""Is there anything inside a mat that repetition does not explain?

Two proposed architectures -- periodic-subtraction and phase-locked
demodulation -- both rest on the same premise: that after removing the
repeating lattice, deterministic site-specific structure remains. Neither
works if a mat interior is exactly periodic.

The premise has not been tested on Applied Materials' data. The evidence
for it came from canvas_ambiguity.py, which runs on OUR generator, and
ours adds line-edge roughness (ler_sigma_nm = 2.0) that theirs does not.
Their collapse defects fire only when a gap falls under 10 nm, and their
dram_1x gaps are 32 nm, so defects likely never fire either.

This measures it directly. Take a mat interior, compare a patch against
the same patch shifted by exactly one lattice period, and compare that
difference against the image's own noise floor.

    difference ~ noise floor  -> exactly periodic. Nothing distinguishes
                                 one period from the next. Residual
                                 methods have nothing to work with and
                                 the failures are unsolvable.

    difference >> noise floor -> aperiodic structure exists. Residual
                                 methods are worth building.

    python experiments/lattice_selfsim.py --data ../am_eval --n 20
    python experiments/lattice_selfsim.py --generator --n 20
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

PATCH = 64


def noise_floor(img, tile=8):
    """Robust per-pixel noise sigma from second differences in flat areas."""
    a = img.astype(np.float32)
    d = a[:, :-2] - 2.0 * a[:, 1:-1] + a[:, 2:]
    g = np.abs(a[:, 2:] - a[:, :-2])
    flat = d[g <= np.quantile(g, 0.25)]
    if len(flat) < 100:
        return float(np.std(d) / np.sqrt(6.0))
    return float(np.median(np.abs(flat)) / (0.6745 * np.sqrt(6.0)))


def dominant_pitch(patch, pmin=3.0, pmax=30.0):
    a = patch.astype(np.float32)
    a = a - a.mean(axis=1, keepdims=True)
    n = a.shape[1]
    spec = np.abs(np.fft.rfft(a * np.hanning(n)[None, :], axis=1)).mean(axis=0)
    kmin, kmax = max(int(n / pmax), 2), min(int(n / pmin), len(spec) - 1)
    if kmax <= kmin:
        return None
    k = int(np.argmax(spec[kmin:kmax + 1])) + kmin
    return n / k


def strip_columns(img):
    a = img.astype(np.float32)
    prof = np.convolve(a.std(axis=0), np.ones(9) / 9, mode="same")
    thr = prof.min() + 0.35 * (prof.max() - prof.min())
    low = prof < thr
    out, s = [], None
    for i, v in enumerate(low):
        if v and s is None:
            s = i
        elif not v and s is not None:
            if i - s >= 12:
                out.append((s, i))
            s = None
    return out


def interior_point(img, patch=PATCH):
    """A location well inside a mat, away from any strip."""
    v = strip_columns(img)
    h = strip_columns(img.T)
    h_, w_ = img.shape

    def ok(c, bands, limit):
        if c - patch < 0 or c + 2 * patch >= limit:
            return False
        return all(not (lo - patch < c < hi + patch) for lo, hi in bands)

    for cx in range(patch, w_ - 3 * patch, 7):
        if not ok(cx, v, w_):
            continue
        for cy in range(patch, h_ - 3 * patch, 7):
            if ok(cy, h, h_):
                return cx, cy
    return None


def measure(img, label):
    pt = interior_point(img)
    if pt is None:
        return None
    cx, cy = pt
    base = img[cy:cy + PATCH, cx:cx + PATCH]
    p = dominant_pitch(base)
    if p is None or p < 3:
        return None

    sigma = noise_floor(img)
    # a difference between two independent noisy patches has sqrt(2)*sigma
    expected = sigma * np.sqrt(2.0)

    rows = []
    for k in (1, 2, 3):
        dx = int(round(k * p))
        if cx + dx + PATCH >= img.shape[1]:
            break
        shifted = img[cy:cy + PATCH, cx + dx:cx + dx + PATCH]
        d = base.astype(np.float64) - shifted.astype(np.float64)
        rmse = float(np.sqrt((d ** 2).mean()))
        rows.append((k, dx, rmse))

    return {"label": label, "pitch": p, "sigma": sigma,
            "expected": expected, "rows": rows}


def report(res):
    if res is None:
        print("    could not find a clean mat interior")
        return None
    print(f"    pitch {res['pitch']:.2f} px   noise sigma {res['sigma']:.2f}"
          f"   expected diff if identical {res['expected']:.2f}")
    ratios = []
    for k, dx, rmse in res["rows"]:
        ratio = rmse / max(res["expected"], 1e-6)
        ratios.append(ratio)
        verdict = "periodic" if ratio < 1.3 else \
            ("some structure" if ratio < 2.0 else "STRUCTURE")
        print(f"      shift {k} period ({dx:>3} px): rmse {rmse:6.2f}"
              f"   ratio {ratio:5.2f}   {verdict}")
    return float(np.median(ratios)) if ratios else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data")
    ap.add_argument("--generator", action="store_true")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--kind", default="dram")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args(argv)

    print("=" * 72)
    print("LATTICE SELF-SIMILARITY")
    print("=" * 72)
    print("\n  Comparing a mat-interior patch against itself shifted by whole")
    print("  lattice periods. Ratio near 1.0 means the two are identical up")
    print("  to noise -- exactly periodic, nothing to distinguish them.\n")

    all_ratios = []

    if a.data:
        root = None
        for r, _, files in os.walk(a.data):
            if "manifest.csv" in files:
                root = r
                rows = list(csv.DictReader(open(os.path.join(r, "manifest.csv"))))
                break
        if root is None:
            raise SystemExit(f"no manifest.csv under {a.data}")
        for row in rows[:a.n]:
            i = int(row["id"])
            sp = os.path.join(root, "search", f"{i:05d}.png")
            img = cv2.imread(sp, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            print(f"  [{i}] {os.path.basename(sp)}")
            r = report(measure(img, str(i)))
            if r:
                all_ratios.append(r)

    if a.generator or not a.data:
        from driftsense.generator.sample import generate_sample, build_params
        p = build_params(noise_level="medium")
        for i in range(min(a.n, 10)):
            s = generate_sample(i, a.kind, base_seed=a.seed, params=p)
            print(f"  [gen {i}]")
            r = report(measure(s.search, f"gen{i}"))
            if r:
                all_ratios.append(r)

    if not all_ratios:
        print("\n  nothing measured")
        return 1

    med = float(np.median(all_ratios))
    print("\n" + "=" * 72)
    print(f"  median ratio across {len(all_ratios)} images : {med:.2f}")
    print("=" * 72)
    if med < 1.3:
        print("  VERDICT: mat interiors are EXACTLY PERIODIC.")
        print("  One period is indistinguishable from the next. Periodic")
        print("  subtraction leaves noise, 1-D collapse gives a constant,")
        print("  and an informativeness map has nothing to weight. The")
        print("  mat-interior failures cannot be solved from the pixels, and")
        print("  the confidence score is the correct deliverable.")
    elif med < 2.0:
        print("  VERDICT: weak aperiodic structure. Residual methods may")
        print("  work but the margin will be thin. Measure carefully before")
        print("  committing.")
    else:
        print("  VERDICT: substantial aperiodic structure. Periodic")
        print("  subtraction and phase-locked demodulation are worth")
        print("  building -- the signal they need is present.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
