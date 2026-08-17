#!/usr/bin/env python3
"""Do collapsed lattice lines carry any signal?

Every residual-matching stage in the proposed pipeline rests on one
assumption: that after the periodic component is removed, what remains
varies from one lattice line to the next. If consecutive lines are
identical, the residual is a constant vector and there is nothing to
match.

There is a specific reason to expect variation. Lines are drawn at exact
nanometre positions and sampled onto a 10 nm grid. At a 64 nm pitch,
consecutive lines land at sub-pixel offsets 0, 0.4, 0.8, 0.2, 0.6 --
repeating every five. Each line therefore renders slightly differently,
and the pattern only truly repeats after the SUPER-PERIOD, which we
measured at 16-43 px rather than the 5-8 px lattice pitch.

That distinction is load-bearing. Folding the periodic model at the
lattice pitch averages away exactly the sub-pixel variation that carries
the signal. Folding at the super-period preserves it.

This measures:

  1. the per-line collapsed envelope, and how much it varies
  2. its autocorrelation, to find the true repeat length
  3. whether the variation exceeds what noise alone would produce
  4. how many distinct line classes exist before the pattern repeats

    python experiments/line_envelope.py --data ../am_eval --n 10
    python experiments/line_envelope.py --generator --n 6
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

BAND = 96


def noise_sigma(img):
    a = img.astype(np.float32)
    d = a[:, :-2] - 2.0 * a[:, 1:-1] + a[:, 2:]
    g = np.abs(a[:, 2:] - a[:, :-2])
    flat = d[g <= np.quantile(g, 0.25)]
    if len(flat) < 100:
        return float(np.std(d) / np.sqrt(6.0))
    return float(np.median(np.abs(flat)) / (0.6745 * np.sqrt(6.0)))


def strip_bands(img, axis):
    a = img.astype(np.float32)
    prof = np.convolve(a.std(axis=0 if axis == 1 else 1),
                       np.ones(9) / 9, mode="same")
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


def interior_window(img, size=260):
    """Largest clean mat interior we can find, as (x0, y0, w, h)."""
    v, h = strip_bands(img, 1), strip_bands(img, 0)
    H, W = img.shape

    def gaps(bands, limit):
        out, prev = [], 0
        for lo, hi in bands:
            if lo - prev > 40:
                out.append((prev, lo))
            prev = hi
        if limit - prev > 40:
            out.append((prev, limit))
        return out

    gx, gy = gaps(v, W), gaps(h, H)
    if not gx or not gy:
        return None
    bx = max(gx, key=lambda t: t[1] - t[0])
    by = max(gy, key=lambda t: t[1] - t[0])
    x0, x1 = bx[0] + 6, min(bx[1] - 6, bx[0] + 6 + size)
    y0, y1 = by[0] + 6, min(by[1] - 6, by[0] + 6 + size)
    if x1 - x0 < 60 or y1 - y0 < 60:
        return None
    return x0, y0, x1 - x0, y1 - y0


def line_envelope(patch, axis=1):
    """Collapse each column (axis=1) or row to a scalar, then find the
    lattice positions and sample the envelope at them."""
    a = patch.astype(np.float64)
    prof = a.mean(axis=0) if axis == 1 else a.mean(axis=1)
    prof = prof - prof.mean()

    n = len(prof)
    spec = np.abs(np.fft.rfft(prof * np.hanning(n)))
    kmin = max(int(n / 40), 2)
    kmax = min(int(n / 3), len(spec) - 1)
    if kmax <= kmin:
        return None
    k = int(np.argmax(spec[kmin:kmax + 1])) + kmin
    pitch = n / k
    if pitch < 3:
        return None

    # peak position of each line, and its collapsed amplitude
    centres, amps = [], []
    for i in range(int(n / pitch)):
        lo = int(round(i * pitch))
        hi = int(round((i + 1) * pitch))
        if hi - lo < 2 or hi > n:
            break
        seg = prof[lo:hi]
        centres.append(lo + int(np.argmax(seg)))
        amps.append(float(seg.max() - seg.min()))
    if len(amps) < 6:
        return None
    return pitch, np.asarray(amps), np.asarray(centres)


def repeat_length(amps):
    """Autocorrelation of the envelope: after how many lines does it
    repeat?"""
    x = amps - amps.mean()
    if np.allclose(x, 0):
        return None, 0.0
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / max(ac[0], 1e-9)
    best_k, best_v = None, -2.0
    for k in range(2, min(len(ac), len(x) // 2 + 1)):
        if ac[k] > best_v:
            best_v, best_k = ac[k], k
    return best_k, float(best_v)


def analyse(img, label):
    win = interior_window(img)
    if win is None:
        print(f"    {label}: no clean mat interior")
        return None
    x0, y0, w, h = win
    patch = img[y0:y0 + h, x0:x0 + w]
    sigma = noise_sigma(img)

    res = line_envelope(patch, axis=1)
    if res is None:
        print(f"    {label}: could not resolve lines")
        return None
    pitch, amps, centres = res

    # variation between lines, against what noise alone would give
    spread = float(amps.std())
    n_per_line = patch.shape[0] * max(pitch, 1)
    noise_spread = sigma / np.sqrt(max(n_per_line, 1))
    ratio = spread / max(noise_spread, 1e-6)

    k, acv = repeat_length(amps)
    print(f"    {label}: pitch {pitch:5.2f} px, {len(amps):>3} lines"
          f"   envelope sd {spread:6.3f}  noise sd {noise_spread:6.3f}"
          f"   ratio {ratio:6.2f}")
    print(f"        first 12 amplitudes: "
          + " ".join(f"{v:5.1f}" for v in amps[:12]))
    if k:
        print(f"        envelope repeats every {k} lines"
              f"  (autocorr {acv:+.2f})")
    return ratio, k


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data")
    ap.add_argument("--generator", action="store_true")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--kind", default="dram")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args(argv)

    print("=" * 74)
    print("LINE ENVELOPE")
    print("=" * 74)
    print("\n  Collapsing each lattice line to a scalar. If consecutive lines")
    print("  differ by more than noise, residual matching has something to")
    print("  work with. If they are identical, it does not.\n")

    ratios, ks = [], []

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
            img = cv2.imread(os.path.join(root, "search", f"{i:05d}.png"),
                             cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            r = analyse(img, f"[{i}]")
            if r:
                ratios.append(r[0])
                if r[1]:
                    ks.append(r[1])

    if a.generator or not a.data:
        from driftsense.generator.sample import generate_sample, build_params
        p = build_params(noise_level="medium")
        for i in range(min(a.n, 8)):
            s = generate_sample(i, a.kind, base_seed=a.seed, params=p)
            r = analyse(s.search, f"[gen {i}]")
            if r:
                ratios.append(r[0])
                if r[1]:
                    ks.append(r[1])

    if not ratios:
        print("\n  nothing measured")
        return 1

    med = float(np.median(ratios))
    print("\n" + "=" * 74)
    print(f"  median envelope-to-noise ratio : {med:.2f}")
    if ks:
        print(f"  median repeat length           : {int(np.median(ks))} lines")
    print("=" * 74)
    if med > 3.0:
        print("  VERDICT: consecutive lines differ well beyond noise. The")
        print("  residual carries real structure and the pipeline's stages")
        print("  4-8 are worth building. Fold the periodic model at the")
        print("  REPEAT LENGTH above, not at the lattice pitch -- folding at")
        print("  the pitch averages away exactly this variation.")
    elif med > 1.5:
        print("  VERDICT: weak but present. Buildable, thin margin. Measure")
        print("  top-1 inclusion on the known failures before committing.")
    else:
        print("  VERDICT: consecutive lines are identical up to noise. The")
        print("  collapsed residual is a constant vector. Stages 4-8 have")
        print("  nothing to match and should be dropped.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
