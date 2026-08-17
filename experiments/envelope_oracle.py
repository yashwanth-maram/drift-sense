#!/usr/bin/env python3
"""Can the line envelope fix the axis that 2-D correlation gets wrong?

Measured across both generators, consecutive lattice lines differ by
5-13x the noise level, because lines drawn at exact nanometre positions
land on a 10 nm sampling grid at steadily shifting sub-pixel offsets. That
variation is real and large. Two-dimensional correlation destroys it by
averaging over 10,000 pixels.

Collapsing a band of the image to a 1-D profile keeps it. The periodic
lattice contributes a constant to every position and vanishes; what
survives is the envelope.

This tests the idea as an ORACLE. It is given the correct band on the
pinned axis -- which the matcher already gets right, to within 0.3 px in
the measured failures -- and asked to find the free axis using the 1-D
envelope alone. That isolates the question:

    does the envelope contain enough information to place the free axis?

Comparisons reported for each failing axis:

    2-D ZNCC        what we do now
    1-D envelope    the proposed replacement for that axis
    rank of truth   under each, since a method can be better on average
                    while still not putting the right answer first

    python experiments/envelope_oracle.py --data ../am_eval --n 100
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
T = 100


def profile(img, axis):
    """Collapse to 1-D. axis=1 keeps the x extent, axis=0 keeps y."""
    a = img.astype(np.float32)
    return a.mean(axis=0) if axis == 1 else a.mean(axis=1)


def amplitude_envelope(prof, pitch=None):
    """Per-line amplitude sequence, and the pixel position of each line.

    Matching the RAW collapsed profile does not work: it still contains the
    full periodic lattice, so 1-D matching on it is ambiguous at the pitch
    by construction. What carries the information is how each line's
    amplitude differs from its neighbours, which is what this extracts.
    """
    x = np.asarray(prof, np.float64)
    x = x - x.mean()
    n = len(x)
    if n < 24:
        return None, None
    if pitch is None:
        spec = np.abs(np.fft.rfft(x * np.hanning(n)))
        kmin, kmax = max(int(n / 40), 2), min(int(n / 3), len(spec) - 1)
        if kmax <= kmin:
            return None, None
        k = int(np.argmax(spec[kmin:kmax + 1])) + kmin
        pitch = n / k
    if pitch < 2.5:
        return None, None

    amps, pos = [], []
    i = 0.0
    while i + pitch <= n:
        lo, hi = int(round(i)), int(round(i + pitch))
        if hi - lo < 2:
            break
        seg = x[lo:hi]
        amps.append(float(seg.max() - seg.min()))
        pos.append((lo + hi) / 2.0)
        i += pitch
    if len(amps) < 5:
        return None, None
    return np.asarray(amps), np.asarray(pos), pitch


def match_1d(long_prof, short_prof):
    """Normalised cross-correlation of a short profile against a long one."""
    a = long_prof.reshape(1, -1).astype(np.float32)
    b = short_prof.reshape(1, -1).astype(np.float32)
    if b.shape[1] >= a.shape[1]:
        return None
    return cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED).ravel()


def rank_of(surface, idx, sep=6):
    """How many distinct peaks outscore the value at idx."""
    if surface is None or not (0 <= idx < len(surface)):
        return None
    v = surface[idx]
    better, i = 0, 0
    order = np.argsort(surface)[::-1]
    taken = []
    for j in order:
        if surface[j] <= v:
            break
        if all(abs(j - t) > sep for t in taken):
            taken.append(j)
            better += 1
        i += 1
        if i > 4000:
            break
    return better


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
    print("ENVELOPE ORACLE")
    print("=" * 76)
    print("\n  Given the correct band on the pinned axis, can the 1-D line")
    print("  envelope place the free axis that 2-D correlation misses?\n")
    print(f"  {'id':>4} {'axis':>5} {'2D err':>9} {'1D err':>9}"
          f" {'2D rank':>8} {'1D rank':>8}   outcome")
    print("  " + "-" * 66)

    fixed = broken = same = 0
    e2, e1 = [], []
    r2, r1 = [], []

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

        free = 1 if abs(dx) >= abs(dy) else 0
        err2 = abs(dx) if free == 1 else abs(dy)

        tmpl = cv2.resize(ref, (T, T), interpolation=cv2.INTER_AREA)

        # oracle band on the PINNED axis, taken from ground truth
        if free == 1:
            y0 = int(np.clip(round(gy - T / 2), 0, srch.shape[0] - T))
            band = srch[y0:y0 + T, :]
            long_p = profile(band, 1)
            short_p = profile(tmpl, 1)
            true_idx = int(round(gx - T / 2))
        else:
            x0 = int(np.clip(round(gx - T / 2), 0, srch.shape[1] - T))
            band = srch[:, x0:x0 + T]
            long_p = profile(band, 0)
            short_p = profile(tmpl, 0)
            true_idx = int(round(gy - T / 2))

        env_l = amplitude_envelope(long_p)
        env_s = amplitude_envelope(short_p, pitch=env_l[2] if env_l[0] is not None else None)
        if env_l[0] is None or env_s[0] is None or len(env_s[0]) >= len(env_l[0]):
            continue
        surf = match_1d(env_l[0], env_s[0])
        if surf is None or len(surf) == 0:
            continue
        best_line = int(np.argmax(surf))
        pred_px = float(env_l[1][best_line]) - float(env_s[1][0])
        err1 = abs(pred_px - true_idx)
        # rank is now over line indices, so convert the truth to one
        true_line = int(np.argmin(np.abs(env_l[1] - env_s[1][0] - true_idx)))

        # rank of the truth under 2-D correlation, for comparison
        full = cv2.matchTemplate(srch, tmpl, cv2.TM_CCOEFF_NORMED)
        line = full[int(np.clip(round(gy - T / 2), 0, full.shape[0] - 1)), :] \
            if free == 1 else \
            full[:, int(np.clip(round(gx - T / 2), 0, full.shape[1] - 1))]
        rk2 = rank_of(line, true_idx)
        rk1 = rank_of(surf, true_line, sep=1)

        e2.append(err2); e1.append(err1)
        if rk2 is not None:
            r2.append(rk2)
        if rk1 is not None:
            r1.append(rk1)

        if err1 <= a.tol:
            fixed += 1
            out = "FIXED"
        elif err1 < err2 * 0.5:
            same += 1
            out = "closer"
        else:
            broken += 1
            out = "still wrong"

        ax = "x" if free == 1 else "y"
        print(f"  {i:>4} {ax:>5} {err2:>9.1f} {err1:>9.1f}"
              f" {str(rk2):>8} {str(rk1):>8}   {out}")

    n = fixed + broken + same
    if not n:
        print("\n  no failures")
        return 0

    print(f"\n  failures analysed          : {n}")
    print(f"  fixed by 1-D envelope      : {fixed}  ({fixed / n * 100:.0f}%)")
    print(f"  closer but still wrong     : {same}")
    print(f"  no better                  : {broken}")
    print(f"\n  median error, 2-D          : {np.median(e2):7.1f} px")
    print(f"  median error, 1-D envelope : {np.median(e1):7.1f} px")
    if r2 and r1:
        print(f"  median rank of truth, 2-D  : {np.median(r2):7.1f}")
        print(f"  median rank of truth, 1-D  : {np.median(r1):7.1f}")

    print("\n" + "=" * 76)
    if fixed / n > 0.4:
        print("  VERDICT: the envelope carries enough to place the free axis.")
        print("  Build it -- but the oracle band came from ground truth, so a")
        print("  real implementation must first establish the pinned axis on")
        print("  its own, and that must be measured separately.")
    elif (fixed + same) / n > 0.4:
        print("  VERDICT: partial. The envelope moves the answer closer but")
        print("  does not usually land it. Worth combining with 2-D rather")
        print("  than replacing it.")
    else:
        print("  VERDICT: the envelope does not place the free axis, even")
        print("  with the other axis handed to it. The variation is real but")
        print("  not usable this way. Drop it.")
    print("=" * 76)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
