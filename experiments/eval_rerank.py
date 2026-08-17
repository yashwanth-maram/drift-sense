#!/usr/bin/env python3
"""Does the pairwise gain survive competition among candidates?

The verifier separates a true patch from ONE lattice impostor at 0.9802
against 0.9670 for plain ZNCC. But the pipeline must beat every impostor
at once. Back-solving from 75% accuracy gives roughly nine effective
impostors per image, so 0.967^8.6 = 0.75 and 0.980^8.6 = 0.84. That is an
upper bound -- impostors are not independent -- and the point of this
script is to replace the estimate with a measurement.

    ZNCC surface -> top-K peaks at lattice spacing -> linear re-rank
    -> centre tie-break -> coordinate

Reports:
    top-1 accuracy and median error, baseline versus re-ranked
    the margin S(true) - S(best impostor) under both scores
    the previously-failing subset, separately

Coefficients are fitted on synthetic data only.

    python experiments/eval_rerank.py --data ../am_eval --n 100
    python experiments/eval_rerank.py --data ../am_eval --n 100 --blend 0.5
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from driftsense.verifier import LinearVerifier
from driftsense.localize import localize, _strip_centres

T = 100
TOL = 5.0


def peaks(surface, sep, k=30):
    r = max(int(round(sep)), 2)
    dil = cv2.dilate(surface, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
    ys, xs = np.nonzero(surface >= dil - 1e-6)
    if len(ys) == 0:
        i = int(np.argmax(surface))
        y, x = np.unravel_index(i, surface.shape)
        return [(int(y), int(x), float(surface[y, x]))]
    v = surface[ys, xs]
    o = np.argsort(v)[::-1][:k]
    return [(int(ys[i]), int(xs[i]), float(v[i])) for i in o]


def lattice_pitch(img):
    a = img.astype(np.float32)
    a = a - a.mean(axis=1, keepdims=True)
    n = a.shape[1]
    spec = np.abs(np.fft.rfft(a * np.hanning(n)[None, :], axis=1)).mean(axis=0)
    kmin, kmax = max(int(n / 30), 2), min(int(n / 3.0), len(spec) - 1)
    if kmax <= kmin:
        return 8.0
    return n / (int(np.argmax(spec[kmin:kmax + 1])) + kmin)


def run(ref, srch, ver, blend, topk):
    tmpl = cv2.resize(ref, (T, T), interpolation=cv2.INTER_AREA)
    surf = cv2.matchTemplate(srch, tmpl, cv2.TM_CCOEFF_NORMED)

    i = int(np.argmax(surf))
    by, bx = np.unravel_index(i, surf.shape)
    base = (bx + T / 2.0, by + T / 2.0, float(surf[by, bx]))

    if not ver.ready:
        return base, base, []

    pk = peaks(surf, max(lattice_pitch(srch) * 0.7, 3.0), topk)
    scored = []
    for py, px, z in pk:
        cand = srch[py:py + T, px:px + T]
        if cand.shape != (T, T):
            continue
        p = ver.score(tmpl, cand)
        if p is None:
            continue
        s = p if blend <= 0 else blend * z + (1.0 - blend) * p
        scored.append((s, px + T / 2.0, py + T / 2.0, z, p))
    if not scored:
        return base, base, []

    scored.sort(key=lambda t: -t[0])
    best = scored[0]
    return base, (best[1], best[2], best[0]), scored


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--blend", type=float, default=0.0,
                    help="0 = verifier only, 1 = ZNCC only")
    ap.add_argument("--topk", type=int, default=30)
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

    ver = LinearVerifier.load()
    print("=" * 74)
    print(f"RE-RANK EVALUATION   blend={a.blend}  topK={a.topk}")
    print("=" * 74)
    if not ver.ready:
        print("\n  no coefficients found -- run experiments/train_verifier.py")
        return 1
    print("\n  Coefficients fitted on synthetic data only.\n")

    eb, er, cond = [], [], []
    fixed, broken = [], []
    margins_z, margins_v = [], []

    for row in rows[:a.n]:
        i = int(row["id"])
        ref = cv2.imread(os.path.join(root, "reference", f"{i:05d}.png"),
                         cv2.IMREAD_GRAYSCALE)
        srch = cv2.imread(os.path.join(root, "search", f"{i:05d}.png"),
                          cv2.IMREAD_GRAYSCALE)
        if ref is None or srch is None:
            continue
        gx, gy = float(row["gt_x"]), float(row["gt_y"])

        tmpl_c = cv2.resize(ref, (T, T), interpolation=cv2.INTER_AREA)
        pinned = (len(_strip_centres(tmpl_c, 1, 0.45, 6)) > 0) \
            + (len(_strip_centres(tmpl_c, 0, 0.45, 6)) > 0)
        cond.append(int(pinned))

        base, rr, scored = run(ref, srch, ver, a.blend, a.topk)
        e0 = float(np.hypot(base[0] - gx, base[1] - gy))
        e1 = float(np.hypot(rr[0] - gx, rr[1] - gy))
        eb.append(e0)
        er.append(e1)

        if e0 > a.tol >= e1:
            fixed.append(i)
        elif e1 > a.tol >= e0:
            broken.append(i)

        # margin between the true candidate and the best impostor
        if scored:
            tru = [s for s in scored
                   if np.hypot(s[1] - gx, s[2] - gy) <= a.tol]
            imp = [s for s in scored
                   if np.hypot(s[1] - gx, s[2] - gy) > a.tol]
            if tru and imp:
                margins_z.append(max(t[3] for t in tru) - max(t[3] for t in imp))
                margins_v.append(max(t[4] for t in tru) - max(t[4] for t in imp))

    eb, er = np.asarray(eb), np.asarray(er)
    n = len(eb)
    ab, ar = float((eb <= a.tol).mean()), float((er <= a.tol).mean())

    print(f"  samples                : {n}")
    print(f"\n  {'':22}{'baseline':>12}{'re-ranked':>12}")
    print(f"  {'accuracy at 5 px':22}{ab * 100:>11.1f}%{ar * 100:>11.1f}%")
    print(f"  {'median error':22}{np.median(eb):>11.2f}{np.median(er):>11.2f}")
    print(f"  {'mean error':22}{eb.mean():>11.1f}{er.mean():>11.1f}")

    print(f"\n  fixed by re-ranking    : {len(fixed)}   {fixed[:14]}")
    print(f"  broken by re-ranking   : {len(broken)}   {broken[:14]}")
    net = (len(fixed) - len(broken)) / max(n, 1) * 100
    print(f"  net                    : {net:+.1f} pp")

    if margins_z:
        mz, mv = np.asarray(margins_z), np.asarray(margins_v)
        print(f"\n  margin  S(true) - S(best impostor), where both are in top-K")
        print(f"    ZNCC       median {np.median(mz):+.4f}"
              f"   positive in {int((mz > 0).sum())}/{len(mz)}")
        print(f"    verifier   median {np.median(mv):+.4f}"
              f"   positive in {int((mv > 0).sum())}/{len(mv)}")

    cond = np.asarray(cond)
    print("\n  by template condition")
    print(f"    {'condition':<20}{'n':>5}{'baseline':>11}{'re-ranked':>11}")
    for c, name in ((2, "both axes pinned"), (1, "one axis pinned"),
                    (0, "neither pinned")):
        m = cond == c
        if m.sum():
            print(f"    {name:<20}{int(m.sum()):>5}"
                  f"{float((eb[m] <= a.tol).mean()) * 100:>10.1f}%"
                  f"{float((er[m] <= a.tol).mean()) * 100:>10.1f}%")

    prev_fail = np.nonzero(eb > a.tol)[0]
    if len(prev_fail):
        pf_acc = float((er[prev_fail] <= a.tol).mean())
        print(f"\n  previously failing subset  : {len(prev_fail)} samples")
        print(f"    now correct              : {int((er[prev_fail] <= a.tol).sum())}"
              f"  ({pf_acc * 100:.1f}%)")
        print(f"    median error  {np.median(eb[prev_fail]):.1f}"
              f" -> {np.median(er[prev_fail]):.1f} px")

    print("\n" + "=" * 74)
    d = ar - ab
    se = float(np.sqrt(max(len(fixed) + len(broken), 1)) / max(n, 1))
    if d > 2 * se:
        print(f"  IMPROVED by {d * 100:+.1f} pp. Keep it, and investigate the")
        print("  line-index head and tuning next.")
    elif abs(d) <= 2 * se:
        print(f"  NO CHANGE ({d * 100:+.1f} pp, within noise). The pairwise")
        print("  gain does not survive competition among candidates. Inspect")
        print("  where the discriminative structure sits spatially before")
        print("  going further.")
    else:
        print(f"  WORSE by {d * 100:.1f} pp. Fall back to plain ZNCC and")
        print("  record this as a negative result.")
    print("=" * 74)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
