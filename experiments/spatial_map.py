#!/usr/bin/env python3
"""Where, spatially, does the true-versus-impostor evidence live?

The 13-feature verifier won isolated pairwise comparison (0.980 vs 0.967)
and lost competition ranking (59/91 vs 75/91). One explanation is that
global descriptors flatten spatially localised evidence -- a defect in one
corner, a phase relationship between two particular lines. If that is what
is happening, the evidence should be concentrated in a few regions of the
patch, and a patch-based method would be justified.

Two things are measured.

CONCENTRATION
    Per pixel, ZNCC decomposes into a sum of contributions. Computing that
    sum separately against the true patch and against the impostor, and
    subtracting, gives a map of where the true candidate is favoured. If
    the map is concentrated, weighting could exploit it. If it is diffuse
    and near-symmetric, there is nothing to weight.

ORACLE WEIGHTING
    The bound on any weighting scheme. Derive the ideal weights from the
    known true-impostor pair, apply weighted correlation across the WHOLE
    search image, and ask whether the true position now wins outright.

    This is deliberately unfair to reality -- the weights come from the
    answer. If the true position still does not win, no learned weighting
    can help, because the oracle is the best case.

    python experiments/spatial_map.py --data ../am_eval --n 100 --save ../spatial
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

T = 100
TOL = 5.0
BLOCK = 10


def patch_at(img, cx, cy, half=T // 2):
    h, w = img.shape
    x = int(np.clip(round(cx - half), 0, w - 2 * half))
    y = int(np.clip(round(cy - half), 0, h - 2 * half))
    return img[y:y + 2 * half, x:x + 2 * half]


def contribution(ref, cand):
    """Per-pixel contribution to ZNCC. Summing this map gives the score."""
    a = ref.astype(np.float64)
    b = cand.astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return np.zeros_like(a)
    return (a * b) / (na * nb)


def block_sum(m, k=BLOCK):
    h, w = m.shape
    h, w = (h // k) * k, (w // k) * k
    return m[:h, :w].reshape(h // k, k, w // k, k).sum(axis=(1, 3))


def weighted_surface(ref, search, w):
    """Correlation with a per-pixel weight map, over the whole search image.

    Implemented directly: sums of w, w*I, w*I^2 over every window come from
    box filters, so the whole surface costs a handful of convolutions.
    """
    ref = ref.astype(np.float32)
    src = search.astype(np.float32)
    w = w.astype(np.float32)
    sw = float(w.sum())
    if sw < 1e-6:
        return None

    rw = ref * w
    mu_r = float(rw.sum() / sw)
    dr = (ref - mu_r) * w
    nr = float(np.sqrt((w * (ref - mu_r) ** 2).sum()))
    if nr < 1e-6:
        return None

    num = cv2.filter2D(src, cv2.CV_32F, dr[::-1, ::-1],
                       anchor=(-1, -1), borderType=cv2.BORDER_CONSTANT)
    s1 = cv2.filter2D(src, cv2.CV_32F, w[::-1, ::-1],
                      borderType=cv2.BORDER_CONSTANT)
    s2 = cv2.filter2D(src * src, cv2.CV_32F, w[::-1, ::-1],
                      borderType=cv2.BORDER_CONSTANT)

    var = s2 - (s1 * s1) / sw
    den = np.sqrt(np.maximum(var, 1e-9)) * nr
    surf = num / np.maximum(den, 1e-9)

    off = T // 2
    h, w_ = search.shape
    valid = surf[off:h - T + 1 + off, off:w_ - T + 1 + off]
    return valid


def render(ref, true_p, imp_p, diff, path):
    def norm(a):
        lo, hi = float(a.min()), float(a.max())
        if hi - lo < 1e-9:
            return np.zeros(a.shape, np.uint8)
        return ((a - lo) * (255.0 / (hi - lo))).astype(np.uint8)

    heat = cv2.applyColorMap(norm(cv2.resize(diff, (T, T),
                                             interpolation=cv2.INTER_NEAREST)),
                             cv2.COLORMAP_JET)
    gap = np.full((T, 6, 3), 30, np.uint8)
    row = np.hstack([cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR), gap,
                     cv2.cvtColor(true_p, cv2.COLOR_GRAY2BGR), gap,
                     cv2.cvtColor(imp_p, cv2.COLOR_GRAY2BGR), gap, heat])
    cv2.imwrite(path, cv2.resize(row, None, fx=3, fy=3,
                                 interpolation=cv2.INTER_NEAREST))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--save", default=None)
    a = ap.parse_args(argv)

    root = None
    for r, _, files in os.walk(a.data):
        if "manifest.csv" in files:
            root = r
            rows = list(csv.DictReader(open(os.path.join(r, "manifest.csv"))))
            break
    if root is None:
        raise SystemExit(f"no manifest.csv under {a.data}")
    if a.save:
        os.makedirs(a.save, exist_ok=True)

    print("=" * 74)
    print("SPATIAL DISCRIMINATION MAP")
    print("=" * 74)
    print("\n  Concentration: how much of the true-vs-impostor evidence sits")
    print("  in the strongest blocks. Oracle: does ideal weighting, derived")
    print("  from the answer itself, let the true position win outright?\n")
    print(f"  {'id':>4} {'err':>8} {'top10%':>8} {'blocks+':>8}"
          f" {'zncc gap':>9} {'oracle':>8}")
    print("  " + "-" * 52)

    conc, favour, gaps = [], [], []
    oracle_win = oracle_tot = 0

    for row in rows[:a.n]:
        i = int(row["id"])
        ref = cv2.imread(os.path.join(root, "reference", f"{i:05d}.png"),
                         cv2.IMREAD_GRAYSCALE)
        srch = cv2.imread(os.path.join(root, "search", f"{i:05d}.png"),
                          cv2.IMREAD_GRAYSCALE)
        if ref is None or srch is None:
            continue
        gx, gy = float(row["gt_x"]), float(row["gt_y"])

        tmpl = cv2.resize(ref, (T, T), interpolation=cv2.INTER_AREA)
        surf = cv2.matchTemplate(srch, tmpl, cv2.TM_CCOEFF_NORMED)
        k = int(np.argmax(surf))
        py, px = np.unravel_index(k, surf.shape)
        pred = (px + T / 2.0, py + T / 2.0)
        err = float(np.hypot(pred[0] - gx, pred[1] - gy))
        if err <= a.tol:
            continue

        true_p = patch_at(srch, gx, gy)
        imp_p = patch_at(srch, *pred)
        d = contribution(tmpl, true_p) - contribution(tmpl, imp_p)
        b = block_sum(d)

        flat = np.sort(np.abs(b).ravel())[::-1]
        top = float(flat[:max(len(flat) // 10, 1)].sum() / max(flat.sum(), 1e-9))
        pos = float((b > 0).mean())
        gap = float(contribution(tmpl, true_p).sum()
                    - contribution(tmpl, imp_p).sum())
        conc.append(top)
        favour.append(pos)
        gaps.append(gap)

        # oracle: weights from the answer, applied over the whole image
        w = np.maximum(cv2.resize(np.maximum(b, 0), (T, T),
                                  interpolation=cv2.INTER_LINEAR), 0)
        w = w / max(w.max(), 1e-9) + 0.05
        ws = weighted_surface(tmpl, srch, w)
        ok = False
        if ws is not None and ws.size:
            kk = int(np.argmax(ws))
            oy, ox = np.unravel_index(kk, ws.shape)
            ok = np.hypot(ox + T / 2.0 - gx, oy + T / 2.0 - gy) <= a.tol
        oracle_tot += 1
        oracle_win += int(ok)

        print(f"  {i:>4} {err:>8.1f} {top:>8.3f} {pos:>8.3f}"
              f" {gap:>+9.4f} {'WIN' if ok else 'lose':>8}")

        if a.save:
            render(tmpl, true_p, imp_p, b,
                   os.path.join(a.save, f"spatial_{i:05d}.png"))

    if not conc:
        print("\n  no failures")
        return 0

    print(f"\n  failures analysed        : {len(conc)}")
    print(f"  median top-10% share     : {np.median(conc):.3f}"
          f"   (0.10 = perfectly uniform)")
    print(f"  median blocks favouring  : {np.median(favour):.3f}"
          f"   (0.50 = no preference)")
    print(f"  median zncc gap          : {np.median(gaps):+.4f}"
          f"   (negative = impostor wins)")
    print(f"\n  ORACLE weighting wins    : {oracle_win}/{oracle_tot}"
          f"  ({oracle_win / max(oracle_tot, 1) * 100:.0f}%)")

    print("\n" + "=" * 74)
    med = float(np.median(conc))
    rate = oracle_win / max(oracle_tot, 1)
    if rate < 0.3:
        print("  Oracle weighting mostly fails. Weights derived from the")
        print("  answer itself cannot make the true position win, so no")
        print("  learned spatial weighting will either. This closes the")
        print("  weighting line, including a CNN verifier.")
    elif med > 0.35:
        print("  Evidence is concentrated and oracle weighting works. A")
        print("  patch-based or attention method is justified -- the signal")
        print("  is spatially localised and global features flatten it.")
    else:
        print("  Oracle weighting works but the evidence is diffuse, so the")
        print("  weights would be hard to predict from the reference alone.")
        print("  Check whether the strong blocks share a visible cause.")
    print("=" * 74)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
