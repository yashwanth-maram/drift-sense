#!/usr/bin/env python3
"""Spatial discrimination, with the registration confound removed.

The original version of this experiment was unfair and its conclusion is
suspect. It compared:

    impostor patch  at the argmax position -- by construction the place
                    where the template aligns BEST
    true patch      at ground truth -- which the registration check showed
                    is typically 1 px away from where the template
                    actually aligns, costing a median 0.089 of correlation

So the impostor was handed its optimal alignment and the true position was
not. The resulting "impostor wins in 85% of blocks", and the several cases
at exactly 100%, may be that artifact rather than a property of the data.

This version searches a small neighbourhood around BOTH positions and
takes each one's own best alignment before comparing. Both figures are
reported, so the size of the artifact is visible.

The stakes: "oracle weighting 0/25" was the argument for closing spatial
weighting, attention, patch verification and the CNN -- and by extension
the multi-correspondence and geometric-consistency family, since the claim
was that all local evidence favours the impostor. If that measurement was
confounded, those lines reopen.

Synthetic only. The Applied Materials failures stay sealed: iterating
diagnostics on them risks tuning to those particular cases.

    python experiments/spatial_map2.py --n 40
    python experiments/spatial_map2.py --n 40 --kind finfet --save ../sm2
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

T = 100
TOL = 5.0
BLOCK = 10
RAD = 3


def zncc(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float((a * b).sum() / (na * nb))


def patch_at(img, cx, cy, half=T // 2):
    h, w = img.shape
    x = int(np.clip(round(cx - half), 0, w - 2 * half))
    y = int(np.clip(round(cy - half), 0, h - 2 * half))
    return img[y:y + 2 * half, x:x + 2 * half]


def best_local(tmpl, img, cx, cy, rad=RAD):
    """Patch at whichever nearby offset the template aligns to best."""
    best = (-2.0, None, (0, 0))
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            p = patch_at(img, cx + dx, cy + dy)
            if p.shape != (T, T):
                continue
            v = zncc(tmpl, p)
            if v > best[0]:
                best = (v, p, (dx, dy))
    return best


def contribution(ref, cand):
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
    ref = ref.astype(np.float32)
    src = search.astype(np.float32)
    w = w.astype(np.float32)
    sw = float(w.sum())
    if sw < 1e-6:
        return None
    mu_r = float((ref * w).sum() / sw)
    dr = (ref - mu_r) * w
    nr = float(np.sqrt((w * (ref - mu_r) ** 2).sum()))
    if nr < 1e-6:
        return None
    num = cv2.filter2D(src, cv2.CV_32F, dr[::-1, ::-1],
                       borderType=cv2.BORDER_CONSTANT)
    s1 = cv2.filter2D(src, cv2.CV_32F, w[::-1, ::-1],
                      borderType=cv2.BORDER_CONSTANT)
    s2 = cv2.filter2D(src * src, cv2.CV_32F, w[::-1, ::-1],
                      borderType=cv2.BORDER_CONSTANT)
    var = s2 - (s1 * s1) / sw
    surf = num / np.maximum(np.sqrt(np.maximum(var, 1e-9)) * nr, 1e-9)
    off = T // 2
    h, wd = search.shape
    return surf[off:h - T + 1 + off, off:wd - T + 1 + off]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--kind", default="dram", choices=["dram", "finfet"])
    ap.add_argument("--seed", type=int, default=3131)
    ap.add_argument("--level", default="medium")
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--save", default=None)
    a = ap.parse_args(argv)

    from driftsense.generator.sample import generate_sample, build_params
    p = build_params(noise_level=a.level)
    if a.save:
        os.makedirs(a.save, exist_ok=True)

    print("=" * 80)
    print("SPATIAL MAP  (registration-corrected)")
    print("=" * 80)
    print("\n  Both patches are compared at their OWN best local alignment,")
    print("  searched over +/-3 px. The naive column repeats the original,")
    print("  unfair comparison so the size of the artifact is visible.\n")
    print(f"  {'id':>4} {'err':>8} {'t off':>8} {'i off':>8}"
          f" {'gain':>7} {'naive+':>8} {'fair+':>7} {'oracle':>8}")
    print("  " + "-" * 66)

    naive, fair, gains = [], [], []
    ow_n = ow_f = tot = 0

    for i in range(a.n):
        s = generate_sample(i, a.kind, base_seed=a.seed, params=p)
        tmpl = cv2.resize(s.reference, (T, T), interpolation=cv2.INTER_AREA)
        surf = cv2.matchTemplate(s.search, tmpl, cv2.TM_CCOEFF_NORMED)
        k = int(np.argmax(surf))
        py, px = np.unravel_index(k, surf.shape)
        pred = (px + T / 2.0, py + T / 2.0)
        err = float(np.hypot(pred[0] - s.gt_x, pred[1] - s.gt_y))
        if err <= a.tol:
            continue
        tot += 1

        # naive: truth at ground truth, impostor at its argmax
        t_naive = patch_at(s.search, s.gt_x, s.gt_y)
        i_naive = patch_at(s.search, *pred)
        b_naive = block_sum(contribution(tmpl, t_naive)
                            - contribution(tmpl, i_naive))

        # fair: each at its own best local alignment
        zt, t_fair, t_off = best_local(tmpl, s.search, s.gt_x, s.gt_y)
        zi, i_fair, i_off = best_local(tmpl, s.search, *pred)
        b_fair = block_sum(contribution(tmpl, t_fair)
                           - contribution(tmpl, i_fair))

        pn = float((b_naive > 0).mean())
        pf = float((b_fair > 0).mean())
        naive.append(pn)
        fair.append(pf)
        gains.append(zt - zncc(tmpl, t_naive))

        def oracle(b):
            w = np.maximum(cv2.resize(np.maximum(b, 0), (T, T),
                                      interpolation=cv2.INTER_LINEAR), 0)
            w = w / max(w.max(), 1e-9) + 0.05
            ws = weighted_surface(tmpl, s.search, w)
            if ws is None or ws.size == 0:
                return False
            kk = int(np.argmax(ws))
            oy, ox = np.unravel_index(kk, ws.shape)
            return bool(np.hypot(ox + T / 2.0 - s.gt_x,
                                 oy + T / 2.0 - s.gt_y) <= a.tol)

        wn, wf = oracle(b_naive), oracle(b_fair)
        ow_n += int(wn)
        ow_f += int(wf)

        print(f"  {i:>4} {err:>8.1f} {str(t_off):>8} {str(i_off):>8}"
              f" {zt - zncc(tmpl, t_naive):>+7.3f}"
              f" {pn:>8.3f} {pf:>7.3f} {'WIN' if wf else 'lose':>8}")

        if a.save:
            def norm(m):
                lo, hi = float(m.min()), float(m.max())
                return np.zeros(m.shape, np.uint8) if hi - lo < 1e-9 else \
                    ((m - lo) * (255.0 / (hi - lo))).astype(np.uint8)
            heat = cv2.applyColorMap(
                norm(cv2.resize(b_fair, (T, T),
                                interpolation=cv2.INTER_NEAREST)),
                cv2.COLORMAP_JET)
            gap = np.full((T, 6, 3), 30, np.uint8)
            strip = np.hstack([cv2.cvtColor(tmpl, cv2.COLOR_GRAY2BGR), gap,
                               cv2.cvtColor(t_fair, cv2.COLOR_GRAY2BGR), gap,
                               cv2.cvtColor(i_fair, cv2.COLOR_GRAY2BGR),
                               gap, heat])
            cv2.imwrite(os.path.join(a.save, f"fair_{i:05d}.png"),
                        cv2.resize(strip, None, fx=3, fy=3,
                                   interpolation=cv2.INTER_NEAREST))

    if not fair:
        print("\n  no failures at this setting")
        return 0

    nn, ff = np.asarray(naive), np.asarray(fair)
    print(f"\n  failures analysed          : {tot}")
    print(f"  median correlation recovered by fair alignment : "
          f"{np.median(gains):+.4f}")
    print(f"\n  blocks favouring truth, naive : {np.median(nn):.3f}")
    print(f"  blocks favouring truth, fair  : {np.median(ff):.3f}")
    print(f"  change                        : {np.median(ff) - np.median(nn):+.3f}")
    print(f"\n  oracle weighting, naive  : {ow_n}/{tot}")
    print(f"  oracle weighting, fair   : {ow_f}/{tot}")

    print("\n" + "=" * 80)
    med = float(np.median(ff))
    if med > 0.42:
        print("  The original result was an artifact of unequal registration.")
        print("  Under fair alignment the evidence is roughly balanced, so")
        print("  the spatial and geometric branch is NOT closed and the")
        print("  multi-correspondence family is live again.")
    elif med > 0.30:
        print("  Partly artifact. The evidence is less lopsided than reported")
        print("  but still favours the impostor. Weighting alone looks weak;")
        print("  judge the geometric branch on the oracle row, not this one.")
    else:
        print("  The original result stands. Even under fair alignment the")
        print("  impostor is favoured across most of the patch, so local")
        print("  evidence genuinely does not distinguish the sites and the")
        print("  weighting family remains closed.")
    print("=" * 80)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
