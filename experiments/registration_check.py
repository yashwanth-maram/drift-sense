#!/usr/bin/env python3
"""Does our template actually align with its own origin?

On the failures, the impostor beats the true position in 85% of blocks --
and in several cases in EVERY block, by margins up to 0.46. That is not
what a wrong-site match looks like. A different instance of the same
lattice should still lose somewhere: at its own defects, its own edges.
Losing everywhere, uniformly, is the signature of misregistration.

If the template is systematically offset from its own origin, the true
position is penalised everywhere while impostors are untouched. That would
make part of the failure rate our bug rather than the problem's
difficulty.

Three checks:

  SELF-MATCH        take the true patch straight out of the search image
                    and correlate it back. The peak must land exactly on
                    the ground-truth coordinate. Anything else means the
                    coordinate convention is wrong.

  ALIGNMENT OFFSET  correlate the reference-derived template against a
                    neighbourhood of the true position. Where does it
                    peak? A consistent non-zero offset across samples is a
                    systematic error in the reference-to-template path.

  COST              how much correlation is lost at the true position
                    versus at the best-aligned position nearby. That is
                    the score misregistration is throwing away.

    python experiments/registration_check.py --data ../am_eval --n 40
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
WIN = 12


def zncc(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float((a * b).sum() / (na * nb))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args(argv)

    root = None
    for r, _, files in os.walk(a.data):
        if "manifest.csv" in files:
            root = r
            rows = list(csv.DictReader(open(os.path.join(r, "manifest.csv"))))
            break
    if root is None:
        raise SystemExit(f"no manifest.csv under {a.data}")

    print("=" * 74)
    print("REGISTRATION CHECK")
    print("=" * 74)
    print("\n  self  : peak of the true patch correlated back into the search")
    print("  align : where the reference-derived template actually peaks")
    print("  cost  : correlation lost by scoring at ground truth instead\n")
    print(f"  {'id':>4} {'self dx,dy':>12} {'align dx,dy':>13}"
          f" {'z at gt':>9} {'z at best':>10} {'cost':>8}")
    print("  " + "-" * 62)

    self_off, align_off, costs = [], [], []

    for row in rows[:a.n]:
        i = int(row["id"])
        ref = cv2.imread(os.path.join(root, "reference", f"{i:05d}.png"),
                         cv2.IMREAD_GRAYSCALE)
        srch = cv2.imread(os.path.join(root, "search", f"{i:05d}.png"),
                          cv2.IMREAD_GRAYSCALE)
        if ref is None or srch is None:
            continue
        gx, gy = float(row["gt_x"]), float(row["gt_y"])

        x0 = int(round(gx - T / 2))
        y0 = int(round(gy - T / 2))
        if not (0 <= x0 <= srch.shape[1] - T and 0 <= y0 <= srch.shape[0] - T):
            continue
        true_patch = srch[y0:y0 + T, x0:x0 + T]

        # 1. self-match: the true patch correlated back into the search
        s = cv2.matchTemplate(srch, true_patch, cv2.TM_CCOEFF_NORMED)
        k = int(np.argmax(s))
        sy, sx = np.unravel_index(k, s.shape)
        sdx, sdy = sx - x0, sy - y0
        self_off.append((sdx, sdy))

        # 2. where does the reference-derived template actually peak?
        tmpl = cv2.resize(ref, (T, T), interpolation=cv2.INTER_AREA)
        lo_x, hi_x = max(x0 - WIN, 0), min(x0 + WIN, srch.shape[1] - T)
        lo_y, hi_y = max(y0 - WIN, 0), min(y0 + WIN, srch.shape[0] - T)
        best, bxy = -2.0, (0, 0)
        for yy in range(lo_y, hi_y + 1):
            for xx in range(lo_x, hi_x + 1):
                v = zncc(tmpl, srch[yy:yy + T, xx:xx + T])
                if v > best:
                    best, bxy = v, (xx - x0, yy - y0)
        z_gt = zncc(tmpl, true_patch)
        align_off.append(bxy)
        costs.append(best - z_gt)

        flag = "  <-- self-match off" if (sdx or sdy) else ""
        print(f"  {i:>4} {f'{sdx:+d},{sdy:+d}':>12}"
              f" {f'{bxy[0]:+d},{bxy[1]:+d}':>13}"
              f" {z_gt:>9.4f} {best:>10.4f} {best - z_gt:>+8.4f}{flag}")

    if not costs:
        print("\n  nothing measured")
        return 1

    so = np.array(self_off)
    ao = np.array(align_off)
    cs = np.array(costs)
    n = len(cs)

    print(f"\n  samples                     : {n}")
    print(f"  self-match exact             : "
          f"{int((np.abs(so).sum(axis=1) == 0).sum())}/{n}")
    print(f"  median alignment offset      : "
          f"({np.median(ao[:, 0]):+.1f}, {np.median(ao[:, 1]):+.1f}) px")
    print(f"  mean alignment offset        : "
          f"({ao[:, 0].mean():+.2f}, {ao[:, 1].mean():+.2f}) px")
    print(f"  fraction with any offset     : "
          f"{float((np.abs(ao).sum(axis=1) > 0).mean()):.2f}")
    print(f"  median correlation cost      : {np.median(cs):+.4f}")
    print(f"  max correlation cost         : {cs.max():+.4f}")

    print("\n" + "=" * 74)
    if int((np.abs(so).sum(axis=1) == 0).sum()) < n:
        print("  COORDINATE BUG: the true patch does not correlate back to")
        print("  its own position. The ground-truth convention or the patch")
        print("  extraction is wrong, and every measurement built on it")
        print("  needs revisiting.")
    elif abs(np.median(ao[:, 0])) >= 1 or abs(np.median(ao[:, 1])) >= 1:
        print("  SYSTEMATIC MISREGISTRATION: the template peaks consistently")
        print("  away from ground truth. The reference-to-template path is")
        print("  introducing a shift, which penalises the true position")
        print("  everywhere while leaving impostors untouched. Fixing this")
        print("  may recover a large part of the failure rate.")
    elif np.median(cs) > 0.02:
        print("  Alignment is centred but noisy, and the true position loses")
        print("  measurable correlation to sub-pixel misfit. Worth a")
        print("  sub-pixel template alignment step -- though note that a")
        print("  phase oracle previously moved nothing.")
    else:
        print("  REGISTRATION IS SOUND. The template aligns with its own")
        print("  origin. The failures are genuine wrong-site matches, not a")
        print("  bug, and the earlier conclusions stand.")
    print("=" * 74)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
