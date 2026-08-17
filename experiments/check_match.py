#!/usr/bin/env python3
"""Visual check: is it actually finding the right place?

Draws, for each pair:

  left    the reference pattern, as given
  middle  the search image with the TRUE box in green and the PREDICTED
          box in red, joined by a line when they differ
  right   the two 100x100 regions side by side -- what the pattern really
          looks like, and what the algorithm matched instead

The right-hand panel is the important one. When the algorithm is wrong,
those two crops usually look nearly identical, which is the whole problem
in one picture: a repeating layout means the wrong place genuinely
resembles the right one.

    python experiments/check_match.py --data ../ourdata --n 5 --out ../inspect
    python experiments/check_match.py --data ../ourdata --pick     # best + worst

Writes annotated PNGs and prints the coordinates.
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

GREEN = (60, 200, 60)
RED = (60, 60, 235)
WHITE = (245, 245, 245)
GREY = (150, 150, 150)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _panel(img, size):
    out = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def _crop(img, cx, cy, half=50):
    h, w = img.shape
    x0 = int(np.clip(cx - half, 0, w - 2 * half))
    y0 = int(np.clip(cy - half, 0, h - 2 * half))
    return img[y0:y0 + 2 * half, x0:x0 + 2 * half]


def render(reference, search, gt, pred, err, tol=5.0):
    S = 420
    pad = 14
    canvas = np.full((S + 90, S * 2 + 260 + pad * 4, 3), 26, np.uint8)

    # --- reference
    canvas[pad + 40:pad + 40 + S, pad:pad + S] = _panel(reference, S)
    cv2.putText(canvas, "reference (1 nm/px)", (pad, pad + 28),
                FONT, 0.5, WHITE, 1, cv2.LINE_AA)

    # --- search with boxes
    x1 = pad * 2 + S
    view = _panel(search, S)
    k = S / search.shape[0]
    gx, gy = int(gt[0] * k), int(gt[1] * k)
    px, py = int(pred[0] * k), int(pred[1] * k)
    hb = int(50 * k)

    cv2.rectangle(view, (gx - hb, gy - hb), (gx + hb, gy + hb), GREEN, 2)
    cv2.rectangle(view, (px - hb, py - hb), (px + hb, py + hb), RED, 2)
    if err > tol:
        cv2.line(view, (gx, gy), (px, py), GREY, 1, cv2.LINE_AA)
    cv2.drawMarker(view, (gx, gy), GREEN, cv2.MARKER_CROSS, 12, 2)
    cv2.drawMarker(view, (px, py), RED, cv2.MARKER_TILTED_CROSS, 12, 2)

    canvas[pad + 40:pad + 40 + S, x1:x1 + S] = view
    cv2.putText(canvas, "search (10 nm/px)  green = true, red = predicted",
                (x1, pad + 28), FONT, 0.5, WHITE, 1, cv2.LINE_AA)

    # --- the two candidate regions
    x2 = pad * 3 + S * 2
    C = 190
    true_c = cv2.resize(_crop(search, *gt), (C, C), interpolation=cv2.INTER_NEAREST)
    pred_c = cv2.resize(_crop(search, *pred), (C, C), interpolation=cv2.INTER_NEAREST)
    canvas[pad + 40:pad + 40 + C, x2:x2 + C] = cv2.cvtColor(true_c, cv2.COLOR_GRAY2BGR)
    canvas[pad + 60 + C:pad + 60 + 2 * C, x2:x2 + C] = cv2.cvtColor(pred_c, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(canvas, (x2 - 2, pad + 38), (x2 + C + 1, pad + 41 + C), GREEN, 2)
    cv2.rectangle(canvas, (x2 - 2, pad + 58 + C), (x2 + C + 1, pad + 61 + 2 * C), RED, 2)
    cv2.putText(canvas, "what it should match", (x2, pad + 28),
                FONT, 0.45, GREEN, 1, cv2.LINE_AA)
    cv2.putText(canvas, "what it matched", (x2, pad + 52 + C),
                FONT, 0.45, RED, 1, cv2.LINE_AA)

    # --- footer
    ok = err <= tol
    msg = (f"true ({gt[0]:.1f}, {gt[1]:.1f})    "
           f"predicted ({pred[0]:.1f}, {pred[1]:.1f})    "
           f"error {err:.2f} px    "
           + ("WITHIN TOLERANCE" if ok else f"MISS (tolerance {tol:.0f} px)"))
    cv2.putText(canvas, msg, (pad, canvas.shape[0] - 18), FONT, 0.55,
                GREEN if ok else RED, 1, cv2.LINE_AA)
    return canvas


def load_manifest(data_dir):
    for root, _, files in os.walk(data_dir):
        if "manifest.csv" in files:
            with open(os.path.join(root, "manifest.csv")) as fh:
                return list(csv.DictReader(fh)), root
    raise SystemExit(f"no manifest.csv under {data_dir}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="./inspect")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--rung", type=int, default=6)
    ap.add_argument("--tol", type=float, default=5.0)
    ap.add_argument("--pick", action="store_true",
                    help="render only the best success and the worst failure")
    a = ap.parse_args(argv)

    rows, root = load_manifest(a.data)
    os.makedirs(a.out, exist_ok=True)
    results = []

    n = len(rows) if a.pick else min(a.n, len(rows))
    print(f"{'id':>4}  {'true':>17}  {'predicted':>17}  {'error':>8}   result")
    print("-" * 68)

    for row in rows[:n]:
        rp = row["reference_path"]
        sp = row["search_path"]
        if not os.path.exists(rp):
            rp = os.path.join(root, "reference", f"{int(row['id']):05d}.png")
            sp = os.path.join(root, "search", f"{int(row['id']):05d}.png")
        ref = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
        srch = cv2.imread(sp, cv2.IMREAD_GRAYSCALE)
        if ref is None or srch is None:
            print(f"  skipping {row['id']}: cannot read images")
            continue

        gt = (float(row["gt_x"]), float(row["gt_y"]))
        r = localize(ref, srch, rung=a.rung)
        pred = (r.x, r.y)
        err = float(np.hypot(pred[0] - gt[0], pred[1] - gt[1]))
        results.append((int(row["id"]), ref, srch, gt, pred, err))

        tag = "hit " if err <= a.tol else "MISS"
        print(f"{row['id']:>4}  ({gt[0]:7.1f},{gt[1]:7.1f})  "
              f"({pred[0]:7.1f},{pred[1]:7.1f})  {err:8.2f}   {tag}")

    if not results:
        return 1

    if a.pick:
        hits = [r for r in results if r[5] <= a.tol]
        misses = [r for r in results if r[5] > a.tol]
        chosen = []
        if hits:
            chosen.append(("success", min(hits, key=lambda r: r[5])))
        if misses:
            chosen.append(("failure", max(misses, key=lambda r: r[5])))
    else:
        chosen = [(f"{r[0]:05d}", r) for r in results]

    for name, (idx, ref, srch, gt, pred, err) in chosen:
        img = render(ref, srch, gt, pred, err, a.tol)
        path = os.path.join(a.out, f"{name}_{idx:05d}.png")
        cv2.imwrite(path, img)
        print(f"  wrote {path}")

    errs = np.array([r[5] for r in results])
    print(f"\n  accuracy at {a.tol:.0f} px : "
          f"{(errs <= a.tol).mean() * 100:.1f}%  ({(errs <= a.tol).sum()}/{len(errs)})")
    print(f"  median error       : {np.median(errs):.2f} px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
