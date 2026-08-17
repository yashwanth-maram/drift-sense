#!/usr/bin/env python3
"""Slide 6 visuals: success and honest failure.

The submission template asks for the reference image, the search image,
the predicted location and the true location, for one success case and one
honest failure case.

Each figure has four panels:

    reference     as supplied, 1000x1000 at 1 nm/px
    search        with the matched region outlined and both centres marked,
                  1000x1000 at 10 nm/px
    inset         the region around the prediction, magnified
    comparison    the true 100x100 region above the predicted one

The comparison panel is the point of the failure figure. On a miss the two
crops usually look nearly identical, which shows why a repeating layout is
hard rather than making the algorithm look careless -- and the problem
statement asks specifically for an honest failure and its cause.

    python experiments/make_figures.py --data ../am_eval --out ../figures
    python experiments/make_figures.py --data ../am_eval --out ../figures --all
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from localize import localize, SCALE
from driftsense.localize import _strip_centres

T = 100
TOL = 5.0
GREEN = (70, 200, 70)
RED = (60, 60, 240)
WHITE = (245, 245, 245)
GREY = (130, 130, 130)
DARK = (24, 24, 26)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _panel(img, size):
    return cv2.cvtColor(cv2.resize(img, (size, size),
                                   interpolation=cv2.INTER_AREA),
                        cv2.COLOR_GRAY2BGR)


def _crop(img, cx, cy, half):
    h, w = img.shape
    x = int(np.clip(round(cx - half), 0, w - 2 * half))
    y = int(np.clip(round(cy - half), 0, h - 2 * half))
    return img[y:y + 2 * half, x:x + 2 * half]


def condition(ref):
    t = cv2.resize(ref, (T, T), interpolation=cv2.INTER_AREA)
    n = (len(_strip_centres(t, 1, 0.45, 6)) > 0) \
        + (len(_strip_centres(t, 0, 0.45, 6)) > 0)
    return n, ("both axes pinned by strips", "one axis pinned",
               "no strip on either axis")[2 - n]


def render(ref, srch, gt, pred, err, conf, ms, idx, tol=TOL):
    S = 400
    INS = 250
    pad = 16
    top = 46
    W = pad * 4 + S * 2 + INS
    H = top + S + 78
    canvas = np.full((H, W), DARK[0], np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    canvas[:] = DARK

    # reference
    canvas[top:top + S, pad:pad + S] = _panel(ref, S)
    cv2.putText(canvas, "REFERENCE   1000x1000 @ 1 nm/px", (pad, top - 14),
                FONT, 0.46, WHITE, 1, cv2.LINE_AA)

    # search, with the matched region outlined
    x1 = pad * 2 + S
    view = _panel(srch, S)
    k = S / srch.shape[0]
    gx, gy = int(gt[0] * k), int(gt[1] * k)
    px, py = int(pred[0] * k), int(pred[1] * k)
    hb = int(T / 2 * k)

    cv2.rectangle(view, (gx - hb, gy - hb), (gx + hb, gy + hb), GREEN, 2)
    cv2.rectangle(view, (px - hb, py - hb), (px + hb, py + hb), RED, 2)
    if err > tol:
        cv2.line(view, (gx, gy), (px, py), GREY, 1, cv2.LINE_AA)
    cv2.drawMarker(view, (gx, gy), GREEN, cv2.MARKER_CROSS, 13, 2)
    cv2.drawMarker(view, (px, py), RED, cv2.MARKER_TILTED_CROSS, 13, 2)
    canvas[top:top + S, x1:x1 + S] = view
    cv2.putText(canvas, "SEARCH  @ 10 nm/px", (x1, top - 14),
                FONT, 0.46, WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas, "true", (x1 + 196, top - 14), FONT, 0.46, GREEN,
                1, cv2.LINE_AA)
    cv2.putText(canvas, "/", (x1 + 234, top - 14), FONT, 0.46, WHITE,
                1, cv2.LINE_AA)
    cv2.putText(canvas, "predicted", (x1 + 246, top - 14), FONT, 0.46,
                RED, 1, cv2.LINE_AA)

    # magnified inset around the prediction
    x2 = pad * 3 + S * 2
    ins = _crop(srch, pred[0], pred[1], 90)
    ins = cv2.cvtColor(cv2.resize(ins, (INS, INS),
                                  interpolation=cv2.INTER_NEAREST),
                       cv2.COLOR_GRAY2BGR)
    ik = INS / 180.0
    c = INS // 2
    hb2 = int(T / 2 * ik)
    cv2.rectangle(ins, (c - hb2, c - hb2), (c + hb2, c + hb2), RED, 2)
    gdx = int((gt[0] - pred[0]) * ik)
    gdy = int((gt[1] - pred[1]) * ik)
    if abs(gdx) < c and abs(gdy) < c:
        cv2.drawMarker(ins, (c + gdx, c + gdy), GREEN, cv2.MARKER_CROSS, 13, 2)
    canvas[top:top + INS, x2:x2 + INS] = ins
    cv2.putText(canvas, "MATCHED REGION", (x2, top - 14), FONT, 0.46,
                WHITE, 1, cv2.LINE_AA)

    # true region above predicted region
    C = 112
    y2 = top + INS + 30
    tc = cv2.resize(_crop(srch, gt[0], gt[1], T // 2), (C, C),
                    interpolation=cv2.INTER_NEAREST)
    pc = cv2.resize(_crop(srch, pred[0], pred[1], T // 2), (C, C),
                    interpolation=cv2.INTER_NEAREST)
    canvas[y2:y2 + C, x2:x2 + C] = cv2.cvtColor(tc, cv2.COLOR_GRAY2BGR)
    canvas[y2:y2 + C, x2 + C + 12:x2 + 2 * C + 12] = \
        cv2.cvtColor(pc, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(canvas, (x2 - 1, y2 - 1), (x2 + C, y2 + C), GREEN, 2)
    cv2.rectangle(canvas, (x2 + C + 11, y2 - 1), (x2 + 2 * C + 12, y2 + C),
                  RED, 2)
    cv2.putText(canvas, "true site", (x2, y2 - 6), FONT, 0.4, GREEN, 1,
                cv2.LINE_AA)
    cv2.putText(canvas, "matched", (x2 + C + 12, y2 - 6), FONT, 0.4, RED, 1,
                cv2.LINE_AA)

    # footer
    ok = err <= tol
    n_pinned, cond_txt = condition(ref)
    l1 = (f"sample {idx}    true ({gt[0]:.1f}, {gt[1]:.1f})    "
          f"predicted ({pred[0]:.1f}, {pred[1]:.1f})    "
          f"error {err:.2f} px")
    l2 = (f"{'WITHIN TOLERANCE' if ok else 'MISS'}  (tolerance {tol:.0f} px)"
          f"    confidence {conf:.3f}    {ms:.0f} ms    {cond_txt}")
    cv2.putText(canvas, l1, (pad, H - 40), FONT, 0.52, WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas, l2, (pad, H - 16), FONT, 0.52,
                GREEN if ok else RED, 1, cv2.LINE_AA)
    return canvas


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


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="./figures")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--all", action="store_true",
                    help="render every sample, not just the two chosen")
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    pairs = load(a.data, a.n)

    print("=" * 66)
    print(f"FIGURES   {a.data}   n={len(pairs)}")
    print("=" * 66)
    print(f"\n  {'id':>4} {'error':>9} {'conf':>8} {'cond':>6}   result")
    print("  " + "-" * 42)

    rows = []
    for idx, ref, srch, gx, gy in pairs:
        r = localize(ref, srch)
        err = float(np.hypot(r.x - gx, r.y - gy))
        n_pin, _ = condition(ref)
        rows.append((idx, ref, srch, (gx, gy), (r.x, r.y), err,
                     r.confidence, r.ms, n_pin))
        print(f"  {idx:>4} {err:>9.2f} {r.confidence:>8.3f} {n_pin:>6}"
              f"   {'hit' if err <= a.tol else 'MISS'}")

    hits = [r for r in rows if r[5] <= a.tol]
    miss = [r for r in rows if r[5] > a.tol]

    chosen = []
    if hits:
        # prefer a success with no strip to pin it -- the hardest class
        hard = [h for h in hits if h[8] == 0]
        chosen.append(("success", min(hard or hits, key=lambda r: r[5])))
    if miss:
        # prefer a mat-interior failure: the two crops look alike, which is
        # the honest illustration of why a repeating layout is hard
        hard = [m for m in miss if m[8] == 0]
        chosen.append(("failure", max(hard or miss, key=lambda r: r[5])))

    if a.all:
        chosen = [(f"{r[0]:05d}", r) for r in rows]

    print()
    for name, r in chosen:
        img = render(r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[0], a.tol)
        path = os.path.join(a.out, f"{name}_{r[0]:05d}.png")
        cv2.imwrite(path, img)
        print(f"  wrote {path}")

    e = np.array([r[5] for r in rows])
    print(f"\n  accuracy at {a.tol:.0f} px : {float((e <= a.tol).mean()) * 100:.1f}%")
    print(f"  median error       : {np.median(e):.2f} px")
    if miss:
        print(f"\n  failures: " + ", ".join(str(m[0]) for m in miss))
        print("  Use a mat-interior failure for slide 6 -- the two crops look")
        print("  nearly identical, which shows why the layout is hard rather")
        print("  than making the algorithm look careless.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
