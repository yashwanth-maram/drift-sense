#!/usr/bin/env python3
"""Fit the linear verifier on synthetic hard negatives.

Positives are the true patch. Negatives are the same patch displaced by
whole lattice periods -- the impostors the matcher actually loses to.
Random negatives would teach nothing, since they are already easy.

Training is synthetic only. Applied Materials' pairs are never used to fit
a coefficient; they are the test set.

    python experiments/train_verifier.py --n 400
    python experiments/train_verifier.py --n 600 --mixed
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from driftsense.verifier import (pair_features, LinearVerifier,
                                 FEATURE_NAMES, COEF_PATH)

T = 100
OFFSETS = (1, 2, 3, 4, 6, 8)


def dominant_pitch(img):
    a = img.astype(np.float32)
    a = a - a.mean(axis=1, keepdims=True)
    n = a.shape[1]
    spec = np.abs(np.fft.rfft(a * np.hanning(n)[None, :], axis=1)).mean(axis=0)
    kmin, kmax = max(int(n / 30), 2), min(int(n / 3.0), len(spec) - 1)
    if kmax <= kmin:
        return None
    return n / (int(np.argmax(spec[kmin:kmax + 1])) + kmin)


def patch_at(img, cx, cy, half=T // 2):
    h, w = img.shape
    x = int(np.clip(round(cx - half), 0, w - 2 * half))
    y = int(np.clip(round(cy - half), 0, h - 2 * half))
    return img[y:y + 2 * half, x:x + 2 * half]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=90210)
    ap.add_argument("--kind", default="dram", choices=["dram", "finfet"])
    ap.add_argument("--mixed", action="store_true",
                    help="train on both families, alternating")
    ap.add_argument("--levels", nargs="+",
                    default=["medium", "high", "severe"])
    ap.add_argument("--out", default=COEF_PATH)
    a = ap.parse_args(argv)

    from driftsense.generator.sample import generate_sample, build_params

    print("=" * 70)
    print(f"TRAIN VERIFIER   n={a.n}  levels={a.levels}"
          f"  {'mixed' if a.mixed else a.kind}")
    print("=" * 70)
    print("\n  Synthetic only. Applied Materials pairs are the test set and")
    print("  are not used to fit any coefficient.\n")

    X, y = [], []
    for i in range(a.n):
        level = a.levels[i % len(a.levels)]
        kind = (("dram", "finfet")[i % 2]) if a.mixed else a.kind
        p = build_params(noise_level=level)
        s = generate_sample(i, kind, base_seed=a.seed, params=p)

        pitch = dominant_pitch(s.search)
        if pitch is None or pitch < 3:
            continue
        tmpl = cv2.resize(s.reference, (T, T), interpolation=cv2.INTER_AREA)

        X.append(pair_features(tmpl, patch_at(s.search, s.gt_x, s.gt_y)))
        y.append(1)

        for k in OFFSETS:
            for axis in (0, 1):
                for sign in (-1, 1):
                    d = sign * k * pitch
                    cx = s.gt_x + (d if axis == 1 else 0)
                    cy = s.gt_y + (d if axis == 0 else 0)
                    if not (50 <= cx <= 950 and 50 <= cy <= 950):
                        continue
                    X.append(pair_features(tmpl, patch_at(s.search, cx, cy)))
                    y.append(0)

        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{a.n}   pairs so far {len(y)}", flush=True)

    X = np.vstack(X)
    y = np.asarray(y)
    print(f"\n  positives {int((y == 1).sum())}   negatives {int((y == 0).sum())}")

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    # negatives outnumber positives ~24:1, so balance the classes
    model = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xs, y)

    v = LinearVerifier(scaler.mean_, scaler.scale_,
                       model.coef_.ravel(), float(model.intercept_[0]))
    v.save(a.out)

    print(f"\n  {'feature':<18} {'coefficient':>12}")
    print("  " + "-" * 32)
    order = np.argsort(-np.abs(model.coef_.ravel()))
    for i in order:
        print(f"  {FEATURE_NAMES[i]:<18} {model.coef_.ravel()[i]:>+12.4f}")
    print(f"\n  intercept          {float(model.intercept_[0]):>+12.4f}")
    print(f"\n  written to {a.out}")
    print(f"  train accuracy {model.score(Xs, y):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
