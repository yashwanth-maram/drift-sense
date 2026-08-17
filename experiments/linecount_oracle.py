#!/usr/bin/env python3
"""Can the structure identify WHICH integer offset a candidate is?

Twelve rejected approaches all asked the same question in different forms:
does this candidate score highest? That is an argmax over hundreds of
near-ties, and it fails because the ties are near.

This asks a different question. Writing

    x = x_anchor + n * p_x + delta_x

and supplying the true pitch and the true candidate family, can observable
structure identify the integer n? That is nine-way classification, not an
argmax over eight hundred positions.

It is an information-existence test. If n cannot be recovered even with
the pitch and family handed over, then the multi-correspondence and
geometric-consistency branch has nothing to be consistent about, and it
closes the way pixel weighting did.

Two estimators, deliberately:

    nearest neighbour   parameter-free. Says whether the information is
                        intrinsically present in the representation.
    logistic regression whether a simple model can exploit it.

A gap between them means the representation carries the signal but needs
better decoding. Both at chance means the signal is not there.

Features are ORDERED structural observations, not global histograms --
line positions, spacings, widths, envelope, second differences -- because
global descriptors were already measured to destroy the arrangement.

Synthetic only. The Applied Materials failures stay sealed.

    python experiments/linecount_oracle.py --n 300
    python experiments/linecount_oracle.py --n 300 --kind finfet
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

T = 100
CLASSES = (-8, -4, -2, -1, 0, 1, 2, 4, 8)
N_LINES = 6
MIN_LINES = 4


def dominant_pitch(img, axis=1):
    a = img.astype(np.float32)
    if axis == 1:
        a = a - a.mean(axis=1, keepdims=True)
        prof = a.mean(axis=0)
    else:
        a = a - a.mean(axis=0, keepdims=True)
        prof = a.mean(axis=1)
    n = len(prof)
    spec = np.abs(np.fft.rfft(prof * np.hanning(n)))
    kmin, kmax = max(int(n / 40), 2), min(int(n / 3), len(spec) - 1)
    if kmax <= kmin:
        return None
    return n / (int(np.argmax(spec[kmin:kmax + 1])) + kmin)


def structural_observations(patch, pitch, axis=1, n_lines=N_LINES):
    """Ordered per-line measurements, kept in sequence.

    Global histograms were already measured to destroy the arrangement,
    which is the thing that distinguishes one lattice position from the
    next. These stay ordered.
    """
    a = patch.astype(np.float64)
    prof = a.mean(axis=0) if axis == 1 else a.mean(axis=1)
    prof = prof - prof.mean()
    n = len(prof)
    if pitch is None or pitch < 2.5:
        return None

    pos, amp, wid, cen = [], [], [], []
    i = 0.0
    while i + pitch <= n and len(amp) < n_lines:
        lo, hi = int(round(i)), int(round(i + pitch))
        seg = prof[lo:hi]
        if hi - lo < 2:
            break
        k = int(np.argmax(seg))
        pos.append(lo + k)
        amp.append(float(seg.max() - seg.min()))
        half = (seg.max() + seg.min()) / 2.0
        wid.append(float((seg > half).sum()))
        w = np.maximum(seg - seg.min(), 0)
        cen.append(float((np.arange(len(seg)) * w).sum() / max(w.sum(), 1e-9)))
        i += pitch

    # pad rather than reject: pitch ranges from about 4 to 25 px, so a
    # 100 px template holds anywhere from 4 to 25 lines
    if len(amp) < MIN_LINES:
        return None
    while len(amp) < n_lines:
        amp.append(0.0); wid.append(0.0); cen.append(0.0); pos.append(0.0)

    amp = np.asarray(amp)
    wid = np.asarray(wid)
    cen = np.asarray(cen)
    pos = np.asarray(pos, float)

    sa = amp.std()
    f = [amp / sa if sa > 1e-9 else amp,
         np.diff(amp, prepend=amp[0]),
         np.diff(amp, n=2, prepend=amp[:2]),
         wid - wid.mean(),
         cen - cen.mean(),
         np.diff(pos, prepend=pos[0]) - pitch]
    return np.concatenate(f)


def build(n, kind, seed, level):
    from driftsense.generator.sample import generate_sample, build_params
    p = build_params(noise_level=level)

    X, y, groups = [], [], []
    for i in range(n):
        s = generate_sample(i, kind, base_seed=seed, params=p)
        # the REFERENCE's own pitch, not the search image's dominant one.
        # The search holds several mats at different pitches, and the one
        # that dominates its spectrum need not be the mat the reference
        # came from.
        ref_t = cv2.resize(s.reference, (T, T), interpolation=cv2.INTER_AREA)
        px = dominant_pitch(ref_t, 1)
        if px is None or px < 3:
            continue
        f_ref = structural_observations(ref_t, px, 1)
        if f_ref is None:
            continue

        for ci, c in enumerate(CLASSES):
            cx = s.gt_x + c * px
            if not (50 <= cx <= 950):
                continue
            x0 = int(np.clip(round(cx - T / 2), 0, s.search.shape[1] - T))
            y0 = int(np.clip(round(s.gt_y - T / 2), 0,
                             s.search.shape[0] - T))
            cand = s.search[y0:y0 + T, x0:x0 + T]
            f_c = structural_observations(cand, px, 1)
            if f_c is None:
                continue
            # comparative, but ORDERED: difference and product per line
            X.append(np.concatenate([f_ref - f_c, f_ref * f_c]))
            y.append(ci)
            groups.append(i)
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{n}   examples {len(y)}", flush=True)
    return np.asarray(X), np.asarray(y), np.asarray(groups)


def report(name, yt, yp):
    acc = float((yt == yp).mean())
    bal = float(balanced_accuracy_score(yt, yp))
    zero = yt == CLASSES.index(0)
    near = np.isin(yt, [CLASSES.index(v) for v in (-2, -1, 1, 2)])
    print(f"\n  {name}")
    print(f"    accuracy            : {acc:.4f}   (chance {1 / len(CLASSES):.4f})")
    print(f"    balanced accuracy   : {bal:.4f}")
    if zero.sum():
        rec = float((yp[zero] == yt[zero]).mean())
        pred0 = yp == CLASSES.index(0)
        prec = float((yt[pred0] == CLASSES.index(0)).mean()) if pred0.sum() else 0.0
        print(f"    n = 0 recall        : {rec:.4f}   the true site")
        print(f"    n = 0 precision     : {prec:.4f}"
              f"   of {int(pred0.sum())} called n=0")
        if rec + prec > 0:
            print(f"    n = 0 F1            : {2 * rec * prec / (rec + prec):.4f}")
    if near.sum():
        print(f"    +/-1, +/-2 accuracy : "
              f"{float((yp[near] == yt[near]).mean()):.4f}"
              f"   the slips that cost accuracy")
    return acc, bal


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--kind", default="dram", choices=["dram", "finfet"])
    ap.add_argument("--seed", type=int, default=24680)
    ap.add_argument("--level", default="medium")
    a = ap.parse_args(argv)

    print("=" * 76)
    print(f"LINE-COUNT ORACLE   kind={a.kind}  n={a.n}")
    print("=" * 76)
    print(f"\n  Classes: {CLASSES}")
    print("  True pitch and candidate family supplied. Synthetic only.\n")

    X, y, g = build(a.n, a.kind, a.seed, a.level)
    if len(y) < 200:
        print("  too few examples")
        return 1

    # split by SAMPLE so no reference appears on both sides
    samples = np.unique(g)
    cut = int(len(samples) * 0.7)
    tr = np.isin(g, samples[:cut])
    te = ~tr
    print(f"\n  examples {len(y)}   train {int(tr.sum())}"
          f"   held out {int(te.sum())}")

    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])

    knn = KNeighborsClassifier(n_neighbors=5).fit(Xtr, y[tr])
    a_knn, b_knn = report("nearest neighbour  (is the information present?)",
                          y[te], knn.predict(Xte))

    lr = LogisticRegression(max_iter=4000).fit(
        Xtr, y[tr])
    a_lr, b_lr = report("logistic regression  (can a simple model use it?)",
                        y[te], lr.predict(Xte))

    # The measurement that decides it. Classification accuracy is not the
    # question -- the pipeline does not classify each candidate in
    # isolation, it picks one from a competing set. So: within each
    # sample's own family of candidates, does argmax P(n=0) land on the
    # true one? That is exactly what the localizer would do.
    print("\n  RANKING WITHIN EACH SAMPLE'S CANDIDATE SET")
    zero_idx = CLASSES.index(0)
    for nm, mdl in (("nearest neighbour", knn), ("logistic", lr)):
        prob = mdl.predict_proba(Xte)
        cols = list(mdl.classes_)
        if zero_idx not in cols:
            continue
        p0 = prob[:, cols.index(zero_idx)]
        gte, yte = g[te], y[te]
        wins = tot = 0
        for sm in np.unique(gte):
            m = gte == sm
            if m.sum() < 3 or not (yte[m] == zero_idx).any():
                continue
            tot += 1
            wins += int(yte[m][int(np.argmax(p0[m]))] == zero_idx)
        if tot:
            print(f"    {nm:<20} true site ranked first in {wins}/{tot}"
                  f"  ({wins / tot * 100:.0f}%)")

    print("\n  confusion, logistic  (rows true, cols predicted)")
    cm = confusion_matrix(y[te], lr.predict(Xte),
                          labels=list(range(len(CLASSES))))
    print("        " + "".join(f"{c:>6}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"  {c:>5} " + "".join(f"{v:>6}" for v in cm[i]))

    best = max(b_knn, b_lr)
    chance = 1.0 / len(CLASSES)
    print("\n" + "=" * 76)
    if best > 0.5:
        print("  The integer offset IS recoverable from ordered structure.")
        print("  That is an information-existence result the twelve")
        print("  score-based attempts could not have found. Build the")
        print("  geometric branch on it -- but note the pitch and candidate")
        print("  family were supplied here, and a real implementation must")
        print("  establish both on its own.")
    elif best > 2 * chance:
        print("  Weak but above chance. Some information about the integer")
        print("  offset survives in the arrangement. Check the confusion")
        print("  matrix: if it separates sign but not magnitude, a coarse")
        print("  direction cue may still be worth having.")
    else:
        print("  The integer offset is NOT recoverable, even with pitch and")
        print("  candidate family supplied. Ordered structure carries no")
        print("  more information than the scores did. This closes the")
        print("  geometric-consistency branch the way the oracle closed")
        print("  pixel weighting.")
    print("=" * 76)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
