#!/usr/bin/env python3
"""Is the remaining 25% solvable at all?

Every remaining failure is a mat-interior crop whose free axis slipped by
an integer number of lattice periods. The claim to test is that those
candidates carry no information distinguishing them -- that the failure is
a property of the data rather than of the matcher.

Two independent measurements:

PART A -- within-image similarity  (runs on any dataset you already have)
    For each failure, pull the true 100x100 region and the predicted one
    from the SAME search image and compare them directly. If they are
    near-identical, nothing could have told them apart. Reported against
    the similarity between the template and each, so noise is controlled
    for.

PART B -- the clean oracle  (needs our generator; AM's cannot do this)
    Regenerate the identical sample at near-zero noise and re-run. Our
    generator derives geometry from (seed, index) independently of imaging
    parameters, so the crop position is unchanged when noise is removed.
    AM's shares one RNG stream across the loop, so changing any imaging
    parameter shifts the geometry too and the comparison is impossible.
    If a failure survives at zero noise, it is not a noise problem.

    python experiments/measure_ceiling.py --data ../am_eval --n 100
    python experiments/measure_ceiling.py --data ../am_eval --n 100 --oracle
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
HALF = 50


def zncc(a, b):
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float((a * b).sum() / (na * nb))


def region(img, cx, cy, half=HALF):
    h, w = img.shape
    x = int(np.clip(round(cx - half), 0, w - 2 * half))
    y = int(np.clip(round(cy - half), 0, h - 2 * half))
    return img[y:y + 2 * half, x:x + 2 * half]


def load_manifest(data_dir):
    for root, _, files in os.walk(data_dir):
        if "manifest.csv" in files:
            with open(os.path.join(root, "manifest.csv")) as fh:
                return list(csv.DictReader(fh)), root
    raise SystemExit(f"no manifest.csv under {data_dir}")


def part_a(rows, root, rung, tol):
    print("=" * 70)
    print("PART A -- how similar are the true and predicted regions?")
    print("=" * 70)
    print("\n  For each failure: the two candidate regions, taken from the")
    print("  same search image, compared to each other and to the template.\n")
    print(f"  {'id':>4} {'error':>8} {'true~pred':>10} {'tmpl~true':>10}"
          f" {'tmpl~pred':>10}   verdict")
    print("  " + "-" * 62)

    stats = []
    for row in rows:
        rp, sp = row["reference_path"], row["search_path"]
        if not os.path.exists(rp):
            rp = os.path.join(root, "reference", f"{int(row['id']):05d}.png")
            sp = os.path.join(root, "search", f"{int(row['id']):05d}.png")
        ref = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
        srch = cv2.imread(sp, cv2.IMREAD_GRAYSCALE)
        if ref is None or srch is None:
            continue

        gx, gy = float(row["gt_x"]), float(row["gt_y"])
        r = localize(ref, srch, rung=rung)
        err = float(np.hypot(r.x - gx, r.y - gy))
        if err <= tol:
            continue

        tmpl = cv2.resize(ref, (2 * HALF, 2 * HALF), interpolation=cv2.INTER_AREA)
        rt = region(srch, gx, gy)
        rp_ = region(srch, r.x, r.y)
        s_tp = zncc(rt, rp_)
        s_mt = zncc(tmpl, rt)
        s_mp = zncc(tmpl, rp_)

        if s_mp > s_mt:
            v = "wrong scores HIGHER"
        elif s_tp > 0.85:
            v = "regions near-identical"
        else:
            v = "regions differ"
        stats.append((s_tp, s_mt, s_mp))
        print(f"  {int(row['id']):>4} {err:>8.1f} {s_tp:>10.3f} {s_mt:>10.3f}"
              f" {s_mp:>10.3f}   {v}")

    if not stats:
        print("\n  no failures found")
        return
    a = np.array(stats)
    print(f"\n  failures                     : {len(a)}")
    print(f"  median true~pred similarity  : {np.median(a[:, 0]):.3f}")
    print(f"  median tmpl~true             : {np.median(a[:, 1]):.3f}")
    print(f"  median tmpl~pred             : {np.median(a[:, 2]):.3f}")
    beats = int((a[:, 2] > a[:, 1]).sum())
    print(f"  wrong region scores higher   : {beats}/{len(a)}"
          f"  ({beats / len(a) * 100:.0f}%)")
    print("\n  Reading this: if tmpl~pred exceeds tmpl~true, the wrong region")
    print("  genuinely matches the template better. No selection rule can")
    print("  recover that -- the information is not there to be used.")


def part_b(n, kind, seed, rung, tol):
    from driftsense.generator.sample import generate_sample, build_params

    print("\n" + "=" * 70)
    print("PART B -- does removing the noise fix it?")
    print("=" * 70)
    print("\n  Same crop, same layout, noise removed. Our generator derives")
    print("  geometry from (seed, index) independently of imaging settings,")
    print("  so these are genuinely paired samples.\n")

    noisy = build_params(noise_level="medium")
    clean = build_params(noise_level="medium", dose_search=4000.0,
                         detector_sigma_search=0.5, speckle_sigma=0.0,
                         salt_pepper_prob=0.0, shear_amplitude_px=0.0,
                         drift_jitter_px=0.0)

    rows = []
    for i in range(n):
        sn = generate_sample(i, kind, base_seed=seed, params=noisy)
        rn = localize(sn.reference, sn.search, rung=rung)
        en = float(np.hypot(rn.x - sn.gt_x, rn.y - sn.gt_y))

        sc = generate_sample(i, kind, base_seed=seed, params=clean)
        rc = localize(sc.reference, sc.search, rung=rung)
        ec = float(np.hypot(rc.x - sc.gt_x, rc.y - sc.gt_y))
        rows.append((i, en, ec, sn.gt_x, sc.gt_x))

    paired = all(abs(r[3] - r[4]) < 1e-6 for r in rows)
    print(f"  pairing check (same crop both runs): "
          f"{'PASS' if paired else 'FAIL -- results not comparable'}")

    print(f"\n  {'id':>4} {'noisy err':>11} {'clean err':>11}   verdict")
    print("  " + "-" * 46)
    fixed = survived = 0
    for i, en, ec, _, _ in rows:
        if en <= tol:
            continue
        if ec <= tol:
            fixed += 1
            v = "noise was the cause"
        else:
            survived += 1
            v = "SURVIVES at zero noise"
        print(f"  {i:>4} {en:>11.1f} {ec:>11.1f}   {v}")

    tot = fixed + survived
    acc_n = np.mean([r[1] <= tol for r in rows]) * 100
    acc_c = np.mean([r[2] <= tol for r in rows]) * 100
    print(f"\n  accuracy noisy : {acc_n:5.1f}%")
    print(f"  accuracy clean : {acc_c:5.1f}%   <-- the achievable ceiling")
    if tot:
        print(f"\n  failures fixed by removing noise : {fixed}/{tot}")
        print(f"  failures that survive             : {survived}/{tot}")
        print("\n  Failures that survive at zero noise are not noise problems.")
        print("  They are cases where the data itself does not distinguish the")
        print("  candidates, and 100% is not the right thing to measure against.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="dataset for Part A")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--rung", type=int, default=1)
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--oracle", action="store_true", help="also run Part B")
    ap.add_argument("--kind", default="dram", choices=["dram", "finfet"])
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args(argv)

    if a.data:
        rows, root = load_manifest(a.data)
        part_a(rows[:a.n], root, a.rung, a.tol)
    if a.oracle or not a.data:
        part_b(min(a.n, 30), a.kind, a.seed, a.rung, a.tol)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
