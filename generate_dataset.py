#!/usr/bin/env python3
"""Drift-Sense synthetic dataset generator.

Generates Reference/Search image pairs with recorded ground truth for
navigation-error recovery in wafer inspection.

    python generate_dataset.py --style DRAM --n 30 --out ./data
    python generate_dataset.py --style FinFET --n 30 --out ./data --noise severe

Flag aliases are accepted so the script works with either the problem
statement's naming (--style, --n, --out) or the reference generator's
(--architectures, --num-samples, --output-dir).

Every sample is a pure function of (seed, index, parameters). The manifest
records all of them, so any pair can be regenerated exactly rather than
stored -- see --regenerate.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

# Run from anywhere: put this script's directory on the import path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2

from driftsense.generator.presets import resolve_style, PRESETS
from driftsense.generator.sample import (
    generate_sample, build_params, NOISE_LEVELS, DEFAULTS)


MANIFEST_FIELDS = ["id", "reference_path", "search_path",
                   "gt_x", "gt_y", "gt_box_x", "gt_box_y",
                   "gt_box_w", "gt_box_h", "kind", "noise_level"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument("--style", "--architecture", "--architectures",
                    dest="style", default="DRAM",
                    help="DRAM or FinFET (a preset name is also accepted)")
    ap.add_argument("--n", "--num-samples", "--count", dest="n",
                    type=int, default=30)
    ap.add_argument("--out", "--output-dir", "--output", dest="out",
                    default="./data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", dest="noise_level", default="medium",
                    choices=sorted(NOISE_LEVELS))
    ap.add_argument("--split", default=None,
                    help="optional subdirectory name under --out")
    ap.add_argument("--homogeneous", action="store_true",
                    help="force one preset per canvas (ablation only)")

    for key, val in DEFAULTS.items():
        ap.add_argument(f"--{key.replace('_', '-')}", dest=key,
                        type=type(val), default=None)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    kind = resolve_style(a.style)

    overrides = {k: getattr(a, k) for k in DEFAULTS}
    params = build_params(a.noise_level, **overrides)

    root = os.path.join(a.out, a.split) if a.split else a.out
    ref_dir = os.path.join(root, "reference")
    srch_dir = os.path.join(root, "search")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(srch_dir, exist_ok=True)

    manifest = os.path.join(root, "manifest.csv")
    extra = sorted(params)
    t0 = time.time()

    with open(manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS +
                           ["seed", "index", "mat_presets"] + extra)
        w.writeheader()

        for i in range(a.n):
            s = generate_sample(i, kind, base_seed=a.seed, params=params)
            rp = os.path.join(ref_dir, f"{i:05d}.png")
            sp = os.path.join(srch_dir, f"{i:05d}.png")
            cv2.imwrite(rp, s.reference)
            cv2.imwrite(sp, s.search)

            row = {"id": i, "reference_path": rp, "search_path": sp,
                   "gt_x": round(s.gt_x, 3), "gt_y": round(s.gt_y, 3),
                   "gt_box_x": round(s.box[0], 3), "gt_box_y": round(s.box[1], 3),
                   "gt_box_w": s.box[2], "gt_box_h": s.box[3],
                   "kind": kind, "noise_level": a.noise_level,
                   "seed": a.seed, "index": i,
                   "mat_presets": s.meta["mat_presets"]}
            row.update({k: params[k] for k in extra})
            w.writerow(row)

            print(f"[{i + 1}/{a.n}] {kind}/{a.noise_level} "
                  f"-> gt=({s.gt_x:.1f}, {s.gt_y:.1f})")

    dt = time.time() - t0
    print(f"\nWrote {a.n} pairs to {root} in {dt:.1f}s "
          f"({dt / max(a.n, 1):.1f}s per pair)")
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
