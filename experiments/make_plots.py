#!/usr/bin/env python3
"""Figures for the README, the presentation and the documentation.

Everything is computed from a real evaluation run, not hard-coded. Point it
at a dataset and it measures, then plots.

    python experiments/make_plots.py --data ./data --n 100 --out ./figures

Produces:

    01_accuracy.png       headline comparison against plain matching
    02_conditions.png     accuracy by whether a strip pins each axis
    03_error_dist.png     the bimodal error distribution
    04_margin.png         the mechanism -- the margin changes sign
    05_confidence.png     precision-recall, and hit/miss separation
    06_noise.png          degradation across noise levels  (--noise)

The margin figure is the important one. It shows why the method works
rather than only that it does.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from localize import localize, template_at_offset, SCALE, OFFSET_STEP
from driftsense.localize import _strip_centres

T = 100
TOL = 5.0

# a restrained technical palette -- dark ground, two accents, no decoration
BG = "#0d1117"
FG = "#e6edf3"
GRID = "#21262d"
ACCENT = "#2f81f7"
GOOD = "#3fb950"
BAD = "#f85149"
MUTED = "#8b949e"


def style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG,
        "savefig.facecolor": BG, "text.color": FG,
        "axes.labelcolor": FG, "axes.edgecolor": GRID,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": GRID, "axes.grid": True, "grid.alpha": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11, "axes.titlesize": 13,
        "axes.titleweight": "bold", "figure.dpi": 130,
        "font.family": "DejaVu Sans",
    })


def plain_locate(ref, srch):
    size = max(int(round(min(srch.shape) / SCALE)), 8)
    t = cv2.resize(ref, (size, size), interpolation=cv2.INTER_AREA)
    s = cv2.matchTemplate(srch, t, cv2.TM_CCOEFF_NORMED)
    k = int(np.argmax(s))
    y, x = np.unravel_index(k, s.shape)
    return x + size / 2.0, y + size / 2.0, s, size


def margins(ref, srch, gx, gy, tol=TOL):
    """Margin between the true site and the best competitor, both ways."""
    _, _, surf, size = plain_locate(ref, srch)
    ty = int(np.clip(round(gy - size / 2), 0, surf.shape[0] - 1))
    tx = int(np.clip(round(gx - size / 2), 0, surf.shape[1] - 1))

    def gap(s):
        v = float(s[max(ty - 2, 0):ty + 3, max(tx - 2, 0):tx + 3].max())
        m = s.copy()
        y0, y1 = max(ty - 20, 0), min(ty + 21, s.shape[0])
        x0, x1 = max(tx - 20, 0), min(tx + 21, s.shape[1])
        m[y0:y1, x0:x1] = -2.0
        return v - float(m.max())

    plain = gap(surf)

    best = None
    resample = max(ref.shape[0] / float(size), 1.0)
    pmax = max(int(round(resample)), 1)
    k = int(2 * round(3 * 1.0) + 1)
    s_lp = cv2.GaussianBlur(srch, (k, k), 1.0)
    for py in range(0, pmax, OFFSET_STEP):
        for px in range(0, pmax, OFFSET_STEP):
            t = template_at_offset(ref, float(px), float(py), size, resample)
            s = cv2.matchTemplate(s_lp, cv2.GaussianBlur(t, (k, k), 1.0),
                                  cv2.TM_CCOEFF_NORMED)
            if best is None or s.max() > best.max():
                best = s
    return plain, gap(best)


def condition(ref):
    t = cv2.resize(ref, (T, T), interpolation=cv2.INTER_AREA)
    return (len(_strip_centres(t, 1, 0.45, 6)) > 0) \
        + (len(_strip_centres(t, 0, 0.45, 6)) > 0)


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


def average_precision(conf, correct):
    conf, correct = np.asarray(conf), np.asarray(correct, bool)
    n = len(conf)
    o = np.argsort(-conf)
    c = correct[o]
    tp, fp = np.cumsum(c), np.cumsum(~c)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / n
    ap, prev = 0.0, 0.0
    for p, r in zip(prec, rec):
        ap += p * (r - prev)
        prev = r
    return float(ap), prec, rec


def bar_labels(ax, bars, fmt="{:.1f}%"):
    for b in bars:
        h = b.get_height()
        if not np.isfinite(h):
            continue
        ax.text(b.get_x() + b.get_width() / 2, h + 1.5, fmt.format(h),
                ha="center", va="bottom", color=FG, fontweight="bold",
                fontsize=10.5)


# ------------------------------------------------------------------ figures

def fig_accuracy(res, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    vals = [res["plain_acc"] * 100, res["ours_acc"] * 100]
    bars = ax.bar(["Plain template\nmatching", "Drift-Sense"], vals,
                  color=[MUTED, ACCENT], width=0.55, zorder=3)
    bar_labels(ax, bars)
    d = vals[1] - vals[0]
    ax.annotate(f"+{d:.1f} pp", xy=(1, vals[1]), xytext=(0.5, vals[1] + 12),
                ha="center", color=GOOD, fontweight="bold", fontsize=14)
    ax.set_ylim(0, 118)
    ax.set_ylabel("accuracy at 5 px")
    ax.set_title(f"Localisation accuracy   ·   n = {res['n']}", loc="left")
    ax.text(0, -18, "Applied Materials reference generator",
            color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "01_accuracy.png"), bbox_inches="tight")
    plt.close(fig)


def fig_conditions(res, out):
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    names = ["both axes\npinned", "one axis\npinned", "neither\npinned"]
    x = np.arange(3)
    p = [res["cond_plain"].get(c, np.nan) * 100 for c in (2, 1, 0)]
    o = [res["cond_ours"].get(c, np.nan) * 100 for c in (2, 1, 0)]
    b1 = ax.bar(x - 0.2, p, 0.38, label="plain matching", color=MUTED, zorder=3)
    b2 = ax.bar(x + 0.2, o, 0.38, label="Drift-Sense", color=ACCENT, zorder=3)
    bar_labels(ax, b1)
    bar_labels(ax, b2)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 118)
    ax.set_ylabel("accuracy at 5 px")
    ax.set_title("Accuracy by reference condition", loc="left", pad=34)
    ax.text(0, -22, "whether a peripheral strip anchors each axis — "
                    "known before matching runs", color=MUTED, fontsize=9)
    ax.legend(frameon=False, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "02_conditions.png"), bbox_inches="tight")
    plt.close(fig)


def fig_error_dist(res, out):
    e = res["errs"]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    edges = [0, 1, 2, 5, 10, 20, 50, 100, 1e4]
    labels = ["<1", "1–2", "2–5", "5–10", "10–20", "20–50", "50–100", ">100"]
    counts = [int(((e >= lo) & (e < hi)).sum())
              for lo, hi in zip(edges[:-1], edges[1:])]
    cols = [GOOD if i < 3 else BAD for i in range(len(counts))]
    bars = ax.bar(labels, counts, color=cols, width=0.6, zorder=3)
    for b, c in zip(bars, counts):
        if c:
            ax.text(b.get_x() + b.get_width() / 2, c + max(counts) * 0.02,
                    str(c), ha="center", color=FG, fontweight="bold")
    ax.set_xlabel("error (pixels)")
    ax.set_ylabel("samples")
    ax.set_title("Error distribution is bimodal", loc="left")
    ax.text(0, -max(counts) * 0.22,
            "either sub-pixel, or a different mat entirely — "
            "nothing in between",
            color=MUTED, fontsize=9)
    ax.legend(handles=[Patch(color=GOOD, label="within tolerance"),
                       Patch(color=BAD, label="miss")],
              frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "03_error_dist.png"), bbox_inches="tight")
    plt.close(fig)


def fig_margin(res, out):
    """The mechanism. Left panel is the readable number; right is detail."""
    mp, mo = res["margin_plain"], res["margin_ours"]
    n = len(mp)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3),
                                 gridspec_kw={"width_ratios": [1, 1.5]})

    wins = [int((mp > 0).sum()), int((mo > 0).sum())]
    bars = a1.bar(["plain\nmatching", "Drift-Sense"],
                  [w / n * 100 for w in wins],
                  color=[MUTED, ACCENT], width=0.55, zorder=3)
    for b, w in zip(bars, wins):
        a1.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                f"{w}/{n}", ha="center", color=FG, fontweight="bold",
                fontsize=12)
    a1.set_ylim(0, 118)
    a1.set_ylabel("% of samples where the true site outscores\nits best competitor")
    a1.set_title("Does the true site win?", loc="left")

    # zoom the distribution to where the decision actually happens
    lo, hi = np.percentile(np.r_[mp, mo], [2, 88])
    pad = (hi - lo) * 0.12
    bins = np.linspace(lo - pad, hi + pad, 30)
    a2.hist(np.clip(mp, bins[0], bins[-1]), bins=bins, color=MUTED,
            alpha=0.9, label="plain matching", zorder=3)
    a2.hist(np.clip(mo, bins[0], bins[-1]), bins=bins, color=ACCENT,
            alpha=0.7, label="Drift-Sense", zorder=4)
    a2.axvline(0, color=FG, lw=1.3, ls="--", zorder=5)
    ymax = a2.get_ylim()[1]
    a2.text(-0.004, ymax * 0.92, "loses", color=MUTED, fontsize=9,
            ha="right", va="top")
    a2.text(0.004, ymax * 0.92, "wins", color=MUTED, fontsize=9,
            ha="left", va="top")
    a2.set_xlabel("ZNCC(true site) − ZNCC(best competitor)")
    a2.set_ylabel("samples")
    a2.set_title(f"median {np.median(mp):+.4f}  →  {np.median(mo):+.4f}",
                 loc="left")
    a2.legend(frameon=False, loc="upper right")

    fig.suptitle("The mechanism: taking the maximum over a resampling "
                 "family flips the margin",
                 x=0.012, ha="left", fontsize=13, fontweight="bold",
                 color=FG, y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "04_margin.png"), bbox_inches="tight")
    plt.close(fig)


def fig_confidence(res, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ap, prec, rec = res["pr"]
    a1.plot(rec, prec, color=ACCENT, lw=2.4, zorder=3)
    a1.fill_between(rec, prec, alpha=0.18, color=ACCENT, zorder=2)
    a1.set_xlabel("recall")
    a1.set_ylabel("precision")
    a1.set_ylim(0, 1.05)
    a1.set_xlim(0, 1.0)
    a1.set_title(f"Precision–recall   ·   AP = {ap:.4f}", loc="left")

    e, c = res["errs"], res["confs"]
    hit, miss = c[e <= TOL], c[e > TOL]
    parts = [hit, miss] if len(miss) else [hit]
    labs = ["correct", "incorrect"] if len(miss) else ["correct"]
    cols = [GOOD, BAD][:len(parts)]
    bp = a2.boxplot(parts, tick_labels=labs, patch_artist=True, widths=0.5,
                    medianprops=dict(color=FG, lw=2))
    for p, col in zip(bp["boxes"], cols):
        p.set_facecolor(col)
        p.set_alpha(0.75)
    for k in ("whiskers", "caps", "fliers"):
        for it in bp[k]:
            it.set_color(MUTED)
    a2.set_ylabel("confidence")
    a2.set_title("Confidence separates hits from misses", loc="left")
    if len(miss):
        a2.text(0.5, -0.22,
                f"median {np.median(hit):.3f} vs {np.median(miss):.3f}   ·   "
                f"top 80% by confidence: "
                f"{float((e[np.argsort(-c)[:int(len(e) * .8)]] <= TOL).mean()) * 100:.0f}% accurate",
                transform=a2.transAxes, color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "05_confidence.png"), bbox_inches="tight")
    plt.close(fig)


def fig_noise(levels, accs, aps, out):
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(len(levels))
    ax.plot(x, [a * 100 for a in accs], "-o", color=ACCENT, lw=2.4, ms=8,
            label="accuracy at 5 px", zorder=4)
    ax.plot(x, [a * 100 for a in aps], "--s", color=GOOD, lw=2, ms=6,
            label="average precision ×100", zorder=3)
    for i, a in enumerate(accs):
        ax.text(i, a * 100 + 4, f"{a * 100:.0f}%", ha="center", color=FG,
                fontweight="bold", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylim(0, 112)
    ax.set_ylabel("percent")
    ax.set_title("Degradation across noise levels", loc="left")
    ax.text(0, -18, "confidence tracks accuracy all the way down — "
                    "the score does not stay falsely high",
            color=MUTED, fontsize=9)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "06_noise.png"), bbox_inches="tight")
    plt.close(fig)


def evaluate(pairs, tol=TOL, with_margin=True):
    errs, confs, conds, times = [], [], [], []
    perr, mp, mo = [], [], []
    for i, (idx, ref, srch, gx, gy) in enumerate(pairs):
        r = localize(ref, srch)
        errs.append(float(np.hypot(r.x - gx, r.y - gy)))
        confs.append(r.confidence)
        times.append(r.ms)
        conds.append(condition(ref))
        px, py, _, _ = plain_locate(ref, srch)
        perr.append(float(np.hypot(px - gx, py - gy)))
        if with_margin:
            a, b = margins(ref, srch, gx, gy, tol)
            mp.append(a)
            mo.append(b)
        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{len(pairs)}", flush=True)

    e, pe = np.asarray(errs), np.asarray(perr)
    c, cond = np.asarray(confs), np.asarray(conds)
    return {
        "n": len(e), "errs": e, "plain_errs": pe, "confs": c, "cond": cond,
        "times": np.asarray(times),
        "ours_acc": float((e <= tol).mean()),
        "plain_acc": float((pe <= tol).mean()),
        "cond_ours": {k: float((e[cond == k] <= tol).mean())
                      for k in (0, 1, 2) if (cond == k).any()},
        "cond_plain": {k: float((pe[cond == k] <= tol).mean())
                       for k in (0, 1, 2) if (cond == k).any()},
        "pr": average_precision(c, e <= tol),
        "margin_plain": np.asarray(mp) if mp else np.zeros(0),
        "margin_ours": np.asarray(mo) if mo else np.zeros(0),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="./figures")
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--noise", action="store_true",
                    help="also sweep noise levels using the generator")
    ap.add_argument("--noise-n", type=int, default=30)
    ap.add_argument("--no-margin", action="store_true",
                    help="skip the margin figure, which is the slow one")
    a = ap.parse_args(argv)

    style()
    os.makedirs(a.out, exist_ok=True)
    pairs = load(a.data, a.n)

    print("=" * 62)
    print(f"PLOTS   {a.data}   n={len(pairs)}")
    print("=" * 62)
    print("\n  measuring ...")
    t0 = time.time()
    res = evaluate(pairs, a.tol, not a.no_margin)
    print(f"  done in {time.time() - t0:.0f}s\n")

    print(f"  accuracy   plain {res['plain_acc'] * 100:.1f}%"
          f"  ->  ours {res['ours_acc'] * 100:.1f}%")
    print(f"  median err {np.median(res['errs']):.2f} px")
    print(f"  AP         {res['pr'][0]:.4f}")
    print(f"  time       {res['times'].mean():.0f} ms/pair\n")

    fig_accuracy(res, a.out)
    fig_conditions(res, a.out)
    fig_error_dist(res, a.out)
    if not a.no_margin:
        fig_margin(res, a.out)
    fig_confidence(res, a.out)

    if a.noise:
        from driftsense.generator.sample import generate_sample, build_params
        levels = ["low", "medium", "high", "severe", "extreme"]
        accs, aps = [], []
        for lv in levels:
            print(f"  noise sweep: {lv} ...", flush=True)
            p = build_params(noise_level=lv)
            gp = []
            for i in range(a.noise_n):
                s = generate_sample(i, "dram", base_seed=3131, params=p)
                gp.append((i, s.reference, s.search, s.gt_x, s.gt_y))
            r = evaluate(gp, a.tol, with_margin=False)
            accs.append(r["ours_acc"])
            aps.append(r["pr"][0])
        fig_noise(levels, accs, aps, a.out)

    print(f"\n  figures written to {a.out}")
    for f in sorted(os.listdir(a.out)):
        if f.endswith(".png"):
            print(f"    {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
