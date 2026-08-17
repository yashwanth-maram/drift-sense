"""Blind noise estimation and variance stabilisation.

The SEM capture chain gives, for a mean level mu,

    Var(y) = c * mu^2  +  a * mu  +  b

with c from multiplicative speckle, a = 255/dose from Poisson electron
counting, and b from additive detector noise.

Zero-normalised cross-correlation is the maximum-likelihood matcher only
when noise is homoscedastic. Under this model, bright regions carry more
variance than dark ones, so they are effectively over-weighted and the
correlation peak is pulled toward them. Stabilising the variance first
makes the matcher well-posed.

The transform follows from the delta method: requiring Var[f(y)] to be
constant gives f'(mu) = 1 / sqrt(c mu^2 + a mu + b), which integrates to

    f(y) = (1/sqrt(c)) * arcsinh( (2 c y + a) / sqrt(4 c b - a^2) )

This is not a choice among options -- it is the unique function that
flattens the measured variance. It degenerates correctly:

    c -> 0   generalized Anscombe, (2/a) sqrt(a y + b)
    a -> 0   arcsinh form, the pure multiplicative case
    c, a -> 0   plain scaling by 1/sqrt(b)

The a -> 0 limit is exactly the transform derived for the KLA restoration
problem, where the operator was Var = a mu^2 + b. The same derivation
covers both problem statements.
"""

from __future__ import annotations

import numpy as np


def estimate_nlf(img, tile=16, keep_frac=0.25):
    """Blind estimate of (c, a, b) for Var(y) = c mu^2 + a mu + b.

    Tiles the image, takes (mean, variance) per tile, and fits the LOWER
    envelope of variance against mean -- structured tiles inflate variance,
    so the floor of the distribution is the noise. Never raises; falls back
    to a mid-range default on degenerate input.
    """
    a_img = img.astype(np.float32)
    h, w = a_img.shape
    ny, nx = h // tile, w // tile
    if ny < 4 or nx < 4:
        return 0.0, 1.275, 25.0

    blocks = (a_img[:ny * tile, :nx * tile]
              .reshape(ny, tile, nx, tile).transpose(0, 2, 1, 3)
              .reshape(-1, tile * tile))
    mu = blocks.mean(axis=1)
    var = blocks.var(axis=1)

    order = np.argsort(mu)
    mu, var = mu[order], var[order]

    n_bins = 12
    edges = np.linspace(mu.min(), mu.max(), n_bins + 1)
    xs, ys = [], []
    for i in range(n_bins):
        m = (mu >= edges[i]) & (mu < edges[i + 1])
        if m.sum() < 8:
            continue
        v = np.sort(var[m])
        k = max(int(len(v) * keep_frac), 1)
        xs.append(float(mu[m].mean()))
        ys.append(float(v[:k].mean()))

    if len(xs) < 4:
        return 0.0, 1.275, 25.0

    xs, ys = np.asarray(xs), np.asarray(ys)
    try:
        coef = np.polyfit(xs, ys, 2)
        c, a, b = float(coef[0]), float(coef[1]), float(coef[2])
    except Exception:
        return 0.0, 1.275, 25.0

    c = max(c, 0.0)
    a = max(a, 0.0)
    b = max(b, 1.0)
    if 4 * c * b <= a * a:
        c = 0.0
    return c, a, b


def vst(y, c, a, b):
    """Forward variance-stabilising transform. The inverse is never needed:
    matching happens in transform space and no image is reconstructed."""
    y = y.astype(np.float32)
    if c > 1e-6 and 4 * c * b > a * a:
        d = np.sqrt(4 * c * b - a * a)
        return (np.arcsinh((2 * c * y + a) / d) / np.sqrt(c)).astype(np.float32)
    if a > 1e-6:
        return (2.0 / a * np.sqrt(np.maximum(a * y + 3 * a * a / 8 + b, 0))
                ).astype(np.float32)
    return (y / np.sqrt(max(b, 1e-6))).astype(np.float32)


def heteroscedasticity_ratio(img, transformed=None, tile=16, n_bins=8):
    """Diagnostic: ratio of noise variance in the brightest intensity bin to
    the darkest. 1.0 means fully stabilised. Reported before and after the
    transform as evidence that it does what it claims."""
    src = img.astype(np.float32)
    ref = src if transformed is None else transformed.astype(np.float32)
    h, w = src.shape
    ny, nx = h // tile, w // tile
    if ny < 4 or nx < 4:
        return float("nan")

    def blocks(x):
        return (x[:ny * tile, :nx * tile]
                .reshape(ny, tile, nx, tile).transpose(0, 2, 1, 3)
                .reshape(-1, tile * tile))

    mu = blocks(src).mean(axis=1)
    var = blocks(ref).var(axis=1)
    edges = np.quantile(mu, np.linspace(0, 1, n_bins + 1))
    floors = []
    for i in range(n_bins):
        m = (mu >= edges[i]) & (mu <= edges[i + 1])
        if m.sum() < 8:
            continue
        v = np.sort(var[m])
        floors.append(float(v[:max(len(v) // 4, 1)].mean()))
    if len(floors) < 3:
        return float("nan")
    lo = min(f for f in floors if f > 0)
    return float(max(floors) / lo) if lo > 0 else float("nan")
