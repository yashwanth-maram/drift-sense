"""Template construction at an exact fractional grid offset.

Measured mechanism. Four template variants were compared at n=60 on both
families:

    A  noisy reference, crop then downsample      66.7% / 80.0%
    B  denoised reference, crop then downsample   66.7% / 80.0%
    C  CLEAN canvas,     crop then downsample     66.7% / 80.0%
    E  CLEAN canvas,     downsample then crop     75.0% / 95.0%

A, B and C agree to four decimals, so reference capture noise contributes
nothing -- at dose 2000 it is sigma ~4, and averaging 100 pixels leaves
0.4. C and E are both noise-free and differ in exactly one respect:
whether the template lands on the SEARCH image's pixel grid. That is worth
+8.3 and +15.0 points, and it closes most of the FinFET score margin
(-0.0239 to -0.0097).

The geometry. The crop origin x0 is an integer in canvas pixels; the
search samples the canvas in blocks of ten. Template pixel m averages
canvas [x0 + 10m, x0 + 10m + 10), while search pixel j covers
[10j, 10j + 10). They coincide only when x0 = 0 (mod 10).

An earlier phase experiment reported this as a null. It approximated the
alignment with warpAffine + INTER_LINEAR before area-averaging, and
bilinear interpolation applies its own low-pass filter, so the result was
a BLURRED approximation of the aligned template rather than the aligned
template. A lossy approximation of the right operation is not a test of
it.

This module resamples exactly. An integral image gives the mean over any
axis-aligned box in constant time, including boxes that start and end at
fractional coordinates, with linear interpolation only at the two
boundary pixels rather than across the whole image.
"""

from __future__ import annotations

import numpy as np
import cv2

T = 100
SCALE = 10


def _integral(img):
    return cv2.integral(img.astype(np.float64))


def _box_mean_rows(ii, y0, y1, x0, x1):
    """Mean over a box with fractional edges, from an integral image.

    Interpolating the integral image at fractional bounds is equivalent to
    a box filter whose partial edge pixels are weighted by their overlap.
    That is the exact area-average, not an interpolated approximation of
    one.
    """
    def interp(y, x):
        y = np.clip(y, 0, ii.shape[0] - 1.000001)
        x = np.clip(x, 0, ii.shape[1] - 1.000001)
        y0i, x0i = np.floor(y).astype(int), np.floor(x).astype(int)
        fy, fx = y - y0i, x - x0i
        a = ii[y0i, x0i]
        b = ii[y0i, x0i + 1]
        c = ii[y0i + 1, x0i]
        d = ii[y0i + 1, x0i + 1]
        return (a * (1 - fy) * (1 - fx) + b * (1 - fy) * fx
                + c * fy * (1 - fx) + d * fy * fx)

    s = interp(y1, x1) - interp(y0, x1) - interp(y1, x0) + interp(y0, x0)
    area = np.maximum((y1 - y0) * (x1 - x0), 1e-9)
    return s / area


def template_at_offset(reference, phi_x, phi_y, size=T, scale=SCALE):
    """Area-average the reference by `scale`, starting at a fractional offset.

    phi_x, phi_y are in reference pixels, within [0, scale). phi = 0
    reproduces a plain INTER_AREA downsample.
    """
    ii = _integral(reference)
    h, w = reference.shape
    n = size
    ys = phi_y + np.arange(n + 1, dtype=np.float64) * scale
    xs = phi_x + np.arange(n + 1, dtype=np.float64) * scale
    ys = np.clip(ys, 0, h)
    xs = np.clip(xs, 0, w)

    Y0, X0 = np.meshgrid(ys[:-1], xs[:-1], indexing="ij")
    Y1, X1 = np.meshgrid(ys[1:], xs[1:], indexing="ij")
    m = _box_mean_rows(ii, Y0, Y1, X0, X1)
    return np.clip(m, 0, 255).astype(np.uint8)


def estimate_offset(reference, scale=SCALE, axis=1):
    """Sub-pixel offset of the reference's lattice from its own pixel grid.

    The offset is a property of the reference alone, not of any candidate,
    so it can be estimated once before matching rather than swept. That
    matters: a sweep of ten offsets per axis is a hundred candidate
    templates, and forty-one candidate SCALES was measured to cost fifteen
    points against five.

    The lattice phase is read from the reference's dominant spectral
    component. The template grid should start where the search grid would
    have, which is where the lattice phase is a multiple of the pitch.
    """
    a = reference.astype(np.float64)
    prof = a.mean(axis=0) if axis == 1 else a.mean(axis=1)
    prof = prof - prof.mean()
    n = len(prof)
    spec = np.fft.rfft(prof * np.hanning(n))
    mag = np.abs(spec)
    kmin = max(int(n / 400), 2)
    kmax = min(int(n / 20), len(mag) - 1)
    if kmax <= kmin:
        return 0.0
    k = int(np.argmax(mag[kmin:kmax + 1])) + kmin
    pitch = n / k
    phase = float(np.angle(spec[k]))
    # position of the first lattice peak, folded into one sampling block
    peak = (-phase / (2 * np.pi)) * pitch
    return float(peak % scale)


def build_template(reference, mode="plain", scale=SCALE, size=T):
    """mode: plain | estimated"""
    if mode == "plain":
        return cv2.resize(reference, (size, size),
                          interpolation=cv2.INTER_AREA)
    if mode == "estimated":
        return template_at_offset(reference,
                                  estimate_offset(reference, scale, 1),
                                  estimate_offset(reference, scale, 0),
                                  size, scale)
    raise ValueError(mode)


def sweep_templates(reference, scale=SCALE, size=T, step=1):
    """All fractional offsets, for the sweep variant."""
    for py in range(0, scale, step):
        for px in range(0, scale, step):
            yield (px, py), template_at_offset(reference, float(px),
                                               float(py), size, scale)
