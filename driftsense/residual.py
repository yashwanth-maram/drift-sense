"""Residual matching: correlate on what is unique, not what repeats.

Measured on 19 failures across both architectures, the true and predicted
regions differ by a median of 24 grey levels on the noise-free canvas.
Nothing is ambiguous. The information distinguishing them is present and
ZNCC is not using it.

The reason is structural. ZNCC weights all 10,000 template pixels equally,
and the overwhelming majority of the image energy sits in the periodic
lattice -- which is identical at every candidate position and therefore
contributes score without contributing discrimination. The aperiodic
content that actually differs (lattice phase relative to the mat, edge
roughness, defects) is a small fraction of the total and gets averaged
away.

So: identify the lattice harmonics in the Fourier spectrum, notch them out
of both template and search, and correlate the residual. The lattice
carries position modulo its period; the residual carries which period.

Combining the two surfaces gives both:

    combined = zncc_raw * w + zncc_residual * (1 - w)

Notching is done with a soft Gaussian stop-band rather than a hard zero,
because a hard notch rings in the spatial domain and the ringing is itself
periodic.
"""

from __future__ import annotations

import numpy as np
import cv2


def _spectrum(img):
    a = img.astype(np.float32)
    a = a - a.mean()
    win = (np.hanning(a.shape[0])[:, None] * np.hanning(a.shape[1])[None, :])
    return np.fft.fftshift(np.fft.fft2(a * win.astype(np.float32)))


def find_harmonics(img, n_peaks=8, min_r=3, max_frac=0.45):
    """Locate the strongest periodic components.

    Returns a list of (dy, dx) offsets from the DC term, in shifted-FFT
    coordinates. Conjugate pairs are both returned since the notch must be
    symmetric for the result to stay real.
    """
    F = _spectrum(img)
    P = np.abs(F)
    h, w = P.shape
    cy, cx = h // 2, w // 2

    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - cy, xx - cx)
    band = (r >= min_r) & (r <= max_frac * min(h, w))

    work = np.where(band, P, 0.0)
    peaks = []
    for _ in range(n_peaks):
        idx = int(np.argmax(work))
        py, px = np.unravel_index(idx, work.shape)
        if work[py, px] <= 0:
            break
        peaks.append((py - cy, px - cx))
        # suppress this peak and its conjugate so the next iteration
        # finds a genuinely different harmonic
        for sy, sx in ((py, px), (2 * cy - py, 2 * cx - px)):
            y0, y1 = max(sy - min_r, 0), min(sy + min_r + 1, h)
            x0, x1 = max(sx - min_r, 0), min(sx + min_r + 1, w)
            work[y0:y1, x0:x1] = 0.0
    return peaks


def notch(img, harmonics, width=2.5):
    """Remove the listed harmonics with a soft Gaussian stop-band.

    A hard notch rings in the spatial domain, and the ringing is periodic,
    which reintroduces exactly the structure being removed.
    """
    a = img.astype(np.float32)
    mean = a.mean()
    F = np.fft.fftshift(np.fft.fft2(a - mean))
    h, w = F.shape
    cy, cx = h // 2, w // 2

    mask = np.ones((h, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for dy, dx in harmonics:
        for sy, sx in ((cy + dy, cx + dx), (cy - dy, cx - dx)):
            d2 = (yy - sy) ** 2 + (xx - sx) ** 2
            mask *= 1.0 - np.exp(-d2 / (2.0 * width ** 2)).astype(np.float32)

    out = np.fft.ifft2(np.fft.ifftshift(F * mask)).real
    return out.astype(np.float32)


def to_uint8(a):
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-6:
        return np.zeros(a.shape, np.uint8)
    return np.clip((a - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def residual_surface(template, search, n_peaks=8, width=2.5):
    """ZNCC on the aperiodic residual of both images.

    The harmonics are found in the SEARCH image, not the template: the
    search contains many mats and therefore a fuller picture of which
    frequencies are lattice rather than content, and using one harmonic
    set for both keeps the two residuals comparable.
    """
    harm = find_harmonics(search, n_peaks=n_peaks)
    if not harm:
        return None
    # scale harmonic offsets from the search grid to the template grid
    sy, sx = search.shape
    ty, tx = template.shape
    t_harm = [(int(round(dy * ty / sy)), int(round(dx * tx / sx)))
              for dy, dx in harm]
    t_harm = [(dy, dx) for dy, dx in t_harm if abs(dy) + abs(dx) >= 2]
    if not t_harm:
        return None

    r_s = to_uint8(notch(search, harm, width))
    r_t = to_uint8(notch(template, t_harm, width))
    return cv2.matchTemplate(r_s, r_t, cv2.TM_CCOEFF_NORMED)


def combine(raw, residual, w_raw=0.5):
    """Blend the raw and residual surfaces.

    Both are ZNCC surfaces on the same grid, so a convex combination is
    well defined. w_raw = 1 recovers plain matching, w_raw = 0 uses the
    residual alone.
    """
    if residual is None:
        return raw
    if residual.shape != raw.shape:
        residual = cv2.resize(residual, (raw.shape[1], raw.shape[0]))
    return (w_raw * raw + (1.0 - w_raw) * residual).astype(np.float32)
