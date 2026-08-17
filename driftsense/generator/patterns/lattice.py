"""Mat-level pattern synthesis.

Everything here works in the nm domain on the fine canvas (1 nm/px). Two
things are deliberately present that the reference generator omits:

  line-edge roughness (LER)
      Real lithographic edges wander. The wander is spatially correlated
      with a correlation length of tens of nm, not white noise, so it is
      generated as smoothed Gaussian noise along each line.

  contact-level defects
      Missing contacts and bridged neighbours. Applied once to the canvas,
      so they appear consistently in both the reference and the search
      image -- they are properties of the device, not of the capture.
"""

from __future__ import annotations

import numpy as np
import cv2


def _correlated_noise(n, sigma, corr_len, rng):
    """Gaussian noise with a correlation length, for edge wander."""
    if sigma <= 0:
        return np.zeros(n, dtype=np.float32)
    raw = rng.normal(0.0, 1.0, n + int(6 * corr_len)).astype(np.float32)
    k = max(int(round(corr_len)) | 1, 3)
    sm = cv2.GaussianBlur(raw.reshape(-1, 1), (1, k), corr_len / 2.0).ravel()
    sd = sm.std()
    if sd > 1e-6:
        sm *= sigma / sd
    return sm[:n]


def _draw_bands(canvas, positions, width, value, axis, rng,
                ler_sigma=0.0, ler_corr=25.0):
    """Draw parallel bands with optional edge roughness.

    axis=0 -> horizontal bands (constant y), axis=1 -> vertical bands.
    Only the rows/columns spanned by each band are touched, so cost is
    O(width) per band rather than O(canvas).
    """
    h, w = canvas.shape
    span = w if axis == 0 else h
    extent = h if axis == 0 else w
    half = width / 2.0
    pad = int(np.ceil(half + 3 * ler_sigma)) + 2

    coords = np.arange(span, dtype=np.float32)
    for c in positions:
        wander = _correlated_noise(span, ler_sigma, ler_corr, rng)
        lo_edge = c - half + wander
        hi_edge = c + half + wander
        r0 = int(np.floor(lo_edge.min())) - 1
        r1 = int(np.ceil(hi_edge.max())) + 1
        r0, r1 = max(r0, 0), min(r1, extent)
        if r1 <= r0:
            continue
        rows = np.arange(r0, r1, dtype=np.float32)[:, None]
        mask = (rows >= lo_edge[None, :]) & (rows <= hi_edge[None, :])
        if axis == 0:
            block = canvas[r0:r1, :]
            block[mask] = value
        else:
            block = canvas[:, r0:r1]
            block[mask.T] = value
    return canvas


def _positions(size, pitch, rng):
    """Band centres across `size`, with a random phase so mats do not all
    start on the same grid line."""
    phase = rng.uniform(0, pitch)
    return np.arange(phase, size, pitch, dtype=np.float32)


def generate_mat(size_px, preset, rng, ler_sigma_nm=2.0,
                 linewidth_bias_nm=0.0, corner_rounding_px=0.0,
                 defect_rate=0.004):
    """One mat of periodic device pattern, `size_px` square, at 1 nm/px."""
    p = preset
    canvas = np.full((size_px, size_px), p.val_bg, dtype=np.uint8)

    fine_w = max(p.fine_width_nm + linewidth_bias_nm, 1.0)
    coarse_w = max(p.coarse_width_nm + linewidth_bias_nm, 1.0)

    fine_pos = _positions(size_px, p.fine_pitch_nm, rng)
    coarse_pos = _positions(size_px, p.coarse_pitch_nm, rng)

    if p.kind == "dram":
        # word-lines horizontal, bit-lines vertical, contact at each crossing
        _draw_bands(canvas, fine_pos, fine_w, p.val_fine, axis=0, rng=rng,
                    ler_sigma=ler_sigma_nm)
        _draw_bands(canvas, coarse_pos, coarse_w, p.val_coarse, axis=1, rng=rng,
                    ler_sigma=ler_sigma_nm)
        _draw_contacts(canvas, fine_pos, coarse_pos, p, rng, defect_rate)
    else:
        # fins vertical (dense), gate bars horizontal (sparse), contacts on gates
        _draw_bands(canvas, fine_pos, fine_w, p.val_fine, axis=1, rng=rng,
                    ler_sigma=ler_sigma_nm)
        _draw_bands(canvas, coarse_pos, coarse_w, p.val_coarse, axis=0, rng=rng,
                    ler_sigma=ler_sigma_nm * 0.6)
        _draw_contacts(canvas, coarse_pos, fine_pos, p, rng, defect_rate)

    if corner_rounding_px > 0:
        k = int(corner_rounding_px) | 1
        canvas = cv2.morphologyEx(
            canvas, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return canvas


def _draw_contacts(canvas, rows, cols, preset, rng, defect_rate):
    """Contacts at lattice crossings, with occasional missing ones."""
    r = max(int(round(preset.contact_nm / 2.0)), 1)
    size = canvas.shape[0]
    for y in rows:
        yi = int(round(y))
        if not (r <= yi < size - r):
            continue
        for x in cols:
            xi = int(round(x))
            if not (r <= xi < size - r):
                continue
            if rng.random() < defect_rate:
                continue                      # missing contact
            cv2.circle(canvas, (xi, yi), r, int(preset.val_contact), -1)
    return canvas
