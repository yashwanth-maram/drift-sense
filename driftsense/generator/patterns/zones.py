"""Large-scale zone composition.

A memory die is not one uniform array. It is built from sub-array mats
separated by strips of peripheral circuitry, sense amplifiers, decoders
and global routing. Each mat is generated independently and may use a
different preset of the same family, so a single field of view contains
several distinct pitches.

That heterogeneity is the main thing that makes localisation tractable in
a repeating layout: local pitch identifies which mat you are in, and the
strips give absolute landmarks.
"""

from __future__ import annotations

import numpy as np
import cv2

from ..presets import presets_for_kind
from .lattice import generate_mat

STRIP_BASE_VAL = 95
STRIP_LINE_VAL = 132
STRIP_LINE_PITCH_NM = 220.0
STRIP_LINE_WIDTH_NM = 9.0


def _strip_texture(size_px, rng):
    """Flat peripheral material with sparse orthogonal routing."""
    canvas = np.full((size_px, size_px), STRIP_BASE_VAL, dtype=np.uint8)
    half = STRIP_LINE_WIDTH_NM / 2.0
    for is_row in (True, False):
        start = rng.uniform(0, STRIP_LINE_PITCH_NM)
        for c in np.arange(start, size_px, STRIP_LINE_PITCH_NM):
            lo = max(int(round(c - half)), 0)
            hi = min(int(round(c + half)) + 1, size_px)
            if hi <= lo:
                continue
            if is_row:
                canvas[lo:hi, :] = STRIP_LINE_VAL
            else:
                canvas[:, lo:hi] = STRIP_LINE_VAL
    return canvas


def zone_spans(size_px, mat_nm, strip_nm):
    """Alternating [mat, strip, mat, strip, ...] spans covering size_px."""
    spans, pos, is_mat = [], 0.0, True
    while pos < size_px:
        end = min(pos + (mat_nm if is_mat else strip_nm), size_px)
        spans.append((is_mat, int(round(pos)), int(round(end))))
        pos = end
        is_mat = not is_mat
    return spans


def generate_zone_canvas(size_px, kind, rng, mat_nm=2600.0, strip_nm=320.0,
                         ler_sigma_nm=2.0, linewidth_bias_nm=0.0,
                         corner_rounding_px=0.0, defect_rate=0.004,
                         heterogeneous=True):
    """Tile mats of `kind` across the canvas, separated by strips."""
    presets = presets_for_kind(kind)
    canvas = _strip_texture(size_px, rng)

    rows = zone_spans(size_px, mat_nm, strip_nm)
    cols = zone_spans(size_px, mat_nm, strip_nm)

    mat_rects, strip_rects, mat_presets = [], [], []

    for r_is_mat, y0, y1 in rows:
        for c_is_mat, x0, x1 in cols:
            if r_is_mat and c_is_mat and y1 > y0 and x1 > x0:
                mh, mw = y1 - y0, x1 - x0
                preset = presets[int(rng.integers(0, len(presets)))] \
                    if heterogeneous else presets[0]
                child = np.random.default_rng(rng.integers(0, 2 ** 31 - 1))
                mat = generate_mat(max(mh, mw), preset, child,
                                   ler_sigma_nm=ler_sigma_nm,
                                   linewidth_bias_nm=linewidth_bias_nm,
                                   corner_rounding_px=corner_rounding_px,
                                   defect_rate=defect_rate)
                canvas[y0:y1, x0:x1] = mat[:mh, :mw]
                mat_rects.append((x0, y0, mw, mh))
                mat_presets.append(preset.name)
            else:
                strip_rects.append((x0, y0, x1 - x0, y1 - y0))

    return {"canvas": canvas, "mat_rects": mat_rects,
            "strip_rects": strip_rects, "mat_presets": mat_presets}
