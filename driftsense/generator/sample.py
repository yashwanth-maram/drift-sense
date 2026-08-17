"""One sample: canvas, crop, two captures, ground truth.

D13 -- per-sample derived RNG. Three independent streams are spawned from
(base_seed, index): one for geometry, one for the reference capture, one
for the search capture. Changing an imaging parameter therefore cannot
shift the crop position, so two runs that differ only in noise level are
still directly comparable sample by sample.

The reference generator shares a single stream across the whole loop, so
changing --shear-amplitude-px desynchronises everything after the first
sample. That was observed directly, and it silently breaks paired
comparison, which is what makes an ablation ladder statistically
efficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .patterns.zones import generate_zone_canvas, zone_spans
from . import sem

FINE_CANVAS_PX = 10000
REFERENCE_PX = 1000
SCALE = 10
SEARCH_PX = FINE_CANVAS_PX // SCALE          # 1000


DEFAULTS = {
    "beam_spot_size_nm": 5.0,
    "edge_brighten_strength": 0.55,
    "dose_reference": 2000.0,
    "dose_search": 200.0,
    "detector_sigma_reference": 2.0,
    "detector_sigma_search": 5.0,
    "shear_amplitude_px": 1.5,
    "drift_jitter_px": 0.5,
    "barrel_distortion_k": 0.0,
    "vignette_strength": 0.0,
    "gamma": 1.0,
    "charging_streak_prob": 0.0,
    "charging_streak_intensity": 0.0,
    "speckle_sigma": 0.0,
    "salt_pepper_prob": 0.0,
    "ler_sigma_nm": 2.0,
    "linewidth_bias_nm": 0.0,
    "corner_rounding_px": 0.0,
    "defect_rate": 0.004,
    "mat_size_nm": 2600.0,
    "strip_width_nm": 320.0,
    "boundary_bias": 0.35,
}


NOISE_LEVELS = {
    "clean":   dict(dose_search=900.0, detector_sigma_search=2.0,
                    shear_amplitude_px=0.3, drift_jitter_px=0.1),
    "low":     dict(dose_search=800.0, detector_sigma_search=2.0,
                    shear_amplitude_px=0.5, drift_jitter_px=0.2),
    "medium":  dict(dose_search=200.0, detector_sigma_search=5.0,
                    shear_amplitude_px=1.5, drift_jitter_px=0.5),
    "high":    dict(dose_search=60.0, detector_sigma_search=8.0,
                    shear_amplitude_px=2.5, drift_jitter_px=1.0,
                    speckle_sigma=0.15),
    "severe":  dict(dose_search=25.0, detector_sigma_search=12.0,
                    shear_amplitude_px=4.0, drift_jitter_px=1.8,
                    speckle_sigma=0.30, salt_pepper_prob=0.01),
    # beyond what the reference generator produces -- their test set is
    # stated to be more degraded than their published levels
    "extreme": dict(dose_search=12.0, detector_sigma_search=18.0,
                    shear_amplitude_px=5.5, drift_jitter_px=2.5,
                    speckle_sigma=0.40, salt_pepper_prob=0.02,
                    charging_streak_prob=3.0, charging_streak_intensity=2.0,
                    vignette_strength=0.25, gamma=1.2),
}


@dataclass
class Sample:
    reference: np.ndarray
    search: np.ndarray
    gt_x: float
    gt_y: float
    box: tuple
    meta: dict = field(default_factory=dict)


def build_params(noise_level=None, **overrides):
    p = dict(DEFAULTS)
    if noise_level:
        p.update(NOISE_LEVELS[noise_level])
    p.update({k: v for k, v in overrides.items() if v is not None})
    return p


def _pick_crop(size, mat_nm, strip_nm, boundary_bias, rng):
    """Choose the crop origin. With probability `boundary_bias` the crop is
    forced to straddle a strip, which is the easier, landmark-rich case;
    otherwise it is uniform, which often lands in a uniform mat interior."""
    limit = size - REFERENCE_PX
    if rng.random() < boundary_bias:
        spans = zone_spans(size, mat_nm, strip_nm)
        strips = [(a, b) for is_mat, a, b in spans if not is_mat]
        if strips:
            sx = strips[int(rng.integers(0, len(strips)))]
            sy = strips[int(rng.integers(0, len(strips)))]
            x0 = int(np.clip(rng.integers(sx[0] - REFERENCE_PX + 60, sx[1] - 60),
                             0, limit))
            y0 = int(np.clip(rng.integers(sy[0] - REFERENCE_PX + 60, sy[1] - 60),
                             0, limit))
            return x0, y0
    return int(rng.integers(0, limit + 1)), int(rng.integers(0, limit + 1))


def generate_sample(index, kind, base_seed=0, params=None):
    p = params or build_params()

    ss = np.random.SeedSequence([int(base_seed), int(index)])
    geom_rng, ref_rng, srch_rng = (np.random.default_rng(s)
                                   for s in ss.spawn(3))

    zones = generate_zone_canvas(
        FINE_CANVAS_PX, kind, geom_rng,
        mat_nm=p["mat_size_nm"], strip_nm=p["strip_width_nm"],
        ler_sigma_nm=p["ler_sigma_nm"],
        linewidth_bias_nm=p["linewidth_bias_nm"],
        corner_rounding_px=p["corner_rounding_px"],
        defect_rate=p["defect_rate"])
    canvas = zones["canvas"]

    canvas = sem.edge_brighten(canvas, p["edge_brighten_strength"])
    canvas = sem.psf_blur(canvas, p["beam_spot_size_nm"])

    x0, y0 = _pick_crop(FINE_CANVAS_PX, p["mat_size_nm"], p["strip_width_nm"],
                        p["boundary_bias"], geom_rng)

    reference = sem.capture(canvas[y0:y0 + REFERENCE_PX, x0:x0 + REFERENCE_PX],
                            p, ref_rng, is_reference=True)
    search = sem.capture(sem.area_downsample(canvas, SCALE),
                         p, srch_rng, is_reference=False)

    bw = REFERENCE_PX // SCALE
    bx, by = x0 / SCALE, y0 / SCALE
    meta = dict(p)
    meta.update({"index": index, "kind": kind, "base_seed": base_seed,
                 "crop_x0_nm": x0, "crop_y0_nm": y0,
                 "mat_presets": "|".join(sorted(set(zones["mat_presets"])))})

    return Sample(reference=reference, search=search,
                  gt_x=bx + bw / 2.0, gt_y=by + bw / 2.0,
                  box=(bx, by, bw, bw), meta=meta)
