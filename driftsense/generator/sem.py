"""SEM imaging chain.

Two stages, and the split matters:

  emission  (once, on the shared canvas)
      Edge-brightening. Secondary-electron yield rises with local surface
      tilt roughly as 1/cos(theta), so steep sidewalls emit more electrons
      than flat tops and appear bright. This is a property of the specimen,
      so it is applied ONCE and appears identically in both images.
      The reference generator omits this entirely despite the problem
      statement listing it as mandatory.

  capture   (twice, independently)
      Beam PSF, then geometry (downsample, raster drift, distortion),
      then detection (shot noise, detector noise, speckle, salt-and-pepper),
      then response (vignette, gamma, charging).

The beam PSF is applied to the shared canvas BEFORE any downsampling,
because there is one physical beam. That is what makes a 1/10 area-average
of the reference carry exactly the same blur as the search image.

Noise model. add_shot_noise draws Poisson counts at `dose`, so

    Var(y) = (255/dose) * mu + sigma_det^2 + sigma_speckle^2 * mu^2
           =      a     * mu +      b      +        c        * mu^2

quadratic in mu. The variance-stabilising transform for this family is
derived in driftsense/noise/vst.py.
"""

from __future__ import annotations

import numpy as np
import cv2


# ---------------------------------------------------------------- emission

def edge_brighten(canvas, strength=0.55, width_nm=3.0):
    """Secondary-electron edge bloom.

    Yield rises where the surface is tilted; in a projected image that is
    exactly where the material map has a gradient. Gradient magnitude is
    computed with Sobel, softened to the escape depth of secondary
    electrons, and added.
    """
    if strength <= 0:
        return canvas
    gx = cv2.Sobel(canvas, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(canvas, cv2.CV_16S, 0, 1, ksize=3)
    mag = cv2.convertScaleAbs(gx) // 2 + cv2.convertScaleAbs(gy) // 2
    k = int(round(width_nm)) | 1
    mag = cv2.GaussianBlur(mag, (k, k), width_nm / 2.0)
    return cv2.add(canvas, cv2.convertScaleAbs(mag, alpha=strength))


# ---------------------------------------------------------------- capture

def psf_blur(img, spot_nm):
    if spot_nm <= 0:
        return img
    k = int(round(spot_nm * 3)) | 1
    return cv2.GaussianBlur(img, (k, k), spot_nm)


def area_downsample(img, factor):
    h, w = img.shape
    return cv2.resize(img, (w // factor, h // factor),
                      interpolation=cv2.INTER_AREA)


def raster_drift(img, shear_px, jitter_px, rng):
    """Row-dependent lateral shift: progressive stage drift plus per-line
    jitter. Row r is displaced by shear * r / (H-1), so content at true x
    appears at x - shear * r / (H-1)."""
    if shear_px == 0 and jitter_px == 0:
        _ = rng.normal(0.0, 1.0, img.shape[0])   # keep the stream aligned
        return img
    h, w = img.shape
    rows = np.arange(h, dtype=np.float32)
    shift = shear_px * rows / max(h - 1, 1)
    shift += rng.normal(0.0, jitter_px, h).astype(np.float32) if jitter_px > 0 \
        else np.zeros(h, dtype=np.float32)
    map_x = np.arange(w, dtype=np.float32)[None, :] + shift[:, None]
    map_y = np.repeat(rows[:, None], w, axis=1)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def barrel(img, k):
    if abs(k) < 1e-9:
        return img
    h, w = img.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = (xx - cx) / cx, (yy - cy) / cy
    r2 = nx * nx + ny * ny
    f = 1.0 + k * r2
    return cv2.remap(img, (cx + nx * f * cx).astype(np.float32),
                     (cy + ny * f * cy).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def shot_noise(img, dose, rng):
    """Poisson electron counting statistics."""
    if dose <= 0:
        return img
    counts = img.astype(np.float32) / 255.0 * dose
    noisy = rng.poisson(np.maximum(counts, 0)).astype(np.float32)
    return np.clip(noisy / dose * 255.0, 0, 255).astype(np.uint8)


def detector_noise(img, sigma, rng):
    if sigma <= 0:
        return img
    out = img.astype(np.float32) + rng.normal(0.0, sigma, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def speckle(img, sigma, rng):
    """Multiplicative gain fluctuation -> variance proportional to mu^2."""
    if sigma <= 0:
        return img
    g = rng.normal(1.0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8)


def salt_pepper(img, prob, rng):
    if prob <= 0:
        return img
    out = img.copy()
    r = rng.random(img.shape)
    out[r < prob / 2] = 0
    out[r > 1 - prob / 2] = 255
    return out


def vignette(img, strength):
    if strength <= 0:
        return img
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    return np.clip(img.astype(np.float32) * (1.0 - strength * r2),
                   0, 255).astype(np.uint8)


def gamma_response(img, gamma):
    if abs(gamma - 1.0) < 1e-6:
        return img
    lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0,
                  0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def charging_streaks(img, prob, intensity, rng):
    """Horizontal bright streaks from local charge build-up on insulators."""
    if prob <= 0 or intensity <= 0:
        return img
    h, w = img.shape
    n = rng.poisson(prob)
    if n == 0:
        return img
    out = img.astype(np.float32)
    for _ in range(int(n)):
        y = int(rng.integers(0, h))
        thick = int(rng.integers(1, 6))
        x0 = int(rng.integers(0, w // 2))
        length = int(rng.integers(w // 8, w - x0))
        prof = np.linspace(intensity * 12.0, 0.0, length, dtype=np.float32)
        y1 = min(y + thick, h)
        out[y:y1, x0:x0 + length] += prof[None, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def capture(img, p, rng, is_reference):
    """Per-image capture chain. The reference is a shorter, cleaner
    exposure path: no stage shear, reduced jitter and field effects."""
    att = 0.2 if is_reference else 1.0
    img = raster_drift(img,
                       0.0 if is_reference else p["shear_amplitude_px"],
                       p["drift_jitter_px"] * att, rng)
    img = barrel(img, p["barrel_distortion_k"] * (0.3 if is_reference else 1.0))
    img = shot_noise(img, p["dose_reference"] if is_reference
                     else p["dose_search"], rng)
    img = detector_noise(img, p["detector_sigma_reference"] if is_reference
                         else p["detector_sigma_search"], rng)
    img = speckle(img, p["speckle_sigma"] * (0.4 if is_reference else 1.0), rng)
    img = salt_pepper(img, p["salt_pepper_prob"] * (0.3 if is_reference else 1.0),
                      rng)
    img = vignette(img, p["vignette_strength"] * (0.5 if is_reference else 1.0))
    img = gamma_response(img, p["gamma"])
    img = charging_streaks(img, p["charging_streak_prob"] * att,
                           p["charging_streak_intensity"], rng)
    return img
