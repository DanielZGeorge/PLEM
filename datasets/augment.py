"""
Synthetic low-light / dark-image simulation -- the augmentation half of
PLEM's two-pronged night/dark-image robustness approach (the other half is
the real illumination-invariant SAR source, `datasets/spacenet6.py`).

No public dataset exists with real night-labeled optical satellite imagery
paired with full road+building+point semantic labels (verified during
planning) -- so night-robustness for optical input has to come from
simulating dark/low-light conditions on the existing daytime-labeled RGB
tiles, not from real night ground truth.

Pure numpy/PIL/skimage, image-only: every function here takes and returns an
H×W×3 uint8 array and never touches a label map, consistent with datasets/'s
"decoupled, H×W numpy" contract -- severity is purely a function of pixel
values, never geometry, so callers can apply this to an image without any
special-casing of its paired label map.
"""

import numpy as np
from PIL import Image, ImageEnhance
from skimage import exposure


def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """
    Power-law gamma correction: I' = 255 * (I/255)**gamma. gamma > 1 darkens
    (the night/low-light direction), gamma < 1 brightens, gamma == 1 is the
    identity.
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    normalized = image.astype(np.float64) / 255.0
    out = 255.0 * np.power(normalized, gamma)
    return np.clip(out, 0, 255).astype(np.uint8)


def jitter_brightness_contrast_saturation(
    image: np.ndarray,
    rng: np.random.Generator,
    brightness_range=(0.3, 0.8),
    contrast_range=(0.7, 1.0),
    saturation_range=(0.5, 1.0),
) -> np.ndarray:
    """
    Randomized brightness/contrast/saturation jitter via PIL.ImageEnhance
    (pillow is already a PLEM dependency, used elsewhere in datasets/potsdam.py).
    Ranges default to < 1.0 (dimmer/flatter/desaturated) since this simulates
    dusk/dawn/overcast/night capture conditions, not general-purpose jitter.
    """
    img = Image.fromarray(image)
    brightness = rng.uniform(*brightness_range)
    contrast = rng.uniform(*contrast_range)
    saturation = rng.uniform(*saturation_range)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(saturation)
    return np.array(img)


def clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid=(8, 8)) -> np.ndarray:
    """
    Contrast-limited adaptive histogram equalization, applied per-channel.
    Models a night-capable camera's own auto-exposure/local-contrast
    compensation -- real low-light footage is rarely just uniformly dim, it's
    also locally contrast-stretched by the sensor/ISP -- so this is chained
    after the raw darkening steps in `simulate_low_light`, not a substitute
    for them.
    """
    normalized = image.astype(np.float64) / 255.0
    out = np.empty_like(normalized)
    for c in range(image.shape[-1]):
        out[..., c] = exposure.equalize_adapthist(
            normalized[..., c], kernel_size=tile_grid, clip_limit=clip_limit / 100.0,
        )
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def inject_sensor_noise(
    image: np.ndarray,
    rng: np.random.Generator,
    gain: float = 1.0,
    gaussian_sigma: float = 6.0,
) -> np.ndarray:
    """
    Simulates high-ISO night-capture grain: Poisson shot-noise, modeled by
    reducing the effective photon budget by `gain` before sampling and
    rescaling back up (`gain` >= 1 -- higher gain means fewer effective
    photons per pixel, so relative shot-noise variance goes up exactly as a
    real sensor's ISO boost to compensate for low light amplifies noise more
    than signal), plus additive Gaussian read-noise. Clipped back to a valid
    uint8 image.
    """
    img_f = image.astype(np.float64)
    gain = max(gain, 1.0)
    photon_budget = np.clip(img_f, 0, None) / gain
    noisy = rng.poisson(photon_budget) * gain
    noisy = noisy + rng.normal(0, gaussian_sigma, size=img_f.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def simulate_low_light(
    image: np.ndarray,
    rng: np.random.Generator,
    severity: float = None,
) -> np.ndarray:
    """
    Top-level composed dark/low-light simulator.

    `severity` in [0, 1] controls how extreme every stage below is; if None
    (the default), a random severity ~Uniform(0.3, 1.0) is drawn per call --
    used for randomized train-time augmentation. A fixed severity (e.g. 0.3 /
    0.6 / 0.9) is used instead for the reproducible dark-test evaluation
    sweep, so the same tile can be compared across severities.

    Pipeline (each stage's own parameters scaled by `severity`):
      1. adjust_gamma      -- gamma = 1 + severity * 2.5
      2. brightness/contrast/saturation jitter -- ranges narrowed toward
         darker/flatter as severity increases
      3. clahe             -- fixed modest clip_limit (sensor auto-contrast)
      4. inject_sensor_noise -- gain scaled by severity

    Returns a uint8 array with the same shape as `image`.
    """
    if severity is None:
        severity = rng.uniform(0.3, 1.0)
    severity = float(np.clip(severity, 0.0, 1.0))

    out = adjust_gamma(image, gamma=1.0 + severity * 2.5)
    out = jitter_brightness_contrast_saturation(
        out, rng,
        brightness_range=(1.0 - 0.55 * severity, 1.0 - 0.15 * severity),
        contrast_range=(1.0 - 0.3 * severity, 1.0 - 0.05 * severity),
        saturation_range=(1.0 - 0.5 * severity, 1.0 - 0.05 * severity),
    )
    out = clahe(out, clip_limit=2.0)
    out = inject_sensor_noise(out, rng, gain=1.0 + severity * 3.0, gaussian_sigma=2.0 + severity * 6.0)
    return out
