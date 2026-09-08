"""
Sanity / unit tests for datasets/augment.py (synthetic low-light/dark-image
simulation -- the augmentation half of PLEM's night/dark-image robustness
approach; see CLAUDE.md's datasets/ table and datasets/spacenet6.py for the
real-SAR-data half).

Run with: pytest tests/test_augment_sanity.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets.augment import (
    adjust_gamma,
    jitter_brightness_contrast_saturation,
    clahe,
    inject_sensor_noise,
    simulate_low_light,
)

SHAPE = (32, 32, 3)


def make_gradient_image(shape=SHAPE, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 220, size=shape[:2], dtype=np.int32)
    img = np.stack([base] * shape[2], axis=-1)
    return np.clip(img, 0, 255).astype(np.uint8)


class TestAdjustGamma:
    def test_identity_at_gamma_one(self):
        img = make_gradient_image()
        out = adjust_gamma(img, gamma=1.0)
        np.testing.assert_array_equal(out, img)

    def test_gamma_greater_than_one_darkens(self):
        img = make_gradient_image()
        out = adjust_gamma(img, gamma=2.5)
        assert out.astype(np.float64).mean() < img.astype(np.float64).mean()

    def test_invalid_gamma_raises(self):
        img = make_gradient_image()
        with pytest.raises(ValueError):
            adjust_gamma(img, gamma=0.0)

    def test_shape_dtype_preserved(self):
        img = make_gradient_image()
        out = adjust_gamma(img, gamma=1.8)
        assert out.shape == img.shape
        assert out.dtype == np.uint8


class TestJitterBrightnessContrastSaturation:
    def test_darkens_on_average(self):
        img = make_gradient_image()
        rng = np.random.default_rng(1)
        out = jitter_brightness_contrast_saturation(img, rng)
        assert out.astype(np.float64).mean() < img.astype(np.float64).mean()

    def test_shape_dtype_preserved(self):
        img = make_gradient_image()
        rng = np.random.default_rng(1)
        out = jitter_brightness_contrast_saturation(img, rng)
        assert out.shape == img.shape
        assert out.dtype == np.uint8


class TestClahe:
    def test_output_finite_valid_range(self):
        img = make_gradient_image()
        out = clahe(img)
        assert np.isfinite(out).all()
        assert out.min() >= 0 and out.max() <= 255

    def test_shape_dtype_preserved(self):
        img = make_gradient_image()
        out = clahe(img)
        assert out.shape == img.shape
        assert out.dtype == np.uint8


class TestInjectSensorNoise:
    def test_output_finite_no_nan_inf(self):
        img = make_gradient_image()
        rng = np.random.default_rng(2)
        out = inject_sensor_noise(img, rng)
        assert np.isfinite(out).all()

    def test_higher_gain_increases_variance(self):
        img = make_gradient_image()
        low = inject_sensor_noise(img, np.random.default_rng(3), gain=1.0)
        high = inject_sensor_noise(img, np.random.default_rng(3), gain=8.0)
        low_residual_std = (low.astype(np.float64) - img.astype(np.float64)).std()
        high_residual_std = (high.astype(np.float64) - img.astype(np.float64)).std()
        assert high_residual_std > low_residual_std


class TestSimulateLowLight:
    def test_darkens_mean_brightness(self):
        img = make_gradient_image()
        rng = np.random.default_rng(4)
        out = simulate_low_light(img, rng, severity=0.8)
        assert out.astype(np.float64).mean() < img.astype(np.float64).mean()

    def test_severity_monotonic(self):
        img = make_gradient_image()
        means = []
        for severity in (0.1, 0.5, 0.9):
            out = simulate_low_light(img, np.random.default_rng(5), severity=severity)
            means.append(out.astype(np.float64).mean())
        assert means[0] > means[1] > means[2]

    def test_shape_dtype_preserved(self):
        img = make_gradient_image()
        out = simulate_low_light(img, np.random.default_rng(6), severity=0.5)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_reproducible_with_seeded_rng(self):
        img = make_gradient_image()
        out1 = simulate_low_light(img, np.random.default_rng(42), severity=0.6)
        out2 = simulate_low_light(img, np.random.default_rng(42), severity=0.6)
        np.testing.assert_array_equal(out1, out2)

    def test_different_seed_gives_different_output(self):
        img = make_gradient_image()
        out1 = simulate_low_light(img, np.random.default_rng(1), severity=0.6)
        out2 = simulate_low_light(img, np.random.default_rng(2), severity=0.6)
        assert not np.array_equal(out1, out2)

    def test_random_severity_when_none(self):
        img = make_gradient_image()
        out = simulate_low_light(img, np.random.default_rng(7), severity=None)
        assert out.shape == img.shape
        assert np.isfinite(out).all()
