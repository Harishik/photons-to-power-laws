"""Tests for the grb_afterglow pipeline."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grb_afterglow import (  # noqa: E402
    LightCurve,
    aperture_photometry_at,
    broken_power_law,
    build_master_bias,
    build_master_flat,
    calibrate_frame,
    calibrate_magnitudes,
    compute_zeropoint,
    detect_sources,
    fit_broken_power_law,
    fit_single_power_law,
    flux_to_mag,
    mag_to_flux,
    single_power_law,
)


def test_master_bias_is_median():
    frames = [np.full((8, 8), v, dtype=float) for v in (10, 12, 14)]
    assert np.allclose(build_master_bias(frames), 12.0)


def test_master_flat_normalised_to_unity():
    rng = np.random.default_rng(0)
    flats = [rng.normal(1000.0, 5.0, (16, 16)) for _ in range(5)]
    mf = build_master_flat(flats)
    assert mf.shape == (16, 16)
    assert np.isclose(np.median(mf), 1.0, atol=1e-2)


def test_calibrate_removes_bias_and_flat():
    scene = np.full((8, 8), 100.0)
    flat = np.full((8, 8), 2.0)
    bias = np.full((8, 8), 500.0)
    raw = scene * flat + bias
    cal = calibrate_frame(raw, master_bias=bias, master_flat=flat)
    assert np.allclose(cal, scene)


def test_calibrate_shapes_must_match():
    with pytest.raises(ValueError):
        build_master_bias([np.zeros((4, 4)), np.zeros((5, 5))])


def _make_star_image(positions, flux=5000.0, fwhm=3.0, sky=100.0, shape=(64, 64), seed=1):
    rng = np.random.default_rng(seed)
    img = np.full(shape, sky, dtype=float)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    for (x, y) in positions:
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
        img += flux * g / g.sum()
    return rng.poisson(img).astype(float)


def test_detect_finds_known_star():
    img = _make_star_image([(32, 32)])
    sources = detect_sources(img, fwhm=3.0, threshold_sigma=5.0)
    assert len(sources) >= 1
    d = np.hypot(sources["x_centroid"] - 32, sources["y_centroid"] - 32)
    assert d.min() < 1.5


def test_aperture_photometry_positive_and_snr():
    img = _make_star_image([(32, 32)], flux=8000.0)
    res = aperture_photometry_at(img, 32, 32, r_aper=5.0, gain=1.5, read_noise=5.0, exptime=1.0)
    assert res.net_counts > 0
    assert res.snr > 10
    assert np.isfinite(res.instrumental_mag)


def test_brighter_star_has_smaller_magnitude():
    faint = aperture_photometry_at(_make_star_image([(32, 32)], flux=3000.0), 32, 32, r_aper=5.0)
    bright = aperture_photometry_at(_make_star_image([(32, 32)], flux=30000.0), 32, 32, r_aper=5.0)
    assert bright.instrumental_mag < faint.instrumental_mag


def test_zeropoint_recovers_constant_offset():
    rng = np.random.default_rng(3)
    cat = rng.uniform(15, 20, 50)
    inst = cat - 24.0 + rng.normal(0, 0.01, 50)
    zp = compute_zeropoint(inst, cat)
    assert abs(zp.value - 24.0) < 0.05
    assert zp.n_stars > 40


def test_calibrate_magnitudes_applies_zeropoint():
    from grb_afterglow.zeropoint import ZeroPoint

    zp = ZeroPoint(value=25.0, error=0.02, n_stars=30, scatter=0.05)
    cal, err = calibrate_magnitudes(np.array([-7.0]), zp, np.array([0.01]))
    assert np.isclose(cal[0], 18.0)
    assert err[0] > 0.02


def test_mag_flux_roundtrip():
    mag = np.array([18.0, 19.5, 21.0])
    flux = mag_to_flux(mag)
    back = flux_to_mag(flux)
    assert np.allclose(mag, back, atol=1e-9)


def test_lightcurve_sorts_and_filters():
    lc = LightCurve(
        time=np.array([100.0, 10.0, -5.0]),
        mag=np.array([19.0, 18.0, np.nan]),
        mag_err=np.array([0.1, 0.1, 0.1]),
    ).finite()
    assert len(lc) == 2
    assert lc.time[0] < lc.time[1]


def test_single_power_law_recovers_index():
    t = np.logspace(2, 5, 20)
    true_alpha = 1.25
    flux = single_power_law(t, amplitude=500.0, alpha=true_alpha, t_ref=1000.0)
    rng = np.random.default_rng(7)
    err = 0.03 * flux
    noisy = flux + rng.normal(0, err)
    fit = fit_single_power_law(t, noisy, err)
    assert abs(fit.params["alpha"] - true_alpha) < 0.05
    assert fit.reduced_chi2 < 3.0


def test_broken_power_law_recovers_break():
    t = np.logspace(2, 6, 40)
    flux = broken_power_law(t, amplitude=300.0, t_break=2e4, alpha1=0.8, alpha2=1.9, smooth=3.0)
    rng = np.random.default_rng(11)
    err = 0.04 * flux
    noisy = flux + rng.normal(0, err)
    fit = fit_broken_power_law(t, noisy, err, smooth=3.0)
    assert abs(fit.params["alpha1"] - 0.8) < 0.2
    assert abs(fit.params["alpha2"] - 1.9) < 0.3
    assert fit.params["alpha2"] > fit.params["alpha1"]
