import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from grb_afterglow.detection import detect_sources
from grb_afterglow.psf import EmpiricalPSF, build_psf, forced_psf_flux, moffat_pixel, psf_photometry
from grb_afterglow.synthetic import _moffat_stamp


def _add(img, x, y, flux, fwhm, beta=3.0):
    half = int(np.ceil(6 * fwhm))
    ix, iy = round(x), round(y)
    stamp = _moffat_stamp(x - ix, y - iy, fwhm, beta, half) * flux
    ny, nx = img.shape
    x0, x1, y0, y1 = max(ix - half, 0), min(ix + half + 1, nx), max(iy - half, 0), min(iy + half + 1, ny)
    img[y0:y1, x0:x1] += stamp[y0 - (iy - half) : y1 - (iy - half), x0 - (ix - half) : x1 - (ix - half)]


def _field(seed, fwhm=2.6, n=60, shape=(128, 128), sky=400.0):
    """Moffat stars with known fluxes on a flat sky, Poisson + read noise (gain 1)."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(10, shape[0] - 10, (n, 2))
    flux = 10 ** rng.uniform(2.6, 5.0, n)
    img = np.full(shape, sky)
    for (x, y), f in zip(xy, flux, strict=True):
        _add(img, x, y, f, fwhm)
    return rng.poisson(img) + rng.normal(0, 5.0, shape), xy, flux


def _sources(img, fwhm):
    s = detect_sources(img, fwhm=fwhm)
    return np.column_stack([np.asarray(s["x_centroid"]), np.asarray(s["y_centroid"])])


def test_moffat_pixel_has_unit_total_flux():
    g = np.arange(-60, 61)
    total = moffat_pixel(g[None, :] + 0.3, g[:, None] - 0.2, alpha=2.0, beta=3.0, g1=0.1, g2=-0.05).sum()
    assert total == pytest.approx(1.0, abs=2e-3)


def test_moffat_from_fwhm_round_trips():
    assert EmpiricalPSF.moffat_from_fwhm(3.1, beta=2.8).fwhm == pytest.approx(3.1)


def test_psf_photometry_recovers_total_fluxes_with_honest_errors():
    pulls, ratios = [], []
    for seed in range(2):
        img, xy, flux = _field(seed)
        res = psf_photometry(img, xy, source_xy=_sources(img, 2.6), fwhm=2.6, gain=1.0)
        pulls.extend((res.flux - flux) / res.flux_err)
        ratios.extend(res.flux[flux > 20000] / flux[flux > 20000])
    pulls = np.asarray(pulls)
    robust_std = 1.4826 * np.median(np.abs(pulls - np.median(pulls)))
    assert np.median(ratios) == pytest.approx(1.0, abs=0.01)  # total flux, not an aperture fraction
    assert abs(np.median(pulls)) < 0.25
    assert 0.8 < robust_std < 1.25


def test_build_psf_measures_the_seeing():
    img, _, _ = _field(3, fwhm=3.2)
    psf = build_psf(img, _sources(img, 3.0), fwhm=3.0, gain=1.0)
    assert psf.n_stars >= 5 and psf.residual is not None
    assert psf.fwhm == pytest.approx(3.2, rel=0.08)
    g = np.arange(-30, 31)
    assert psf(g[None, :], g[:, None]).sum() == pytest.approx(1.0, abs=0.02)


def test_saturated_cores_are_ignored():
    img, xy, flux = _field(4)
    bright = int(np.argmax(flux))
    ix, iy = (int(v) for v in np.rint(xy[bright]))
    img = img.astype(float)
    img[iy - 1 : iy + 2, ix - 1 : ix + 2] = np.nan
    res = psf_photometry(img, xy, source_xy=_sources(np.nan_to_num(img, nan=400.0), 2.6), fwhm=2.6, gain=1.0)
    assert res.flux[bright] == pytest.approx(flux[bright], rel=0.05)  # recovered from the wings
    others = np.arange(len(flux)) != bright
    assert np.median(res.flux[others][flux[others] > 20000] / flux[others][flux[others] > 20000]) == pytest.approx(
        1.0, abs=0.01
    )


def test_forced_psf_flux_on_correlated_noise():
    rng = np.random.default_rng(5)
    fwhm, flux = 3.0, 3000.0
    psf = EmpiricalPSF.moffat_from_fwhm(fwhm, beta=3.0)
    fluxes, errs = [], []
    for _ in range(30):
        # Smoothing correlates the noise, as PSF matching does on a difference image.
        img = gaussian_filter(rng.normal(0.0, 20.0, (96, 96)), 0.8) * 2.0
        _add(img, 47.3, 51.6, flux, fwhm, beta=3.0)
        res = forced_psf_flux(img, 47.3, 51.6, psf)
        fluxes.append(res.flux)
        errs.append(res.flux_err)
    fluxes = np.asarray(fluxes)
    assert np.mean(fluxes) == pytest.approx(flux, rel=0.03)
    assert np.std(fluxes) == pytest.approx(np.median(errs), rel=0.35)


def test_forced_psf_flux_tolerates_bad_pixels():
    psf = EmpiricalPSF.moffat_from_fwhm(3.0, beta=3.0)
    img = np.random.default_rng(6).normal(0.0, 5.0, (80, 80))
    _add(img, 40.0, 40.0, 5000.0, 3.0)
    img[41, 41] = np.nan
    res = forced_psf_flux(img, 40.0, 40.0, psf)
    assert np.isfinite(res.flux) and res.flux == pytest.approx(5000.0, rel=0.03)
    assert np.isnan(forced_psf_flux(img, 2.0, 40.0, psf).flux)  # too close to the edge
