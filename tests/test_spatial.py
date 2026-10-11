"""Spatially varying PSF and subtraction kernel, for full frames whose seeing changes across the field."""

import numpy as np
import pytest
from astropy.wcs import WCS

from grb_afterglow.psf import EmpiricalPSF, forced_psf_flux, psf_photometry, psf_photometry_tiled
from grb_afterglow.subtraction import subtract_images
from grb_afterglow.synthetic import _add_source


def _wcs(shape):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [150.0, 20.0]
    w.wcs.crpix = [shape[1] / 2 + 0.5, shape[0] / 2 + 0.5]
    w.wcs.cd = [[-1 / 3600, 0.0], [0.0, 1 / 3600]]
    return w


def _gradient(x, y, shape, lo, hi):
    return lo + (hi - lo) * (x + y) / (shape[0] + shape[1])


def _field(fwhm, *, shape=(300, 300), n=220, seed=1, sky=200.0):
    """Stars on a sky whose Moffat FWHM changes linearly from one corner to the other."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(6, shape[0] - 6, (n, 2))
    flux = 10 ** rng.uniform(3.0, 4.8, n)
    img = np.zeros(shape)
    for (x, y), f in zip(xy, flux, strict=True):
        _add_source(img, x, y, f, _gradient(x, y, shape, *fwhm), 3.0)
    return np.random.default_rng(seed + 100).poisson(img + sky).astype(float), xy, flux


@pytest.fixture(scope="module")
def gradient_field():
    return _field((3.0, 3.6))  # 20 % across the frame


def _ratios(result, xy, flux):
    bright = flux > 3000
    corner = (xy[:, 0] + xy[:, 1] < 200) | (xy[:, 0] + xy[:, 1] > 400)
    ratio = result.flux / flux
    pull = (result.flux - flux) / result.flux_err
    return np.median(ratio[bright & corner]), np.std(pull[bright])


def test_spatial_psf_follows_a_seeing_gradient(gradient_field):
    img, xy, flux = gradient_field
    src = xy[np.argsort(-flux)]
    tiled = psf_photometry_tiled(img, xy, source_xy=src, fwhm=3.3, gain=1.0, tiles=(3, 3))
    psf = tiled.psf
    assert psf.gradients["log_alpha"][0] > 0 and psf.gradients["log_alpha"][1] > 0  # the gradient is found
    corners = np.array([20.0, 280.0])
    assert psf.fwhm_at(corners, corners) == pytest.approx(_gradient(corners, corners, img.shape, 3.0, 3.6), rel=0.04)
    assert np.isfinite(tiled.flux).all()
    for x, y in ((20.0, 20.0), (280.0, 280.0)):  # the local PSF is a normal EmpiricalPSF
        local = psf.at(x, y)
        assert isinstance(local, EmpiricalPSF)
        g = np.arange(-15, 16)
        assert local(g[None, :], g[:, None]).sum() == pytest.approx(1.0, abs=0.02)
    ratio, pull_std = _ratios(tiled, xy, flux)
    assert ratio == pytest.approx(1.0, abs=0.01)
    assert pull_std < 1.4


def test_spatial_psf_finds_no_gradient_in_a_uniform_field():
    img, xy, flux = _field((3.2, 3.2))
    src = xy[np.argsort(-flux)]
    tiled = psf_photometry_tiled(img, xy, source_xy=src, fwhm=3.3, gain=1.0, tiles=(3, 3))
    assert all(v == (0.0, 0.0) for v in tiled.psf.gradients.values())
    single = psf_photometry(img, xy, source_xy=src, fwhm=3.3, gain=1.0)
    bright = flux > 3000
    assert np.median(tiled.flux[bright] / single.flux[bright]) == pytest.approx(1.0, abs=0.003)


def test_tiled_kernel_subtracts_a_field_with_a_seeing_gradient():
    """Science PSF from 2.6 to 4.2 px across the frame, template uniform and sharper."""
    shape = (256, 256)
    rng = np.random.default_rng(0)
    xy = rng.uniform(5, shape[0] - 5, (260, 2))
    flux = 10 ** rng.uniform(2.5, 4.7, 260)
    transient = (200.3, 128.6, 3000.0)
    keep = np.hypot(xy[:, 0] - transient[0], xy[:, 1] - transient[1]) > 15
    xy, flux = xy[keep], flux[keep]

    def fwhm_at(x):
        return 2.6 + 1.6 * x / shape[1]

    sci, tpl = np.zeros(shape), np.zeros(shape)
    for (x, y), f in zip(xy, flux, strict=True):
        _add_source(sci, x, y, f, fwhm_at(x), 3.0)
        _add_source(tpl, x, y, 4 * f, 2.2, 3.0)
    _add_source(sci, transient[0], transient[1], transient[2], fwhm_at(transient[0]), 3.0)
    sci = rng.poisson(sci + 100.0).astype(float)
    tpl = rng.poisson(tpl + 400.0).astype(float)
    w = _wcs(shape)

    def star_chi(diff):
        yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
        chi = diff.data / np.sqrt(diff.variance)
        out = []
        for (x, y), f in zip(xy, flux, strict=True):
            m = (np.hypot(xx - x, yy - y) <= 4) & diff.valid
            if f > 5000 and m.sum() > 30:
                out.append(np.mean(chi[m] ** 2))
        return float(np.sqrt(np.mean(out)))

    single = subtract_images(sci, w, tpl, w, science_fwhm=3.4, template_fwhm=2.2)
    tiled = subtract_images(sci, w, tpl, w, science_fwhm=3.4, template_fwhm=2.2, tiles=(3, 3))
    assert tiled.info["tiles"] == (3, 3) and len(tiled.kernels) == 3
    assert star_chi(single) > 4.0  # one kernel leaves residuals of every star away from the centre
    assert star_chi(tiled) < 2.5
    # The kernel broadens from left to right, as the science PSF does.
    left, right = tiled.kernel_at(20.0, 128.0), tiled.kernel_at(236.0, 128.0)
    g = np.arange(-(left.shape[0] // 2), left.shape[0] // 2 + 1)
    width = [np.sqrt(np.sum(k * g[None, :] ** 2) / np.sum(k)) for k in (left, right)]
    assert width[1] > width[0] + 0.3
    # The transient keeps its flux, measured with the science PSF at its position.
    psf = EmpiricalPSF.moffat_from_fwhm(fwhm_at(transient[0]), beta=3.0)
    fit = forced_psf_flux(tiled.data, transient[0], transient[1], psf)
    assert fit.flux == pytest.approx(transient[2], abs=3 * fit.flux_err)
    with pytest.raises(ValueError, match="fewer tiles"):
        subtract_images(sci, w, tpl, w, science_fwhm=3.4, template_fwhm=2.2, tiles=(8, 8))


def test_pipeline_with_tiles(small_campaign, calibration):
    """A uniform-seeing frame reduced with tiles gives the same answer as without."""
    from astropy.wcs import WCS as _WCS

    from grb_afterglow import PipelineConfig
    from grb_afterglow.calibration import calibrate_frame, load_fits
    from grb_afterglow.pipeline import Template, reduce_frame

    _, truth = small_campaign
    raw, bias, flats, catalog, target, templates = calibration
    info = sorted((f for f in truth["frames"] if f["band"] == "r"), key=lambda f: f["time_s"])[2]
    image_raw, header = load_fits(raw / info["file"])
    image = calibrate_frame(image_raw, master_bias=bias, master_flat=flats["r"])
    template = Template.from_fits(templates["r"])
    kwargs = {"band": "r", "time": header["TSINCE"], "exptime": header["EXPTIME"], "gain": header["GAIN"]}
    result = reduce_frame(
        image, _WCS(header), target, catalog, config=PipelineConfig(photometry="psf", tiles=(2, 2)),
        template=template, **kwargs,
    )  # fmt: skip
    assert result.difference.kernels is not None
    assert result.flux_ujy == pytest.approx(info["afterglow_flux_ujy"], abs=4 * result.flux_err_ujy)
    with pytest.raises(ValueError):
        PipelineConfig(tiles=(0, 2))
