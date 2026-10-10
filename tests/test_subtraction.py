import numpy as np
import pytest
from astropy.wcs import WCS

from grb_afterglow.subtraction import resample_to_grid


def _wcs(crpix, rot_deg=0.0):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [150.0, 20.0]
    w.wcs.crpix = list(crpix)
    s, t = 1.0 / 3600.0, np.radians(rot_deg)
    w.wcs.cd = [[-s * np.cos(t), s * np.sin(t)], [s * np.sin(t), s * np.cos(t)]]
    return w


def _blob(shape, x0, y0, sigma=2.0):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma**2))


def test_resample_moves_sources_to_the_same_sky_position():
    target = _wcs((32.5, 32.5))
    source = _wcs((35.2, 30.7), rot_deg=0.3)
    ra, dec = target.pixel_to_world_values(30.0, 34.0)
    sx, sy = source.world_to_pixel_values(ra, dec)
    image = _blob((64, 64), float(sx), float(sy))
    out, valid = resample_to_grid(image, source, target, (64, 64))
    expected = _blob((64, 64), 30.0, 34.0)
    assert np.abs(out - expected)[valid].max() < 0.02
    assert valid[32, 32]
    assert valid.mean() > 0.8


def test_resample_flags_pixels_outside_the_input():
    target = _wcs((32.5, 32.5))
    source = _wcs((52.5, 32.5))  # shifted by 20 px: part of the output falls outside
    out, valid = resample_to_grid(np.ones((64, 64)), source, target, (64, 64))
    assert 0.5 < valid.mean() < 0.85
    assert np.allclose(out[valid], 1.0)


@pytest.mark.parametrize("bad", [np.nan])
def test_resample_flags_non_finite_input(bad):
    w = _wcs((32.5, 32.5))
    image = np.ones((64, 64))
    image[30:34, 30:34] = bad
    out, valid = resample_to_grid(image, w, w, (64, 64))
    assert not valid[31, 31] and valid[10, 10]
    assert np.all(np.isfinite(out))


def _pair(seed, sci_fwhm, tpl_fwhm, transient, *, shape=(96, 96), host=20000.0, tpl_scale=0.8, depth=4.0):
    """Science and deeper template of one field (stars + a PSF-convolved host galaxy), on different grids."""
    from scipy.signal import fftconvolve

    from grb_afterglow.synthetic import _moffat_stamp

    rng = np.random.default_rng(seed)
    centre = (shape[1] / 2 + 0.5, shape[0] / 2 + 0.5)
    sci_wcs = _wcs(centre)
    tpl_wcs = _wcs((centre[0] + 3.37, centre[1] - 2.61), rot_deg=0.15)
    xy = rng.uniform(8, shape[0] - 8, (40, 2))
    flux = 10 ** rng.uniform(2.6, 4.9, 40)
    tx, ty = 45.3, 50.8
    keep = np.hypot(xy[:, 0] - tx, xy[:, 1] - ty) > 12
    xy, flux = xy[keep], flux[keep]
    world = sci_wcs.pixel_to_world_values(xy[:, 0], xy[:, 1])
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]

    def render(wcs, fwhm, scale, dep, sky, with_transient):
        img = np.zeros(shape)
        sx, sy = wcs.world_to_pixel_values(*world)
        sources = [(x, y, f * scale * dep) for x, y, f in zip(sx, sy, flux, strict=True)]
        if with_transient and transient > 0:
            px, py = wcs.world_to_pixel_values(*sci_wcs.pixel_to_world_values(tx, ty))
            sources.append((float(px), float(py), transient * scale * dep))
        half = int(np.ceil(6 * fwhm))
        for x, y, f in sources:
            ix, iy = round(x), round(y)
            stamp = _moffat_stamp(x - ix, y - iy, fwhm, 3.0, half) * f
            x0, x1, y0, y1 = (
                max(ix - half, 0),
                min(ix + half + 1, shape[1]),
                max(iy - half, 0),
                min(iy + half + 1, shape[0]),
            )
            img[y0:y1, x0:x1] += stamp[y0 - (iy - half) : y1 - (iy - half), x0 - (ix - half) : x1 - (ix - half)]
        hx, hy = wcs.world_to_pixel_values(*sci_wcs.pixel_to_world_values(tx + 1.2, ty - 0.8))
        disk = np.exp(-1.678 * np.hypot(xx - hx, (yy - hy) / 0.6) / 3.5)
        psf = _moffat_stamp(0.0, 0.0, fwhm, 3.0, int(np.ceil(5 * fwhm)))
        img += fftconvolve(disk / disk.sum() * host * scale * dep, psf / psf.sum(), mode="same")
        img += sky * dep
        return (rng.poisson(np.maximum(img, 0)) + rng.normal(0, 5.0, shape)) / dep

    sci = render(sci_wcs, sci_fwhm, 1.0, 1.0, 400.0, True)
    tpl = render(tpl_wcs, tpl_fwhm, tpl_scale, depth, 520.0, False)
    return sci, sci_wcs, tpl, tpl_wcs, (tx, ty), xy, flux


def _measure(diff, xy, r, mask):
    from grb_afterglow.photometry import blank_aperture_noise, forced_photometry

    f = float(forced_photometry(diff.data, np.array([xy]), r_aper=r, r_in=r + 3, r_out=r + 9)["net_counts"][0])
    return f, blank_aperture_noise(diff.data, r, mask=mask, r_in=r + 3, r_out=r + 9)


@pytest.mark.parametrize(("sci_fwhm", "tpl_fwhm", "psf_from"), [(3.4, 2.4, "science"), (2.2, 3.2, "template")])
def test_subtraction_cancels_static_light_and_keeps_the_transient(sci_fwhm, tpl_fwhm, psf_from):
    from grb_afterglow.photometry import forced_photometry
    from grb_afterglow.subtraction import subtract_images

    transient = 4000.0
    sci, sci_wcs, tpl, tpl_wcs, txy, xy, flux = _pair(1, sci_fwhm, tpl_fwhm, transient)
    diff = subtract_images(sci, sci_wcs, tpl, tpl_wcs, science_fwhm=sci_fwhm, template_fwhm=tpl_fwhm)
    assert diff.psf_from == psf_from
    assert diff.fwhm == (sci_fwhm if psf_from == "science" else tpl_fwhm)
    assert diff.info["flux_ratio"] == pytest.approx(0.8, rel=0.02)
    assert diff.valid.mean() > 0.6

    # Static stars vanish ...
    bright = flux > 20000
    r = 1.5 * max(sci_fwhm, tpl_fwhm)
    resid = forced_photometry(diff.data, xy[bright], r_aper=r, r_in=2 * r + 1, r_out=3 * r + 2)["net_counts"]
    assert np.median(np.abs(np.asarray(resid)) / flux[bright]) < 0.01

    # ... and the transient keeps its flux. Calibrate the aperture with the same stars measured on the
    # science image given the difference image's PSF, relative to their total flux.
    r_d = 2.0 * diff.fwhm
    mask = np.zeros(sci.shape, bool)
    conv = diff.as_difference_psf(sci)
    ap = np.asarray(forced_photometry(conv, xy[bright], r_aper=r_d, r_in=r_d + 3, r_out=r_d + 9)["net_counts"])
    frac = np.median(ap / flux[bright])
    f, sigma = _measure(diff, txy, r_d, mask)
    assert f / frac == pytest.approx(transient, abs=3 * sigma / frac)


def test_difference_is_science_minus_matched_template():
    from grb_afterglow.subtraction import subtract_images

    for sci_fwhm, tpl_fwhm in ((3.0, 2.2), (2.0, 3.0)):
        sci, sci_wcs, tpl, tpl_wcs, *_ = _pair(2, sci_fwhm, tpl_fwhm, 1000.0)
        diff = subtract_images(sci, sci_wcs, tpl, tpl_wcs, science_fwhm=sci_fwhm, template_fwhm=tpl_fwhm)
        inner = diff.valid & np.pad(np.ones((76, 76), bool), 10)  # away from the convolution edges
        recon = diff.as_difference_psf(sci) - diff.matched_template
        assert np.allclose(recon[inner], diff.data[inner], atol=1e-6 * np.nanmax(np.abs(diff.data)))


def test_host_galaxy_leaves_no_residual_at_the_transient_position():
    from grb_afterglow.subtraction import subtract_images

    pulls = []
    for seed in range(4):
        sci, sci_wcs, tpl, tpl_wcs, txy, *_ = _pair(10 + seed, 3.4, 2.4, 0.0)
        diff = subtract_images(sci, sci_wcs, tpl, tpl_wcs, science_fwhm=3.4, template_fwhm=2.4)
        f, sigma = _measure(diff, txy, 3.4, np.zeros(sci.shape, bool))
        pulls.append(f / sigma)
    assert abs(np.mean(pulls)) < 1.5 and np.max(np.abs(pulls)) < 3.5
