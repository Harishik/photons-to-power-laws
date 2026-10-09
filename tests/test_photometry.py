import numpy as np
import pytest

from grb_afterglow import aperture_photometry_at, forced_photometry
from grb_afterglow.photometry import instrumental_magnitude


def test_aperture_photometry_recovers_flux(star_image):
    img = star_image([(32, 32)], flux=10000.0, fwhm=3.0, noise=False)
    res = aperture_photometry_at(img, 32, 32, r_aper=8.0, r_in=12.0, r_out=18.0)
    assert res.net_counts == pytest.approx(10000.0, rel=0.01)
    assert res.sky_per_pixel == pytest.approx(100.0, rel=1e-3)


def test_brighter_star_has_smaller_magnitude(star_image):
    faint = aperture_photometry_at(star_image([(32, 32)], flux=3000.0), 32, 32, r_aper=5.0)
    bright = aperture_photometry_at(star_image([(32, 32)], flux=30000.0), 32, 32, r_aper=5.0)
    assert bright.instrumental_mag < faint.instrumental_mag
    assert bright.snr > faint.snr > 5


def test_error_bars_match_scatter(star_image):
    """Across noise realisations, (measured - true) / error should have unit spread."""
    truth = aperture_photometry_at(star_image([(32, 32)], flux=4000.0, noise=False), 32, 32, r_aper=4.0)
    pulls = []
    for seed in range(200):
        res = aperture_photometry_at(star_image([(32, 32)], flux=4000.0, seed=seed), 32, 32, r_aper=4.0, gain=1.0)
        pulls.append((res.net_counts - truth.net_counts) / res.net_counts_err)
    assert abs(np.mean(pulls)) < 0.2
    assert 0.85 < np.std(pulls) < 1.15


def test_blank_sky_is_unbiased():
    rng = np.random.default_rng(5)
    img = rng.normal(500.0, 10.0, (200, 200))
    pos = rng.uniform(30, 170, (300, 2))
    table = forced_photometry(img, pos, r_aper=3.0)
    pull = np.asarray(table["net_counts"]) / np.asarray(table["net_counts_err"])
    assert abs(np.mean(pull)) < 0.15
    assert 0.85 < np.std(pull) < 1.15


def test_source_mask_removes_crowded_field_sky_bias(star_image):
    """Faint-star wings inflate an annulus median; masking them removes the bias.

    Forced photometry on empty sky in a crowded field should average to zero.
    Without a source mask it comes out negative because the sky is overestimated.
    """
    from scipy import ndimage

    from grb_afterglow import estimate_background
    from grb_afterglow.detection import make_source_mask

    rng = np.random.default_rng(0)
    stars = rng.uniform(4, 196, (150, 2))
    img = star_image(stars, flux=10 ** rng.uniform(2.7, 4.3, 150), fwhm=3.5, sky=200.0, shape=(200, 200), seed=0)
    mask = make_source_mask(img, estimate_background(img), dilate=5)
    occupied = np.zeros(img.shape, dtype=bool)
    occupied[stars[:, 1].astype(int), stars[:, 0].astype(int)] = True
    clearance = ndimage.distance_transform_edt(~occupied)
    pos = rng.uniform(25, 175, (2000, 2))
    pos = pos[clearance[pos[:, 1].astype(int), pos[:, 0].astype(int)] > 9][:300]

    def mean_pull(sky_mask):
        t = forced_photometry(img, pos, r_aper=3.5, r_in=10.5, r_out=17.5, sky_mask=sky_mask)
        return float(np.mean(np.asarray(t["net_counts"]) / np.asarray(t["net_counts_err"])))

    assert mean_pull(None) < -0.3
    assert abs(mean_pull(mask)) < 0.2


def test_negative_flux_has_no_magnitude():
    mag, _ = instrumental_magnitude(np.array([-5.0, 100.0]), np.array([1.0, 1.0]))
    assert np.isnan(mag[0]) and np.isfinite(mag[1])


def test_radii_are_validated():
    with pytest.raises(ValueError):
        forced_photometry(np.zeros((20, 20)), np.array([[10, 10]]), r_aper=5.0, r_in=4.0, r_out=6.0)
