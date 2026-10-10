import numpy as np
import pytest

import grb_afterglow.detection as detection
from grb_afterglow import detect_sources, estimate_background, estimate_fwhm
from grb_afterglow.detection import make_source_mask


def test_detect_finds_known_star(star_image):
    sources = detect_sources(star_image([(32, 32)]), fwhm=3.0)
    assert len(sources) >= 1
    d = np.hypot(sources["x_centroid"] - 32, sources["y_centroid"] - 32)
    assert d.min() < 0.5


def test_detect_returns_empty_table_for_blank_sky():
    rng = np.random.default_rng(0)
    sources = detect_sources(rng.normal(100.0, 5.0, (64, 64)), fwhm=3.0, threshold_sigma=8.0)
    assert len(sources) == 0 and "x_centroid" in sources.colnames


def test_detect_normalises_legacy_column_names(monkeypatch):
    class LegacyFinder:
        def __init__(self, **kwargs):
            pass

        def __call__(self, residual):
            return detection.Table({"xcentroid": [10.0], "ycentroid": [11.0], "flux": [100.0]})

    monkeypatch.setattr(detection, "DAOStarFinder", LegacyFinder)
    img = np.zeros((16, 16))
    sources = detection.detect_sources(img, background=np.zeros_like(img))
    assert {"x_centroid", "y_centroid"} <= set(sources.colnames)


@pytest.mark.parametrize("true_fwhm", [2.5, 3.5, 5.0])
def test_estimate_fwhm_measures_seeing(star_image, true_fwhm):
    positions = [(x, y) for x in (20, 50, 80) for y in (20, 50, 80)]
    img = star_image(positions, flux=20000.0, fwhm=true_fwhm, shape=(100, 100))
    sources = detect_sources(img, fwhm=3.0)
    fwhm = estimate_fwhm(img, sources, guess=3.0, background=estimate_background(img))
    assert abs(fwhm - true_fwhm) / true_fwhm < 0.08


def test_estimate_fwhm_falls_back_to_guess_without_stars():
    empty = detection.Table(names=("x_centroid", "y_centroid", "flux"))
    assert estimate_fwhm(np.zeros((10, 10)), empty, guess=4.2) == 4.2


def test_estimate_fwhm_is_not_set_by_a_lone_galaxy(star_image):
    """Stars in close pairs and one isolated, extended galaxy: the isolation radius relaxes to the stars."""
    fwhm = 3.0
    pairs = [
        (x, y) for x0, y0 in ((20, 20), (20, 70), (70, 20), (100, 100), (60, 100)) for x, y in ((x0, y0), (x0 + 9, y0))
    ]
    img = star_image(pairs, flux=40000.0, fwhm=fwhm, shape=(130, 130), noise=False)
    yy, xx = np.mgrid[0:130, 0:130]
    img += 2000.0 * np.exp(-np.hypot(xx - 105, yy - 45) / 2.5)  # an exponential disk, wider than a star
    img = np.random.default_rng(3).poisson(img).astype(float)
    sources = detect_sources(img, fwhm=fwhm)
    assert estimate_fwhm(img, sources, guess=fwhm) == pytest.approx(fwhm, rel=0.1)


def test_background_recovers_gradient():
    _, xx = np.mgrid[0:128, 0:128]
    sky = 100.0 + 0.2 * xx
    rng = np.random.default_rng(3)
    bkg = estimate_background(sky + rng.normal(0, 2.0, sky.shape))
    inner = (slice(16, -16), slice(16, -16))
    assert np.allclose(bkg.background[inner], sky[inner], atol=1.0)


def test_background_rms_matches_noise():
    rng = np.random.default_rng(4)
    bkg = estimate_background(rng.normal(250.0, 6.0, (128, 128)))
    assert abs(bkg.median_rms - 6.0) < 0.3


def test_source_mask_covers_stars_not_noise(star_image):
    img = star_image([(32, 32)], flux=20000.0)
    mask = make_source_mask(img, estimate_background(img))
    assert mask[32, 32]
    assert mask.mean() < 0.1
