import numpy as np
import pytest

from grb_afterglow import build_master_bias, build_master_flat, calibrate_frame, combine_frames
from grb_afterglow.calibration import load_fits, save_fits


def test_master_bias_is_median_of_constant_frames():
    frames = [np.full((8, 8), v, dtype=float) for v in (10, 12, 14)]
    assert np.allclose(build_master_bias(frames), 12.0)


def test_combine_rejects_cosmic_ray():
    rng = np.random.default_rng(0)
    frames = [rng.normal(100.0, 1.0, (16, 16)) for _ in range(7)]
    frames[3][5, 5] = 5000.0
    combined = combine_frames(frames)
    assert abs(combined[5, 5] - 100.0) < 3.0


def test_master_flat_is_normalised_and_scale_independent():
    rng = np.random.default_rng(1)
    pattern = 1.0 + 0.05 * rng.standard_normal((16, 16))
    flats = [level * pattern for level in (8000.0, 15000.0, 22000.0)]
    flat = build_master_flat(flats)
    assert np.isclose(np.median(flat), 1.0, atol=1e-3)
    assert np.allclose(flat / np.median(flat), pattern / np.median(pattern), rtol=1e-6)


def test_master_flat_subtracts_bias():
    bias = np.full((8, 8), 300.0)
    flats = [bias + 1000.0 * np.ones((8, 8)) for _ in range(3)]
    assert np.allclose(build_master_flat(flats, master_bias=bias), 1.0)


def test_calibrate_inverts_detector_model():
    scene = np.full((8, 8), 100.0)
    flat = np.full((8, 8), 2.0)
    bias = np.full((8, 8), 500.0)
    assert np.allclose(calibrate_frame(scene * flat + bias, master_bias=bias, master_flat=flat), scene)


def test_calibrate_ignores_bad_flat_pixels():
    flat = np.ones((4, 4))
    flat[0, 0] = 0.0
    flat[1, 1] = np.nan
    out = calibrate_frame(np.full((4, 4), 10.0), master_flat=flat)
    assert np.all(np.isfinite(out)) and np.allclose(out, 10.0)


@pytest.mark.parametrize("bad", [[np.zeros((4, 4)), np.zeros((5, 5))], []])
def test_combine_validates_input(bad):
    with pytest.raises(ValueError):
        build_master_bias(bad)


def test_fits_roundtrip(tmp_path):
    from astropy.io import fits

    data = np.arange(12.0).reshape(3, 4)
    header = fits.Header({"EXPTIME": 30.0})
    save_fits(tmp_path / "x.fits", data, header)
    loaded, hdr = load_fits(tmp_path / "x.fits")
    assert np.allclose(loaded, data) and hdr["EXPTIME"] == 30.0


def test_load_fits_finds_compressed_extension(tmp_path):
    from astropy.io import fits

    data = np.ones((8, 8), dtype=np.float32)
    fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(data)]).writeto(tmp_path / "x.fits.fz")
    loaded, _ = load_fits(tmp_path / "x.fits.fz")
    assert loaded.shape == (8, 8)
