import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from grb_afterglow import ZeroPoint, calibrate_magnitudes, compute_zeropoint, match_sky, match_xy


def test_zeropoint_recovers_offset_and_clips_outliers():
    rng = np.random.default_rng(3)
    cat = rng.uniform(15, 20, 60)
    inst = cat - 24.0 + rng.normal(0, 0.01, 60)
    inst[:5] += 0.8  # blended or variable stars
    zp = compute_zeropoint(inst, cat, instrumental_errs=np.full(60, 0.01))
    assert zp.value == pytest.approx(24.0, abs=0.01)
    assert zp.n_stars == 55
    assert not zp.used[:5].any()


def test_zeropoint_fits_colour_term():
    rng = np.random.default_rng(4)
    colour = rng.uniform(0.2, 1.5, 80)
    cat = rng.uniform(15, 20, 80)
    inst = cat - 25.0 - 0.07 * colour + rng.normal(0, 0.005, 80)
    zp = compute_zeropoint(inst, cat, color=colour)
    assert zp.color_term == pytest.approx(0.07, abs=0.01)
    assert zp.value == pytest.approx(25.0, abs=0.01)


def test_zeropoint_requires_valid_pairs():
    with pytest.raises(ValueError):
        compute_zeropoint(np.array([np.nan]), np.array([15.0]))


def test_calibrate_magnitudes_adds_zeropoint_error():
    zp = ZeroPoint(value=25.0, error=0.02, n_stars=30, scatter=0.05)
    cal, err = calibrate_magnitudes(np.array([-7.0]), zp, np.array([0.01]))
    assert cal[0] == pytest.approx(18.0)
    assert err[0] == pytest.approx(np.hypot(0.01, 0.02))


def test_match_xy_is_one_to_one():
    measured = Table({"x": [10.0, 50.0], "y": [10.0, 50.0]})
    reference = Table({"x": [10.5, 10.8, 49.0, 90.0], "y": [10.0, 10.0, 50.0, 90.0]})
    idx_m, idx_r, sep = match_xy(measured, reference, max_sep=2.0)
    assert sorted(idx_m.tolist()) == [0, 1]
    assert dict(zip(idx_m.tolist(), idx_r.tolist(), strict=True)) == {0: 0, 1: 2}
    assert np.all(sep <= 2.0)


def test_match_sky_respects_radius():
    ref = SkyCoord([150.0, 150.01] * u.deg, [20.0, 20.0] * u.deg)
    meas = SkyCoord([150.0 + 1 / 3600, 150.2] * u.deg, [20.0, 20.0] * u.deg)
    idx_m, idx_r, sep = match_sky(meas, ref, max_sep_arcsec=2.0)
    assert idx_m.tolist() == [0] and idx_r.tolist() == [0] and sep[0] < 1.0
