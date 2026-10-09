import numpy as np
import pytest

from grb_afterglow import LightCurve, flux_to_mag, mag_to_flux


def test_mag_flux_roundtrip():
    mag = np.array([18.0, 19.5, 21.0])
    assert np.allclose(flux_to_mag(mag_to_flux(mag)), mag)


def test_ab_reference_points():
    assert mag_to_flux(23.9) == pytest.approx(1.0, rel=1e-3)
    assert np.isnan(flux_to_mag(-1.0))


def test_lightcurve_sorts_and_filters():
    lc = LightCurve([100.0, 10.0, -5.0], [5.0, 8.0, np.nan], [0.1, 0.1, 0.1], "r").finite()
    assert len(lc) == 2
    assert lc.time.tolist() == [10.0, 100.0]


def test_from_magnitudes_propagates_errors():
    lc = LightCurve.from_magnitudes([1.0], [20.0], [0.1], "g")
    assert lc.flux_err[0] / lc.flux[0] == pytest.approx(0.4 * np.log(10) * 0.1)
    assert lc.mag_err[0] == pytest.approx(0.1)


def test_detections_and_upper_limits():
    lc = LightCurve([1.0, 2.0], [10.0, 0.5], [1.0, 1.0])
    assert lc.detected().tolist() == [True, False]
    assert lc.upper_limit_mag()[1] == pytest.approx(flux_to_mag(3.5))


def test_select_band():
    lc = LightCurve([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.1] * 3, ["g", "r", "g"])
    assert lc.bands == ["g", "r"]
    assert len(lc.select("g")) == 2


def test_csv_roundtrip(tmp_path):
    lc = LightCurve([10.0, 20.0], [5.0, -1.0], [0.5, 0.5], ["r", "i"])
    lc.save_csv(tmp_path / "lc.csv")
    back = LightCurve.read_csv(tmp_path / "lc.csv")
    assert np.allclose(back.flux, lc.flux) and back.band.tolist() == ["r", "i"]


def test_read_csv_accepts_magnitudes(tmp_path):
    (tmp_path / "m.csv").write_text("time_s,mag,mag_err\n100,18.0,0.05\n200,18.8,0.05\n")
    lc = LightCurve.read_csv(tmp_path / "m.csv")
    assert lc.mag == pytest.approx([18.0, 18.8])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        LightCurve([1.0, 2.0], [1.0], [1.0])
