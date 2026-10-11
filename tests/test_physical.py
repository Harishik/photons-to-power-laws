import numpy as np
import pytest

from grb_afterglow import LightCurve
from grb_afterglow.extinction import ccm89

grb = pytest.importorskip("afterglowpy")

from grb_afterglow.physical import AfterglowModel, fit_afterglow, sample_afterglow  # noqa: E402

WAVELENGTHS = {"g": 478.0, "r": 642.0, "i": 787.0}
TRUTH = {"log10_E0": 53.5, "theta_core": 0.05, "log10_n0": -1.0, "p": 2.4, "log10_eps_B": -2.5}
FIXED = {"log10_n0": -1.0, "log10_eps_B": -2.5}


def _model(**kwargs):
    return AfterglowModel(0.876, wavelengths_nm=WAVELENGTHS, **kwargs)


def _lightcurve(model, seed=0, frac=0.03):
    t = np.repeat(np.geomspace(0.02, 8.0, 12) * 86400.0, 3)
    bands = np.array(list("gri") * 12, dtype=object)
    flux = model.predict_band(t, bands, TRUTH)
    err = frac * flux
    return LightCurve(t, flux + np.random.default_rng(seed).normal(0, err), err, bands)


def test_model_is_afterglowpy_behind_galactic_dust():
    model = _model(ebv_mw=0.3)
    t = np.geomspace(1e3, 1e6, 7)
    kwargs = model.afterglowpy_kwargs(TRUTH)
    assert kwargs["E0"] == pytest.approx(10**53.5) and kwargs["epsilon_e"] == pytest.approx(0.1)
    assert "thetaWing" not in kwargs
    nu = np.full(t.size, 299_792_458.0 / 642e-9)
    expected = 1e3 * grb.fluxDensity(t, nu, **kwargs) * 10 ** (-0.4 * 3.1 * 0.3 * ccm89(642.0))
    assert model.predict_band(t, "r", TRUTH) == pytest.approx(expected, rel=1e-9)
    assert "thetaWing" in _model(jet="gaussian").afterglowpy_kwargs(TRUTH)
    with pytest.raises(ValueError):
        _model(jet="cone")


def test_fit_recovers_an_injected_jet():
    model = _model()
    lc = _lightcurve(model)
    start = [{"theta_core": 0.1, "p": 2.2, "log10_E0": 53.0}, {"theta_core": 0.03, "p": 2.6, "log10_E0": 54.0}]
    fit = fit_afterglow(lc, model, ["theta_core", "p", "log10_E0"], fixed=FIXED, start=start)
    for name in fit.free:
        assert fit.params[name] == pytest.approx(TRUTH[name], abs=4 * fit.errors[name])
    assert fit.reduced_chi2 < 2.0
    assert fit.predict_band(lc.time[:3], lc.band[:3]) == pytest.approx(
        model.predict_band(lc.time[:3], lc.band[:3], fit.params)
    )
    assert "afterglowpy tophat jet" in fit.summary()


def test_sampling_and_argument_checks():
    model = _model()
    lc = _lightcurve(model, seed=1)
    fit = fit_afterglow(
        lc, model, ["p", "log10_E0"], fixed={**FIXED, "theta_core": 0.05}, start={"p": 2.4, "log10_E0": 53.5}
    )
    post = sample_afterglow(lc, fit, n_walkers=8, n_steps=80, burn=20, thin=2, seed=3)
    assert post.names == ["p", "log10_E0"] and post.samples.shape[1] == 2
    assert 2.0001 <= post["p"].min() and post["p"].max() <= 4.0
    with pytest.raises(ValueError, match="Unknown"):
        fit_afterglow(lc, model, ["theta_wing"])
    with pytest.raises(ValueError):
        fit_afterglow(lc, model, [])


def test_one_epoch_spectrum():
    model = _model()
    nu = np.array([1e9, 4.67e14, 2.4e17])
    kwargs = model.afterglowpy_kwargs(TRUTH)
    expected = 1e3 * grb.fluxDensity(np.full(3, 86400.0), nu, **kwargs)
    assert model.flux_density(86400.0, nu, TRUTH) == pytest.approx(expected, rel=1e-12)
