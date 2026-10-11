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


def test_self_absorption_follows_granot_and_sari():
    params = {**TRUTH, "log10_n0": -2.0}
    plain, absorbed = _model(), _model(self_absorption=True)
    nu_sa = absorbed.self_absorption_frequency(params)
    # Granot & Sari (2002), Table 2, b = 1: 1.24e9 (p-1)^(3/5) (3p+2)^(-3/5) (1+z)^-1 epsbar_e^-1 eps_B^(1/5) n^(3/5) E52^(1/5)
    eps_bar = 0.1 * (2.4 - 2.0) / (2.4 - 1.0)
    expected = 1.24e9 * (1.4 / 9.2) ** 0.6 / 1.876 / eps_bar * 10 ** (-2.5 / 5 - 2.0 * 0.6 + 1.5 / 5)
    assert nu_sa == pytest.approx(expected, rel=1e-12)
    nu = np.array([0.1 * nu_sa, nu_sa, 1e3 * nu_sa, 4.67e14])
    ratio = absorbed.flux_density(86400.0, nu, params) / plain.flux_density(86400.0, nu, params)
    tau = (nu / nu_sa) ** (-5 / 3)
    assert ratio == pytest.approx(-np.expm1(-tau) / tau, rel=1e-9)
    assert ratio[-1] == 1.0 and ratio[0] < 0.1  # untouched in the optical, nu^2 well below nu_sa
    samples = {**params, "log10_n0": np.array([-3.0, -2.0])}
    assert absorbed.self_absorption_frequency(samples)[1] == pytest.approx(nu_sa)
    # Eichler & Waxman (2005): E0 / f, n0 / f, eps_e f, eps_B f and xi_N = f give the same afterglow
    shift = np.log10(0.1)
    scaled = {
        **params,
        "log10_E0": params["log10_E0"] - shift,
        "log10_n0": params["log10_n0"] - shift,
        "log10_eps_e": -1.0 + shift,
        "log10_eps_B": params["log10_eps_B"] + shift,
        "xi_N": 0.1,
    }
    assert absorbed.self_absorption_frequency(scaled) == pytest.approx(nu_sa, rel=1e-12)
    radio = np.array([1e9, 5e9])
    assert absorbed.flux_density(86400.0, radio, scaled) == pytest.approx(
        absorbed.flux_density(86400.0, radio, params), rel=1e-6
    )


def test_broadband_figure_renders(tmp_path):
    from grb_afterglow.plotting import plot_broadband

    model = _model()
    t = np.concatenate([np.geomspace(0.1, 5.0, 6), [2.0, 3.0], [16.0, 30.0]]) * 86400.0
    bands = np.array(["r"] * 6 + ["xrt:3keV"] * 2 + ["gmrt:1.25GHz"] * 2, dtype=object)
    flux = model.predict_band(t, bands, TRUTH)
    lc = LightCurve(t, flux * 1.05, 0.1 * flux, bands)
    fit = fit_afterglow(lc, model, ["log10_E0"], fixed={k: v for k, v in TRUTH.items() if k != "log10_E0"})
    late = LightCurve(
        np.array([10.0, 12.0, 40.0]) * 86400.0,
        np.array([5.0, -1.0, 80.0]),
        np.array([0.5, 1.0, 10.0]),
        np.array(["r", "r", "gmrt:0.65GHz"], dtype=object),
    )
    _, ax = plot_broadband(
        lc, fit, alternative=fit, unfitted=late, title="t", subtitle="s", savepath=tmp_path / "b.png"
    )
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels[:3] == ["xrt:3keV", "r", "gmrt:1.25GHz"]  # by frequency, X-rays first
    assert "not fitted" in labels and "gmrt:0.65GHz" in labels  # a band seen only in the unfitted data
    assert (tmp_path / "b.png").stat().st_size > 5_000
