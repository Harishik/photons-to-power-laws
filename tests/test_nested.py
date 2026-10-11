import numpy as np
import pytest

from grb_afterglow import LightCurve, broken_power_law, fit_lightcurve

pytest.importorskip("nautilus")

from grb_afterglow.nested import AMPLITUDE_DEX, Prior, bayes_factor, nested_posterior


def test_prior_transforms():
    u = np.linspace(0.01, 0.99, 99)
    uniform = np.array([Prior("uniform", 2.0, 4.0).transform(x) for x in u])
    assert uniform == pytest.approx(2.0 + 2.0 * u)
    normal = np.array([Prior("normal", 0.0, 0.1).transform(x) for x in u])
    assert normal[49] == pytest.approx(0.0, abs=1e-12) and normal[83] == pytest.approx(0.1 * 0.994, abs=0.01)
    sine = np.array([Prior("sine", 0.0, np.pi / 2).transform(x) for x in u])
    assert np.cos(sine) == pytest.approx(1.0 - u)  # CDF of p ∝ sin x on [0, pi/2] is 1 - cos x
    truncated = np.array([Prior("normal", 0.0, 1.0, lower=-0.5, upper=1.0).transform(x) for x in u])
    assert truncated.min() > -0.5 and truncated.max() < 1.0 and np.all(np.diff(truncated) > 0)
    assert Prior("normal", 0.0, 1.0, lower=-0.5, upper=0.5).transform(0.5) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        Prior("uniform", 1.0, 0.0)
    with pytest.raises(ValueError):
        Prior("cauchy", 0.0, 1.0)


@pytest.fixture(scope="module")
def jet_break():
    rng = np.random.default_rng(3)
    t = np.repeat(np.geomspace(1e3, 1e6, 14), 2)
    bands = np.array(["g", "r"] * 14, dtype=object)
    flux = broken_power_law(t, 400.0, 6e4, 0.8, 2.0, t_ref=1e4) * np.where(bands == "g", (617 / 481) ** -0.6, 1.0)
    err = 0.04 * flux
    lc = LightCurve(t, flux + rng.normal(0, err), err, bands)
    return lc, fit_lightcurve(lc, "single", reference_band="r"), fit_lightcurve(lc, "broken", reference_band="r")


def test_nested_posterior_and_bayes_factor(jet_break):
    lc, single, broken = jet_break
    post = nested_posterior(lc, broken, n_live=300, n_eff=2000, n_networks=1, seed=1)
    assert post.names == list(broken.params) and post.n_calls > 0
    for name in ("alpha1", "alpha2", "beta"):
        assert post.interval(name)[0] == pytest.approx(broken.params[name], abs=0.6 * broken.errors[name])
    assert np.median(post["t_break"]) == pytest.approx(broken.params["t_break"], rel=0.2)
    alt = nested_posterior(
        lc, single, n_live=300, n_eff=2000, n_networks=1, seed=1, n_jobs=2
    )  # workers need a picklable likelihood
    ln_b, err = bayes_factor(post, alt)
    assert ln_b > 10 and err < 0.1  # the break is decisively preferred
    assert "nested sampling" in post.summary()
    with pytest.raises(ValueError):
        nested_posterior(lc, broken, priors={"gamma": Prior("uniform", 0, 1)})


def test_evidence_matches_the_laplace_approximation(jet_break):
    lc, single, _ = jet_break  # a single power law to a broken one: a poor fit, but a Gaussian posterior
    post = nested_posterior(lc, single, n_live=300, n_eff=2000, n_networks=1, seed=2)
    names = list(single.params)
    jac = np.diag([1 / (single.params[n] * np.log(10)) if n == "amplitude" else 1.0 for n in names])
    cov = jac @ single.covariance @ jac.T  # in the sampled coordinates: log10 amplitude, alpha, beta
    e = lc.flux_err
    ln_l_max = -0.5 * single.chi2 - np.sum(np.log(e)) - 0.5 * e.size * np.log(2 * np.pi)
    volume = 2 * AMPLITUDE_DEX * 9.0 * 8.0  # uniform priors: 4 dex, alpha in [-3, 6], beta in [-3, 5]
    laplace = ln_l_max + 0.5 * len(names) * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(cov)) - np.log(volume)
    assert post.log_evidence == pytest.approx(laplace, abs=0.3)


def test_short_runs_warn_and_floors_by_band(jet_break):
    lc, single, _ = jet_break
    with pytest.warns(RuntimeWarning, match="before it converged"):
        rough = nested_posterior(lc, single, n_live=100, n_networks=1, max_calls=300, sys_err={"g": 0.05})
    assert rough.log_evidence_err == np.inf and rough.n_calls <= 400
    with pytest.raises(RuntimeError):
        nested_posterior(lc, single, max_calls=0)


def test_nested_physical_fit_with_calibration_offset():
    pytest.importorskip("afterglowpy")
    from grb_afterglow.physical import AfterglowModel, fit_afterglow

    model = AfterglowModel(0.876, wavelengths_nm={"r": 642.0})
    truth = {"log10_E0": 53.5, "theta_core": 0.05, "log10_n0": -1.0, "p": 2.4, "log10_eps_B": -2.5}
    t = np.concatenate([np.geomspace(0.05, 5.0, 10), np.geomspace(1.5, 4.0, 6)]) * 86400.0
    bands = np.array(["r"] * 10 + ["xrt:3keV"] * 6, dtype=object)
    flux = model.predict_band(t, bands, truth) * np.where(bands == "xrt:3keV", 10 ** (-0.4 * 0.15), 1.0)
    err = np.where(bands == "r", 0.03, 0.15) * flux
    lc = LightCurve(t, flux + np.random.default_rng(2).normal(0, err), err, bands)
    fixed = {k: truth[k] for k in ("theta_core", "log10_n0", "log10_eps_B")}
    fit = fit_afterglow(
        lc, model, ["log10_E0", "p"], fixed=fixed, start={"log10_E0": 53.3, "p": 2.3},
        sys_err={"r": 0.02, "xrt": 0.05}, calibration_offsets={"xrt": 0.2},
    )  # fmt: skip
    assert fit.free == ["log10_E0", "p", "offset_xrt"] and fit.priors == {"xrt": 0.2}
    assert fit.offsets["xrt"] == pytest.approx(0.15, abs=3 * fit.errors["offset_xrt"])
    post = nested_posterior(lc, fit, n_live=200, n_eff=1000, n_networks=1, seed=4)
    for name in ("log10_E0", "p"):
        assert post.interval(name)[0] == pytest.approx(truth[name], abs=4 * fit.errors[name])
    assert np.isfinite(post.log_evidence)
