import numpy as np
import pytest

from grb_afterglow import LightCurve, fit_lightcurve
from grb_afterglow.mcmc import grid_posterior, sample_posterior


@pytest.fixture(scope="module")
def jet_break():
    """A two-band jet-break light curve with 3 % errors."""
    rng = np.random.default_rng(11)
    t = np.repeat(np.geomspace(1e3, 1e6, 18), 2)
    bands = np.array(["g", "r"] * 18, dtype=object)
    nu_ratio = np.where(bands == "g", 617.0 / 481.0, 1.0)
    x = t / 8e4
    flux = 500 * x**-0.9 * (1 + x**3) ** (-(2.2 - 0.9) / 3) * nu_ratio**-0.6
    err = 0.03 * flux
    lc = LightCurve(t, flux + rng.normal(0, err), err, bands)
    return lc, fit_lightcurve(lc, "broken", reference_band="r")


def test_posterior_agrees_with_least_squares_and_truth(jet_break):
    lc, fit = jet_break
    post = sample_posterior(lc, fit, n_walkers=24, n_steps=2500, burn=800, seed=2)
    assert post.names == list(fit.params)
    assert 0.2 < post.acceptance_fraction < 0.8
    for name, truth in (("alpha1", 0.9), ("alpha2", 2.2), ("beta", 0.6)):
        med, lo, hi = post.interval(name)
        assert med == pytest.approx(fit.params[name], abs=0.5 * fit.errors[name])
        assert 0.5 * (hi - lo) == pytest.approx(fit.errors[name], rel=0.35)
        assert abs(med - truth) < 4 * 0.5 * (hi - lo)
    t_med = post.interval("t_break_days")[0]
    assert t_med * 86400 == pytest.approx(8e4, rel=0.25)
    p = post.p("jet_slow")
    assert np.median(p) == pytest.approx(2 * post.interval("beta")[0] + 1, abs=1e-6)
    assert "alpha2" in post.summary()


def test_posterior_is_reproducible(jet_break):
    lc, fit = jet_break
    a = sample_posterior(lc, fit, n_walkers=16, n_steps=200, burn=50, seed=5)
    b = sample_posterior(lc, fit, n_walkers=16, n_steps=200, burn=50, seed=5)
    assert np.array_equal(a.samples, b.samples)


def test_single_band_single_power_law():
    t = np.geomspace(1e3, 1e5, 15)
    flux = 100 * (t / 1e4) ** -1.3
    lc = LightCurve(t, flux, 0.05 * flux, "r")
    fit = fit_lightcurve(lc, "single")
    post = sample_posterior(lc, fit, n_walkers=16, n_steps=1500, burn=500)
    assert post.interval("alpha")[0] == pytest.approx(1.3, abs=0.03)
    assert np.all(post["amplitude"] > 0)


def test_grid_posterior_matches_least_squares_when_gaussian(jet_break):
    lc, fit = jet_break
    # Steps of about a third of the posterior width in both parameters.
    grid = {"t_break": np.geomspace(6.5e4, 9e4, 41), "alpha1": np.linspace(0.86, 0.94, 41)}
    post = grid_posterior(lc, fit, grid, n_samples=20000, seed=3)
    assert post.names == list(fit.params) and np.isnan(post.acceptance_fraction)
    for name in ("alpha1", "alpha2", "beta"):
        med, lo, hi = post.interval(name)
        assert med == pytest.approx(fit.params[name], abs=0.5 * fit.errors[name])
        assert 0.5 * (hi - lo) == pytest.approx(fit.errors[name], rel=0.25)
    assert post.interval("t_break_days")[0] * 86400 == pytest.approx(fit.params["t_break"], rel=0.1)
    assert "grid" in post.summary()
    one = grid_posterior(lc, fit, {"alpha2": np.linspace(1.8, 2.6, 81)}, n_samples=5000)
    assert one.interval("alpha2")[0] == pytest.approx(fit.params["alpha2"], abs=0.3 * fit.errors["alpha2"])
    with pytest.raises(ValueError):
        grid_posterior(lc, fit, {"gamma": np.ones(3)})
