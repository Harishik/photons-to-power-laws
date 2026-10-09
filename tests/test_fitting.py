import numpy as np
import pytest

from grb_afterglow import (
    LightCurve,
    broken_power_law,
    compare_models,
    fit_broken_power_law,
    fit_lightcurve,
    fit_single_power_law,
    single_power_law,
)
from grb_afterglow.fitting import band_frequency_hz


def _noisy(flux, frac, seed):
    rng = np.random.default_rng(seed)
    err = frac * flux
    return flux + rng.normal(0, err), err


def test_single_power_law_recovers_index():
    t = np.logspace(2, 5, 20)
    flux, err = _noisy(single_power_law(t, 500.0, 1.25), 0.03, 7)
    fit = fit_single_power_law(t, flux, err)
    assert fit.params["alpha"] == pytest.approx(1.25, abs=3 * fit.errors["alpha"])
    assert fit.reduced_chi2 < 2.0


def test_broken_power_law_recovers_break():
    t = np.logspace(2, 6, 40)
    flux, err = _noisy(broken_power_law(t, 300.0, 2e4, 0.8, 1.9), 0.04, 11)
    fit = fit_broken_power_law(t, flux, err)
    assert fit.params["alpha1"] == pytest.approx(0.8, abs=0.1)
    assert fit.params["alpha2"] == pytest.approx(1.9, abs=0.15)
    assert fit.params["t_break"] == pytest.approx(2e4, rel=0.3)


def test_model_comparison_prefers_true_model():
    t = np.logspace(2, 6, 40)
    flux, err = _noisy(broken_power_law(t, 300.0, 2e4, 0.8, 2.0), 0.03, 2)
    broken = compare_models(fit_single_power_law(t, flux, err), fit_broken_power_law(t, flux, err))
    assert broken.best.model_name == "broken_power_law"
    flux, err = _noisy(single_power_law(t, 300.0, 1.1), 0.03, 3)
    single = compare_models(fit_single_power_law(t, flux, err), fit_broken_power_law(t, flux, err))
    assert single.best.model_name == "single_power_law"


def test_multiband_fit_measures_spectral_index():
    t = np.tile(np.logspace(3, 5, 10), 3)
    bands = np.repeat(["g", "r", "i"], 10)
    nu = np.array([band_frequency_hz(b) for b in bands])
    truth = single_power_law(t, 200.0, 1.1) * (nu / band_frequency_hz("r")) ** -0.7
    flux, err = _noisy(truth, 0.02, 5)
    fit = fit_lightcurve(LightCurve(t, flux, err, bands), "single", reference_band="r")
    assert fit.params["beta"] == pytest.approx(0.7, abs=3 * fit.errors["beta"])
    assert fit.params["alpha"] == pytest.approx(1.1, abs=3 * fit.errors["alpha"])


def test_negative_forced_fluxes_are_used():
    t = np.logspace(3, 6, 15)
    flux, err = _noisy(single_power_law(t, 100.0, 1.5), 0.0, 0)
    err = np.full_like(flux, 0.5)
    flux = flux + np.random.default_rng(1).normal(0, 0.5, flux.size)
    assert np.any(flux < 0)
    fit = fit_single_power_law(t, flux, err)
    assert fit.dof == t.size - 2


def test_predict_matches_model_and_information_criteria():
    t = np.logspace(2, 5, 12)
    flux, err = _noisy(single_power_law(t, 50.0, 1.0), 0.02, 9)
    fit = fit_single_power_law(t, flux, err)
    assert np.allclose(fit.predict(t), single_power_law(t, fit.params["amplitude"], fit.params["alpha"], fit.t_ref))
    assert fit.bic > fit.aic > fit.chi2


def test_systematic_floor_inflates_errors():
    t = np.logspace(2, 5, 12)
    flux, err = _noisy(single_power_law(t, 50.0, 1.0), 0.001, 9)
    assert fit_single_power_law(t, flux, err, sys_err=0.05).chi2 < fit_single_power_law(t, flux, err).chi2


def test_too_few_points():
    with pytest.raises(ValueError):
        fit_single_power_law([1.0, 2.0], [1.0, 0.5], [0.1, 0.1])
