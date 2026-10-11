import numpy as np
import pytest

from grb_afterglow import LightCurve, filters, fit_lightcurve
from grb_afterglow.extinction import ccm89, pei92_smc
from grb_afterglow.filters import (
    Bandpass,
    available_filters,
    band_wavelength_nm,
    filter_source,
    get_bandpass,
    instrument_of,
    register_bandpass,
)
from grb_afterglow.mcmc import sample_posterior
from grb_afterglow.spectral import SpectralModel

NIR_BANDS = ["lt:g", "lt:r", "lt:i", "lt:z", "f2:J", "f2:H", "f2:Ks", "wfc3:F160W"]


def test_registry_reproduces_svo_pivot_wavelengths():
    names = available_filters()
    assert {"ztf:r", "lt:g", "f2:Ks", "wirc:J", "wfc3:F160W", "2mass:H"} <= set(names)
    for name in names:
        # Energy-counter curves (lt:r, GMOS-S, WIRC) only match once converted to photon counting.
        assert get_bandpass(name).pivot_nm == pytest.approx(float(filter_source(name)["svo_pivot_nm"]), rel=1e-4)
    with pytest.raises(KeyError):
        get_bandpass("lt:H")


def test_band_names_and_wavelengths(monkeypatch):
    assert instrument_of("f2:Ks") == "f2" and instrument_of("r") is None
    assert band_wavelength_nm("ztf:r") == pytest.approx(641.71, abs=0.01)  # registry pivot
    assert band_wavelength_nm("r") == 617.0  # plain names keep the Pan-STARRS1 default
    assert band_wavelength_nm("zr", {"zr": 640.0}) == 640.0
    lam = np.linspace(500.0, 600.0, 51)
    monkeypatch.setattr(filters, "_REGISTERED", {})
    register_bandpass(Bandpass("test:box", lam, np.ones_like(lam)))
    assert "test:box" in available_filters()
    assert band_wavelength_nm("test:box") == pytest.approx(get_bandpass("test:box").pivot_nm)


def test_band_factor_matches_direct_integration():
    """The model's precomputed quadrature against the filter curve integrated directly."""
    beta, ebv, av, z = 0.7, 1.32, 0.4, 0.151
    model = SpectralModel.for_reference_band("lt:r", ebv_mw=ebv, redshift=z)
    for band in ("lt:g", "lt:r", "f2:Ks", "wirc:J", "ztf:i", "wfc3:F775W"):
        bp = get_bandpass(band)
        dust = 3.1 * ebv * ccm89(bp.wavelength_nm) + av * pei92_smc(bp.wavelength_nm / (1 + z))
        expected = bp.power_law_ratio(beta, model.reference_nm, dust)
        assert model.band_factors([band], beta, av)[0] == pytest.approx(expected, rel=2e-3)


def test_monochromatic_bands_reproduce_the_point_model():
    model = SpectralModel.for_reference_band("r", ebv_mw=0.3, redshift=1.0)
    factors = model.band_factors(["g", "r", "z"], 0.8, 0.5)
    lam = np.array([481.0, 617.0, 866.0])
    dust = 3.1 * 0.3 * ccm89(lam) + 0.5 * pei92_smc(lam / 2.0)
    assert factors == pytest.approx((lam / 617.0) ** 0.8 * 10 ** (-0.4 * dust), rel=1e-12)
    assert not model.grid("g").integrated and model.grid("ztf:g").integrated
    assert SpectralModel.for_reference_band("r", bandpasses=False).grid("ztf:g").wavelength_nm.size == 1


def _multi_instrument(beta=0.7, ebv=1.3, offsets=None, seed=3):
    """griz from one telescope, JHKs from another, F160W from a third, behind Galactic dust.

    The truth is computed from the filter curves directly, independently of SpectralModel.
    """
    offsets = {"f2": 0.08, "wfc3": -0.05} if offsets is None else offsets
    ref = get_bandpass("lt:r").pivot_nm

    def band_flux(band):
        bp = get_bandpass(band)
        ratio = bp.power_law_ratio(beta, ref, 3.1 * ebv * ccm89(bp.wavelength_nm))
        return ratio * 10 ** (-0.4 * offsets.get(instrument_of(band), 0.0))

    t = np.repeat(np.geomspace(2e5, 2e6, 6), len(NIR_BANDS))
    bands = np.array(NIR_BANDS * 6, dtype=object)
    flux = 100.0 * (t / 6e5) ** -1.1 * np.array([band_flux(b) for b in bands])
    err = 0.02 * flux
    return LightCurve(t, flux + np.random.default_rng(seed).normal(0, err), err, bands)


def test_multi_instrument_fit_recovers_spectrum_and_offsets():
    lc = _multi_instrument()
    kwargs = {"reference_band": "lt:r", "t_ref": 6e5, "ebv_mw": 1.3, "calibration_offsets": {"f2": None, "wfc3": None}}
    fit = fit_lightcurve(lc, "single", **kwargs)
    assert set(fit.params) == {"amplitude", "alpha", "beta", "offset_f2", "offset_wfc3"}
    for name, truth in (("beta", 0.7), ("alpha", 1.1), ("amplitude", 100.0)):
        assert fit.params[name] == pytest.approx(truth, abs=3 * fit.errors[name])
    assert fit.offsets["f2"] == pytest.approx(0.08, abs=3 * fit.errors["offset_f2"])
    assert fit.offsets["wfc3"] == pytest.approx(-0.05, abs=3 * fit.errors["offset_wfc3"])
    assert fit.reduced_chi2 < 1.6
    # predict_band is the model the data were fitted with: offsets and filter curves included.
    resid = (lc.flux - fit.predict_band(lc.time, lc.band)) / lc.flux_err
    assert np.sum(resid**2) == pytest.approx(fit.chi2)
    # Evaluating every band at its pivot wavelength instead biases the spectral index behind this much dust.
    pivot = fit_lightcurve(lc, "single", bandpasses=False, **kwargs)
    assert abs(pivot.params["beta"] - 0.7) > 3 * pivot.errors["beta"]
    assert pivot.chi2 > fit.chi2 + 15


def test_calibration_priors():
    lc = _multi_instrument(offsets={})
    tight = fit_lightcurve(lc, "single", reference_band="lt:r", ebv_mw=1.3, calibration_offsets={"f2": 0.01})
    assert tight.priors == {"f2": 0.01}
    assert tight.errors["offset_f2"] < 0.01
    # Every instrument free: the amplitude is degenerate with the offsets.
    with pytest.raises(ValueError, match="degenerate"):
        fit_lightcurve(lc, "single", ebv_mw=1.3, calibration_offsets={"lt": None, "f2": None, "wfc3": None})
    # ...unless one of them has a prior.
    anchored = fit_lightcurve(lc, "single", ebv_mw=1.3, calibration_offsets={"lt": 0.03, "f2": None, "wfc3": None})
    assert anchored.params["beta"] == pytest.approx(0.7, abs=3 * anchored.errors["beta"])
    with pytest.raises(ValueError):
        fit_lightcurve(lc, "single", calibration_offsets={"f2": 0.0})
    with pytest.raises(ValueError):
        fit_lightcurve(lc.select("lt:r"), "single", ebv_mw=0.1)


def test_posterior_with_calibration_offsets():
    lc = _multi_instrument()
    fit = fit_lightcurve(
        lc, "single", reference_band="lt:r", ebv_mw=1.3, calibration_offsets={"f2": 0.05, "wfc3": None}
    )
    post = sample_posterior(lc, fit, n_walkers=24, n_steps=1500, burn=500, seed=4)
    for name in ("beta", "offset_f2", "offset_wfc3"):
        med, lo, hi = post.interval(name)
        assert med == pytest.approx(fit.params[name], abs=0.5 * fit.errors[name])
        assert 0.5 * (hi - lo) == pytest.approx(fit.errors[name], rel=0.3)


def test_dusty_fit_extrapolates_to_xray_and_radio():
    """Forward-modelled Galactic dust must not stop a fit from being evaluated far from the optical."""
    lc = _multi_instrument(offsets={})
    fit = fit_lightcurve(lc, "single", reference_band="lt:r", ebv_mw=1.3)
    model = fit.spectral
    lam = np.array([150.0, 617.0, 3333.0, 3334.0])
    assert model.galactic_extinction(lam[:3]) == pytest.approx(3.1 * 1.3 * ccm89(lam[:3]))
    assert model.galactic_extinction(lam[3]) == pytest.approx(model.galactic_extinction(lam[2]), rel=1e-3)
    xray, radio = fit.predict(np.array([6e5]), 2.418e17), fit.predict(np.array([6e5]), 1e10)
    p = fit.params
    no_dust = fit.temporal(np.array([6e5]))[0] * (2.418e17 / fit.nu_ref) ** -p["beta"]
    assert xray[0] == pytest.approx(no_dust, rel=1e-9)
    assert np.isfinite(radio[0]) and radio[0] > 0


def test_quadrature_does_not_depend_on_the_curve_normalisation():
    bp = get_bandpass("sdss:g")
    scaled = Bandpass("scaled:g", bp.wavelength_nm, bp.transmission * 1.5e-4)
    model = SpectralModel(500.0, ebv_mw=1.32, bandpasses={"g": bp, "scaled:g": scaled})
    assert model.band_factors(["scaled:g"], 0.66)[0] == pytest.approx(model.band_factors(["g"], 0.66)[0], rel=1e-12)
    with pytest.raises(ValueError, match="no positive transmission"):
        SpectralModel(500.0, bandpasses={"dark": Bandpass("dark", bp.wavelength_nm, 0 * bp.transmission)}).grid("dark")


def test_bands_named_after_their_frequency():
    from grb_afterglow.filters import band_frequency_hz, tagged_frequency_hz

    assert tagged_frequency_hz("xrt:10keV") == pytest.approx(2.418e18, rel=1e-3)  # E = h nu
    assert tagged_frequency_hz("gmrt:1.25GHz") == 1.25e9
    assert tagged_frequency_hz("alma:3mm") == pytest.approx(299_792_458.0 / 3e-3)
    assert band_frequency_hz("2.2um") == pytest.approx(299_792_458.0 / 2.2e-6)
    assert instrument_of("xrt:10keV") == "xrt"
    assert tagged_frequency_hz("lt:r") is None and tagged_frequency_hz("zr") is None
    assert band_wavelength_nm("zr") == 617.0  # plain names still use the table
