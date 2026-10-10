import numpy as np
import pytest

from grb_afterglow import LightCurve, fit_lightcurve
from grb_afterglow.extinction import (
    Bandpass,
    band_extinction,
    ccm89,
    deredden,
    pei92_smc,
    power_law_color,
)
from grb_afterglow.fitting import band_frequency_hz


def test_ccm89_reference_values():
    assert ccm89(549.0) == pytest.approx(1.0, abs=0.01)  # V band defines A_V
    # E(B - V) = A_V / R_V, so A_B / A_V = 1 + 1 / R_V for R_V = 3.1
    assert ccm89(440.0) == pytest.approx(1.0 + 1.0 / 3.1, abs=0.01)
    assert ccm89(2190.0) == pytest.approx(0.574 * (1000 / 2190) ** 1.61 - 0.527 * (1000 / 2190) ** 1.61 / 3.1)
    # The 2175 Å bump: a local maximum between the near-UV and the far-UV rise.
    assert ccm89(217.5) > ccm89(260.0) and ccm89(217.5) > ccm89(180.0)
    assert np.all(np.diff(ccm89(np.linspace(400, 3000, 50))) < 0)  # monotonic in the optical/IR
    with pytest.raises(ValueError):
        ccm89(50.0)


def test_pei92_smc_shape():
    assert pei92_smc(550.0) == pytest.approx(0.97, abs=0.02)  # normalised as A_B (1 + R_V) / R_V
    lam = np.geomspace(120, 2000, 80)
    assert np.all(np.diff(pei92_smc(lam)) < 0)  # no 2175 Å bump, steep far-UV rise
    assert pei92_smc(150.0) > 4 * pei92_smc(550.0)


def test_band_extinction_and_deredden_round_trip():
    a_r = band_extinction("r", ebv_mw=0.5)
    assert a_r == pytest.approx(3.1 * 0.5 * ccm89(617.0))
    assert band_extinction("zr", ebv_mw=0.5) == pytest.approx(a_r)  # "zr" falls back to "r"
    assert band_extinction("g", host_av=1.0, redshift=1.0) == pytest.approx(pei92_smc(481.0 / 2.0))
    lc = LightCurve([1e4, 2e4], [10.0, 5.0], [1.0, 0.5], ["r", "r"])
    out = deredden(lc, ebv_mw=0.5)
    assert out.flux == pytest.approx(lc.flux * 10 ** (0.4 * a_r))
    assert out.flux_err == pytest.approx(lc.flux_err * 10 ** (0.4 * a_r))
    assert out.meta["extinction"]["ebv_mw"] == 0.5


def test_bandpass_pivot_and_mean_attenuation():
    lam = np.linspace(550.0, 700.0, 301)
    flat = Bandpass("r", lam, np.ones_like(lam))
    # For a top hat: sqrt((l2^2 - l1^2) / (2 ln(l2 / l1)))
    assert flat.pivot_nm == pytest.approx(np.sqrt((700.0**2 - 550.0**2) / (2 * np.log(700.0 / 550.0))), rel=1e-4)
    narrow = Bandpass("n", np.linspace(615.0, 619.0, 41), np.ones(41))
    a_curve = 3.1 * ccm89(narrow.wavelength_nm)
    assert narrow.mean_attenuation(a_curve) == pytest.approx(3.1 * ccm89(617.0), abs=1e-3)
    # Behind heavy dust a broad band is dominated by its red side: less extinction than at the pivot.
    heavy = 3.1 * 1.5 * ccm89(lam)
    assert flat.mean_attenuation(heavy) < 3.1 * 1.5 * ccm89(flat.pivot_nm)
    assert band_extinction("r", ebv_mw=1.5, bandpasses={"r": flat}) == pytest.approx(flat.mean_attenuation(heavy))


def test_bandpass_reads_svo_files(tmp_path):
    path = tmp_path / "f.dat"
    path.write_text("# comment\n5000 0.0\n6000 1.0\n7000 0.0\n")
    bp = Bandpass.from_file(path, "r")
    assert bp.wavelength_nm == pytest.approx([500.0, 600.0, 700.0])
    assert bp.pivot_nm == pytest.approx(600.0, rel=0.01)


def test_power_law_color():
    assert power_law_color(0.0, "g", "r") == pytest.approx(0.0)
    assert power_law_color(1.0, "g", "r") == pytest.approx(2.5 * np.log10(617.0 / 481.0))
    reddened = power_law_color(1.0, "g", "r", ebv_mw=0.3)
    assert reddened - power_law_color(1.0, "g", "r") == pytest.approx(
        band_extinction("g", ebv_mw=0.3) - band_extinction("r", ebv_mw=0.3)
    )


def test_fit_recovers_spectral_index_and_host_extinction():
    """A power law behind SMC-like host dust at z = 1, observed in grizy over a decaying light curve."""
    rng = np.random.default_rng(4)
    bands = np.array(list("grizy") * 12, dtype=object)
    t = np.repeat(np.geomspace(3e3, 3e5, 12), 5)
    nu = np.array([band_frequency_hz(b) for b in bands])
    beta, av, z = 0.8, 0.5, 1.0
    a_host = np.array([band_extinction(b, host_av=av, redshift=z) for b in bands])
    flux = 300 * (t / 3e4) ** -1.1 * (nu / band_frequency_hz("r")) ** -beta * 10 ** (-0.4 * a_host)
    err = 0.02 * flux
    lc = LightCurve(t, flux + rng.normal(0, err), err, bands)
    plain = fit_lightcurve(lc, "single", reference_band="r")
    fit = fit_lightcurve(lc, "single", reference_band="r", host_redshift=z)
    assert fit.params["beta"] == pytest.approx(beta, abs=4 * fit.errors["beta"])
    assert fit.params["host_av"] == pytest.approx(av, abs=4 * fit.errors["host_av"])
    assert plain.params["beta"] > beta + 0.2  # ignoring the dust reddens the spectrum
    assert fit.chi2 < plain.chi2
    assert fit.predict(np.array([3e4]), band_frequency_hz("r"))[0] == pytest.approx(
        300 * 10 ** (-0.4 * band_extinction("r", host_av=av, redshift=z)), rel=0.03
    )


def test_host_extinction_needs_several_bands():
    lc = LightCurve([1e3, 2e3, 4e3], [3.0, 2.0, 1.0], [0.1, 0.1, 0.1], "r")
    with pytest.raises(ValueError):
        fit_lightcurve(lc, "single", host_redshift=1.0)
