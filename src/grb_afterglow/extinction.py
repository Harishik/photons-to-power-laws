"""Dust extinction: correcting afterglow photometry for Milky Way and host-galaxy dust.

Dust dims and reddens the afterglow. The temporal decay index is unaffected
(the extinction does not change with time), but the spectral index ``beta``
measured from optical colours is biased unless the photometry is corrected.

Two extinction curves are provided, both as ``A_lambda / A_V``:

* :func:`ccm89`, the Milky Way curve of Cardelli, Clayton & Mathis (1989,
  ApJ 345, 245) for a given ``R_V`` (3.1 for the diffuse interstellar medium).
  It is the curve of the XSPEC ``redden`` model used in many GRB papers.
* :func:`pei92_smc`, the Small Magellanic Cloud curve of Pei (1992, ApJ 395,
  130), the usual choice for GRB host galaxies, whose dust rarely shows the
  2175 Å bump. Host extinction acts at the rest-frame wavelength
  ``lambda_obs / (1 + z)``.

Band corrections use either a band's effective wavelength or, better, the
filter's transmission curve (:class:`Bandpass`): the extinction is then
averaged over the band, weighted by the photons a power-law source delivers.
Behind several magnitudes of dust the two differ, and the effective
wavelength itself must be that of the actual filter: a few nanometres matter.
"""

from __future__ import annotations

import numpy as np

from .filters import Bandpass, band_wavelength_nm
from .lightcurve import LightCurve

__all__ = [
    "PEI92_SMC_RV",
    "Bandpass",
    "band_extinction",
    "band_wavelength_nm",
    "ccm89",
    "deredden",
    "extinction_curve",
    "pei92_smc",
    "power_law_color",
]

# Pei (1992), Table 4: SMC terms (a_i, lambda_i [um], b_i, n_i) of xi = A_lambda / A_B.
_PEI92_SMC = np.array(
    [
        (185.0, 0.042, 90.0, 2.0),
        (27.0, 0.08, 5.50, 4.0),
        (0.005, 0.22, -1.95, 2.0),
        (0.010, 9.7, -1.95, 2.0),
        (0.012, 18.0, -1.80, 2.0),
        (0.030, 25.0, 0.00, 2.0),
    ]
)
PEI92_SMC_RV = 2.93


def ccm89(wavelength_nm, r_v: float = 3.1) -> np.ndarray:
    """Milky Way extinction ``A_lambda / A_V`` (Cardelli, Clayton & Mathis 1989).

    Valid from 0.1 to 3.3 micron (``x = 1 / lambda`` from 0.3 to 10 per micron);
    raises ``ValueError`` outside that range.
    """
    lam = np.atleast_1d(np.asarray(wavelength_nm, dtype=float))
    x = 1000.0 / lam  # inverse microns
    if np.any((x < 0.3) | (x > 10.0)):
        raise ValueError("CCM89 is defined for 0.1 to 3.3 micron (100 to 3333 nm).")
    a, b = np.zeros_like(x), np.zeros_like(x)

    ir = x < 1.1
    a[ir], b[ir] = 0.574 * x[ir] ** 1.61, -0.527 * x[ir] ** 1.61

    opt = (x >= 1.1) & (x < 3.3)
    y = x[opt] - 1.82
    a[opt] = np.polyval([0.32999, -0.77530, 0.01979, 0.72085, -0.02427, -0.50447, 0.17699, 1.0], y)
    b[opt] = np.polyval([-2.09002, 5.30260, -0.62251, -5.38434, 1.07233, 2.28305, 1.41338, 0.0], y)

    uv = (x >= 3.3) & (x < 8.0)
    xu = x[uv]
    fa = np.where(xu >= 5.9, -0.04473 * (xu - 5.9) ** 2 - 0.009779 * (xu - 5.9) ** 3, 0.0)
    fb = np.where(xu >= 5.9, 0.2130 * (xu - 5.9) ** 2 + 0.1207 * (xu - 5.9) ** 3, 0.0)
    a[uv] = 1.752 - 0.316 * xu - 0.104 / ((xu - 4.67) ** 2 + 0.341) + fa
    b[uv] = -3.090 + 1.825 * xu + 1.206 / ((xu - 4.62) ** 2 + 0.263) + fb

    fuv = x >= 8.0
    y = x[fuv] - 8.0
    a[fuv] = np.polyval([-0.070, 0.137, -0.628, -1.073], y)
    b[fuv] = np.polyval([0.374, -0.420, 4.257, 13.670], y)
    out = a + b / r_v
    return out if np.ndim(wavelength_nm) else out[0]


def pei92_smc(wavelength_nm) -> np.ndarray:
    """SMC extinction ``A_lambda / A_V`` (Pei 1992), from the far UV to the far infrared."""
    lam = np.atleast_1d(np.asarray(wavelength_nm, dtype=float)) / 1000.0  # micron
    if np.any(lam <= 0):
        raise ValueError("Wavelengths must be positive.")
    a, lam_i, b, n = (_PEI92_SMC[:, k][:, None] for k in range(4))
    ratio = (lam[None, :] / lam_i) ** n
    xi = np.sum(a / (ratio + 1.0 / ratio + b), axis=0)  # A_lambda / A_B
    out = xi * (1.0 + PEI92_SMC_RV) / PEI92_SMC_RV  # A_B / A_V = (R_V + 1) / R_V
    return out if np.ndim(wavelength_nm) else out[0]


def extinction_curve(
    wavelength_nm, *, ebv_mw: float = 0.0, r_v: float = 3.1, host_av: float = 0.0, redshift: float = 0.0
) -> np.ndarray:
    """Total extinction (mag) at observed wavelengths: Milky Way (CCM89) plus host (SMC at ``z``)."""
    lam = np.asarray(wavelength_nm, dtype=float)
    a_mw = r_v * ebv_mw * ccm89(lam, r_v) if ebv_mw else 0.0
    a_host = host_av * pei92_smc(lam / (1.0 + redshift)) if host_av else 0.0
    return a_mw + a_host


def band_extinction(
    band: str,
    *,
    ebv_mw: float = 0.0,
    r_v: float = 3.1,
    host_av: float = 0.0,
    redshift: float = 0.0,
    wavelengths_nm: dict[str, float] | None = None,
    bandpasses: dict[str, Bandpass] | None = None,
    beta: float = 0.5,
) -> float:
    """Total extinction in magnitudes in ``band``: Milky Way (CCM89) plus host (SMC at ``z``).

    With a :class:`Bandpass` for the band the extinction is averaged over the
    filter for a source of spectral index ``beta``; otherwise it is evaluated at
    the band's effective wavelength.
    """
    dust = {"ebv_mw": ebv_mw, "r_v": r_v, "host_av": host_av, "redshift": redshift}
    if bandpasses and band in bandpasses:
        bp = bandpasses[band]
        return bp.mean_attenuation(extinction_curve(bp.wavelength_nm, **dust), beta)
    return float(extinction_curve(band_wavelength_nm(band, wavelengths_nm), **dust))


def deredden(
    lc: LightCurve,
    *,
    ebv_mw: float = 0.0,
    r_v: float = 3.1,
    host_av: float = 0.0,
    redshift: float = 0.0,
    wavelengths_nm: dict[str, float] | None = None,
    bandpasses: dict[str, Bandpass] | None = None,
    beta: float = 0.5,
) -> LightCurve:
    """Correct a light curve for extinction, band by band.

    Fluxes and errors are multiplied by ``10**(0.4 A_band)``, with ``A_band``
    from :func:`band_extinction` (filter-averaged for a spectral index ``beta``
    where ``bandpasses`` are given). ``ebv_mw`` is the Galactic colour excess
    ``E(B-V)``, e.g. from the Schlafly & Finkbeiner (2011) recalibration of the
    Schlegel et al. (1998) dust map.
    """
    kwargs = {
        "ebv_mw": ebv_mw,
        "r_v": r_v,
        "host_av": host_av,
        "redshift": redshift,
        "wavelengths_nm": wavelengths_nm,
        "bandpasses": bandpasses,
        "beta": beta,
    }
    factor = np.array([10 ** (0.4 * band_extinction(str(b), **kwargs)) for b in lc.band])
    meta = {**lc.meta, "extinction": {"ebv_mw": ebv_mw, "r_v": r_v, "host_av": host_av, "redshift": redshift}}
    return LightCurve(lc.time, lc.flux * factor, lc.flux_err * factor, lc.band, meta)


def power_law_color(
    beta: float,
    blue: str,
    red: str,
    *,
    ebv_mw: float = 0.0,
    r_v: float = 3.1,
    host_av: float = 0.0,
    redshift: float = 0.0,
    wavelengths_nm: dict[str, float] | None = None,
) -> float:
    """Observed AB colour ``m_blue - m_red`` of a power-law spectrum ``F ∝ nu^-beta`` behind dust.

    Bands with a transmission curve in the registry (e.g. ``"ps1:g"``) are
    integrated over it, which matters behind several magnitudes of dust; other
    bands are evaluated at their effective wavelength. Useful as the
    ``target_color`` of a colour-term zero point: an afterglow seen through
    heavy Galactic dust is much redder than the field stars that calibrate it.
    """
    from .spectral import SpectralModel

    model = SpectralModel.for_reference_band(
        red, ebv_mw=ebv_mw, r_v=r_v, redshift=redshift, wavelengths_nm=wavelengths_nm
    )
    f_blue, f_red = model.band_factors([blue, red], beta, host_av)
    return float(-2.5 * np.log10(f_blue / f_red))
