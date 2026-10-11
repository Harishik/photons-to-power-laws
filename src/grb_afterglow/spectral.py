"""The spectral part of a multi-band light-curve model.

A multi-band fit models the afterglow spectrum as a power law
``F_nu ∝ (lambda / lambda_ref)^beta`` behind two screens of dust, the Milky
Way's (CCM89, fixed ``E(B-V)``) and the host galaxy's (SMC curve at the
rest-frame wavelength, fitted ``A_V``). What an instrument measures is that
spectrum averaged over its filter, weighted by the photons it detects:

    F_band / F_nu(lambda_ref) = ∫ T (lambda / lambda_ref)^beta 10^(-0.4 A(lambda)) dlambda / lambda
                                / ∫ T dlambda / lambda.

:class:`SpectralModel` evaluates this factor for every band of a light curve.
Bands with a transmission curve (named ``"instrument:filter"`` in the
registry of :mod:`grb_afterglow.filters`, or given explicitly) are integrated
over the curve; other bands are treated as monochromatic at their effective
wavelength, which is the model used before version 0.5. Integrals are
precomputed as quadrature weights so that a model evaluation is a single
weighted sum, fast enough for MCMC.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .extinction import ccm89, pei92_smc
from .filters import SPEED_OF_LIGHT_M_S, Bandpass, band_wavelength_nm, get_bandpass, has_bandpass

_MAX_SAMPLES = 400
"""Bandpasses sampled more finely than this are resampled to it (an error well below 0.1 %)."""

_LN10_04 = 0.4 * np.log(10.0)


def _quadrature(bp: Bandpass) -> tuple[np.ndarray, np.ndarray]:
    """Wavelengths and weights of ``∫ f T dlambda / lambda / ∫ T dlambda / lambda`` (trapezoid rule)."""
    lam, t = bp.wavelength_nm, bp.transmission
    peak = t.max() if t.size else 0.0
    if not peak > 0:
        raise ValueError(f"Bandpass {bp.name!r} has no positive transmission.")
    inside = np.flatnonzero(t > 1e-4 * peak)  # relative: a curve's normalisation is irrelevant
    lo, hi = max(inside[0] - 1, 0), min(inside[-1] + 1, lam.size - 1)
    lam, t = lam[lo : hi + 1], t[lo : hi + 1]
    if lam.size > _MAX_SAMPLES:
        grid = np.linspace(lam[0], lam[-1], _MAX_SAMPLES)
        lam, t = grid, np.interp(grid, lam, t)
    widths = np.diff(lam)
    trap = np.zeros_like(lam)
    trap[:-1] += widths / 2
    trap[1:] += widths / 2
    w = t / lam * trap
    return lam, w / w.sum()


@dataclass(frozen=True)
class BandGrid:
    """Quadrature of one band: wavelengths (nm), weights summing to 1, and whether a bandpass is used."""

    band: str
    wavelength_nm: np.ndarray = field(repr=False)
    weight: np.ndarray = field(repr=False)
    integrated: bool

    @property
    def effective_nm(self) -> float:
        """Pivot wavelength of an integrated band, or the band's single wavelength."""
        if not self.integrated:
            return float(self.wavelength_nm[0])
        return float(np.sqrt(np.sum(self.weight * self.wavelength_nm**2)))


@dataclass(frozen=True)
class SpectralStack:
    """The bands of a fit stacked into padded arrays, for vectorised evaluation."""

    bands: tuple[str, ...]
    log_ratio: np.ndarray = field(repr=False)  # ln(lambda / lambda_ref)
    weight_mw: np.ndarray = field(repr=False)  # quadrature weight times the Galactic attenuation
    host_exponent: np.ndarray = field(repr=False)  # -0.4 ln(10) k_SMC(lambda / (1 + z))

    def factors(self, beta: float, host_av: float = 0.0) -> np.ndarray:
        """Band flux in units of the intrinsic ``F_nu`` at the reference wavelength, one per band."""
        expo = beta * self.log_ratio
        if host_av:
            expo = expo + host_av * self.host_exponent
        return np.sum(self.weight_mw * np.exp(expo), axis=1)


@dataclass(frozen=True)
class SpectralModel:
    """How each band sees a power law behind Galactic and host-galaxy dust.

    Parameters
    ----------
    reference_nm
        Wavelength at which the fit's ``amplitude`` is the intrinsic (dust-free)
        flux density: the reference band's pivot or effective wavelength.
    ebv_mw, r_v
        Galactic colour excess and ``R_V``, applied with the CCM89 curve
        (forward-modelled, so the data are fitted as observed).
    redshift
        Host redshift for the SMC host-extinction curve, or None if the fit has
        no host dust.
    wavelengths_nm
        Effective wavelengths of monochromatic bands (see
        :func:`grb_afterglow.filters.band_wavelength_nm`).
    bandpasses
        True to integrate over the registry's curve for bands named
        ``"instrument:filter"``; False to treat every band as monochromatic; or
        a mapping of band name to :class:`~grb_afterglow.filters.Bandpass`
        (bands not in it fall back to the registry).
    """

    reference_nm: float
    ebv_mw: float = 0.0
    r_v: float = 3.1
    redshift: float | None = None
    wavelengths_nm: Mapping[str, float] | None = None
    bandpasses: Mapping[str, Bandpass] | bool = True
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def for_reference_band(cls, reference_band: str, **kwargs) -> SpectralModel:
        """A model whose reference wavelength is that of ``reference_band``."""
        model = cls(1.0, **kwargs)
        return cls(model.grid(reference_band).effective_nm, **kwargs)

    @property
    def nu_ref(self) -> float:
        """Reference frequency in Hz."""
        return SPEED_OF_LIGHT_M_S / (self.reference_nm * 1e-9)

    def bandpass_for(self, band: str) -> Bandpass | None:
        """The transmission curve used for ``band``, or None if it is treated as monochromatic."""
        if isinstance(self.bandpasses, Mapping) and band in self.bandpasses:
            return self.bandpasses[band]
        if self.wavelengths_nm and band in self.wavelengths_nm:
            return None
        if self.bandpasses is not False and has_bandpass(band):
            return get_bandpass(band)
        return None

    def grid(self, band: str) -> BandGrid:
        band = str(band)
        if band not in self._cache:
            bp = self.bandpass_for(band)
            if bp is None:
                lam, w = np.array([band_wavelength_nm(band, self.wavelengths_nm)]), np.ones(1)
            else:
                lam, w = _quadrature(bp)
            self._cache[band] = BandGrid(band, lam, w, bp is not None)
        return self._cache[band]

    def galactic_extinction(self, wavelength_nm) -> np.ndarray:
        """Milky Way extinction (mag) at observed wavelengths.

        CCM89 between 100 nm and 3.3 micron; its infrared power law continued to
        longer wavelengths (toward zero in the radio); none shortward of 100 nm,
        where absorption by gas, which is not modelled, dominates anyway. So a
        fit can be evaluated at X-ray or radio frequencies.
        """
        lam = np.asarray(wavelength_nm, dtype=float)
        if not self.ebv_mw:
            return np.zeros_like(lam)
        flat = np.atleast_1d(lam).astype(float)
        x = 1000.0 / flat  # inverse microns
        curve = np.zeros_like(flat)
        ok = (x >= 0.3) & (x <= 10.0)
        curve[ok] = ccm89(flat[ok], self.r_v)
        ir = x < 0.3
        curve[ir] = (0.574 - 0.527 / self.r_v) * x[ir] ** 1.61
        return (self.r_v * self.ebv_mw * curve).reshape(lam.shape)

    def host_curve(self, wavelength_nm) -> np.ndarray:
        """Host ``A_lambda / A_V`` at observed wavelengths (zero without a redshift)."""
        lam = np.asarray(wavelength_nm, dtype=float)
        if self.redshift is None:
            return np.zeros_like(lam)
        return pei92_smc(lam / (1.0 + self.redshift))

    def stack(self, bands: Sequence[str]) -> SpectralStack:
        """Precompute the quadrature of ``bands`` for :meth:`SpectralStack.factors`."""
        grids = [self.grid(b) for b in bands]
        n = max(g.wavelength_nm.size for g in grids)
        shape = (len(grids), n)
        log_ratio, weight_mw, host = np.zeros(shape), np.zeros(shape), np.zeros(shape)
        for j, g in enumerate(grids):
            m = g.wavelength_nm.size
            log_ratio[j, :m] = np.log(g.wavelength_nm / self.reference_nm)
            weight_mw[j, :m] = g.weight * 10 ** (-0.4 * self.galactic_extinction(g.wavelength_nm))
            host[j, :m] = -_LN10_04 * self.host_curve(g.wavelength_nm)
        return SpectralStack(tuple(str(b) for b in bands), log_ratio, weight_mw, host)

    def band_factors(self, bands: Sequence[str], beta: float, host_av: float = 0.0) -> np.ndarray:
        """Band flux in units of the intrinsic ``F_nu`` at the reference wavelength."""
        return self.stack(bands).factors(beta, host_av)

    def monochromatic(self, nu_hz, beta: float, host_av: float = 0.0) -> np.ndarray:
        """The same factor at single frequencies (Hz) rather than through a band."""
        lam = SPEED_OF_LIGHT_M_S / np.asarray(nu_hz, dtype=float) * 1e9
        dust = self.galactic_extinction(lam) + host_av * self.host_curve(lam)
        return (lam / self.reference_nm) ** beta * 10 ** (-0.4 * dust)
