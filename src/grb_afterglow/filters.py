"""Photometric systems: the transmission curves of real instruments, for synthetic photometry.

A broad-band measurement is an average of the source spectrum over the
filter, weighted by the instrument's response. Two instruments' "r" bands are
not the same band, and behind several magnitudes of dust the difference
matters. This module keeps the system transmission curves of the instruments
used in this package (from the SVO Filter Profile Service, committed under
``data/filters``) under names of the form ``"instrument:filter"``, for example
``"ztf:r"``, ``"lt:g"`` or ``"f2:Ks"``, and computes synthetic photometry of a
power-law spectrum seen through dust.

All curves are converted to photon-counting response: SVO marks some curves
as energy-counter responses, which are divided by the wavelength. Band fluxes
are then AB-like averages of ``F_nu`` weighted by ``T(lambda) / lambda``.

Plain band names (``"g"``, ``"r"``, ``"zr"``) are not in the registry; for them
the effective wavelengths in :data:`EFFECTIVE_WAVELENGTH_NM` are used, as before.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache, lru_cache
from importlib import resources
from os import PathLike

import numpy as np

_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # noqa: NPY201 (numpy < 2.0 has only trapz)

SPEED_OF_LIGHT_M_S = 299_792_458.0

EFFECTIVE_WAVELENGTH_NM: dict[str, float] = {
    "g": 481.0,
    "r": 617.0,
    "i": 752.0,
    "z": 866.0,
    "y": 962.0,
}
"""Pan-STARRS1 effective wavelengths (Tonry et al. 2012), used as defaults for plain band names."""


@dataclass(frozen=True)
class Bandpass:
    """A filter's photon-counting system response.

    ``transmission`` is the relative photon response at ``wavelength_nm``
    (filter, optics and detector where available); its normalisation is
    irrelevant.
    """

    name: str
    wavelength_nm: np.ndarray = field(repr=False)
    transmission: np.ndarray = field(repr=False)

    @classmethod
    def from_file(
        cls, path: str | PathLike[str], name: str, *, unit_nm: float = 0.1, detector: str = "photon"
    ) -> Bandpass:
        """Read two columns, wavelength and transmission (SVO format, wavelengths in Angstrom).

        ``detector="energy"`` marks an energy-counter response, which is
        converted to a photon response by dividing by the wavelength.
        """
        if detector not in ("photon", "energy"):
            raise ValueError("detector must be 'photon' or 'energy'.")
        lam, trans = np.loadtxt(path, unpack=True, comments="#")
        lam = np.asarray(lam, float) * unit_nm
        trans = np.clip(np.asarray(trans, float), 0.0, None)
        if detector == "energy":
            trans = trans / lam
        order = np.argsort(lam)
        trans = trans[order] / trans.max()
        return cls(name, lam[order], trans)

    @property
    def pivot_nm(self) -> float:
        """Pivot wavelength ``sqrt(int T lam dlam / int T / lam dlam)``: where ``F_nu`` and ``F_lambda`` agree."""
        lam, t = self.wavelength_nm, self.transmission
        return float(np.sqrt(_trapezoid(t * lam, lam) / _trapezoid(t / lam, lam)))

    def power_law_ratio(self, beta: float, reference_nm: float, extinction_mag=0.0) -> float:
        """Synthetic band flux of ``F_nu = (lambda / reference_nm)^beta`` behind dust.

        ``extinction_mag`` is the extinction ``A(lambda)`` on :attr:`wavelength_nm`
        (or a scalar). The result is the band-averaged ``F_nu``,
        ``int T F_nu 10^(-0.4 A) dlam / lam / int T dlam / lam``, in units of the
        intrinsic ``F_nu`` at ``reference_nm``: the factor that turns the
        spectrum's normalisation into the measured flux density in this band.
        """
        lam, t = self.wavelength_nm, self.transmission
        w = t / lam
        flux = (lam / reference_nm) ** beta * 10 ** (-0.4 * np.asarray(extinction_mag, float))
        return float(_trapezoid(w * flux, lam) / _trapezoid(w, lam))

    def mean_attenuation(self, extinction_mag: np.ndarray, beta: float = 0.5) -> float:
        """Band-averaged extinction (mag) of a source ``F_nu ∝ nu^-beta`` for extinction ``A(lambda)``."""
        pivot = self.pivot_nm
        return float(
            -2.5 * np.log10(self.power_law_ratio(beta, pivot, extinction_mag) / self.power_law_ratio(beta, pivot))
        )


def _data_dir():
    return resources.files("grb_afterglow").joinpath("data", "filters")


@lru_cache(maxsize=1)
def _index() -> dict[str, dict[str, str]]:
    text = _data_dir().joinpath("index.csv").read_text()
    rows = csv.DictReader(line for line in text.splitlines() if not line.startswith("#"))
    return {row["name"]: row for row in rows}


_REGISTERED: dict[str, Bandpass] = {}


def available_filters() -> list[str]:
    """Names of every bandpass in the registry."""
    return sorted(set(_index()) | set(_REGISTERED))


def has_bandpass(name: str) -> bool:
    return str(name) in _REGISTERED or str(name) in _index()


@cache
def _load(name: str) -> Bandpass:
    row = _index()[name]
    with resources.as_file(_data_dir().joinpath(row["file"])) as path:
        return Bandpass.from_file(path, name, detector=row["detector"])


def get_bandpass(name: str) -> Bandpass:
    """The bandpass registered under ``name`` (e.g. ``"ztf:r"``)."""
    name = str(name)
    if name in _REGISTERED:
        return _REGISTERED[name]
    if name not in _index():
        raise KeyError(f"No bandpass {name!r}; known: {', '.join(available_filters())}.")
    return _load(name)


def register_bandpass(bandpass: Bandpass) -> None:
    """Add (or replace) a bandpass in the registry under its name."""
    _REGISTERED[bandpass.name] = bandpass


def filter_source(name: str) -> dict[str, str]:
    """Provenance of a committed bandpass: SVO ID, detector type, SVO's pivot wavelength."""
    return dict(_index()[str(name)])


def instrument_of(band: str) -> str | None:
    """The instrument part of an ``"instrument:filter"`` band name, or None for plain names."""
    band = str(band)
    return band.split(":", 1)[0] if ":" in band else None


def band_wavelength_nm(band: str, wavelengths_nm: Mapping[str, float] | None = None) -> float:
    """Wavelength of a band in nm: from ``wavelengths_nm``, the registry (pivot), or the default table."""
    band = str(band)
    if wavelengths_nm and band in wavelengths_nm:
        return float(wavelengths_nm[band])
    if has_bandpass(band):
        return get_bandpass(band).pivot_nm
    table = wavelengths_nm or EFFECTIVE_WAVELENGTH_NM
    key = band if band in table else band[-1:]
    if key not in table:
        raise KeyError(f"No effective wavelength known for band {band!r}.")
    return float(table[key])


def band_frequency_hz(band: str, wavelengths_nm: Mapping[str, float] | None = None) -> float:
    """Effective frequency of a band in Hz (see :func:`band_wavelength_nm`)."""
    return SPEED_OF_LIGHT_M_S / (band_wavelength_nm(band, wavelengths_nm) * 1e-9)
