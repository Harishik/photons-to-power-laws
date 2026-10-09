"""Light-curve container and AB magnitude / flux-density conversions.

Light curves are stored in flux density (microjansky) rather than magnitudes.
Forced photometry of a faded afterglow can return zero or negative flux, which
has no magnitude but is still a valid, unbiased measurement that a fit should
use. Magnitudes and upper limits are derived on demand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from os import PathLike

import numpy as np
from astropy.table import Table

AB_ZEROPOINT_UJY = 3631e6
"""Flux density of an AB magnitude 0 source, in microjansky."""

AB_MAG_1UJY = 23.9
"""AB magnitude of a 1 microjansky source."""

SECONDS_PER_DAY = 86400.0


def mag_to_flux(mag: np.ndarray | float) -> np.ndarray:
    """Convert AB magnitude to flux density in microjansky."""
    return AB_ZEROPOINT_UJY * 10 ** (-0.4 * np.asarray(mag, dtype=float))


def flux_to_mag(flux_ujy: np.ndarray | float) -> np.ndarray:
    """Convert flux density in microjansky to AB magnitude (NaN if not positive)."""
    flux = np.asarray(flux_ujy, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(flux > 0, -2.5 * np.log10(np.where(flux > 0, flux, 1.0) / AB_ZEROPOINT_UJY), np.nan)


def mag_err_to_flux_err(mag: np.ndarray, mag_err: np.ndarray) -> np.ndarray:
    """Propagate a magnitude error to a flux-density error."""
    return mag_to_flux(mag) * 0.4 * np.log(10.0) * np.asarray(mag_err, dtype=float)


def flux_err_to_mag_err(flux: np.ndarray, flux_err: np.ndarray) -> np.ndarray:
    """Propagate a flux-density error to a magnitude error (NaN if flux is not positive)."""
    flux = np.asarray(flux, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(flux > 0, 2.5 / np.log(10.0) * np.asarray(flux_err, float) / flux, np.nan)


@dataclass
class LightCurve:
    """A (possibly multi-band) optical light curve in flux density.

    Parameters
    ----------
    time
        Time since the burst trigger, in seconds.
    flux, flux_err
        Flux density and its 1-sigma uncertainty, in microjansky.
    band
        Filter name for each point. A single string applies to all points.
    meta
        Free-form metadata, e.g. the object name or trigger time.
    """

    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    band: np.ndarray | str = "unknown"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time = np.atleast_1d(np.asarray(self.time, dtype=float))
        self.flux = np.atleast_1d(np.asarray(self.flux, dtype=float))
        self.flux_err = np.atleast_1d(np.asarray(self.flux_err, dtype=float))
        if isinstance(self.band, str):
            self.band = np.full(self.time.shape, self.band, dtype=object)
        self.band = np.atleast_1d(np.asarray(self.band, dtype=object))
        if not (self.time.shape == self.flux.shape == self.flux_err.shape == self.band.shape):
            raise ValueError("time, flux, flux_err, and band must have the same shape.")
        order = np.argsort(self.time, kind="stable")
        self.time, self.flux = self.time[order], self.flux[order]
        self.flux_err, self.band = self.flux_err[order], self.band[order]

    @classmethod
    def from_magnitudes(
        cls,
        time: np.ndarray,
        mag: np.ndarray,
        mag_err: np.ndarray,
        band: np.ndarray | str = "unknown",
        meta: dict | None = None,
    ) -> LightCurve:
        """Build a light curve from AB magnitudes."""
        mag = np.asarray(mag, dtype=float)
        return cls(time, mag_to_flux(mag), mag_err_to_flux_err(mag, mag_err), band, dict(meta or {}))

    def __len__(self) -> int:
        return int(self.time.size)

    @property
    def time_days(self) -> np.ndarray:
        return self.time / SECONDS_PER_DAY

    @property
    def mag(self) -> np.ndarray:
        """AB magnitude (NaN where the flux is not positive)."""
        return flux_to_mag(self.flux)

    @property
    def mag_err(self) -> np.ndarray:
        return flux_err_to_mag_err(self.flux, self.flux_err)

    @property
    def snr(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.flux_err > 0, self.flux / self.flux_err, 0.0)

    @property
    def bands(self) -> list[str]:
        """Unique band names in order of first appearance."""
        return list(dict.fromkeys(str(b) for b in self.band))

    def detected(self, nsigma: float = 3.0) -> np.ndarray:
        """Mask of points detected above ``nsigma``."""
        return self.snr >= nsigma

    def upper_limit_mag(self, nsigma: float = 3.0) -> np.ndarray:
        """``nsigma`` upper limit in AB mag: the source is fainter than this."""
        limit = np.maximum(self.flux, 0.0) + nsigma * self.flux_err
        return flux_to_mag(limit)

    def subset(self, mask: np.ndarray) -> LightCurve:
        mask = np.asarray(mask, dtype=bool)
        return LightCurve(self.time[mask], self.flux[mask], self.flux_err[mask], self.band[mask], dict(self.meta))

    def select(self, band: str) -> LightCurve:
        """Return the points observed in one band."""
        return self.subset(self.band == band)

    def finite(self) -> LightCurve:
        """Drop points with non-finite values, non-positive time, or zero error."""
        mask = (
            np.isfinite(self.time)
            & np.isfinite(self.flux)
            & np.isfinite(self.flux_err)
            & (self.time > 0)
            & (self.flux_err > 0)
        )
        return self.subset(mask)

    def to_table(self) -> Table:
        """Export to an :class:`astropy.table.Table` with derived columns."""
        table = Table()
        table["time_s"] = self.time
        table["time_days"] = self.time_days
        table["band"] = self.band.astype(str)
        table["flux_ujy"] = self.flux
        table["flux_err_ujy"] = self.flux_err
        table["mag_ab"] = self.mag
        table["mag_err"] = self.mag_err
        table["snr"] = self.snr
        table.meta.update(self.meta)
        return table

    def save_csv(self, path: str | PathLike[str]) -> None:
        """Write the light curve to CSV."""
        table = self.to_table()
        table.meta.clear()
        for col in table.colnames:
            if table[col].dtype.kind == "f":
                table[col].info.format = ".6g"
        table.write(path, format="ascii.csv", overwrite=True)

    @classmethod
    def read_csv(cls, path: str | PathLike[str]) -> LightCurve:
        """Read a CSV written by :meth:`save_csv`.

        Files with ``mag``/``mag_err`` columns instead of flux are also accepted.
        """
        table = Table.read(path, format="ascii.csv")
        band = np.asarray(table["band"], dtype=object) if "band" in table.colnames else "unknown"
        if "flux_ujy" in table.colnames:
            return cls(table["time_s"], table["flux_ujy"], table["flux_err_ujy"], band)
        mag_col = "mag_ab" if "mag_ab" in table.colnames else "mag"
        return cls.from_magnitudes(table["time_s"], table[mag_col], table["mag_err"], band)
