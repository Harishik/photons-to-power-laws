"""Light-curve container and AB magnitude/flux conversions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

AB_ZEROPOINT_UJY = 3631e6


def mag_to_flux(mag: np.ndarray | float) -> np.ndarray:
    """Convert AB magnitude to flux density in microjansky."""
    return AB_ZEROPOINT_UJY * 10 ** (-0.4 * np.asarray(mag, dtype=float))


def flux_to_mag(flux_ujy: np.ndarray | float) -> np.ndarray:
    """Convert flux density in microjansky to AB magnitude."""
    flux = np.asarray(flux_ujy, dtype=float)
    return -2.5 * np.log10(flux / AB_ZEROPOINT_UJY)


def mag_err_to_flux_err(mag: np.ndarray, mag_err: np.ndarray) -> np.ndarray:
    flux = mag_to_flux(mag)
    return flux * np.log(10) * 0.4 * np.asarray(mag_err, dtype=float)


@dataclass
class LightCurve:
    """A single-band optical light curve."""

    time: np.ndarray
    mag: np.ndarray
    mag_err: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float)
        self.mag = np.asarray(self.mag, dtype=float)
        self.mag_err = np.asarray(self.mag_err, dtype=float)
        if not (self.time.shape == self.mag.shape == self.mag_err.shape):
            raise ValueError("time, mag, and mag_err must have the same shape.")
        order = np.argsort(self.time)
        self.time = self.time[order]
        self.mag = self.mag[order]
        self.mag_err = self.mag_err[order]

    def __len__(self) -> int:
        return int(self.time.size)

    @property
    def flux(self) -> np.ndarray:
        return mag_to_flux(self.mag)

    @property
    def flux_err(self) -> np.ndarray:
        return mag_err_to_flux_err(self.mag, self.mag_err)

    def finite(self) -> "LightCurve":
        """Return only finite positive-time points."""
        mask = (
            np.isfinite(self.time)
            & np.isfinite(self.mag)
            & np.isfinite(self.mag_err)
            & (self.time > 0)
            & (self.mag_err > 0)
        )
        return LightCurve(self.time[mask], self.mag[mask], self.mag_err[mask], dict(self.meta))

    def save_csv(self, path: str) -> None:
        """Save time, magnitude, and flux columns to CSV."""
        import csv

        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["time_s", "mag", "mag_err", "flux_ujy", "flux_err_ujy"])
            for row in zip(self.time, self.mag, self.mag_err, self.flux, self.flux_err):
                writer.writerow([float(x) for x in row])
