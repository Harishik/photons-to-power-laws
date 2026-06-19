"""GRB optical afterglow photometry and light-curve fitting pipeline."""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Photons to Power Laws contributors"
__license__ = "MIT"

from .calibration import build_master_bias, build_master_flat, calibrate_frame
from .detection import detect_sources, estimate_background
from .photometry import aperture_photometry_at, aperture_photometry_table
from .zeropoint import compute_zeropoint, calibrate_magnitudes
from .lightcurve import LightCurve, flux_to_mag, mag_to_flux
from .fitting import (
    single_power_law,
    broken_power_law,
    fit_single_power_law,
    fit_broken_power_law,
)

__all__ = [
    "__version__",
    "build_master_bias",
    "build_master_flat",
    "calibrate_frame",
    "detect_sources",
    "estimate_background",
    "aperture_photometry_at",
    "aperture_photometry_table",
    "compute_zeropoint",
    "calibrate_magnitudes",
    "LightCurve",
    "flux_to_mag",
    "mag_to_flux",
    "single_power_law",
    "broken_power_law",
    "fit_single_power_law",
    "fit_broken_power_law",
]
