"""Photons to Power Laws: GRB optical afterglow photometry and light-curve physics."""

from __future__ import annotations

__version__ = "0.5.0"
__license__ = "MIT"

from .calibration import build_master_bias, build_master_flat, calibrate_frame, combine_frames
from .closure import CLOSURE_RELATIONS, evaluate_closure_relations
from .detection import detect_sources, estimate_background, estimate_fwhm
from .extinction import deredden
from .filters import Bandpass, available_filters, get_bandpass, register_bandpass
from .fitting import (
    FitResult,
    broken_power_law,
    compare_models,
    fit_broken_power_law,
    fit_lightcurve,
    fit_single_power_law,
    single_power_law,
)
from .lightcurve import LightCurve, flux_to_mag, mag_to_flux
from .mcmc import Posterior, grid_posterior, sample_posterior
from .photometry import aperture_photometry_at, aperture_photometry_table, forced_photometry
from .pipeline import FrameResult, PipelineConfig, PipelineResult, Template, reduce_frame, run_pipeline
from .psf import EmpiricalPSF, SpatialPSF, build_psf, build_spatial_psf, psf_photometry, psf_photometry_tiled
from .spectral import SpectralModel
from .subtraction import DifferenceImage, subtract_images
from .zeropoint import ZeroPoint, calibrate_magnitudes, compute_zeropoint, match_sky, match_xy

__all__ = [
    "CLOSURE_RELATIONS",
    "Bandpass",
    "DifferenceImage",
    "EmpiricalPSF",
    "FitResult",
    "FrameResult",
    "LightCurve",
    "PipelineConfig",
    "PipelineResult",
    "Posterior",
    "SpatialPSF",
    "SpectralModel",
    "Template",
    "ZeroPoint",
    "__version__",
    "aperture_photometry_at",
    "aperture_photometry_table",
    "available_filters",
    "broken_power_law",
    "build_master_bias",
    "build_master_flat",
    "build_psf",
    "build_spatial_psf",
    "calibrate_frame",
    "calibrate_magnitudes",
    "combine_frames",
    "compare_models",
    "compute_zeropoint",
    "deredden",
    "detect_sources",
    "estimate_background",
    "estimate_fwhm",
    "evaluate_closure_relations",
    "fit_broken_power_law",
    "fit_lightcurve",
    "fit_single_power_law",
    "flux_to_mag",
    "forced_photometry",
    "get_bandpass",
    "grid_posterior",
    "mag_to_flux",
    "match_sky",
    "match_xy",
    "psf_photometry",
    "psf_photometry_tiled",
    "reduce_frame",
    "register_bandpass",
    "run_pipeline",
    "sample_posterior",
    "single_power_law",
    "subtract_images",
]
