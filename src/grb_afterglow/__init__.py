"""Photons to Power Laws: GRB optical afterglow photometry and light-curve physics."""

from __future__ import annotations

__version__ = "0.2.0"
__license__ = "MIT"

from .calibration import build_master_bias, build_master_flat, calibrate_frame, combine_frames
from .closure import CLOSURE_RELATIONS, evaluate_closure_relations
from .detection import detect_sources, estimate_background, estimate_fwhm
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
from .photometry import aperture_photometry_at, aperture_photometry_table, forced_photometry
from .pipeline import FrameResult, PipelineConfig, PipelineResult, reduce_frame, run_pipeline
from .zeropoint import ZeroPoint, calibrate_magnitudes, compute_zeropoint, match_sky, match_xy

__all__ = [
    "CLOSURE_RELATIONS",
    "FitResult",
    "FrameResult",
    "LightCurve",
    "PipelineConfig",
    "PipelineResult",
    "ZeroPoint",
    "__version__",
    "aperture_photometry_at",
    "aperture_photometry_table",
    "broken_power_law",
    "build_master_bias",
    "build_master_flat",
    "calibrate_frame",
    "calibrate_magnitudes",
    "combine_frames",
    "compare_models",
    "compute_zeropoint",
    "detect_sources",
    "estimate_background",
    "estimate_fwhm",
    "evaluate_closure_relations",
    "fit_broken_power_law",
    "fit_lightcurve",
    "fit_single_power_law",
    "flux_to_mag",
    "forced_photometry",
    "mag_to_flux",
    "match_sky",
    "match_xy",
    "reduce_frame",
    "run_pipeline",
    "single_power_law",
]
