"""Image calibration: bias subtraction and flat-field correction."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats


def _stack(frames: Sequence[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(f, dtype=float) for f in frames]
    if not arrays:
        raise ValueError("Need at least one frame to combine.")
    shape = arrays[0].shape
    for arr in arrays:
        if arr.shape != shape:
            raise ValueError("All frames must have the same shape.")
    return np.stack(arrays, axis=0)


def build_master_bias(frames: Sequence[np.ndarray], *, sigma: float = 5.0) -> np.ndarray:
    """Build a master bias frame with a robust median combine."""
    cube = _stack(frames)
    return np.nanmedian(cube, axis=0)


def build_master_flat(
    frames: Sequence[np.ndarray],
    *,
    master_bias: np.ndarray | None = None,
    sigma: float = 5.0,
) -> np.ndarray:
    """Build a normalized master flat frame."""
    cube = _stack(frames)
    if master_bias is not None:
        bias = np.asarray(master_bias, dtype=float)
        if bias.shape != cube.shape[1:]:
            raise ValueError("master_bias shape must match flat frames.")
        cube = cube - bias
    flat = np.nanmedian(cube, axis=0)
    _, med, _ = sigma_clipped_stats(flat, sigma=sigma)
    if not np.isfinite(med) or med == 0:
        raise ValueError("Cannot normalize flat with non-finite or zero median.")
    flat = flat / med
    flat[~np.isfinite(flat)] = 1.0
    flat[flat <= 0] = 1.0
    return flat


def calibrate_frame(
    raw: np.ndarray,
    *,
    master_bias: np.ndarray | None = None,
    master_flat: np.ndarray | None = None,
) -> np.ndarray:
    """Apply `(raw - bias) / flat` to a science frame."""
    data = np.asarray(raw, dtype=float).copy()
    if master_bias is not None:
        bias = np.asarray(master_bias, dtype=float)
        if bias.shape != data.shape:
            raise ValueError("master_bias shape must match raw frame.")
        data -= bias
    if master_flat is not None:
        flat = np.asarray(master_flat, dtype=float)
        if flat.shape != data.shape:
            raise ValueError("master_flat shape must match raw frame.")
        safe = flat.copy()
        safe[~np.isfinite(safe)] = 1.0
        safe[safe <= 0] = 1.0
        data /= safe
    return data


def load_fits(path: str) -> tuple[np.ndarray, fits.Header]:
    """Load the primary image and header from a FITS file."""
    with fits.open(path) as hdul:
        return np.asarray(hdul[0].data, dtype=float), hdul[0].header.copy()


def save_fits(path: str, data: np.ndarray, header: fits.Header | None = None) -> None:
    """Write a primary-image FITS file."""
    fits.PrimaryHDU(np.asarray(data, dtype=float), header=header).writeto(path, overwrite=True)
