"""Image calibration: master frames, bias subtraction, and flat-field correction.

A CCD pixel value is ``raw = bias + flat * sky_signal`` (plus noise). Calibration
inverts that model, ``(raw - bias) / flat``, so that pixel values become
proportional to the light that actually arrived from the sky.
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip, sigma_clipped_stats


def _stack(frames: Sequence[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(f, dtype=float) for f in frames]
    if not arrays:
        raise ValueError("Need at least one frame to combine.")
    shape = arrays[0].shape
    for arr in arrays:
        if arr.shape != shape:
            raise ValueError("All frames must have the same shape.")
    return np.stack(arrays, axis=0)


def combine_frames(frames: Sequence[np.ndarray], *, sigma: float = 3.0) -> np.ndarray:
    """Combine a stack of frames pixel by pixel with sigma clipping.

    Each pixel's values across the stack are clipped at ``sigma`` robust standard
    deviations from their median before the median is taken. This rejects cosmic
    rays and other transient outliers that a plain median of few frames can
    let through.

    Parameters
    ----------
    frames
        Two-dimensional frames of identical shape.
    sigma
        Clipping threshold in units of the MAD-based standard deviation.

    Returns
    -------
    numpy.ndarray
        The combined frame.
    """
    cube = _stack(frames)
    if cube.shape[0] < 3:
        return np.nanmedian(cube, axis=0)
    clipped = sigma_clip(cube, sigma=sigma, axis=0, maxiters=5, cenfunc="median", stdfunc="mad_std", masked=True)
    combined = np.ma.median(clipped, axis=0)
    fallback = np.nanmedian(cube, axis=0)
    return np.where(np.ma.getmaskarray(combined), fallback, np.ma.getdata(combined))


def build_master_bias(frames: Sequence[np.ndarray], *, sigma: float = 3.0) -> np.ndarray:
    """Build a master bias frame from zero-second exposures.

    Parameters
    ----------
    frames
        Raw bias frames.
    sigma
        Outlier-rejection threshold passed to :func:`combine_frames`.
    """
    return combine_frames(frames, sigma=sigma)


def build_master_flat(
    frames: Sequence[np.ndarray],
    *,
    master_bias: np.ndarray | None = None,
    sigma: float = 3.0,
) -> np.ndarray:
    """Build a master flat normalised to a median of one.

    Each flat is bias-subtracted and scaled by its own median before combining,
    so exposures taken at different twilight brightness levels contribute equally.

    Parameters
    ----------
    frames
        Raw flat-field frames.
    master_bias
        Optional master bias to subtract from every flat first.
    sigma
        Outlier-rejection threshold passed to :func:`combine_frames`.
    """
    cube = _stack(frames)
    if master_bias is not None:
        bias = np.asarray(master_bias, dtype=float)
        if bias.shape != cube.shape[1:]:
            raise ValueError("master_bias shape must match flat frames.")
        cube = cube - bias
    scaled = []
    for frame in cube:
        _, med, _ = sigma_clipped_stats(frame, sigma=sigma)
        if not np.isfinite(med) or med <= 0:
            raise ValueError("Cannot normalise a flat frame with a non-positive median.")
        scaled.append(frame / med)
    flat = combine_frames(scaled, sigma=sigma)
    _, med, _ = sigma_clipped_stats(flat, sigma=sigma)
    flat = flat / med
    flat[~np.isfinite(flat) | (flat <= 0)] = 1.0
    return flat


def calibrate_frame(
    raw: np.ndarray,
    *,
    master_bias: np.ndarray | None = None,
    master_flat: np.ndarray | None = None,
) -> np.ndarray:
    """Apply ``(raw - bias) / flat`` to a science frame.

    Non-finite or non-positive flat pixels are treated as one, so a bad flat
    pixel never produces an infinite science pixel.
    """
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
        safe[~np.isfinite(safe) | (safe <= 0)] = 1.0
        data /= safe
    return data


def load_fits(path: str | PathLike[str]) -> tuple[np.ndarray, fits.Header]:
    """Load the first image HDU and its header from a FITS file.

    Works for plain and tile-compressed (``.fits.fz``) files, where the image
    lives in the first extension rather than the primary HDU.
    """
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and np.ndim(hdu.data) == 2:
                return np.asarray(hdu.data, dtype=float), hdu.header.copy()
    raise ValueError(f"No two-dimensional image found in {path}.")


def save_fits(path: str | PathLike[str], data: np.ndarray, header: fits.Header | None = None) -> None:
    """Write a primary-image FITS file, overwriting any existing file."""
    fits.PrimaryHDU(np.asarray(data, dtype=np.float32), header=header).writeto(path, overwrite=True)
