"""Source detection and image background estimation."""

from __future__ import annotations

import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder


def estimate_background(data: np.ndarray, *, box_size: int = 32) -> np.ndarray:
    """Estimate a smooth 2-D sky background."""
    image = np.asarray(data, dtype=float)
    try:
        bkg = Background2D(
            image,
            box_size=box_size,
            filter_size=3,
            bkg_estimator=MedianBackground(),
        )
        return np.asarray(bkg.background, dtype=float)
    except Exception:
        _, med, _ = sigma_clipped_stats(image)
        return np.full_like(image, med, dtype=float)


def detect_sources(
    data: np.ndarray,
    *,
    fwhm: float = 3.0,
    threshold_sigma: float = 5.0,
    background: np.ndarray | None = None,
) -> Table:
    """Detect point sources with a DAOStarFinder-style algorithm."""
    image = np.asarray(data, dtype=float)
    if background is None:
        background = estimate_background(image)
    residual = image - background
    _, _, std = sigma_clipped_stats(residual)
    threshold = threshold_sigma * max(float(std), 1e-6)
    finder = DAOStarFinder(fwhm=fwhm, threshold=threshold)
    sources = finder(residual)
    if sources is None:
        return Table(names=("x_centroid", "y_centroid", "flux"), dtype=(float, float, float))
    sources.sort("flux")
    sources.reverse()
    return sources


def estimate_fwhm(data: np.ndarray, sources: Table, *, default: float = 3.5) -> float:
    """Estimate a rough median FWHM from detected-source second moments."""
    if len(sources) == 0:
        return default
    vals = []
    for col in ("sharpness",):
        if col in sources.colnames:
            arr = np.asarray(sources[col], dtype=float)
            arr = arr[np.isfinite(arr) & (arr > 0)]
            if arr.size:
                # DAOStarFinder sharpness is not a FWHM, but it tracks compactness.
                vals.extend(np.clip(default / arr, 1.5, 8.0).tolist())
    return float(np.median(vals)) if vals else float(default)
