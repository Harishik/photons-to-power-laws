"""Background estimation, source detection, and seeing (FWHM) measurement."""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass

import numpy as np
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.table import Table
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

GAUSSIAN_FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))

# photutils 2.0 renamed several DAOStarFinder columns. Normalise to the new names
# so downstream code works with every supported photutils version.
_COLUMN_ALIASES = {"xcentroid": "x_centroid", "ycentroid": "y_centroid", "npix": "n_pixels"}
_PEAK_MAX_KWARG = "peak_max" if "peak_max" in inspect.signature(DAOStarFinder.__init__).parameters else "peakmax"


@dataclass
class BackgroundModel:
    """A smooth sky-background map and its per-pixel noise."""

    background: np.ndarray
    rms: np.ndarray

    @property
    def median_rms(self) -> float:
        return float(np.nanmedian(self.rms))


def estimate_background(
    data: np.ndarray, *, box_size: int = 32, filter_size: int = 3, sigma: float = 3.0
) -> BackgroundModel:
    """Estimate a spatially varying sky background and its noise.

    The image is divided into ``box_size`` boxes, each box's sigma-clipped median
    is measured with stars rejected, and the coarse grid is median-filtered and
    interpolated back to full resolution. Boxes are used as long as at least
    half their pixels survive clipping, which keeps crowded Galactic-plane
    fields usable. Images too small for a grid fall back to a single global
    sigma-clipped estimate.
    """
    image = np.asarray(data, dtype=float)
    box = int(min(box_size, max(min(image.shape) // 3, 1)))
    # Median-filtering a coarse grid of boxes flattens real gradients at the edges,
    # so only filter when there are enough boxes for it to reject star-hit boxes.
    n_boxes = min(image.shape) // box
    filt = filter_size if n_boxes >= 5 else 1
    try:
        bkg = Background2D(
            image,
            box_size=box,
            filter_size=filt,
            sigma_clip=SigmaClip(sigma=sigma),
            bkg_estimator=MedianBackground(),
            exclude_percentile=50.0,
        )
        return BackgroundModel(np.asarray(bkg.background), np.asarray(bkg.background_rms))
    except ValueError as exc:
        warnings.warn(f"Background2D failed ({exc}); using a global background.", stacklevel=2)
        _, med, std = sigma_clipped_stats(image, sigma=sigma)
        return BackgroundModel(np.full_like(image, med), np.full_like(image, std))


def make_source_mask(
    data: np.ndarray,
    background: BackgroundModel,
    *,
    nsigma: float = 2.5,
    npixels: int = 5,
    dilate: int = 3,
) -> np.ndarray:
    """Mask every pixel that belongs to a source, including its faint wings.

    Pixels more than ``nsigma`` above the background are grouped into connected
    segments; segments with at least ``npixels`` pixels are kept (so isolated
    noise peaks are not) and grown by ``dilate`` pixels. Excluding these pixels
    from sky annuli removes the bias that unresolved star wings otherwise add
    to the local sky estimate.
    """
    residual = np.asarray(data, dtype=float) - background.background
    above = residual > nsigma * background.rms
    labels, n = ndimage.label(above)
    if n == 0:
        return np.zeros(residual.shape, dtype=bool)
    sizes = ndimage.sum(above, labels, index=np.arange(1, n + 1))
    keep = np.isin(labels, np.flatnonzero(sizes >= npixels) + 1)
    if dilate > 0:
        keep = ndimage.binary_dilation(keep, iterations=dilate)
    return keep


def _normalise_columns(table: Table) -> Table:
    for old, new in _COLUMN_ALIASES.items():
        if old in table.colnames and new not in table.colnames:
            table.rename_column(old, new)
    return table


def detect_sources(
    data: np.ndarray,
    *,
    fwhm: float = 3.0,
    threshold_sigma: float = 5.0,
    background: BackgroundModel | np.ndarray | None = None,
    peak_max: float | None = None,
) -> Table:
    """Detect point sources in a background-subtracted image.

    Uses DAOFIND (Stetson 1987): the image is convolved with a Gaussian kernel of
    the given FWHM and local maxima above ``threshold_sigma`` times the
    background noise are kept.

    Parameters
    ----------
    data
        Calibrated image.
    fwhm
        Expected stellar FWHM in pixels.
    threshold_sigma
        Detection threshold in units of the background RMS.
    background
        A :class:`BackgroundModel` or a background array. Estimated if omitted.
    peak_max
        Sources whose peak exceeds this value (e.g. saturation) are rejected.

    Returns
    -------
    astropy.table.Table
        Sources sorted brightest first, with ``x_centroid`` and ``y_centroid``.
    """
    image = np.asarray(data, dtype=float)
    if background is None:
        background = estimate_background(image)
    if isinstance(background, BackgroundModel):
        residual = image - background.background
        noise = background.median_rms
    else:
        residual = image - np.asarray(background, dtype=float)
        _, _, noise = sigma_clipped_stats(residual)
    threshold = threshold_sigma * max(float(noise), 1e-12)
    finder = DAOStarFinder(fwhm=fwhm, threshold=threshold, exclude_border=True, **{_PEAK_MAX_KWARG: peak_max})
    sources = finder(residual)
    if sources is None or len(sources) == 0:
        return Table(names=("x_centroid", "y_centroid", "flux"), dtype=(float, float, float))
    sources = _normalise_columns(sources)
    sources.sort("flux", reverse=True)
    return sources


def _gaussian_stamp(coords, amplitude, x0, y0, sigma, offset):
    yy, xx = coords
    r2 = (xx - x0) ** 2 + (yy - y0) ** 2
    return (amplitude * np.exp(-r2 / (2.0 * sigma**2)) + offset).ravel()


def _fit_stamp_fwhm(stamp: np.ndarray, guess_sigma: float) -> float:
    ny, nx = stamp.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    offset = float(np.median(stamp))
    p0 = [float(stamp.max() - offset), nx / 2.0, ny / 2.0, guess_sigma, offset]
    bounds = ([0.0, 0.0, 0.0, 0.3, -np.inf], [np.inf, nx, ny, nx, np.inf])
    popt, _ = curve_fit(_gaussian_stamp, (yy, xx), stamp.ravel(), p0=p0, bounds=bounds)
    return float(popt[3] * GAUSSIAN_FWHM_PER_SIGMA)


def estimate_fwhm(
    data: np.ndarray,
    sources: Table,
    *,
    guess: float = 3.0,
    max_stars: int = 25,
    min_stars: int = 5,
    isolation: float | None = None,
    background: BackgroundModel | np.ndarray | None = None,
) -> float:
    """Measure the image seeing by fitting 2-D Gaussians to isolated stars.

    The brightest sources without a neighbour inside ``isolation`` pixels are
    fitted individually and the sigma-clipped median FWHM is returned. If
    fewer than ``min_stars`` can be fitted, the isolation radius is relaxed
    (down to half its value), so that a single object, such as a galaxy,
    cannot set the seeing on its own. If no star can be fitted, ``guess`` is
    returned unchanged.
    """
    if len(sources) == 0:
        return float(guess)
    image = np.asarray(data, dtype=float)
    if isinstance(background, BackgroundModel):
        image = image - background.background
    elif background is not None:
        image = image - np.asarray(background, dtype=float)

    half = int(np.ceil(2.5 * guess))
    isolation = float(isolation if isolation is not None else 2 * half)
    xy = np.column_stack([np.asarray(sources["x_centroid"]), np.asarray(sources["y_centroid"])])
    tree = cKDTree(xy)
    neighbour_dist, _ = tree.query(xy, k=2) if len(xy) > 1 else (np.full((1, 2), np.inf), None)

    ny, nx = image.shape
    fitted: dict[int, float | None] = {}  # source index -> FWHM, None if the fit failed
    for radius in (isolation, 0.75 * isolation, 0.5 * isolation):
        fwhms: list[float] = []
        for i, ((x, y), dist) in enumerate(zip(xy, neighbour_dist[:, 1], strict=True)):
            if dist < radius:
                continue
            ix, iy = int(np.rint(x)), int(np.rint(y))
            if ix - half < 0 or iy - half < 0 or ix + half >= nx or iy + half >= ny:
                continue
            if i not in fitted:
                stamp = image[iy - half : iy + half + 1, ix - half : ix + half + 1]
                try:
                    fitted[i] = _fit_stamp_fwhm(stamp, guess / GAUSSIAN_FWHM_PER_SIGMA)
                except (RuntimeError, ValueError):
                    fitted[i] = None
            fwhm = fitted[i]
            if fwhm is not None and 0.5 * guess < fwhm < 4.0 * guess:
                fwhms.append(fwhm)
            if len(fwhms) >= max_stars:
                break
        if len(fwhms) >= min_stars:
            break
    if not fwhms:
        return float(guess)
    _, median, _ = sigma_clipped_stats(np.asarray(fwhms), sigma=3.0)
    return float(median)
