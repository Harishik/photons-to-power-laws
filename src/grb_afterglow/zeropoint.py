"""Catalog cross-matching and photometric zero-point calibration.

Instrumental magnitudes depend on the telescope, filter, exposure, and the
atmosphere on the night. Field stars with known catalog magnitudes fix the
offset, the zero point, between the two scales:

    m_catalog = m_instrumental + ZP  (+ C * colour, optionally)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip
from astropy.table import Table
from scipy.spatial import cKDTree


@dataclass
class ZeroPoint:
    """A photometric zero-point solution.

    Attributes
    ----------
    value
        Zero point, added to instrumental magnitudes.
    error
        Uncertainty of ``value`` (standard error of the mean of the residuals).
    n_stars
        Number of stars kept after sigma clipping.
    scatter
        Standard deviation of the kept residuals, a measure of calibration noise.
    color_term
        Slope ``C`` of the colour dependence, if a colour term was fitted.
    color_term_error
        Uncertainty of ``color_term``.
    reference_color
        Colour at which ``value`` applies when a colour term was fitted
        (0 unless the zero point was moved, see :meth:`at_color`).
    used
        Boolean mask over the input stars marking those kept by the clipping.
    """

    value: float
    error: float
    n_stars: int
    scatter: float
    color_term: float | None = None
    used: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool), repr=False)
    color_term_error: float = 0.0
    reference_color: float = 0.0
    mean_color: float = 0.0

    def at_color(self, color: float) -> ZeroPoint:
        """The zero point for a source of the given catalog colour.

        Without a colour term this is the same zero point. With one, ``value``
        moves along the fitted slope and the slope's uncertainty is added for
        the distance from the calibration stars' mean colour.
        """
        if self.color_term is None:
            return self
        value = self.value + self.color_term * (color - self.reference_color)
        error = float(np.hypot(self.error, self.color_term_error * (color - self.mean_color)))
        return replace(self, value=float(value), error=error, reference_color=float(color))


def _unique_nearest(dist: np.ndarray, idx_meas: np.ndarray, max_sep: float):
    """Keep reference->measured matches within ``max_sep``, one per measured source."""
    order = np.argsort(dist)
    taken: set[int] = set()
    keep_ref, keep_meas = [], []
    for j in order:
        i = int(idx_meas[j])
        if not np.isfinite(dist[j]) or dist[j] > max_sep or i in taken:
            continue
        taken.add(i)
        keep_ref.append(int(j))
        keep_meas.append(i)
    keep_ref_arr = np.asarray(keep_ref, dtype=int)
    return np.asarray(keep_meas, dtype=int), keep_ref_arr, dist[keep_ref_arr]


def match_xy(
    measured: Table,
    reference: Table,
    *,
    max_sep: float = 3.0,
    x_meas: str = "x",
    y_meas: str = "y",
    x_ref: str = "x",
    y_ref: str = "y",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-to-one nearest-neighbour match of two catalogs in pixel space.

    Returns
    -------
    idx_meas, idx_ref, separation
        Index arrays into ``measured`` and ``reference`` and the separations in
        pixels. Each measured source is matched at most once.
    """
    meas = np.column_stack([np.asarray(measured[x_meas], float), np.asarray(measured[y_meas], float)])
    ref = np.column_stack([np.asarray(reference[x_ref], float), np.asarray(reference[y_ref], float)])
    if len(meas) == 0 or len(ref) == 0:
        empty = np.zeros(0, dtype=int)
        return empty, empty, np.zeros(0)
    dist, idx = cKDTree(meas).query(ref, k=1)
    return _unique_nearest(np.asarray(dist), np.asarray(idx), max_sep)


def match_sky(
    measured: SkyCoord,
    reference: SkyCoord,
    *,
    max_sep_arcsec: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-to-one nearest-neighbour match of two catalogs on the sky.

    Returns
    -------
    idx_meas, idx_ref, separation_arcsec
    """
    if len(measured) == 0 or len(reference) == 0:
        empty = np.zeros(0, dtype=int)
        return empty, empty, np.zeros(0)
    idx, sep2d, _ = reference.match_to_catalog_sky(measured)
    return _unique_nearest(sep2d.to_value(u.arcsec), np.asarray(idx), max_sep_arcsec)


def compute_zeropoint(
    instrumental_mags: np.ndarray,
    catalog_mags: np.ndarray,
    *,
    instrumental_errs: np.ndarray | None = None,
    catalog_errs: np.ndarray | None = None,
    color: np.ndarray | None = None,
    color_term: float | None = None,
    color_term_error: float = 0.0,
    sigma: float = 3.0,
) -> ZeroPoint:
    """Solve a robust zero point from matched stars.

    Residuals ``catalog - instrumental`` are iteratively sigma clipped (with a
    MAD-based width, so a few blended or variable stars cannot inflate it), and
    the zero point is the inverse-variance weighted mean of the survivors. If
    ``color`` is given, a linear colour term is fitted alongside.

    Parameters
    ----------
    instrumental_mags, catalog_mags
        Matched magnitudes.
    instrumental_errs, catalog_errs
        Optional per-star uncertainties used as weights.
    color
        Optional catalog colour (e.g. ``g - r``) for a colour-term fit.
    color_term, color_term_error
        A known colour-term slope (e.g. measured once from many frames of the
        same instrument) and its uncertainty; with ``color`` given, the slope
        is then fixed instead of fitted.
    sigma
        Clipping threshold.
    """
    inst = np.asarray(instrumental_mags, dtype=float)
    cat = np.asarray(catalog_mags, dtype=float)
    var = np.zeros_like(inst)
    if instrumental_errs is not None:
        var = var + np.asarray(instrumental_errs, dtype=float) ** 2
    if catalog_errs is not None:
        var = var + np.asarray(catalog_errs, dtype=float) ** 2
    col = None if color is None else np.asarray(color, dtype=float)

    good = np.isfinite(inst) & np.isfinite(cat) & np.isfinite(var)
    if col is not None:
        good &= np.isfinite(col)
    if np.count_nonzero(good) == 0:
        raise ValueError("No valid star pairs to compute a zero point.")

    resid = cat - inst
    used = good.copy()
    slope: float | None = None
    for _ in range(10):
        if col is not None and color_term is not None:
            slope = float(color_term)
            model = resid - slope * col
        elif col is not None and np.count_nonzero(used) >= 3:
            slope = float(np.polyfit(col[used], resid[used], 1)[0])
            model = resid - slope * col
        else:
            model = resid
        clipped = sigma_clip(
            np.where(good, model, np.nan),
            sigma=sigma,
            cenfunc="median",
            stdfunc="mad_std",
            maxiters=1,
            masked=True,
        )
        new_used = good & ~np.ma.getmaskarray(clipped)
        if np.array_equal(new_used, used) or np.count_nonzero(new_used) == 0:
            break
        used = new_used

    detrended = resid - (slope * col if (slope is not None and col is not None) else 0.0)
    vals = detrended[used]
    n = int(vals.size)
    if np.any(var[used] > 0):
        weights = 1.0 / np.maximum(var[used], 1e-8)
        value = float(np.sum(weights * vals) / np.sum(weights))
    else:
        value = float(np.median(vals))
    scatter = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    formal = float(np.sqrt(1.0 / np.sum(1.0 / np.maximum(var[used], 1e-8)))) if np.any(var[used] > 0) else 0.0
    error = max(scatter / np.sqrt(n) if n > 1 else 0.0, formal)
    slope_err, mean_col = 0.0, 0.0
    if slope is not None:
        c = col[used]
        mean_col = float(np.mean(c))
        spread = float(np.sum((c - mean_col) ** 2))
        if color_term is not None:
            slope_err = float(color_term_error)
        else:
            slope_err = scatter / np.sqrt(spread) if spread > 0 else float("inf")
        # The intercept at colour 0 is uncertain by the slope error times the mean colour as well.
        error = float(np.hypot(error, slope_err * mean_col))
    return ZeroPoint(value, error, n, scatter, slope, used, slope_err, 0.0, mean_col)


def calibrate_magnitudes(
    instrumental_mags: np.ndarray,
    zeropoint: ZeroPoint,
    instrumental_errs: np.ndarray | None = None,
    color: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a zero point: ``m = m_inst + ZP (+ C * (colour - reference colour))``.

    The zero-point uncertainty is added in quadrature to every star.
    """
    inst = np.asarray(instrumental_mags, dtype=float)
    cal = inst + zeropoint.value
    if zeropoint.color_term is not None and color is not None:
        cal = cal + zeropoint.color_term * (np.asarray(color, dtype=float) - zeropoint.reference_color)
    if instrumental_errs is None:
        cal_err = np.full_like(cal, zeropoint.error)
    else:
        cal_err = np.hypot(np.asarray(instrumental_errs, dtype=float), zeropoint.error)
    return cal, cal_err
