"""Aperture photometry with a local sky annulus and a CCD-equation error budget."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.stats import SigmaClip
from astropy.table import Table
from photutils.aperture import (
    ApertureStats,
    CircularAnnulus,
    CircularAperture,
    aperture_photometry,
)

# 2.5 / ln(10): converts a fractional flux error into a magnitude error.
MAG_ERR_PER_FRAC_FLUX = 2.5 / np.log(10.0)


@dataclass
class PhotometryResult:
    """Aperture photometry of a single position."""

    x: float
    y: float
    aperture_sum: float
    sky_per_pixel: float
    sky_std: float
    net_counts: float
    net_counts_err: float
    snr: float
    instrumental_mag: float
    instrumental_mag_err: float


def instrumental_magnitude(
    net_counts: np.ndarray | float, net_counts_err: np.ndarray | float, exptime: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Convert net counts to instrumental magnitudes ``-2.5 log10(counts / exptime)``.

    Non-positive fluxes have no magnitude and return NaN.
    """
    counts = np.asarray(net_counts, dtype=float)
    err = np.asarray(net_counts_err, dtype=float)
    if exptime <= 0:
        raise ValueError("exptime must be positive.")
    with np.errstate(divide="ignore", invalid="ignore"):
        positive = counts > 0
        mag = np.where(positive, -2.5 * np.log10(np.where(positive, counts, 1.0) / exptime), np.nan)
        mag_err = np.where(positive, MAG_ERR_PER_FRAC_FLUX * err / np.where(positive, counts, 1.0), np.nan)
    return mag, mag_err


def forced_photometry(
    data: np.ndarray,
    positions: np.ndarray,
    *,
    r_aper: float,
    r_in: float | None = None,
    r_out: float | None = None,
    gain: float | None = None,
    exptime: float = 1.0,
    sky_sigma: float = 3.0,
    sky_mask: np.ndarray | None = None,
) -> Table:
    """Measure aperture photometry at fixed pixel positions.

    The sky level is the sigma-clipped median of a surrounding annulus, so stars
    falling in the annulus do not bias it. The uncertainty follows the CCD
    equation, written in counts:

    .. math::

        \\sigma^2 = \\frac{N}{g} + n_{ap}\\,\\sigma_{sky}^2
                   \\left(1 + \\frac{n_{ap}}{n_{sky}}\\right)

    where ``N`` is the net source counts, ``g`` the gain, ``n_ap`` and ``n_sky``
    the aperture and annulus areas, and ``sigma_sky`` the measured per-pixel sky
    scatter (which already contains sky shot noise and read noise). The last term
    accounts for the uncertainty of the sky estimate itself.

    Parameters
    ----------
    data
        Calibrated image.
    positions
        ``(N, 2)`` array of ``(x, y)`` pixel positions.
    r_aper
        Aperture radius in pixels.
    r_in, r_out
        Sky-annulus radii in pixels. Default to 3 and 5 aperture radii.
    gain
        Detector gain in electrons per count. If omitted, source shot noise is
        neglected (the background-limited approximation).
    exptime
        Exposure time used to normalise the instrumental magnitudes.
    sky_sigma
        Sigma-clipping threshold for the annulus.
    sky_mask
        Boolean image, True for pixels to exclude from the sky annulus (for
        example from :func:`~grb_afterglow.detection.make_source_mask`).

    Returns
    -------
    astropy.table.Table
        One row per position.
    """
    image = np.asarray(data, dtype=float)
    pos = np.atleast_2d(np.asarray(positions, dtype=float))
    r_in = float(r_in if r_in is not None else 3.0 * r_aper)
    r_out = float(r_out if r_out is not None else 5.0 * r_aper)
    if not 0 < r_aper < r_in < r_out:
        raise ValueError("Radii must satisfy 0 < r_aper < r_in < r_out.")

    aperture = CircularAperture(pos, r=r_aper)
    annulus = CircularAnnulus(pos, r_in=r_in, r_out=r_out)
    mask = ~np.isfinite(image)
    annulus_mask = mask if sky_mask is None else (mask | np.asarray(sky_mask, dtype=bool))
    phot = aperture_photometry(image, aperture, mask=mask)
    sky_stats = ApertureStats(image, annulus, mask=annulus_mask, sigma_clip=SigmaClip(sigma=sky_sigma))

    aperture_sum = np.asarray(phot["aperture_sum"], dtype=float)
    sky = np.atleast_1d(np.asarray(sky_stats.median, dtype=float))
    sky_std = np.atleast_1d(np.asarray(sky_stats.std, dtype=float))
    n_sky = np.maximum(np.atleast_1d(np.asarray(sky_stats.sum_aper_area.value, dtype=float)), 1.0)
    n_ap = float(aperture.area)

    net = aperture_sum - sky * n_ap
    variance = n_ap * sky_std**2 * (1.0 + n_ap / n_sky)
    if gain is not None and gain > 0:
        variance = variance + np.maximum(net, 0.0) / gain
    err = np.sqrt(variance)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(err > 0, net / err, 0.0)
    mag, mag_err = instrumental_magnitude(net, err, exptime)

    return Table(
        {
            "x": pos[:, 0],
            "y": pos[:, 1],
            "aperture_sum": aperture_sum,
            "sky_per_pixel": sky,
            "sky_std": sky_std,
            "net_counts": net,
            "net_counts_err": err,
            "snr": snr,
            "instrumental_mag": mag,
            "instrumental_mag_err": mag_err,
        }
    )


def aperture_photometry_at(
    data: np.ndarray,
    x: float,
    y: float,
    *,
    r_aper: float = 4.0,
    r_in: float | None = None,
    r_out: float | None = None,
    gain: float | None = None,
    exptime: float = 1.0,
    sky_mask: np.ndarray | None = None,
) -> PhotometryResult:
    """Measure aperture photometry at a single pixel position.

    See :func:`forced_photometry` for the method and parameters.
    """
    row = forced_photometry(
        data,
        np.array([[x, y]]),
        r_aper=r_aper,
        r_in=r_in,
        r_out=r_out,
        gain=gain,
        exptime=exptime,
        sky_mask=sky_mask,
    )[0]
    return PhotometryResult(**{name: float(row[name]) for name in row.colnames})


def aperture_photometry_table(
    data: np.ndarray,
    sources: Table,
    *,
    r_aper: float = 4.0,
    r_in: float | None = None,
    r_out: float | None = None,
    gain: float | None = None,
    exptime: float = 1.0,
    sky_mask: np.ndarray | None = None,
) -> Table:
    """Run aperture photometry for every row of a detection table."""
    positions = np.column_stack([np.asarray(sources["x_centroid"]), np.asarray(sources["y_centroid"])])
    return forced_photometry(
        data,
        positions,
        r_aper=r_aper,
        r_in=r_in,
        r_out=r_out,
        gain=gain,
        exptime=exptime,
        sky_mask=sky_mask,
    )


def blank_aperture_noise(
    image: np.ndarray,
    r_aper: float,
    *,
    mask: np.ndarray | None = None,
    n_apertures: int = 300,
    r_in: float | None = None,
    r_out: float | None = None,
    seed: int = 0,
) -> float:
    """Empirical 1-sigma noise of an aperture sum, from apertures on empty sky.

    Resampling and PSF-matching convolution correlate the noise of
    neighbouring pixels, so per-pixel variances underestimate the noise of an
    aperture sum on a difference image. Placing many apertures (with the same
    local-sky subtraction) where there are no sources and taking the robust
    scatter of their sums measures the noise that actually applies.

    Parameters
    ----------
    image
        Image to sample, typically a difference image.
    r_aper, r_in, r_out
        Aperture and sky-annulus radii, as used for the measurement itself.
    mask
        Boolean image, True where sources (or bad pixels) are; apertures are
        placed only where no masked pixel lies within ``r_out``.
    n_apertures
        Number of blank apertures to use.
    seed
        Seed for the aperture positions.

    Returns
    -------
    float
        Robust standard deviation (1.4826 x MAD) of the blank-aperture sums,
        or NaN if fewer than 20 clear positions exist.
    """
    from scipy import ndimage

    data = np.asarray(image, dtype=float)
    r_in = float(r_in if r_in is not None else 3.0 * r_aper)
    r_out = float(r_out if r_out is not None else 5.0 * r_aper)
    bad = ~np.isfinite(data)
    if mask is not None:
        bad |= np.asarray(mask, dtype=bool)
    clearance = ndimage.distance_transform_edt(~bad)
    ny, nx = data.shape
    margin = int(np.ceil(r_out)) + 1
    if nx <= 2 * margin or ny <= 2 * margin:
        return float("nan")
    ys, xs = np.nonzero(clearance[margin : ny - margin, margin : nx - margin] > r_aper + 1.0)
    if xs.size < 20:
        return float("nan")
    rng = np.random.default_rng(seed)
    pick = rng.choice(xs.size, size=min(n_apertures, xs.size), replace=False)
    positions = np.column_stack([xs[pick] + margin, ys[pick] + margin]).astype(float)
    sky_mask = None if mask is None else np.asarray(mask, dtype=bool)
    table = forced_photometry(data, positions, r_aper=r_aper, r_in=r_in, r_out=r_out, sky_mask=sky_mask)
    sums = np.asarray(table["net_counts"], dtype=float)
    sums = sums[np.isfinite(sums)]
    if sums.size < 20:
        return float("nan")
    return float(1.4826 * np.median(np.abs(sums - np.median(sums))))
