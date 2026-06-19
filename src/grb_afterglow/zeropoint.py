"""Photometric zero-point matching and calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.table import Table


@dataclass
class ZeroPoint:
    value: float
    error: float
    n_stars: int
    scatter: float


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
    """Nearest-neighbour match two catalogs in pixel space."""
    mx = np.asarray(measured[x_meas], dtype=float)
    my = np.asarray(measured[y_meas], dtype=float)
    rx = np.asarray(reference[x_ref], dtype=float)
    ry = np.asarray(reference[y_ref], dtype=float)
    idx_m, idx_r, seps = [], [], []
    for j in range(len(rx)):
        d = np.hypot(mx - rx[j], my - ry[j])
        if d.size == 0:
            continue
        i = int(np.argmin(d))
        if d[i] <= max_sep:
            idx_m.append(i)
            idx_r.append(j)
            seps.append(float(d[i]))
    return np.asarray(idx_m, dtype=int), np.asarray(idx_r, dtype=int), np.asarray(seps)


def match_sky(
    measured: Table,
    reference: Table,
    *,
    max_sep_arcsec: float = 2.0,
    ra_meas: str = "ra",
    dec_meas: str = "dec",
    ra_ref: str = "ra",
    dec_ref: str = "dec",
):
    """Match two catalogs on the sky with astropy SkyCoord."""
    from astropy import units as u

    m = SkyCoord(ra=measured[ra_meas] * u.deg, dec=measured[dec_meas] * u.deg)
    r = SkyCoord(ra=reference[ra_ref] * u.deg, dec=reference[dec_ref] * u.deg)
    idx, sep2d, _ = r.match_to_catalog_sky(m)
    keep = sep2d.arcsec <= max_sep_arcsec
    return idx[keep], np.nonzero(keep)[0], sep2d.arcsec[keep]


def compute_zeropoint(
    instrumental_mags: np.ndarray,
    catalog_mags: np.ndarray,
    *,
    instrumental_errs: np.ndarray | None = None,
    sigma: float = 3.0,
) -> ZeroPoint:
    """Compute a robust photometric zero point from matched stars."""
    inst = np.asarray(instrumental_mags, dtype=float)
    cat = np.asarray(catalog_mags, dtype=float)
    good = np.isfinite(inst) & np.isfinite(cat)
    if instrumental_errs is not None:
        good &= np.isfinite(np.asarray(instrumental_errs, dtype=float))
    inst, cat = inst[good], cat[good]
    if inst.size == 0:
        raise ValueError("No valid star pairs to compute a zero point.")
    residuals = cat - inst
    _, zp_median, zp_std = sigma_clipped_stats(residuals, sigma=sigma)
    if not np.isfinite(zp_std) or zp_std == 0:
        clipped = np.ones_like(residuals, dtype=bool)
    else:
        clipped = np.abs(residuals - zp_median) <= sigma * zp_std
    n = int(np.count_nonzero(clipped)) or int(inst.size)
    zp_err = float(zp_std / np.sqrt(n)) if n > 0 else float(zp_std)
    return ZeroPoint(float(zp_median), zp_err, n, float(zp_std))


def calibrate_magnitudes(
    instrumental_mags: np.ndarray,
    zeropoint: ZeroPoint,
    instrumental_errs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a zero point: m_cal = m_inst + ZP."""
    inst = np.asarray(instrumental_mags, dtype=float)
    cal = inst + zeropoint.value
    if instrumental_errs is None:
        cal_err = np.full_like(cal, zeropoint.error)
    else:
        ierr = np.asarray(instrumental_errs, dtype=float)
        cal_err = np.sqrt(ierr ** 2 + zeropoint.error ** 2)
    return cal, cal_err
