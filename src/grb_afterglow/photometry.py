"""Aperture photometry with a local-sky uncertainty budget."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.table import Table
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry


@dataclass
class PhotometryResult:
    x: float
    y: float
    aperture_sum: float
    sky_per_pixel: float
    net_counts: float
    net_counts_err: float
    snr: float
    instrumental_mag: float
    instrumental_mag_err: float


def _instrumental_mag(net_counts: float, net_counts_err: float, exptime: float) -> tuple[float, float]:
    if net_counts <= 0 or exptime <= 0:
        return float("nan"), float("nan")
    rate = net_counts / exptime
    mag = -2.5 * np.log10(rate)
    mag_err = 1.0857362047581296 * net_counts_err / max(net_counts, 1e-30)
    return float(mag), float(mag_err)


def aperture_photometry_at(
    data: np.ndarray,
    x: float,
    y: float,
    *,
    r_aper: float = 4.0,
    r_in: float | None = None,
    r_out: float | None = None,
    gain: float = 1.0,
    read_noise: float = 5.0,
    exptime: float = 1.0,
) -> PhotometryResult:
    """Measure circular-aperture photometry at one pixel position."""
    image = np.asarray(data, dtype=float)
    r_in = float(r_in if r_in is not None else 2.5 * r_aper)
    r_out = float(r_out if r_out is not None else 4.0 * r_aper)
    pos = [(float(x), float(y))]
    aper = CircularAperture(pos, r=r_aper)
    ann = CircularAnnulus(pos, r_in=r_in, r_out=r_out)
    phot = aperture_photometry(image, aper)
    ann_mask = ann.to_mask(method="center")[0]
    ann_data = ann_mask.multiply(image)
    ann_vals = ann_data[ann_mask.data > 0]
    ann_vals = ann_vals[np.isfinite(ann_vals)]
    sky = float(np.median(ann_vals)) if ann_vals.size else 0.0
    area = float(aper.area)
    aperture_sum = float(phot["aperture_sum"][0])
    net = aperture_sum - sky * area
    n_pix = max(area, 1.0)
    source_e = max(net * gain, 0.0)
    sky_e = max(sky * gain, 0.0) * n_pix
    rn_e = (read_noise ** 2) * n_pix
    err_counts = float(np.sqrt(source_e + sky_e + rn_e) / max(gain, 1e-12))
    snr = float(net / err_counts) if err_counts > 0 else 0.0
    mag, mag_err = _instrumental_mag(net, err_counts, exptime)
    return PhotometryResult(float(x), float(y), aperture_sum, sky, float(net), err_counts, snr, mag, mag_err)


def aperture_photometry_table(
    data: np.ndarray,
    sources: Table,
    *,
    r_aper: float = 4.0,
    gain: float = 1.0,
    read_noise: float = 5.0,
    exptime: float = 1.0,
) -> Table:
    """Run aperture photometry for all rows in a source table."""
    rows = []
    for row in sources:
        x = float(row["x_centroid"])
        y = float(row["y_centroid"])
        res = aperture_photometry_at(
            data,
            x,
            y,
            r_aper=r_aper,
            gain=gain,
            read_noise=read_noise,
            exptime=exptime,
        )
        rows.append((
            res.x,
            res.y,
            res.net_counts,
            res.net_counts_err,
            res.snr,
            res.instrumental_mag,
            res.instrumental_mag_err,
        ))
    return Table(
        rows=rows,
        names=("x", "y", "net_counts", "net_counts_err", "snr", "instrumental_mag", "instrumental_mag_err"),
    )
