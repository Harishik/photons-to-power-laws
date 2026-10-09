"""End-to-end reduction: from calibrated frames to a calibrated light curve.

:func:`reduce_frame` turns one image into one light-curve point:

1. estimate the sky background and detect sources;
2. measure the seeing (FWHM) and size the aperture from it;
3. cross-match detections with a reference catalog through the WCS;
4. solve the photometric zero point from the matched, unsaturated stars;
5. force photometry at the burst position and convert it to flux density.

:func:`run_pipeline` applies it to a list of FITS files. Synthetic and real
data go through exactly the same code path.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS, FITSFixedWarning

from .calibration import calibrate_frame, load_fits
from .detection import detect_sources, estimate_background, estimate_fwhm, make_source_mask
from .lightcurve import AB_MAG_1UJY, LightCurve
from .photometry import PhotometryResult, aperture_photometry_at, forced_photometry
from .zeropoint import ZeroPoint, compute_zeropoint, match_sky


@dataclass
class PipelineConfig:
    """Tunable parameters of the reduction.

    Aperture and annulus sizes scale with the measured seeing of each frame, so
    the same fraction of a point source's light is captured in good and bad
    seeing. Because field stars and the burst are measured identically, the
    zero point absorbs the aperture correction.
    """

    fwhm_guess: float = 3.5
    threshold_sigma: float = 5.0
    aperture_radius_fwhm: float = 1.0
    annulus_fwhm: tuple[float, float] = (3.0, 5.0)
    match_radius_arcsec: float = 1.5
    min_zp_snr: float = 20.0
    min_zp_stars: int = 5
    zp_sigma: float = 3.0
    saturation_fraction: float = 0.8
    mask_dilate_fwhm: float = 1.5
    catalog_bands: Mapping[str, str] = field(default_factory=dict)


@dataclass
class FrameResult:
    """Everything measured from one science frame."""

    band: str
    time: float
    exptime: float
    fwhm: float
    n_sources: int
    zeropoint: ZeroPoint
    target: PhotometryResult
    flux_ujy: float
    flux_err_ujy: float
    astrometric_offset_px: tuple[float, float]
    r_aper: float
    annulus: tuple[float, float]
    star_instrumental_mag: np.ndarray = field(repr=False)
    star_catalog_mag: np.ndarray = field(repr=False)
    sources_xy: np.ndarray = field(repr=False)
    matched_xy: np.ndarray = field(repr=False)
    path: str | None = None

    @property
    def snr(self) -> float:
        return self.flux_ujy / self.flux_err_ujy if self.flux_err_ujy > 0 else 0.0


@dataclass
class PipelineResult:
    """A light curve together with the per-frame measurements behind it."""

    lightcurve: LightCurve
    frames: list[FrameResult]

    def summary_table(self) -> Table:
        """One row per frame: seeing, zero point, and the burst measurement."""
        rows = [
            (
                Path(f.path).name if f.path else "",
                f.band,
                f.time,
                f.exptime,
                f.fwhm,
                f.n_sources,
                f.zeropoint.n_stars,
                f.zeropoint.value,
                f.zeropoint.error,
                f.zeropoint.scatter,
                f.flux_ujy,
                f.flux_err_ujy,
                f.snr,
            )
            for f in self.frames
        ]
        names = (
            "file",
            "band",
            "time_s",
            "exptime",
            "fwhm_px",
            "n_sources",
            "n_zp_stars",
            "zeropoint",
            "zeropoint_err",
            "zeropoint_scatter",
            "flux_ujy",
            "flux_err_ujy",
            "snr",
        )
        table = Table(rows=rows, names=names) if rows else Table(names=names)
        for col in (
            "time_s",
            "fwhm_px",
            "zeropoint",
            "zeropoint_err",
            "zeropoint_scatter",
            "flux_ujy",
            "flux_err_ujy",
            "snr",
        ):
            table[col].info.format = ".5g"
        return table


def _catalog_column(band: str, catalog: Table, mapping: Mapping[str, str]) -> str:
    if band in mapping:
        return mapping[band]
    if band in catalog.colnames:
        return band
    short = band[-1:]
    if short in catalog.colnames:
        return short
    raise KeyError(f"Reference catalog has no column for band {band!r}.")


def reduce_frame(
    image: np.ndarray,
    wcs: WCS,
    target: SkyCoord,
    catalog: Table,
    *,
    band: str,
    time: float,
    exptime: float,
    gain: float | None = None,
    saturation: float | None = None,
    config: PipelineConfig | None = None,
    path: str | None = None,
) -> FrameResult:
    """Measure the burst flux density in one calibrated frame.

    Parameters
    ----------
    image
        Bias- and flat-corrected image.
    wcs
        World coordinate system of ``image``.
    target
        Sky position of the burst.
    catalog
        Reference stars with ``ra``, ``dec`` (deg), and per-band magnitude
        columns (``r``, ``r_err``, ...) on the AB system.
    band
        Filter of the frame; mapped to a catalog column via
        ``config.catalog_bands`` or by name.
    time
        Mid-exposure time since the trigger, in seconds.
    exptime
        Exposure time in seconds.
    gain
        Detector gain (e-/ADU), used for the source shot-noise term.
    saturation
        Saturation level in calibrated units; brighter stars are not used for
        the zero point.
    """
    cfg = config or PipelineConfig()
    bkg = estimate_background(image)
    peak_max = None
    if saturation is not None:
        peak_max = cfg.saturation_fraction * saturation - float(np.median(bkg.background))

    sources = detect_sources(
        image, fwhm=cfg.fwhm_guess, threshold_sigma=cfg.threshold_sigma, background=bkg, peak_max=peak_max
    )
    fwhm = estimate_fwhm(image, sources, guess=cfg.fwhm_guess, background=bkg)
    if abs(fwhm - cfg.fwhm_guess) > 0.25 * cfg.fwhm_guess:
        sources = detect_sources(
            image, fwhm=fwhm, threshold_sigma=cfg.threshold_sigma, background=bkg, peak_max=peak_max
        )
        fwhm = estimate_fwhm(image, sources, guess=fwhm, background=bkg)

    r_ap = cfg.aperture_radius_fwhm * fwhm
    r_in, r_out = (k * fwhm for k in cfg.annulus_fwhm)
    sky_mask = make_source_mask(image, bkg, dilate=max(2, int(np.rint(cfg.mask_dilate_fwhm * fwhm))))

    # Cross-match detections with the catalog on the sky.
    col = _catalog_column(band, catalog, cfg.catalog_bands)
    det_x = np.asarray(sources["x_centroid"], dtype=float)
    det_y = np.asarray(sources["y_centroid"], dtype=float)
    det_ra, det_dec = wcs.pixel_to_world_values(det_x, det_y)
    ref = SkyCoord(np.asarray(catalog["ra"]) * u.deg, np.asarray(catalog["dec"]) * u.deg)
    meas = SkyCoord(np.atleast_1d(det_ra) * u.deg, np.atleast_1d(det_dec) * u.deg)
    idx_m, idx_r, _ = match_sky(meas, ref, max_sep_arcsec=cfg.match_radius_arcsec)

    tx, ty = (float(v) for v in wcs.world_to_pixel(target))
    offset = (0.0, 0.0)
    if idx_m.size >= 3:
        cat_x, cat_y = wcs.world_to_pixel(ref[idx_r])
        offset = (
            float(np.median(det_x[idx_m] - cat_x)),
            float(np.median(det_y[idx_m] - cat_y)),
        )
        tx, ty = tx + offset[0], ty + offset[1]

    # Keep matched stars away from the burst itself.
    far = np.hypot(det_x[idx_m] - tx, det_y[idx_m] - ty) > r_out
    idx_m, idx_r = idx_m[far], idx_r[far]

    if idx_m.size == 0:
        raise RuntimeError(f"No catalog stars matched in frame {path or ''}.")
    stars = forced_photometry(
        image,
        np.column_stack([det_x[idx_m], det_y[idx_m]]),
        r_aper=r_ap,
        r_in=r_in,
        r_out=r_out,
        gain=gain,
        exptime=exptime,
        sky_mask=sky_mask,
    )
    cat_mag = np.asarray(catalog[col], dtype=float)[idx_r]
    cat_err = np.asarray(catalog[f"{col}_err"], dtype=float)[idx_r] if f"{col}_err" in catalog.colnames else None
    good = (np.asarray(stars["snr"]) >= cfg.min_zp_snr) & np.isfinite(cat_mag)
    if np.count_nonzero(good) < cfg.min_zp_stars:
        raise RuntimeError(f"Only {np.count_nonzero(good)} usable zero-point stars in frame {path or ''}.")
    inst = np.asarray(stars["instrumental_mag"])[good]
    zp = compute_zeropoint(
        inst,
        cat_mag[good],
        instrumental_errs=np.asarray(stars["instrumental_mag_err"])[good],
        catalog_errs=None if cat_err is None else cat_err[good],
        sigma=cfg.zp_sigma,
    )

    burst = aperture_photometry_at(
        image, tx, ty, r_aper=r_ap, r_in=r_in, r_out=r_out, gain=gain, exptime=exptime, sky_mask=sky_mask
    )
    scale = 10 ** (-0.4 * (zp.value - AB_MAG_1UJY)) / exptime
    flux = burst.net_counts * scale
    flux_err = float(np.hypot(burst.net_counts_err * scale, abs(flux) * 0.4 * np.log(10) * zp.error))

    return FrameResult(
        band=band,
        time=float(time),
        exptime=float(exptime),
        fwhm=float(fwhm),
        n_sources=len(sources),
        zeropoint=zp,
        target=burst,
        flux_ujy=float(flux),
        flux_err_ujy=flux_err,
        astrometric_offset_px=offset,
        r_aper=float(r_ap),
        annulus=(float(r_in), float(r_out)),
        star_instrumental_mag=inst,
        star_catalog_mag=cat_mag[good],
        sources_xy=np.column_stack([det_x, det_y]),
        matched_xy=np.column_stack([det_x[idx_m][good], det_y[idx_m][good]]),
        path=path,
    )


def header_time_since_trigger(header: fits.Header, trigger_mjd: float | None = None) -> float:
    """Mid-exposure time since the trigger in seconds.

    Uses a ``TSINCE`` keyword if present, otherwise ``MJD-OBS``/``OBSMJD``
    (start of exposure) plus half the exposure time, relative to ``trigger_mjd``.
    """
    if "TSINCE" in header:
        return float(header["TSINCE"])
    if trigger_mjd is None:
        raise ValueError("Header has no TSINCE; pass trigger_mjd to compute time since trigger.")
    for key in ("OBSMJD", "MJD-OBS"):
        if key in header:
            start = float(header[key])
            return (start - trigger_mjd) * 86400.0 + 0.5 * float(header.get("EXPTIME", 0.0))
    raise KeyError("Header has no TSINCE, OBSMJD, or MJD-OBS keyword.")


def run_pipeline(
    science_files: Sequence[str | PathLike[str]],
    target: SkyCoord,
    catalog: Table,
    *,
    master_bias: np.ndarray | None = None,
    master_flats: Mapping[str, np.ndarray] | None = None,
    config: PipelineConfig | None = None,
    trigger_mjd: float | None = None,
    band_key: str = "FILTER",
    on_frame: Callable[[FrameResult], None] | None = None,
) -> PipelineResult:
    """Reduce a set of science frames into a calibrated light curve.

    Frames whose zero point cannot be solved are skipped with a warning rather
    than aborting the whole run.

    Parameters
    ----------
    science_files
        FITS files with a WCS and a band keyword. ``EXPTIME`` defaults to 1 s if
        absent; it cancels out of the calibrated flux.
    target
        Sky position of the burst.
    catalog
        Reference catalog (see :func:`reduce_frame`).
    master_bias, master_flats
        Optional calibration frames; flats are looked up by band.
    trigger_mjd
        Trigger time, needed when headers carry no ``TSINCE``.
    on_frame
        Optional callback invoked with each :class:`FrameResult` (e.g. for logging).
    """
    frames: list[FrameResult] = []
    for path in science_files:
        raw, header = load_fits(path)
        band = str(header.get(band_key, "unknown")).strip()
        flat = None if master_flats is None else master_flats.get(band)
        image = calibrate_frame(raw, master_bias=master_bias, master_flat=flat)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            wcs = WCS(header)
        saturation = header.get("SATURATE")
        if saturation is not None and master_bias is not None:
            saturation = float(saturation) - float(np.median(master_bias))
        try:
            result = reduce_frame(
                image,
                wcs,
                target,
                catalog,
                band=band,
                time=header_time_since_trigger(header, trigger_mjd),
                exptime=float(header.get("EXPTIME", 1.0)),
                gain=header.get("GAIN"),
                saturation=saturation,
                config=config,
                path=str(path),
            )
        except RuntimeError as exc:
            warnings.warn(f"Skipping {path}: {exc}", stacklevel=2)
            continue
        frames.append(result)
        if on_frame is not None:
            on_frame(result)

    lc = LightCurve(
        [f.time for f in frames],
        [f.flux_ujy for f in frames],
        [f.flux_err_ujy for f in frames],
        np.array([f.band for f in frames], dtype=object) if frames else "unknown",
    )
    return PipelineResult(lc, frames)
