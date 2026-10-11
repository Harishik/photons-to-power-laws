"""End-to-end reduction: from calibrated frames to a calibrated light curve.

:func:`reduce_frame` turns one image into one light-curve point:

1. estimate the sky background and detect sources;
2. measure the seeing (FWHM) and size the aperture from it;
3. cross-match detections with a reference catalog through the WCS;
4. measure the matched, unsaturated stars (aperture or PSF photometry) and
   solve the photometric zero point;
5. optionally subtract a deep template image to remove the host galaxy and
   other static light;
6. force photometry at the burst position (in the science image or the
   difference image) and convert it to flux density.

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
from scipy import ndimage
from scipy.spatial import cKDTree

from .calibration import calibrate_frame, load_fits
from .detection import detect_sources, estimate_background, estimate_fwhm, make_source_mask
from .lightcurve import AB_MAG_1UJY, LightCurve
from .photometry import (
    PhotometryResult,
    aperture_photometry_at,
    blank_aperture_noise,
    forced_photometry,
    instrumental_magnitude,
)
from .psf import ConvolvedPSF, EmpiricalPSF, SpatialPSF, forced_psf_flux, psf_photometry, psf_photometry_tiled
from .subtraction import DifferenceImage, subtract_images
from .zeropoint import ZeroPoint, compute_zeropoint, match_sky


@dataclass
class PipelineConfig:
    """Tunable parameters of the reduction.

    Aperture and annulus sizes scale with the measured seeing of each frame, so
    the same fraction of a point source's light is captured in good and bad
    seeing. Because field stars and the burst are measured identically, the
    zero point absorbs the aperture correction.

    ``photometry`` selects how stars and the burst are measured: ``"aperture"``
    (circular apertures with a local sky annulus) or ``"psf"`` (an empirical
    PSF built from each frame and fitted to all sources simultaneously, see
    :mod:`grb_afterglow.psf`), which is more precise for faint sources and
    deblends neighbours.

    ``color_terms`` maps a frame band to the two catalog columns of a colour
    (e.g. ``{"zr": ("g", "r")}``) when the instrument's bandpass differs from
    the catalog's: the zero point is then fitted with a linear colour term and
    evaluated at ``target_color``, the burst's colour in that catalog system
    (see :func:`grb_afterglow.extinction.power_law_color`), or, if that is not
    given, at the median colour of the calibration stars. A third and fourth
    element fix the slope and its uncertainty (``("g", "r", 0.14, 0.01)``),
    e.g. when it has been measured once from many frames.

    ``tiles = (nx, ny)`` is for full CCD frames, whose PSF changes across the
    field. With ``photometry="psf"`` the PSF varies smoothly over the frame
    (:func:`~grb_afterglow.psf.build_spatial_psf`) and the scene is fitted in
    ``nx`` by ``ny`` tiles; image subtraction fits a kernel in each tile and
    interpolates it; the burst is measured with the PSF and kernel at its
    position. With aperture photometry only the kernel is tiled. Cutouts a few
    arcminutes across do not need it.
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
    photometry: str = "aperture"
    color_terms: Mapping[str, tuple] = field(default_factory=dict)
    target_color: float | Mapping[str, float] | None = None
    tiles: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.photometry not in ("aperture", "psf"):
            raise ValueError(f"photometry must be 'aperture' or 'psf', not {self.photometry!r}.")
        if self.tiles is not None and (len(self.tiles) != 2 or min(self.tiles) < 1):
            raise ValueError("tiles must be two positive integers (nx, ny).")


@dataclass
class Template:
    """A deep reference image of the field, used for difference imaging.

    Ideally deeper and in better seeing than the science frames, and taken
    before the burst or long after the afterglow has faded.
    """

    image: np.ndarray
    wcs: WCS
    fwhm: float
    path: str | None = None

    @classmethod
    def from_fits(cls, path: str | PathLike[str], *, fwhm_guess: float = 3.5) -> Template:
        """Load a calibrated template and measure its seeing."""
        image, header = load_fits(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            wcs = WCS(header)
        bkg = estimate_background(image)
        sources = detect_sources(image, fwhm=fwhm_guess, background=bkg)
        fwhm = estimate_fwhm(image, sources, guess=fwhm_guess, background=bkg)
        if abs(fwhm - fwhm_guess) > 0.25 * fwhm_guess:
            sources = detect_sources(image, fwhm=fwhm, background=bkg)
            fwhm = estimate_fwhm(image, sources, guess=fwhm, background=bkg)
        return cls(image, wcs, float(fwhm), str(path))


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
    difference: DifferenceImage | None = field(default=None, repr=False)
    photometry: str = "aperture"
    psf: EmpiricalPSF | SpatialPSF | None = field(default=None, repr=False)

    @property
    def subtracted(self) -> bool:
        """Whether the burst was measured on a template-subtracted image."""
        return self.difference is not None

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
    template: Template | None = None,
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
    template
        Optional deep reference image. If given, it is subtracted and the burst
        is measured on the difference image, which removes host-galaxy and
        other static light. The zero point is always solved on the science
        image itself.
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
    star_xy = np.column_stack([det_x[idx_m], det_y[idx_m]])
    det_xy = np.column_stack([det_x, det_y])
    cat_mag = np.asarray(catalog[col], dtype=float)[idx_r]
    cat_err = np.asarray(catalog[f"{col}_err"], dtype=float)[idx_r] if f"{col}_err" in catalog.colnames else None

    psf = None
    if cfg.photometry == "psf":
        # Saturated cores are excluded from the fit; their unsaturated wings still deblend neighbours.
        psf_image = _mask_saturated(image, saturation, cfg.saturation_fraction)
        with_burst = template is None
        positions = np.vstack([[tx, ty], star_xy]) if with_burst else star_xy
        if cfg.tiles is not None:
            phot = psf_photometry_tiled(
                psf_image, positions, source_xy=det_xy, fwhm=fwhm, gain=gain, tiles=tuple(cfg.tiles)
            )
        else:
            phot = psf_photometry(psf_image, positions, source_xy=det_xy, fwhm=fwhm, gain=gain)
        psf = phot.psf
        star_counts, star_counts_err = phot.flux[with_burst:], phot.flux_err[with_burst:]
    else:
        stars = forced_photometry(
            image, star_xy, r_aper=r_ap, r_in=r_in, r_out=r_out, gain=gain, exptime=exptime, sky_mask=sky_mask
        )
        star_counts, star_counts_err = np.asarray(stars["net_counts"]), np.asarray(stars["net_counts_err"])
    with np.errstate(divide="ignore", invalid="ignore"):
        star_snr = np.where(star_counts_err > 0, star_counts / star_counts_err, 0.0)
    star_mag, star_mag_err = instrumental_magnitude(star_counts, star_counts_err, exptime)
    color, slope, slope_err = None, None, 0.0
    if band in cfg.color_terms:
        blue, red, *fixed = cfg.color_terms[band]
        color = (np.asarray(catalog[blue], dtype=float) - np.asarray(catalog[red], dtype=float))[idx_r]
        if fixed:
            slope, slope_err = float(fixed[0]), float(fixed[1]) if len(fixed) > 1 else 0.0
    good = (star_snr >= cfg.min_zp_snr) & np.isfinite(cat_mag) & np.isfinite(star_mag)
    if color is not None:
        good &= np.isfinite(color)
    if np.count_nonzero(good) < cfg.min_zp_stars:
        raise RuntimeError(f"Only {np.count_nonzero(good)} usable zero-point stars in frame {path or ''}.")
    inst = star_mag[good]
    zp = compute_zeropoint(
        inst,
        cat_mag[good],
        instrumental_errs=star_mag_err[good],
        catalog_errs=None if cat_err is None else cat_err[good],
        color=None if color is None else color[good],
        color_term=slope,
        color_term_error=slope_err,
        sigma=cfg.zp_sigma,
    )
    if color is not None:
        target = cfg.target_color.get(band) if isinstance(cfg.target_color, Mapping) else cfg.target_color
        zp = zp.at_color(float(np.median(color[good][zp.used]) if target is None else target))

    diff = None
    if template is not None:
        nearest = cKDTree(det_xy).query(star_xy[good], k=2)[0][:, 1] if len(det_xy) > 1 else np.full(good.sum(), np.inf)
        zp_stars = _ZeroPointStars(star_xy[good], star_counts[good], star_snr[good], nearest)
        diff, burst, net, net_err = _measure_on_difference(
            image, wcs, template, tx, ty, fwhm, sky_mask, zp_stars, psf, gain, exptime, cfg
        )
    elif cfg.photometry == "psf":
        burst = _psf_result(tx, ty, phot.flux[0], phot.flux_err[0], exptime)
        net, net_err = burst.net_counts, burst.net_counts_err
    else:
        burst = aperture_photometry_at(
            image, tx, ty, r_aper=r_ap, r_in=r_in, r_out=r_out, gain=gain, exptime=exptime, sky_mask=sky_mask
        )
        net, net_err = burst.net_counts, burst.net_counts_err
    scale = 10 ** (-0.4 * (zp.value - AB_MAG_1UJY)) / exptime
    flux = net * scale
    flux_err = float(np.hypot(net_err * scale, abs(flux) * 0.4 * np.log(10) * zp.error))

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
        sources_xy=det_xy,
        matched_xy=star_xy[good],
        path=path,
        difference=diff,
        photometry=cfg.photometry,
        psf=psf,
    )


@dataclass
class _ZeroPointStars:
    """The stars that set a frame's zero point: positions, measured counts, S/N, nearest-neighbour distance."""

    xy: np.ndarray
    counts: np.ndarray
    snr: np.ndarray
    isolation: np.ndarray


def _mask_saturated(image: np.ndarray, saturation: float | None, fraction: float) -> np.ndarray:
    if saturation is None:
        return image
    saturated = ndimage.binary_dilation(np.asarray(image) >= fraction * saturation, iterations=1)
    return np.where(saturated, np.nan, image) if saturated.any() else image


def _psf_result(x: float, y: float, flux: float, err: float, exptime: float, background: float = np.nan):
    """A :class:`PhotometryResult` for a PSF-fitted flux (``aperture_sum`` holds the fitted flux)."""
    mag, mag_err = instrumental_magnitude(flux, err, exptime)
    snr = flux / err if err > 0 else 0.0
    return PhotometryResult(
        x=float(x),
        y=float(y),
        aperture_sum=float(flux),
        sky_per_pixel=float(background),
        sky_std=np.nan,
        net_counts=float(flux),
        net_counts_err=float(err),
        snr=float(snr),
        instrumental_mag=float(mag),
        instrumental_mag_err=float(mag_err),
    )


def _transfer_ratio(
    counts_diff_psf: np.ndarray, counts_science: np.ndarray, snr: np.ndarray, isolation: np.ndarray, radius: float
) -> float:
    """Median ratio of the same stars measured on the difference-image PSF and on the science image.

    Stars with another detection within ``2 radius`` are skipped when enough
    isolated ones remain: a neighbour inside the larger aperture of the
    broader PSF would inflate the ratio.
    """
    ok = np.isfinite(counts_diff_psf) & (counts_diff_psf > 0) & (counts_science > 0)
    for cut in (ok & (snr >= 30.0) & (isolation > 2 * radius), ok & (isolation > 2 * radius), ok & (snr >= 30.0), ok):
        if np.count_nonzero(cut) >= 3:
            return float(np.median(counts_diff_psf[cut] / counts_science[cut]))
    return 1.0


def _measure_on_difference(image, wcs, template, tx, ty, fwhm, sky_mask, stars, psf, gain, exptime, cfg):
    """Subtract the template and measure the burst on the difference image.

    Returns the difference image, the photometry, and the net counts and their
    1-sigma error on the scale of the zero-point stars' measurements in the
    science image. When the difference image carries the template's PSF (the
    science image was convolved), PSF photometry uses the science PSF seen
    through the matching kernel; aperture photometry transfers the scale with
    the zero-point stars themselves, measured the same way on the science
    image convolved with the kernel. Curves of growth of the template are
    avoided: in a deep, crowded template the light of neighbouring stars
    inflates their large-aperture "total" fluxes.
    """
    diff = subtract_images(
        image, wcs, template.image, template.wcs, science_fwhm=fwhm, template_fwhm=template.fwhm, tiles=cfg.tiles
    )
    mask = sky_mask | ~diff.valid
    ratio = 1.0
    if cfg.photometry == "psf":
        # The science PSF, through the matching kernel if the science image was the one convolved: exact,
        # and normalised to total flux like the zero-point stars' PSF fluxes, so no transfer is needed.
        local = psf.at(tx, ty) if isinstance(psf, SpatialPSF) else psf
        psf_d = local if diff.psf_from == "science" else ConvolvedPSF(local, diff.kernel_at(tx, ty), diff.fwhm)
        fit = forced_psf_flux(diff.data, tx, ty, psf_d, mask=mask, gain=gain)
        burst = _psf_result(tx, ty, fit.flux, fit.flux_err, exptime, fit.background)
        net, net_err = fit.flux, fit.flux_err
    else:
        r_d = cfg.aperture_radius_fwhm * diff.fwhm
        r_in_d, r_out_d = (k * diff.fwhm for k in cfg.annulus_fwhm)
        burst = aperture_photometry_at(diff.data, tx, ty, r_aper=r_d, r_in=r_in_d, r_out=r_out_d, sky_mask=mask)
        # Convolution correlates the noise of the difference image: measure the aperture noise on empty sky,
        # and add the burst's own shot noise, which empty apertures do not contain.
        noise = blank_aperture_noise(diff.data, r_d, mask=mask, r_in=r_in_d, r_out=r_out_d)
        if not np.isfinite(noise):
            noise = burst.net_counts_err
        shot = np.sqrt(max(burst.net_counts, 0.0) / gain) if gain else 0.0
        net, net_err = burst.net_counts, float(np.hypot(noise, shot))
        if diff.psf_from != "science":
            conv = diff.as_difference_psf(image)
            on_conv = forced_photometry(conv, stars.xy, r_aper=r_d, r_in=r_in_d, r_out=r_out_d, sky_mask=sky_mask)
            ratio = _transfer_ratio(np.asarray(on_conv["net_counts"]), stars.counts, stars.snr, stars.isolation, r_d)
    if not np.isfinite(net_err):
        net_err = np.inf
    return diff, burst, float(net / ratio), float(net_err / ratio)


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
    templates: Mapping[str, Template | str | PathLike[str]] | None = None,
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
    templates
        Optional deep reference image per band (a :class:`Template` or a path
        to a calibrated FITS file). Frames in a band with a template are
        measured on the template-subtracted image.
    on_frame
        Optional callback invoked with each :class:`FrameResult` (e.g. for logging).
    """
    cfg = config or PipelineConfig()
    loaded: dict[str, Template] = {}
    for band, tpl in (templates or {}).items():
        loaded[band] = tpl if isinstance(tpl, Template) else Template.from_fits(tpl, fwhm_guess=cfg.fwhm_guess)
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
                config=cfg,
                template=loaded.get(band),
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
