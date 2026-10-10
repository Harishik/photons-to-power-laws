"""Synthetic GRB afterglow observations with known ground truth.

The generator writes the files an observer would actually have after a
follow-up campaign: bias and twilight-flat frames, multi-band science frames
with a world coordinate system (WCS), and a reference star catalog in sky
coordinates. The scene is designed so that each pipeline stage has real work
to do:

* a bias with column structure and a vignetted, non-uniform flat;
* dithered pointings, so the burst lands on different pixels every epoch;
* variable seeing (Moffat PSFs) and variable atmospheric transparency, so the
  zero point must be solved per frame;
* bright stars that saturate in long exposures;
* exposure times that grow as the afterglow fades, as in real follow-up;
* a host galaxy under the afterglow, which dominates the light once the
  afterglow has faded, and a deep reference image per band (taken long after
  the burst) for image subtraction.

The injected afterglow obeys a closure relation of the standard fireball
model, so the full chain from pixels to physics can be validated.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from os import PathLike
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.signal import fftconvolve

from .fitting import band_frequency_hz, broken_power_law
from .lightcurve import SECONDS_PER_DAY, flux_to_mag, mag_to_flux


@dataclass
class AfterglowModel:
    """A jet-break afterglow in the slow-cooling, constant-density (ISM) regime.

    The defaults correspond to an electron index ``p = 2.2`` observed between
    the injection and cooling frequencies: ``alpha1 = 3(p-1)/4``,
    ``beta = (p-1)/2``, and ``alpha2 = p`` after the jet break.
    """

    p: float = 2.2
    t_break: float = 0.5 * SECONDS_PER_DAY
    smooth: float = 3.0
    mag_ref: float = 14.6
    t_ref: float = 600.0
    reference_band: str = "r"

    @property
    def alpha1(self) -> float:
        return 3.0 * (self.p - 1.0) / 4.0

    @property
    def alpha2(self) -> float:
        return self.p

    @property
    def beta(self) -> float:
        return (self.p - 1.0) / 2.0

    def flux_ujy(self, t, band: str) -> np.ndarray:
        """Flux density in microjansky at time ``t`` (s) in ``band``."""
        temporal = broken_power_law(
            t, 1.0, self.t_break, self.alpha1, self.alpha2, smooth=self.smooth, t_ref=self.t_ref
        )
        spectral = (band_frequency_hz(band) / band_frequency_hz(self.reference_band)) ** (-self.beta)
        return mag_to_flux(self.mag_ref) * temporal * spectral


@dataclass
class HostGalaxy:
    """An exponential-disk host galaxy, offset slightly from the afterglow.

    The defaults give a galaxy whose light inside a seeing-sized aperture is
    comparable to the afterglow a day or two after the burst, so measurements
    without image subtraction are visibly biased at late times.
    """

    mag_r: float = 21.0
    r_eff_arcsec: float = 1.0
    axis_ratio: float = 0.6
    position_angle_deg: float = 35.0
    offset_arcsec: tuple[float, float] = (0.45, -0.3)
    color_gr: float = 0.7
    color_ri: float = 0.35

    def mag(self, band: str) -> float:
        return {"g": self.mag_r + self.color_gr, "r": self.mag_r, "i": self.mag_r - self.color_ri}[band[-1:]]


@dataclass
class SyntheticConfig:
    """Instrument, site, and campaign parameters for :func:`generate_dataset`."""

    shape: tuple[int, int] = (256, 256)
    pixel_scale: float = 0.6
    field_ra: float = 150.1
    field_dec: float = 22.4
    afterglow_offset_arcsec: tuple[float, float] = (11.0, -7.0)
    bands: tuple[str, ...] = ("g", "r", "i")
    zeropoint: dict[str, float] = field(default_factory=lambda: {"g": 25.2, "r": 25.0, "i": 24.7})
    extinction_coeff: dict[str, float] = field(default_factory=lambda: {"g": 0.18, "r": 0.10, "i": 0.06})
    sky_mag_arcsec2: dict[str, float] = field(default_factory=lambda: {"g": 21.6, "r": 20.9, "i": 20.2})
    n_epochs: int = 12
    t_first: float = 360.0
    t_last: float = 4.0 * SECONDS_PER_DAY
    exptime_min: float = 30.0
    exptime_max: float = 600.0
    seeing_arcsec: tuple[float, float] = (1.4, 2.6)
    cloud_scatter_mag: float = 0.05
    dither_px: float = 10.0
    moffat_beta: float = 3.0
    n_stars: int = 70
    star_mag_range: tuple[float, float] = (14.0, 21.5)
    min_star_separation_arcsec: float = 9.0
    gain: float = 1.6
    read_noise: float = 6.0
    bias_level: float = 400.0
    saturation: float = 65000.0
    flat_level: float = 22000.0
    n_bias: int = 7
    n_flat: int = 5
    afterglow: AfterglowModel = field(default_factory=AfterglowModel)
    host: HostGalaxy | None = field(default_factory=HostGalaxy)
    template_seeing_arcsec: float = 1.2
    template_exptime: float = 1800.0
    template_ncombine: int = 6


def _moffat_stamp(dx: float, dy: float, fwhm: float, beta: float, half: int, sub: int = 3) -> np.ndarray:
    """Pixel-integrated Moffat profile with unit total flux, centred at (dx, dy)."""
    alpha = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    offsets = (np.arange(sub) + 0.5) / sub - 0.5
    grid = np.arange(-half, half + 1)
    yy = (grid[:, None, None, None] + offsets[None, None, :, None]) - dy
    xx = (grid[None, :, None, None] + offsets[None, None, None, :]) - dx
    profile = (beta - 1.0) / (np.pi * alpha**2) * (1.0 + (xx**2 + yy**2) / alpha**2) ** (-beta)
    return profile.mean(axis=(2, 3))


def _add_source(scene: np.ndarray, x: float, y: float, electrons: float, fwhm: float, beta: float) -> None:
    half = int(np.ceil(6 * fwhm))
    ix, iy = int(np.round(x)), int(np.round(y))
    stamp = _moffat_stamp(x - ix, y - iy, fwhm, beta, half) * electrons
    ny, nx = scene.shape
    x0, x1 = max(ix - half, 0), min(ix + half + 1, nx)
    y0, y1 = max(iy - half, 0), min(iy + half + 1, ny)
    if x0 >= x1 or y0 >= y1:
        return
    scene[y0:y1, x0:x1] += stamp[y0 - (iy - half) : y1 - (iy - half), x0 - (ix - half) : x1 - (ix - half)]


def _add_host(
    scene: np.ndarray,
    x: float,
    y: float,
    electrons: float,
    host: HostGalaxy,
    pixel_scale: float,
    fwhm: float,
    beta: float,
) -> None:
    """Add a PSF-convolved, pixel-integrated exponential disk centred at (x, y)."""
    r_eff = host.r_eff_arcsec / pixel_scale
    half = int(np.ceil(7 * r_eff + 4 * fwhm))
    ix, iy = int(np.rint(x)), int(np.rint(y))
    sub = 3
    off = (np.arange(sub) + 0.5) / sub - 0.5
    grid = np.arange(-half, half + 1)
    yy = grid[:, None, None, None] + off[None, None, :, None] - (y - iy)
    xx = grid[None, :, None, None] + off[None, None, None, :] - (x - ix)
    pa = np.radians(host.position_angle_deg)
    xr = np.cos(pa) * xx + np.sin(pa) * yy
    yr = (-np.sin(pa) * xx + np.cos(pa) * yy) / host.axis_ratio
    disk = np.exp(-1.678 * np.hypot(xr, yr) / r_eff).mean(axis=(2, 3))
    psf = _moffat_stamp(0.0, 0.0, fwhm, beta, int(np.ceil(4 * fwhm)))
    stamp = fftconvolve(disk / disk.sum(), psf / psf.sum(), mode="same") * electrons
    ny, nx = scene.shape
    x0, x1 = max(ix - half, 0), min(ix + half + 1, nx)
    y0, y1 = max(iy - half, 0), min(iy + half + 1, ny)
    if x0 < x1 and y0 < y1:
        scene[y0:y1, x0:x1] += stamp[y0 - (iy - half) : y1 - (iy - half), x0 - (ix - half) : x1 - (ix - half)]


def _make_wcs(cfg: SyntheticConfig, dx: float = 0.0, dy: float = 0.0) -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [cfg.field_ra, cfg.field_dec]
    wcs.wcs.crpix = [cfg.shape[1] / 2 + 0.5 + dx, cfg.shape[0] / 2 + 0.5 + dy]
    scale = cfg.pixel_scale / 3600.0
    wcs.wcs.cd = [[-scale, 0.0], [0.0, scale]]
    return wcs


def _star_catalog(cfg: SyntheticConfig, rng: np.random.Generator, ag_ra: float, ag_dec: float) -> Table:
    wcs = _make_wcs(cfg)
    margin = 4
    ny, nx = cfg.shape
    lo, hi = cfg.star_mag_range
    # Star counts rise steeply towards faint magnitudes, N(<m) ∝ 10^(0.3 m).
    u = rng.uniform(10 ** (0.3 * lo), 10 ** (0.3 * hi), cfg.n_stars * 2)
    r_mag = np.log10(u) / 0.3
    x = rng.uniform(margin, nx - margin, r_mag.size)
    y = rng.uniform(margin, ny - margin, r_mag.size)
    ag_x, ag_y = wcs.world_to_pixel_values(ag_ra, ag_dec)
    far = np.hypot(x - ag_x, y - ag_y) * cfg.pixel_scale > cfg.min_star_separation_arcsec
    x, y, r_mag = x[far][: cfg.n_stars], y[far][: cfg.n_stars], r_mag[far][: cfg.n_stars]
    ra, dec = wcs.pixel_to_world_values(x, y)
    g_r = rng.uniform(0.25, 1.3, r_mag.size)
    r_i = 0.42 * g_r + rng.normal(0.0, 0.04, r_mag.size)
    true = {"g": r_mag + g_r, "r": r_mag, "i": r_mag - r_i}

    catalog = Table()
    catalog["ra"] = ra
    catalog["dec"] = dec
    for band in cfg.bands:
        mag = true[band]
        err = 0.008 + 0.03 * 10 ** (0.4 * (mag - 21.5))
        catalog[band] = mag + rng.normal(0.0, err)
        catalog[f"{band}_err"] = err
        catalog[f"{band}_true"] = mag
    return catalog


def generate_dataset(
    out_dir: str | PathLike[str],
    *,
    config: SyntheticConfig | None = None,
    seed: int = 42,
) -> dict:
    """Generate a synthetic multi-band GRB follow-up campaign on disk.

    Writes ``raw/`` (bias, flat, and science FITS frames), ``templates/``
    (one deep, calibrated reference image per band, taken after the afterglow
    has faded), ``catalog.csv`` (the reference star catalog a real survey would
    provide), and ``truth.json`` (everything the pipeline is supposed to
    recover).

    Returns
    -------
    dict
        The ground truth, as also written to ``truth.json``.
    """
    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    ny, nx = cfg.shape
    gain, rn = cfg.gain, cfg.read_noise

    cos_dec = np.cos(np.radians(cfg.field_dec))
    ag_ra = cfg.field_ra + cfg.afterglow_offset_arcsec[0] / 3600.0 / cos_dec
    ag_dec = cfg.field_dec + cfg.afterglow_offset_arcsec[1] / 3600.0
    catalog = _star_catalog(cfg, rng, ag_ra, ag_dec)
    host = cfg.host
    if host is not None:
        host_ra = ag_ra + host.offset_arcsec[0] / 3600.0 / cos_dec
        host_dec = ag_dec + host.offset_arcsec[1] / 3600.0

    def add_static_scene(scene, wcs, band, fwhm_px, electrons):
        sx, sy = wcs.world_to_pixel_values(catalog["ra"], catalog["dec"])
        for x, y, mag in zip(sx, sy, catalog[f"{band}_true"], strict=True):
            _add_source(scene, x, y, float(electrons(mag)), fwhm_px, cfg.moffat_beta)
        if host is not None:
            hx, hy = wcs.world_to_pixel_values(host_ra, host_dec)
            e_host = float(electrons(host.mag(band)))
            _add_host(scene, float(hx), float(hy), e_host, host, cfg.pixel_scale, fwhm_px, cfg.moffat_beta)

    # Detector signatures shared by every frame.
    column_pattern = rng.normal(0.0, 2.0, nx)[None, :]
    bias_frame = cfg.bias_level + np.broadcast_to(column_pattern, cfg.shape)
    yy, xx = np.mgrid[0:ny, 0:nx]
    r2 = ((xx - nx / 2) ** 2 + (yy - ny / 2) ** 2) / (0.5 * max(nx, ny)) ** 2
    pixel_response = rng.normal(1.0, 0.01, cfg.shape)
    true_flat = (1.0 - 0.06 * r2) * pixel_response

    def read_out(electrons: np.ndarray) -> np.ndarray:
        signal = rng.poisson(np.maximum(electrons, 0.0)) + rng.normal(0.0, rn, cfg.shape)
        return np.minimum(bias_frame + signal / gain, cfg.saturation).astype(np.float32)

    for i in range(cfg.n_bias):
        hdr = fits.Header({"IMAGETYP": "BIAS", "EXPTIME": 0.0})
        fits.PrimaryHDU(read_out(np.zeros(cfg.shape)), hdr).writeto(raw / f"bias_{i:02d}.fits", overwrite=True)
    for band in cfg.bands:
        for i in range(cfg.n_flat):
            level = cfg.flat_level * rng.uniform(0.8, 1.2) * gain
            hdr = fits.Header({"IMAGETYP": "FLAT", "FILTER": band})
            fits.PrimaryHDU(read_out(level * true_flat), hdr).writeto(raw / f"flat_{band}_{i:02d}.fits", overwrite=True)

    times = np.geomspace(cfg.t_first, cfg.t_last, cfg.n_epochs)
    airmass = 1.0 + 0.6 * (0.5 + 0.5 * np.sin(np.linspace(0.0, 3.0 * np.pi, cfg.n_epochs)))
    frames = []
    for j, t_epoch in enumerate(times):
        exptime = float(np.clip(cfg.exptime_min * (t_epoch / cfg.t_first) ** 0.45, cfg.exptime_min, cfg.exptime_max))
        exptime = float(np.round(exptime, -1) or cfg.exptime_min)
        t_now = t_epoch
        for band in cfg.bands:
            t_mid = t_now + exptime / 2.0
            t_now += exptime + 40.0
            dx, dy = rng.uniform(-cfg.dither_px, cfg.dither_px, 2)
            wcs = _make_wcs(cfg, dx, dy)
            seeing = rng.uniform(*cfg.seeing_arcsec)
            fwhm_px = seeing / cfg.pixel_scale
            extinction = cfg.extinction_coeff[band] * airmass[j] + abs(rng.normal(0.0, cfg.cloud_scatter_mag))
            zp = cfg.zeropoint[band] - extinction  # AB mag that yields 1 ADU/s

            def electrons(mag, zp=zp, exptime=exptime):
                return gain * exptime * 10 ** (-0.4 * (np.asarray(mag) - zp))

            sky_pix_mag = cfg.sky_mag_arcsec2[band] - 2.5 * np.log10(cfg.pixel_scale**2)
            scene = np.full(cfg.shape, float(electrons(sky_pix_mag)))
            add_static_scene(scene, wcs, band, fwhm_px, electrons)
            ag_flux = float(cfg.afterglow.flux_ujy(t_mid, band))
            ag_x, ag_y = wcs.world_to_pixel_values(ag_ra, ag_dec)
            _add_source(
                scene, float(ag_x), float(ag_y), float(electrons(flux_to_mag(ag_flux))), fwhm_px, cfg.moffat_beta
            )

            hdr = wcs.to_header()
            hdr["IMAGETYP"] = "SCIENCE"
            hdr["OBJECT"] = "SYNTHETIC-GRB"
            hdr["FILTER"] = band
            hdr["EXPTIME"] = exptime
            hdr["TSINCE"] = (t_mid, "Mid-exposure time since trigger [s]")
            hdr["GAIN"] = (gain, "e-/ADU")
            hdr["RDNOISE"] = (rn, "e-")
            hdr["SATURATE"] = (cfg.saturation, "ADU")
            name = f"sci_{band}_{j:02d}.fits"
            fits.PrimaryHDU(read_out(scene * true_flat), hdr).writeto(raw / name, overwrite=True)
            frames.append(
                {
                    "file": name,
                    "band": band,
                    "time_s": t_mid,
                    "exptime": exptime,
                    "seeing_arcsec": seeing,
                    "zeropoint": zp,
                    "airmass": float(airmass[j]),
                    "afterglow_flux_ujy": ag_flux,
                }
            )

    # Deep reference images, taken long after the burst: stars and host only, already calibrated.
    templates = {}
    tdir = out / "templates"
    tdir.mkdir(exist_ok=True)
    for band in cfg.bands:
        dx, dy = rng.uniform(-cfg.dither_px, cfg.dither_px, 2)
        wcs = _make_wcs(cfg, dx, dy)
        fwhm_px = cfg.template_seeing_arcsec / cfg.pixel_scale
        zp = cfg.zeropoint[band] - cfg.extinction_coeff[band] * 1.1
        texp = cfg.template_exptime

        def t_electrons(mag, zp=zp, texp=texp):
            return gain * texp * 10 ** (-0.4 * (np.asarray(mag) - zp))

        sky_pix_mag = cfg.sky_mag_arcsec2[band] - 2.5 * np.log10(cfg.pixel_scale**2)
        scene = np.full(cfg.shape, float(t_electrons(sky_pix_mag)))
        add_static_scene(scene, wcs, band, fwhm_px, t_electrons)
        n = cfg.template_ncombine
        counts = (rng.poisson(scene) + rng.normal(0.0, rn * np.sqrt(n), cfg.shape)) / gain
        counts = np.minimum(counts, cfg.saturation * n)
        hdr = wcs.to_header()
        hdr["IMAGETYP"] = "TEMPLATE"
        hdr["OBJECT"] = "SYNTHETIC-GRB"
        hdr["FILTER"] = band
        hdr["EXPTIME"] = (texp, "Total exposure time of the coadd [s]")
        hdr["NCOMBINE"] = n
        hdr["GAIN"] = (gain, "e-/ADU")
        hdr["SATURATE"] = (cfg.saturation * n, "ADU")
        name = f"template_{band}.fits"
        fits.PrimaryHDU(counts.astype(np.float32), hdr).writeto(tdir / name, overwrite=True)
        templates[band] = {"file": f"templates/{name}", "seeing_arcsec": cfg.template_seeing_arcsec, "zeropoint": zp}

    catalog_out = catalog.copy()
    catalog_out.remove_columns([f"{b}_true" for b in cfg.bands])
    catalog_out.write(out / "catalog.csv", format="ascii.csv", overwrite=True)

    ag = cfg.afterglow
    truth = {
        "seed": seed,
        "afterglow": {
            "ra": ag_ra,
            "dec": ag_dec,
            "p": ag.p,
            "alpha1": ag.alpha1,
            "alpha2": ag.alpha2,
            "beta": ag.beta,
            "t_break_s": ag.t_break,
            "smooth": ag.smooth,
            "regime": "ISM, slow cooling, nu_m < nu < nu_c, jet break",
        },
        "host": None
        if host is None
        else {"ra": host_ra, "dec": host_dec, **{f"mag_{b}": host.mag(b) for b in cfg.bands}, **asdict(host)},
        "detector": {"gain": gain, "read_noise": rn, "saturation": cfg.saturation},
        "config": {k: v for k, v in asdict(cfg).items() if k not in ("afterglow", "host")},
        "frames": frames,
        "templates": templates,
    }
    with open(out / "truth.json", "w") as fh:
        json.dump(truth, fh, indent=2, default=float)
    return truth
