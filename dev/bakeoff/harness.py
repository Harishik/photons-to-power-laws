"""Common evaluation harness for the PSF-photometry and image-subtraction bake-off.

Usage
-----
    python harness.py subtraction path/to/method.py [--quick]
    python harness.py psf path/to/method.py [--quick]

A *subtraction* method module must define

    subtract(sci, sci_wcs, ref, ref_wcs, *, sci_var, ref_var, sci_fwhm, ref_fwhm) -> dict

returning at least ``{"diff": D}`` where ``D`` is on the SCIENCE pixel grid and in
SCIENCE counts (a transient of F science counts appears in D with total F).
Optional keys: ``"var"`` (per-pixel variance of D), ``"fwhm"`` (PSF FWHM of D, px),
``"kernel"`` and ``"convolved"`` (the kernel applied to the convolved image, in
``scipy.ndimage.correlate`` orientation, and ``"science"`` or ``"reference"``; used to
calibrate the real data), ``"psf"`` (a centred, normalised stamp of the PSF of D, used
for its encircled energy instead of a Moffat profile of FWHM ``fwhm``).

A *psf* method module must define

    psf_photometry(image, positions, *, fwhm, source_xy, gain=None, var=None) -> dict

returning ``{"flux": array, "flux_err": array}`` in image counts for each (x, y) in
``positions``. ``source_xy`` lists detected sources (bright first) that may be used
to build the PSF model and to fit neighbours. ``image`` still contains sky.

Results are written as JSON next to the method file (results_<kind>.json) and printed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.wcs import WCS
from photutils.aperture import CircularAperture, aperture_photometry
from scipy import ndimage
from scipy.signal import fftconvolve

REPO = Path(__file__).resolve().parents[2]  # dev/bakeoff/harness.py -> repository root
sys.path.insert(0, str(REPO / "src"))

from grb_afterglow.calibration import load_fits  # noqa: E402
from grb_afterglow.detection import detect_sources, estimate_background, make_source_mask  # noqa: E402
from grb_afterglow.fitting import fit_single_power_law  # noqa: E402
from grb_afterglow.lightcurve import SECONDS_PER_DAY, LightCurve  # noqa: E402
from grb_afterglow.photometry import forced_photometry  # noqa: E402
from grb_afterglow.pipeline import PipelineConfig, header_time_since_trigger, reduce_frame  # noqa: E402
from grb_afterglow.synthetic import _moffat_stamp  # noqa: E402

warnings.simplefilter("ignore")

# --------------------------------------------------------------------------- synthetic data

SHAPE = (128, 128)


def _add(img, x, y, flux, fwhm, beta=3.0):
    half = int(np.ceil(6 * fwhm))
    ix, iy = int(np.rint(x)), int(np.rint(y))
    stamp = _moffat_stamp(x - ix, y - iy, fwhm, beta, half) * flux
    ny, nx = img.shape
    x0, x1, y0, y1 = max(ix - half, 0), min(ix + half + 1, nx), max(iy - half, 0), min(iy + half + 1, ny)
    if x0 < x1 and y0 < y1:
        img[y0:y1, x0:x1] += stamp[y0 - (iy - half) : y1 - (iy - half), x0 - (ix - half) : x1 - (ix - half)]


def _disk(shape, x0, y0, total, r_e, q=0.6, pa=0.5, sub=3):
    """Pixel-integrated exponential disk (Sersic n=1), normalised to ``total``."""
    ny, nx = shape
    off = (np.arange(sub) + 0.5) / sub - 0.5
    yy = np.arange(ny)[:, None, None, None] + off[None, None, :, None] - y0
    xx = np.arange(nx)[None, :, None, None] + off[None, None, None, :] - x0
    c, s = np.cos(pa), np.sin(pa)
    xr, yr = c * xx + s * yy, (-s * xx + c * yy) / q
    img = np.exp(-1.678 * np.hypot(xr, yr) / r_e).mean(axis=(2, 3))
    return total * img / img.sum()


def _wcs(crpix, scale_arcsec=1.0, rot_deg=0.0):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [150.0, 20.0]
    w.wcs.crpix = list(crpix)
    s = scale_arcsec / 3600.0
    t = np.radians(rot_deg)
    w.wcs.cd = [[-s * np.cos(t), s * np.sin(t)], [s * np.sin(t), s * np.cos(t)]]
    return w


def make_pair(seed, sci_fwhm, ref_fwhm, transient, host=20000.0, shift=(3.37, -2.61), rot=0.15, ref_scale=0.8):
    """A science/reference pair of the same field with a transient on a host galaxy.

    Both images are background-included counts with Poisson + read noise. The
    reference is 4x deeper, has a different PSF, pixel grid (shift + rotation),
    sky level, and photometric scale (``ref_scale`` counts per science count).
    """
    rng = np.random.default_rng(seed)
    sci_wcs = _wcs((64.5, 64.5))
    ref_wcs = _wcs((64.5 + shift[0], 64.5 + shift[1]), rot_deg=rot)
    n = 45
    sky_xy = rng.uniform(6, 122, (n, 2))
    flux = 10 ** rng.uniform(2.6, 4.9, n)
    tx, ty = 61.3, 66.8  # transient, on the host
    keep = np.hypot(sky_xy[:, 0] - tx, sky_xy[:, 1] - ty) > 12
    sky_xy, flux = sky_xy[keep], flux[keep]
    star_world = sci_wcs.pixel_to_world_values(sky_xy[:, 0], sky_xy[:, 1])
    host_x, host_y = tx + 1.2, ty - 0.8

    def render(wcs, fwhm, scale, depth, sky, with_transient):
        img = np.zeros(SHAPE)
        sx, sy = wcs.world_to_pixel_values(*star_world)
        for x, y, f in zip(sx, sy, flux, strict=True):
            _add(img, x, y, f * scale * depth, fwhm)
        hx, hy = wcs.world_to_pixel_values(*sci_wcs.pixel_to_world_values(host_x, host_y))
        # The host is seen through the same PSF as the stars (fixed 09:10: it was unconvolved before).
        disk = _disk(SHAPE, float(hx), float(hy), host * scale * depth, r_e=3.5)
        psf = _moffat_stamp(0.0, 0.0, fwhm, 3.0, int(np.ceil(5 * fwhm)))
        img += fftconvolve(disk, psf / psf.sum(), mode="same")
        if with_transient and transient > 0:
            px, py = wcs.world_to_pixel_values(*sci_wcs.pixel_to_world_values(tx, ty))
            _add(img, float(px), float(py), transient * scale * depth, fwhm)
        img += sky * depth
        rn = 5.0
        noisy = rng.poisson(np.maximum(img, 0)) + rng.normal(0, rn, SHAPE)
        var = np.maximum(img, 0) + rn**2
        return noisy / depth, var / depth**2

    sci, sci_var = render(sci_wcs, sci_fwhm, 1.0, 1.0, 400.0, True)
    ref, ref_var = render(ref_wcs, ref_fwhm, ref_scale, 4.0, 520.0, False)
    return {
        "sci": sci,
        "sci_var": sci_var,
        "sci_wcs": sci_wcs,
        "ref": ref,
        "ref_var": ref_var,
        "ref_wcs": ref_wcs,
        "transient_xy": (tx, ty),
        "transient_flux": transient,
        "star_xy": sky_xy,
        "star_flux": flux,
    }


def make_single(seed, fwhm, crowded=True):
    """A single image of stars with known fluxes (for PSF photometry)."""
    rng = np.random.default_rng(1000 + seed)
    n = 120 if crowded else 40
    xy = rng.uniform(8, 120, (n, 2))
    flux = 10 ** rng.uniform(2.5, 5.0, n)
    img = np.full(SHAPE, 400.0)
    for (x, y), f in zip(xy, flux, strict=True):
        _add(img, x, y, f, fwhm)
    rn = 5.0
    var = img + rn**2
    noisy = rng.poisson(img) + rng.normal(0, rn, SHAPE)
    return noisy, var, xy, flux


# --------------------------------------------------------------------------- measurement helpers


def _random_apertures(diff, r, exclude_xy, exclude_r, n=300, seed=0):
    rng = np.random.default_rng(seed)
    occ = np.zeros(diff.shape, bool)
    for x, y in exclude_xy:
        occ[int(np.clip(np.rint(y), 0, diff.shape[0] - 1)), int(np.clip(np.rint(x), 0, diff.shape[1] - 1))] = True
    dist = ndimage.distance_transform_edt(~occ)
    margin = int(np.ceil(r + 8))
    pos = rng.uniform(margin, diff.shape[1] - margin, (6 * n, 2))
    ok = dist[pos[:, 1].astype(int), pos[:, 0].astype(int)] > exclude_r
    pos = pos[ok][:n]
    if len(pos) < 20:
        return float("nan")
    t = forced_photometry(diff, pos, r_aper=r, r_in=r + 2, r_out=r + 7)
    return float(np.std(np.asarray(t["net_counts"])))


def measure_on_diff(diff, xy, r, exclude_xy, exclude_r):
    t = forced_photometry(diff, np.array([xy]), r_aper=r, r_in=r + 2, r_out=r + 7)
    return float(t["net_counts"][0]), _random_apertures(diff, r, exclude_xy, exclude_r)


def moffat_fraction(r, fwhm, beta=3.0):
    alpha = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    return 1.0 - (1.0 + (r / alpha) ** 2) ** (1.0 - beta)


def encircled_fraction(res, r, fwhm):
    """Fraction of the difference image's PSF within radius r: from the method's PSF stamp if given."""
    psf = res.get("psf")
    if psf is None:
        return moffat_fraction(r, fwhm)
    psf = np.asarray(psf, float)
    c = (psf.shape[1] - 1) / 2.0, (psf.shape[0] - 1) / 2.0
    inside = float(aperture_photometry(psf, CircularAperture(c, r=r))["aperture_sum"][0])
    return inside / float(psf.sum())


def cog_fraction(img, fr, r):
    """Fraction of a point source's light inside radius r, from the curve of growth of bright stars."""
    bkg = estimate_background(img)
    mask = make_source_mask(img, bkg, dilate=max(2, int(np.rint(1.5 * fr.fwhm))))
    xy = fr.matched_xy
    big_r = 4.0 * fr.fwhm
    a = forced_photometry(img, xy, r_aper=r, r_in=big_r + 2, r_out=big_r + 8, sky_mask=mask)
    b = forced_photometry(img, xy, r_aper=big_r, r_in=big_r + 2, r_out=big_r + 8, sky_mask=mask)
    ca, cb = np.asarray(a["net_counts"]), np.asarray(b["net_counts"])
    ok = (np.asarray(b["snr"]) > 15) & (ca > 0) & (cb > 0)
    if ok.sum() >= 3:
        return float(np.median(ca[ok] / cb[ok]))
    return float(moffat_fraction(r, fr.fwhm))  # too few bright stars: analytic Moffat fallback


def transfer_ratio(img, conv, fr, r, fwhm_conv):
    """Median flux ratio of the zero-point stars: aperture r in ``conv`` over the ZP aperture in ``img``."""
    bkg = estimate_background(img)
    mask = make_source_mask(img, bkg, dilate=max(2, int(np.rint(1.5 * fr.fwhm))))
    xy = fr.matched_xy
    a = forced_photometry(img, xy, r_aper=fr.r_aper, r_in=fr.annulus[0], r_out=fr.annulus[1], sky_mask=mask)
    k = fwhm_conv / fr.fwhm
    b = forced_photometry(conv, xy, r_aper=r, r_in=fr.annulus[0] * k, r_out=fr.annulus[1] * k, sky_mask=mask)
    ca, cb = np.asarray(a["net_counts"]), np.asarray(b["net_counts"])
    ok = (np.asarray(a["snr"]) > 30) & (ca > 0) & (cb > 0)
    return float(np.median(cb[ok] / ca[ok])) if ok.sum() >= 3 else float("nan")


def _load_method(path, name):
    spec = importlib.util.spec_from_file_location("method_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["method_under_test"] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, name)


# --------------------------------------------------------------------------- real ZTF data

CS = REPO / "case_studies" / "grb221009a"
TARGET = SkyCoord(288.26452 * u.deg, 19.77350 * u.deg)
TRIGGER_MJD = 59861.553460648
REJECT = {"ztf_20221018154699_zr_sciimg.fits"}
CFG = PipelineConfig(
    fwhm_guess=2.0,
    aperture_radius_fwhm=1.0,
    annulus_fwhm=(4.0, 8.0),
    min_zp_snr=10.0,
    catalog_bands={"zr": "r", "zg": "g"},
)


def _reduce(path, catalog):
    img, hdr = load_fits(path)
    wcs = WCS(hdr)
    t = 0.0 if "refimg" in path.name else header_time_since_trigger(hdr, TRIGGER_MJD)
    fr = reduce_frame(
        img,
        wcs,
        TARGET,
        catalog,
        band="zr",
        time=t,
        exptime=float(hdr.get("EXPTIME", 1.0)),
        gain=hdr.get("GAIN"),
        saturation=hdr.get("SATURATE"),
        config=CFG,
        path=str(path),
    )
    return img, hdr, wcs, fr


def ztf_epochs():
    catalog = Table.read(CS / "data" / "panstarrs_dr1.csv", format="ascii.csv")
    manifest = Table.read(CS / "data" / "ztf_manifest.csv", format="ascii.csv")
    rows = [r for r in manifest if r["download"] == "ok" and r["filtercode"] == "zr" and r["file"] not in REJECT]
    ref = _reduce(CS / "data" / "ztf" / "ztf_ref_zr_refimg.fits", catalog)
    epochs = [_reduce(CS / "data" / "ztf" / str(r["file"]), catalog) for r in rows]
    lt = Table.read(CS / "data" / "laskar2023_liverpool.csv", format="ascii.csv", comment="#")
    lt = lt[(lt["band"] == "r") & (lt["time_days"] >= 1.0)]
    lc = LightCurve.from_magnitudes(np.asarray(lt["time_days"]) * SECONDS_PER_DAY, lt["mag"], lt["mag_err"], "r")
    lt_fit = fit_single_power_law(lc.time, lc.flux, lc.flux_err, sys_err=0.02)
    return ref, epochs, catalog, lt_fit


def _ujy_per_count_large(img, fr, r_large):
    """Calibration for a large aperture: ZP (r = 1 FWHM) times an aperture correction from isolated stars."""
    bkg = estimate_background(img)
    mask = make_source_mask(img, bkg, dilate=max(2, int(np.rint(1.5 * fr.fwhm))))
    xy = fr.matched_xy
    small = forced_photometry(img, xy, r_aper=fr.r_aper, r_in=fr.annulus[0], r_out=fr.annulus[1], sky_mask=mask)
    big = forced_photometry(
        img,
        xy,
        r_aper=r_large,
        r_in=max(fr.annulus[0], r_large + 2),
        r_out=max(fr.annulus[1], r_large + 6),
        sky_mask=mask,
    )
    s, b = np.asarray(small["net_counts"]), np.asarray(big["net_counts"])
    ok = (np.asarray(small["snr"]) > 30) & (s > 0) & (b > 0)
    ratio = float(np.median(b[ok] / s[ok])) if ok.sum() >= 3 else 1.0
    k1 = 10 ** (-0.4 * (fr.zeropoint.value - 23.9)) / fr.exptime
    return k1 / ratio, ratio


# --------------------------------------------------------------------------- evaluations


def eval_subtraction(subtract, quick=False):
    out = {"synthetic": {}, "real": {}}
    configs = [("sci_sharper", 2.4, 3.4), ("ref_sharper", 3.4, 2.4), ("matched", 3.0, 3.0), ("undersampled", 1.6, 2.2)]
    fluxes = [0.0, 600.0, 4000.0]
    seeds = range(4 if quick else 10)
    t0 = time.time()
    all_pulls, null_pulls, bright_bias, noise_ratio, star_resid = [], [], [], [], []
    big_pulls, small_err = [], []
    per_config = {}
    for name, fs, fr_ in configs:
        cfg_pulls = []
        for seed in seeds:
            for F in fluxes:
                pair = make_pair(seed, fs, fr_, F)
                res = subtract(
                    pair["sci"],
                    pair["sci_wcs"],
                    pair["ref"],
                    pair["ref_wcs"],
                    sci_var=pair["sci_var"],
                    ref_var=pair["ref_var"],
                    sci_fwhm=fs,
                    ref_fwhm=fr_,
                )
                d = np.asarray(res["diff"], float)
                fd = float(res.get("fwhm") or max(fs, fr_))
                excl = np.vstack([pair["star_xy"][pair["star_flux"] > 3000], [pair["transient_xy"]]])
                r_big = 2.5 * max(fs, fr_)
                fb, sb = measure_on_diff(d, pair["transient_xy"], r_big, excl, exclude_r=r_big + 3)
                frac_big = encircled_fraction(res, r_big, fd)
                big_pulls.append((fb / frac_big - F) / (sb / frac_big))
                r = 1.2 * fd
                frac = encircled_fraction(res, r, fd)
                f, sig = measure_on_diff(d, pair["transient_xy"], r, excl, exclude_r=r + 4)
                f, sig = f / frac, sig / frac
                small_err.append(sig)
                pull = (f - F) / sig
                all_pulls.append(pull)
                cfg_pulls.append(pull)
                if F == 0:
                    null_pulls.append(pull)
                if F == 4000.0:
                    bright_bias.append((f - F) / F)
                ideal = np.sqrt(np.pi * r**2 * np.median(pair["sci_var"] + pair["ref_var"] / 0.8**2)) / frac
                noise_ratio.append(sig / ideal)
                if F == 0:
                    bright = pair["star_flux"] > 20000
                    sx = pair["star_xy"][bright]
                    rr = 1.5 * max(fs, fr_)
                    t = forced_photometry(d, sx, r_aper=rr, r_in=2 * rr + 1, r_out=3 * rr + 2)
                    star_resid.extend(np.abs(np.asarray(t["net_counts"])) / pair["star_flux"][bright])
        per_config[name] = {"pull_mean": float(np.mean(cfg_pulls)), "pull_std": float(np.std(cfg_pulls))}
    out["synthetic"] = {
        "pull_mean": float(np.mean(all_pulls)),
        "pull_std": float(np.std(all_pulls)),
        "null_pull_std": float(np.std(null_pulls)),
        "bright_bias_frac": float(np.mean(bright_bias)),
        "bright_bias_scatter": float(np.std(bright_bias)),
        "noise_ratio_vs_ideal": float(np.median(noise_ratio)),
        "star_residual_frac_median": float(np.median(star_resid)),
        "per_config": per_config,
        "big_aperture_pull_mean": float(np.mean(big_pulls)),
        "big_aperture_pull_std": float(np.std(big_pulls)),
        "median_flux_err_counts": float(np.median(small_err)),
        "seconds": time.time() - t0,
    }

    ref, epochs, _, lt_fit = ztf_epochs()
    rimg, _, rwcs, rfr = ref
    rows, chi_det, chi_late = [], [], []
    for img, _, wcs, fr in epochs:
        res = subtract(img, wcs, rimg, rwcs, sci_var=None, ref_var=None, sci_fwhm=fr.fwhm, ref_fwhm=rfr.fwhm)
        d = np.asarray(res["diff"], float)
        fd = float(res.get("fwhm") or max(fr.fwhm, rfr.fwhm))
        r_s = 1.2 * fd
        k1 = 10 ** (-0.4 * (fr.zeropoint.value - 23.9)) / fr.exptime
        tx, ty = fr.target.x, fr.target.y
        excl = np.vstack([fr.sources_xy[:40], [[tx, ty]]])
        f, sig = measure_on_diff(d, (tx, ty), r_s, excl, exclude_r=r_s + 2)
        kernel = res.get("kernel")
        if kernel is not None and res.get("convolved") in ("science", "reference"):
            # Aperture correction from the zero-point aperture (science image) to the measurement
            # aperture on the difference-image PSF, from the same stars (see HARNESS_CHANGELOG 19:30).
            conv = (
                ndimage.correlate(img, kernel / kernel.sum(), mode="nearest") if res["convolved"] == "science" else img
            )
            ratio = transfer_ratio(img, conv, fr, r_s, fd)
        else:  # methods that do not expose their kernel: curves of growth (unreliable in crowded fields)
            use_ref_psf = abs(fd - rfr.fwhm) < abs(fd - fr.fwhm)
            psf_img, psf_fr = (rimg, rfr) if use_ref_psf else (img, fr)
            ratio = cog_fraction(psf_img, psf_fr, r_s) / cog_fraction(img, fr, fr.r_aper)
        scale = k1 / ratio
        f_ujy, e_ujy = f * scale, sig * scale
        pred = float(lt_fit.predict(np.array([fr.time]))[0])
        days = fr.time / SECONDS_PER_DAY
        z = (f_ujy - pred) / np.hypot(e_ujy, 0.05 * pred)
        (chi_late if days > 9 else chi_det).append(z)
        # residual RMS (in sigma) in apertures on the 10 brightest unsaturated matched stars
        mx = fr.matched_xy[:10]
        t = forced_photometry(d, mx, r_aper=1.5 * fr.fwhm, r_in=3 * fr.fwhm + 1, r_out=5 * fr.fwhm + 2)
        star_res = float(np.median(np.abs(np.asarray(t["net_counts"])) / sig)) if len(mx) else float("nan")
        rows.append(
            {
                "days": days,
                "flux_ujy": f_ujy,
                "err_ujy": e_ujy,
                "lt_pred_ujy": pred,
                "z": z,
                "star_residual_in_sigma": star_res,
                "ap_ratio": ratio,
            }
        )
    out["real"] = {
        "epochs": rows,
        "chi2_per_pt_detected_epochs": float(np.mean(np.square(chi_det))) if chi_det else None,
        "chi2_per_pt_late_epochs": float(np.mean(np.square(chi_late))) if chi_late else None,
        "median_err_ujy": float(np.median([r["err_ujy"] for r in rows])),
        "median_star_residual_in_sigma": float(np.nanmedian([r["star_residual_in_sigma"] for r in rows])),
    }
    return out


def _aperture_reference(img, xy, fwhm, source_xy):
    """Baseline: aperture r = 1 FWHM with aperture correction from bright isolated stars."""
    bkg = estimate_background(img)
    mask = make_source_mask(img, bkg, dilate=max(2, int(np.rint(1.5 * fwhm))))
    a = forced_photometry(img, xy, r_aper=fwhm, r_in=3 * fwhm, r_out=5 * fwhm, sky_mask=mask)
    return np.asarray(a["net_counts"]), np.asarray(a["net_counts_err"])


def eval_psf(psf_photometry, quick=False):
    out = {}
    t0 = time.time()
    rows = []
    for crowded in (False, True):
        for fwhm in (1.8, 2.6, 3.6):
            for seed in range(2 if quick else 5):
                img, var, xy, flux = make_single(seed, fwhm, crowded)
                srcs = detect_sources(img, fwhm=fwhm)
                src_xy = np.column_stack([np.asarray(srcs["x_centroid"]), np.asarray(srcs["y_centroid"])])
                inner = np.all((xy > 10) & (xy < 118), axis=1)
                res = psf_photometry(img, xy[inner], fwhm=fwhm, source_xy=src_xy, gain=1.0, var=var)
                f, e = np.asarray(res["flux"], float), np.asarray(res["flux_err"], float)
                ap, ap_e = _aperture_reference(img, xy[inner], fwhm, src_xy)
                bright = flux[inner] > 20000
                corr = np.median(ap[bright] / flux[inner][bright]) if bright.sum() > 3 else 1.0
                for fi, ei, ti, ai, aei in zip(f, e, flux[inner], ap, ap_e, strict=True):
                    rows.append((crowded, fwhm, ti, fi, ei, ai / corr, aei / corr))
    rows = np.array(rows, dtype=float)
    crowded, fwhm, true, f, e, ap, ap_e = rows.T
    good = np.isfinite(f) & np.isfinite(e) & (e > 0)
    pull = (f - true) / e
    faint = good & (true < 3000)
    bright = good & (true > 20000)
    s_m = float(np.median(f[bright] / true[bright])) if bright.sum() > 3 else 1.0
    s_a = float(np.median(ap[bright] / true[bright])) if bright.sum() > 3 else 1.0
    frac_err_m = e / (s_m * true)
    frac_err_a = ap_e / (s_a * true)
    pull_scaled = (f / s_m - true) / (e / s_m)

    def rstd(x):
        x = x[np.isfinite(x)]
        return float(1.4826 * np.median(np.abs(x - np.median(x))))

    out["synthetic"] = {
        "n": int(good.sum()),
        "frac_failed": float(1 - good.mean()),
        "flux_scale_bright": s_m,
        "bright_bias_frac": float(np.median((f[bright] - true[bright]) / true[bright])),
        "pull_median": float(np.median(pull[good])),
        "pull_std_robust": rstd(pull[good]),
        "pull_std_robust_after_scale": rstd(pull_scaled[good]),
        "crowded_pull_std_robust": rstd(pull[good & (crowded == 1)]),
        "faint_precision_gain_vs_aperture": float(np.median(frac_err_a[faint] / frac_err_m[faint])),
        "faint_median_abs_error_frac": float(np.median(np.abs(f[faint] / s_m - true[faint]) / true[faint])),
        "faint_median_abs_error_frac_aperture": float(np.median(np.abs(ap[faint] / s_a - true[faint]) / true[faint])),
        "seconds": time.time() - t0,
    }

    ref, epochs, _, _ = ztf_epochs()
    zp_scatter, zp_scatter_ap = [], []
    for img, hdr, _, fr in [ref, *epochs]:
        mx = fr.matched_xy
        res = psf_photometry(img, mx, fwhm=fr.fwhm, source_xy=fr.sources_xy, gain=hdr.get("GAIN"), var=None)
        f = np.asarray(res["flux"], float)
        cat = fr.star_catalog_mag
        ok = f > 0
        resid = cat[ok] + 2.5 * np.log10(f[ok])
        mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        zp_scatter.append(mad)
        apf = fr.star_instrumental_mag
        ra = cat - apf
        zp_scatter_ap.append(1.4826 * np.median(np.abs(ra - np.median(ra))))
    out["real"] = {
        "zp_scatter_psf_median": float(np.median(zp_scatter)),
        "zp_scatter_aperture_median": float(np.median(zp_scatter_ap)),
        "per_frame_psf": [float(x) for x in zp_scatter],
        "per_frame_aperture": [float(x) for x in zp_scatter_ap],
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["subtraction", "psf"])
    ap.add_argument("method")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    path = Path(a.method).resolve()
    if a.kind == "subtraction":
        res = eval_subtraction(_load_method(path, "subtract"), a.quick)
    else:
        res = eval_psf(_load_method(path, "psf_photometry"), a.quick)
    out = path.parent / f"results_{a.kind}{'_quick' if a.quick else ''}.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(
        json.dumps(
            {k: (v if k != "real" else {kk: vv for kk, vv in v.items() if kk != "epochs"}) for k, v in res.items()},
            indent=2,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
