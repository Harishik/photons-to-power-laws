#!/usr/bin/env python3
"""A full ZTF CCD quadrant through the pipeline, with a PSF and kernel that vary across the frame.

The GRB 210204A case study measures the afterglow on 120 arcsec cutouts. This
script reduces the same epoch (r band, 0.94 d after the burst) on the full
3072 x 3080 pixel quadrant (field 515, CCD 5, quadrant 1; 0.86 x 0.87 deg) with
``PipelineConfig(photometry="psf", tiles=(4, 4))``: the PSF varies smoothly
over the frame and the subtraction kernel is fitted in 16 tiles. It then
compares the afterglow flux with the cutout measurement, maps the PSF and
kernel across the quadrant, and measures how well static stars cancel
everywhere with one kernel and with tiles.

Downloads about 80 MB (the quadrant and its reference image from IRSA,
Pan-STARRS1 stars from MAST) into ``--data`` on the first run.

    python dev/full_frame/run_full_frame.py
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.wcs import WCS, FITSFixedWarning

from grb_afterglow import PipelineConfig, Template, reduce_frame
from grb_afterglow.archives import (
    download,
    query_panstarrs,
    query_ztf_science_images,
    ztf_reference_url,
    ztf_science_url,
)
from grb_afterglow.calibration import load_fits
from grb_afterglow.detection import detect_sources, estimate_background
from grb_afterglow.pipeline import header_time_since_trigger
from grb_afterglow.plotting import themed
from grb_afterglow.subtraction import subtract_images

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TARGET = SkyCoord(117.08054 * u.deg, 11.40949 * u.deg)  # ZTF21aagwbjr, GCN Circular 29405
TRIGGER_MJD = 59249.27042824  # Fermi/GBM, 2021-02-04 06:29:25 UT
EPOCH = "20210205214653"  # the r-band epoch at 0.94 d in the case study
FIELD, CCD, QUADRANT = 515, 5, 1
TILES = (4, 4)
CASE_STUDY = ROOT / "case_studies" / "grb210204a"


def fetch(data: Path) -> tuple[Path, Path, Path]:
    data.mkdir(parents=True, exist_ok=True)
    sci, ref, cat = data / f"ztf_{EPOCH}_zr_sciimg.fits", data / "ztf_ref_zr_refimg.fits", data / "panstarrs.csv"
    if not sci.exists():
        meta = query_ztf_science_images(TARGET.ra.deg, TARGET.dec.deg, TRIGGER_MJD + 0.5, TRIGGER_MJD + 1.5)
        row = next(r for r in meta if str(r["filefracday"]) == EPOCH and str(r["filtercode"]) == "zr")
        print("Downloading the science quadrant ...")
        download(ztf_science_url(row), sci)
    if not ref.exists():
        print("Downloading the reference quadrant ...")
        download(ztf_reference_url(FIELD, "zr", CCD, QUADRANT), ref)
    if not cat.exists():
        image, header = load_fits(sci)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            centre = WCS(header).pixel_to_world((image.shape[1] - 1) / 2, (image.shape[0] - 1) / 2)
        print("Querying Pan-STARRS1 over the quadrant ...")
        table = query_panstarrs(centre.ra.deg, centre.dec.deg, 38.0, bands="gr", constraints={"rMeanPSFMag.lte": 19.5})
        table.write(cat, format="ascii.csv", overwrite=True)
    return sci, ref, cat


def star_residuals(diff, image, fwhm: float, n_max: int = 3000) -> np.ndarray:
    """Normalised residual (rms chi) within 1.5 FWHM of bright, isolated stars, with their positions."""
    bkg = estimate_background(image)
    sources = detect_sources(image, fwhm=fwhm, threshold_sigma=30.0, background=bkg)
    x, y = np.asarray(sources["x_centroid"]), np.asarray(sources["y_centroid"])
    chi = diff.data / np.sqrt(diff.variance)
    r = int(np.ceil(1.5 * fwhm))
    g = np.arange(-r, r + 1)
    out = []
    for xi, yi in list(zip(x, y, strict=True))[:n_max]:
        ix, iy = int(np.rint(xi)), int(np.rint(yi))
        if not (r <= ix < image.shape[1] - r and r <= iy < image.shape[0] - r):
            continue
        patch = chi[iy - r : iy + r + 1, ix - r : ix + r + 1][np.hypot(*np.meshgrid(g, g)) <= 1.5 * fwhm]
        if np.all(np.isfinite(patch)):
            out.append((xi, yi, float(np.sqrt(np.mean(patch**2)))))
    return np.array(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=HERE / "data")
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    sci_path, ref_path, cat_path = fetch(args.data)

    summary_cs = json.loads((CASE_STUDY / "results" / "summary.json").read_text())
    term = summary_cs["color_terms"]["zr"]
    blue, red = term["color"].split(" - ")
    config = PipelineConfig(
        fwhm_guess=2.0,
        aperture_radius_fwhm=1.0,
        annulus_fwhm=(4.0, 8.0),
        min_zp_snr=10.0,
        catalog_bands={"zr": "r"},
        photometry="psf",
        color_terms={"zr": (blue, red, term["slope"], term["err"])},
        target_color=summary_cs["afterglow_colors"]["zr"],
        tiles=TILES,
    )
    cutout = Table.read(CASE_STUDY / "results" / "ztf_photometry.csv", format="ascii.csv")
    cutout = cutout[[str(f).startswith(f"ztf_{EPOCH}_zr") for f in cutout["file"]]][0]

    image, header = load_fits(sci_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        wcs = WCS(header)
    catalog = Table.read(cat_path, format="ascii.csv")
    t0 = time.perf_counter()
    template = Template.from_fits(ref_path, fwhm_guess=2.0)
    frame = reduce_frame(
        image,
        wcs,
        TARGET,
        catalog,
        band="zr",
        time=header_time_since_trigger(header, TRIGGER_MJD),
        exptime=float(header["EXPTIME"]),
        gain=header.get("GAIN"),
        saturation=header.get("SATURATE"),
        config=config,
        template=template,
        path=str(sci_path),
    )
    runtime = time.perf_counter() - t0
    diff, psf = frame.difference, frame.psf
    print(
        f"Full quadrant {image.shape[1]} x {image.shape[0]}, {frame.n_sources} sources, "
        f"zero point {frame.zeropoint.value:.3f} ± {frame.zeropoint.error:.3f} from {frame.zeropoint.n_stars} stars; "
        f"{runtime:.0f} s"
    )
    tension = (frame.flux_ujy - cutout["flux_ujy"]) / np.hypot(frame.flux_err_ujy, cutout["flux_err_ujy"])
    print(
        f"Afterglow at {frame.time / 86400:.3f} d: full frame {frame.flux_ujy:.2f} ± {frame.flux_err_ujy:.2f} µJy; "
        f"cutout {cutout['flux_ujy']:.2f} ± {cutout['flux_err_ujy']:.2f} µJy ({tension:+.1f}σ)"
    )

    # PSF and kernel across the frame.
    ny, nx = image.shape
    gx, gy = np.meshgrid(np.linspace(0, nx - 1, 9), np.linspace(0, ny - 1, 9))
    fwhm_grid = psf.fwhm_at(gx, gy)
    centres = [(x, y) for y in diff.tile_y for x in diff.tile_x]

    def kernel_width(k):
        h = k.shape[0] // 2
        g = np.arange(-h, h + 1)
        return float(np.sqrt(np.sum(k * (g[None, :] ** 2 + g[:, None] ** 2) / 2) / np.sum(k)))

    widths = np.array([[kernel_width(k) for k in row] for row in diff.kernels])
    print(f"PSF FWHM across the quadrant: {fwhm_grid.min():.2f} to {fwhm_grid.max():.2f} px; gradients {psf.gradients}")
    print(f"Kernel rms width per tile: {widths.min():.2f} to {widths.max():.2f} px")

    # Static stars: one kernel for the whole quadrant against 16 tiles.
    t1 = time.perf_counter()
    single = subtract_images(
        image, wcs, template.image, template.wcs, science_fwhm=frame.fwhm, template_fwhm=template.fwhm
    )
    t_single = time.perf_counter() - t1
    res_single = star_residuals(single, image, frame.fwhm)
    res_tiled = star_residuals(diff, image, frame.fwhm)
    print(
        f"Bright stars, rms chi within 1.5 FWHM: one kernel {np.median(res_single[:, 2]):.2f} (median), "
        f"{TILES[0]} x {TILES[1]} tiles {np.median(res_tiled[:, 2]):.2f} ({len(res_tiled)} stars)"
    )

    result = {
        "frame": {
            "file": sci_path.name,
            "shape": list(image.shape),
            "field_ccd_quadrant": [FIELD, CCD, QUADRANT],
            "n_sources": frame.n_sources,
            "zero_point": [frame.zeropoint.value, frame.zeropoint.error, frame.zeropoint.n_stars],
            "runtime_s": runtime,
            "tiles": list(TILES),
        },
        "afterglow": {
            "time_days": frame.time / 86400,
            "full_frame_ujy": [frame.flux_ujy, frame.flux_err_ujy],
            "cutout_ujy": [float(cutout["flux_ujy"]), float(cutout["flux_err_ujy"])],
            "cutout_zero_point_stars": int(cutout["n_zp_stars"]),
            "difference_sigma": float(tension),
        },
        "psf": {
            "fwhm_px_min_max": [float(fwhm_grid.min()), float(fwhm_grid.max())],
            "gradients": psf.gradients,
            "n_psf_stars": psf.n_stars,
            "model_error": psf.model_error,
        },
        "kernel_rms_width_px": {"min": float(widths.min()), "max": float(widths.max()), "tile_centres": centres},
        "static_star_residual_rms_chi": {
            "one_kernel_median": float(np.median(res_single[:, 2])),
            "tiled_median": float(np.median(res_tiled[:, 2])),
            "n_stars": len(res_tiled),
            "single_kernel_runtime_s": t_single,
        },
    }
    (args.out / "results.json").write_text(json.dumps(result, indent=2, default=float))

    for theme in ("light", "dark"):
        with themed(theme) as t:
            fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
            im = axes[0].imshow(fwhm_grid, origin="lower", extent=(0, nx, 0, ny), cmap="viridis")
            fig.colorbar(im, ax=axes[0], label="PSF FWHM (px)", shrink=0.85)
            axes[0].set_title("PSF across the quadrant")
            im = axes[1].imshow(widths, origin="lower", extent=(0, nx, 0, ny), cmap="viridis")
            fig.colorbar(im, ax=axes[1], label="kernel rms width (px)", shrink=0.85)
            axes[1].set_title(f"Kernel in {TILES[0]} × {TILES[1]} tiles")
            for label, res, color in (("one kernel", res_single, t.series[2]), ("tiles", res_tiled, t.series[0])):
                dist = np.hypot(res[:, 0] - nx / 2, res[:, 1] - ny / 2)
                order = np.argsort(dist)
                bins = np.array_split(order, 10)
                curve = ([np.median(dist[b]) for b in bins], [np.median(res[b, 2]) for b in bins])
                axes[2].plot(*curve, "o-", color=color, label=label)
            axes[2].axhline(1.0, color=t.muted, lw=0.9)
            axes[2].set_xlabel("Distance from the frame centre (px)")
            axes[2].set_ylabel("Static-star residual, rms χ")
            axes[2].legend()
            axes[2].set_title("How well stars cancel")
            for ax in axes[:2]:
                ax.grid(False)
                ax.plot(*zip(*centres, strict=True), "+", color="white", ms=6)
                ax.set_xlabel("x (px)")
            axes[0].set_ylabel("y (px)")
            fig.suptitle("A full ZTF quadrant (GRB 210204A field, r band, 0.94 d)", x=0.02, ha="left", weight="bold")
            fig.tight_layout()
            fig.savefig(args.out / f"full_frame_{theme}.png", dpi=150)
            plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
