#!/usr/bin/env python3
"""GRB 221009A: the pipeline applied to public Zwicky Transient Facility images.

Runs the same reduction as the synthetic demo on real ZTF survey frames of the
"brightest of all time" gamma-ray burst, then

1. subtracts the flux at the burst position in the deep pre-burst ZTF
   reference image (crowded Galactic-plane field, so this matters);
2. rejects epochs that fail visual vetting, with the reason recorded;
3. fits the temporal decay and compares it with independent Liverpool
   Telescope photometry published by Laskar et al. (2023).

Everything runs offline from the files in ``data/`` (see ``fetch_data.py``).
"""

from __future__ import annotations

import argparse
import json
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

from grb_afterglow import LightCurve, PipelineConfig, fit_single_power_law, reduce_frame
from grb_afterglow.calibration import load_fits
from grb_afterglow.lightcurve import SECONDS_PER_DAY, flux_to_mag
from grb_afterglow.pipeline import FrameResult, header_time_since_trigger
from grb_afterglow.plotting import plot_filmstrip, plot_lightcurve, plot_zeropoints

HERE = Path(__file__).resolve().parent
TARGET = SkyCoord(288.26452 * u.deg, 19.77350 * u.deg)  # Swift/UVOT, GCN 32632
TRIGGER_MJD = 59861.553460648  # Fermi/GBM, 2022-10-09 13:16:59 UT, GCN 32636
LITERATURE_ALPHA = (1.42, 0.01)  # r'-band decay after 0.4 d, Laskar et al. (2023)
FIT_TMIN_DAYS = 1.0

# Epochs rejected after visual inspection of the cutouts (see figures/filmstrip_*.png).
REJECTED = {
    "ztf_20221018154699_zr_sciimg.fits": "satellite trail",
}

CONFIG = PipelineConfig(
    fwhm_guess=2.0,
    aperture_radius_fwhm=1.0,
    annulus_fwhm=(4.0, 8.0),
    min_zp_snr=10.0,
    catalog_bands={"zr": "r", "zg": "g"},
)


def reduce(path: Path, catalog: Table, band: str) -> FrameResult:
    image, header = load_fits(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        wcs = WCS(header)
    time = 0.0 if "refimg" in path.name else header_time_since_trigger(header, TRIGGER_MJD)
    return reduce_frame(
        image,
        wcs,
        TARGET,
        catalog,
        band=band,
        time=time,
        exptime=float(header.get("EXPTIME", 1.0)),
        gain=header.get("GAIN"),
        saturation=header.get("SATURATE"),
        config=CONFIG,
        path=str(path),
    )


def load_literature(path: Path, band: str = "r") -> LightCurve:
    table = Table.read(path, format="ascii.csv", comment="#")
    table = table[table["band"] == band]
    return LightCurve.from_magnitudes(
        np.asarray(table["time_days"]) * SECONDS_PER_DAY,
        table["mag"],
        table["mag_err"],
        band,
        {"source": "Laskar et al. 2023, Liverpool Telescope"},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=HERE / "data")
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--figures", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)

    catalog = Table.read(args.data / "panstarrs_dr1.csv", format="ascii.csv")
    manifest = Table.read(args.data / "ztf_manifest.csv", format="ascii.csv")
    manifest = manifest[manifest["download"] == "ok"]

    rows, frames, references = [], {}, {}
    for band in ("zr", "zg"):
        ref = reduce(args.data / "ztf" / f"ztf_ref_{band}_refimg.fits", catalog, band)
        references[band] = ref
        print(
            f"{band} reference: {ref.flux_ujy:6.2f} ± {ref.flux_err_ujy:4.2f} µJy at the burst position "
            f"(FWHM {ref.fwhm:.2f} px, ZP from {ref.zeropoint.n_stars} stars)"
        )

    for entry in manifest:
        name, band = str(entry["file"]), str(entry["filtercode"])
        status = REJECTED.get(name, "ok")
        try:
            frame = reduce(args.data / "ztf" / name, catalog, band)
        except RuntimeError as exc:
            print(f"  {name}: skipped ({exc})")
            rows.append((name, band, float(entry["obsmjd"] - TRIGGER_MJD), *([np.nan] * 7), "no zero point"))
            continue
        frames[name] = frame
        ref = references[band]
        net = frame.flux_ujy - ref.flux_ujy
        err = float(np.hypot(frame.flux_err_ujy, ref.flux_err_ujy))
        rows.append(
            (
                name,
                band,
                frame.time / SECONDS_PER_DAY,
                frame.fwhm,
                frame.zeropoint.value,
                frame.zeropoint.error,
                frame.zeropoint.n_stars,
                net,
                err,
                net / err,
                status,
            )
        )
        print(
            f"  {name}  {band}  t = {frame.time / SECONDS_PER_DAY:6.2f} d  "
            f"net F = {net:7.2f} ± {err:5.2f} µJy  ({net / err:+5.1f}σ)  {status}"
        )

    table = Table(
        rows=rows,
        names=(
            "file",
            "band",
            "time_days",
            "fwhm_px",
            "zeropoint",
            "zeropoint_err",
            "n_zp_stars",
            "flux_ujy",
            "flux_err_ujy",
            "snr",
            "status",
        ),
    )
    for col in ("time_days", "fwhm_px", "zeropoint", "zeropoint_err", "flux_ujy", "flux_err_ujy", "snr"):
        table[col].info.format = ".4g"
    table.write(args.results / "ztf_photometry.csv", format="ascii.csv", overwrite=True)

    good = table[(table["status"] == "ok")]
    ztf = LightCurve(
        np.asarray(good["time_days"]) * SECONDS_PER_DAY,
        good["flux_ujy"],
        good["flux_err_ujy"],
        np.asarray(good["band"], dtype=object),
        {"source": "ZTF, this pipeline"},
    )
    ztf_r = ztf.select("zr")
    ztf_r.save_csv(args.results / "lightcurve_ztf_r.csv")
    lt_r = load_literature(args.data / "laskar2023_liverpool.csv")

    late = ztf_r.subset(ztf_r.time_days >= FIT_TMIN_DAYS)
    fit_ztf = fit_single_power_law(late.time, late.flux, late.flux_err)
    lt_late = lt_r.subset(lt_r.time_days >= FIT_TMIN_DAYS)
    fit_lt = fit_single_power_law(lt_late.time, lt_late.flux, lt_late.flux_err, sys_err=0.02)
    both = LightCurve(
        np.concatenate([late.time, lt_late.time]),
        np.concatenate([late.flux, lt_late.flux]),
        np.concatenate([late.flux_err, np.hypot(lt_late.flux_err, 0.02 * lt_late.flux)]),
        "r",
    )
    fit_joint = fit_single_power_law(both.time, both.flux, both.flux_err)

    # Photometric cross-check: ZTF detections against the Liverpool power law at the same times.
    det = ztf_r.detected(3.0)
    predicted = fit_lt.predict(ztf_r.time[det])
    delta_mag = flux_to_mag(ztf_r.flux[det]) - flux_to_mag(predicted)

    for label, fit in (("ZTF r only", fit_ztf), ("Liverpool r only", fit_lt), ("ZTF + Liverpool", fit_joint)):
        a, ae = fit.params["alpha"], fit.errors["alpha"]
        tension = abs(a - LITERATURE_ALPHA[0]) / np.hypot(ae, LITERATURE_ALPHA[1])
        print(
            f"{label:<18} alpha = {a:.3f} ± {ae:.3f}  chi2/dof = {fit.chi2:.1f}/{fit.dof}  "
            f"({tension:.1f}σ from Laskar et al. {LITERATURE_ALPHA[0]} ± {LITERATURE_ALPHA[1]})"
        )
    for t, dm in zip(ztf_r.time_days[det], delta_mag, strict=True):
        print(f"ZTF − Liverpool model at {t:.2f} d: {dm:+.3f} mag")

    summary = {
        "target": "GRB 221009A",
        "position_deg": [TARGET.ra.deg, TARGET.dec.deg],
        "trigger_mjd": TRIGGER_MJD,
        "n_ztf_epochs_measured": len(frames),
        "rejected": REJECTED,
        "reference_flux_ujy": {b: [r.flux_ujy, r.flux_err_ujy] for b, r in references.items()},
        "detections_r": [
            {"time_days": float(t), "flux_ujy": float(f), "flux_err_ujy": float(e), "mag_ab": float(m), "snr": float(s)}
            for t, f, e, m, s in zip(
                ztf_r.time_days[det], ztf_r.flux[det], ztf_r.flux_err[det], ztf_r.mag[det], ztf_r.snr[det], strict=True
            )
        ],
        "ztf_minus_liverpool_mag": [float(x) for x in delta_mag],
        "fits_t_ge_1d": {
            name: {"alpha": f.params["alpha"], "alpha_err": f.errors["alpha"], "chi2": f.chi2, "dof": f.dof}
            for name, f in (("ztf_r", fit_ztf), ("liverpool_r", fit_lt), ("joint", fit_joint))
        },
        "literature_alpha": {
            "value": LITERATURE_ALPHA[0],
            "error": LITERATURE_ALPHA[1],
            "source": "Laskar et al. 2023, ApJL, r'-band decay after 0.40 d",
        },
    }
    with open(args.results / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    make_figures(args, frames, references, ztf_r, lt_r, fit_joint, table)
    return 0


def make_figures(args, frames, references, ztf_r, lt_r, fit, table) -> None:
    r_names = [str(n) for n in table["file"][table["band"] == "zr"] if str(n) in frames]
    panels = []
    ref = references["zr"]
    ref_img, _ = load_fits(ref.path)
    panels.append({"image": ref_img, "x": ref.target.x, "y": ref.target.y, "r_aper": ref.r_aper, "label": "reference"})
    for name in r_names[:9]:
        f = frames[name]
        img, _ = load_fits(f.path)
        panels.append(
            {
                "image": img,
                "x": f.target.x,
                "y": f.target.y,
                "r_aper": f.r_aper,
                "label": f"{f.time / SECONDS_PER_DAY:.1f} d",
                "flag": REJECTED.get(name),
            }
        )

    for theme in ("light", "dark"):
        fig, _ = plot_lightcurve(
            ztf_r,
            fit,
            extra=[(lt_r, "Liverpool Telescope r', Laskar et al. 2023")],
            theme=theme,
            band_labels={"zr": "ZTF r, this pipeline"},
            model_range=(FIT_TMIN_DAYS, 60.0),
            title="GRB 221009A in public ZTF images",
            subtitle=(
                "Reference-subtracted forced photometry · "
                f"power law fitted to t ≥ {FIT_TMIN_DAYS:g} d (ZTF + Liverpool)"
            ),
            annotate=True,
            savepath=args.figures / f"lightcurve_{theme}.png",
        )
        plt.close(fig)
        fig, _ = plot_filmstrip(
            panels,
            half=14,
            ncols=len(panels),
            theme=theme,
            title="ZTF r-band cutouts, 29 arcsec across, centred on the burst",
            savepath=args.figures / f"filmstrip_{theme}.png",
            panel_size=1.45,
        )
        plt.close(fig)
        r_frames = [frames[n] for n in r_names if REJECTED.get(n) is None]
        fig, _ = plot_zeropoints(
            r_frames,
            catalog_label="Pan-STARRS1 r",
            trend_title="Zero point per ZTF frame",
            theme=theme,
            savepath=args.figures / f"zeropoints_{theme}.png",
        )
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
