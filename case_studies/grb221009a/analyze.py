#!/usr/bin/env python3
"""GRB 221009A: the pipeline applied to public Zwicky Transient Facility images.

Runs the same reduction as the synthetic demo on real ZTF survey frames of the
"brightest of all time" gamma-ray burst, then

1. subtracts the deep pre-burst ZTF reference image from every epoch
   (crowded Galactic-plane field, so this matters) and measures the burst on
   the difference image with PSF photometry; for comparison it also repeats
   the simpler v0.2 measurement, aperture photometry minus the flux at the
   burst position in the reference image;
2. rejects epochs that fail visual vetting, with the reason recorded;
3. fits the temporal decay and compares it with independent Liverpool
   Telescope photometry published by Laskar et al. (2023);
4. measures the spectral index beta from that griz photometry together with
   the optical and near-infrared photometry of O'Connor et al. (2023) from
   Lowell and Gemini, with the Galactic extinction forward-modelled through
   every instrument's filter curve and a calibration offset per band, with an
   MCMC posterior and systematic errors from the uncertain E(B-V) and from the
   photometric system, and uses it for the colour term of the ZTF zero points.

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

from grb_afterglow import (
    LightCurve,
    PipelineConfig,
    evaluate_closure_relations,
    fit_lightcurve,
    fit_single_power_law,
    reduce_frame,
)
from grb_afterglow.calibration import load_fits
from grb_afterglow.extinction import power_law_color
from grb_afterglow.filters import Bandpass, get_bandpass, register_bandpass
from grb_afterglow.lightcurve import SECONDS_PER_DAY, flux_to_mag
from grb_afterglow.mcmc import sample_posterior
from grb_afterglow.pipeline import FrameResult, Template, header_time_since_trigger
from grb_afterglow.plotting import (
    plot_filmstrip,
    plot_lightcurve,
    plot_posterior,
    plot_spectrum,
    plot_subtraction,
    plot_zeropoints,
)

HERE = Path(__file__).resolve().parent
TARGET = SkyCoord(288.26452 * u.deg, 19.77350 * u.deg)  # Swift/UVOT, GCN 32632
TRIGGER_MJD = 59861.553460648  # Fermi/GBM, 2022-10-09 13:16:59 UT, GCN 32636
LITERATURE_ALPHA = (1.42, 0.01)  # r'-band decay after 0.4 d, Laskar et al. (2023)
FIT_TMIN_DAYS = 1.0
EBV_MW = (1.32, 0.06)  # Schlafly & Finkbeiner (2011) map, as derived by Shrestha et al. (2023)
LITERATURE_BETA = (0.68, 0.05)  # optical/IR after 1.7 d for E(B-V) = 1.32, O'Connor et al. (2023)
LASKAR_BETA = (0.76, 0.04)  # NIR to optical at ~4.4 d, Galactic extinction removed, Laskar et al. (2023)
# From 1.7 d, the interval of the O'Connor et al. measurement, to 20 d, where a supernova may begin to contribute.
SED_WINDOW_DAYS = (1.7, 20.0)
SED_REFERENCE_DAYS = 4.4  # the Gemini optical + NIR epoch
NIR_FILTERS = ("J", "H", "Ks")
# Zero-point uncertainty per band: optical photometry calibrated to Pan-STARRS1 without colour terms, NIR to 2MASS
# and converted from Vega to AB.
CALIBRATION_PRIOR_MAG = {"optical": 0.03, "nir": 0.05}

# Epochs rejected after visual inspection of the cutouts (see figures/filmstrip_*.png).
REJECTED = {
    "ztf_20221018154699_zr_sciimg.fits": "satellite trail",
}

SETTINGS = {
    "fwhm_guess": 2.0,
    "aperture_radius_fwhm": 1.0,
    "annulus_fwhm": (4.0, 8.0),
    "min_zp_snr": 10.0,
    "catalog_bands": {"zr": "r", "zg": "g"},
}
APERTURE = PipelineConfig(**SETTINGS, photometry="aperture")  # v0.2: aperture minus the reference flux


def reduce(
    path: Path, catalog: Table, band: str, *, config: PipelineConfig, template: Template | None = None
) -> FrameResult:
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
        config=config,
        template=template,
        path=str(path),
    )


def load_literature(path: Path, band: str | None = "r") -> LightCurve:
    """Liverpool Telescope photometry of Laskar et al. (2023): one band, or all of griz."""
    table = Table.read(path, format="ascii.csv", comment="#")
    if band is not None:
        table = table[table["band"] == band]
    return LightCurve.from_magnitudes(
        np.asarray(table["time_days"]) * SECONDS_PER_DAY,
        table["mag"],
        table["mag_err"],
        np.asarray(table["band"], dtype=object),
        {"source": "Laskar et al. 2023, Liverpool Telescope"},
    )


def load_multi_instrument(data: Path) -> LightCurve:
    """Liverpool griz (Laskar et al. 2023) and Lowell, Gemini, Palomar and HST photometry (O'Connor et al. 2023).

    Bands are named after the filter curve that models them (``"lt:r"``, ``"f2:Ks"``): calibrated
    magnitudes of a source this red depend on the exact bandpass.
    """
    lt = Table.read(data / "laskar2023_liverpool.csv", format="ascii.csv", comment="#")
    oir = Table.read(data / "oconnor2023_oir.csv", format="ascii.csv", comment="#")
    return LightCurve.from_magnitudes(
        np.concatenate([lt["time_days"], oir["time_days"]]) * SECONDS_PER_DAY,
        np.concatenate([lt["mag"], oir["mag"]]),
        np.concatenate([lt["mag_err"], oir["mag_err"]]),
        np.array([f"lt:{b}" for b in lt["band"]] + [str(b) for b in oir["band"]], dtype=object),
        {"source": "Laskar et al. 2023 (Liverpool); O'Connor et al. 2023 (LDT, Gemini, P200, HST)"},
    )


def register_lmi_filters() -> None:
    """The Lowell LMI griz filters are SDSS-type, but their curves are not in the SVO service: use SDSS's."""
    for f in "griz":
        sdss = get_bandpass(f"sdss:{f}")
        register_bandpass(Bandpass(f"lmi:{f}", sdss.wavelength_nm, sdss.transmission))


def catalog_system(bands) -> dict:
    """Alternative model: each measurement in the system of its calibration catalogue (PS1 optical, 2MASS NIR).

    That is what a zero point with colour terms evaluated at the afterglow's colour would deliver; the published
    photometry has no colour terms, so the instruments' own (natural) curves are the default.
    """
    out = {}
    for band in bands:
        name = band.split(":", 1)[1]
        out[band] = get_bandpass(f"2mass:{name}" if name in NIR_FILTERS else f"ps1:{name}")
    return out


def fit_sed(lc: LightCurve, ebv: float, bandpasses=True):
    """Single power law with a common spectral index, Galactic dust forward-modelled through every filter curve.

    Every band but the reference (Liverpool r, which sets the flux scale) carries a zero-point offset with a
    Gaussian prior (CALIBRATION_PRIOR_MAG): the photometry comes from several instruments and catalogues and is
    calibrated without colour terms.
    """
    priors = {
        b: CALIBRATION_PRIOR_MAG["nir" if b.split(":")[1] in NIR_FILTERS else "optical"]
        for b in lc.bands
        if b != "lt:r"
    }
    return fit_lightcurve(
        lc,
        "single",
        reference_band="lt:r",
        ebv_mw=ebv,
        sys_err=0.02,
        bandpasses=bandpasses,
        calibration_offsets=priors,
    )


def measure_spectral_index(data: Path) -> dict:
    """The optical-to-NIR spectral index: Liverpool griz alone, then with the LDT and Gemini photometry."""
    register_lmi_filters()
    lc = load_multi_instrument(data)
    lo, hi = SED_WINDOW_DAYS
    window = lc.subset((lc.time_days >= lo) & (lc.time_days <= hi))
    liverpool = window.subset(np.array([str(b).startswith("lt:") for b in window.band]))
    rows = {}
    for name, sub in (("liverpool", liverpool), ("joint", window)):
        for system in ("natural", "catalog"):
            bandpasses = True if system == "natural" else catalog_system(sub.bands)
            fits = [fit_sed(sub, ebv, bandpasses) for ebv in (EBV_MW[0] - EBV_MW[1], EBV_MW[0], EBV_MW[0] + EBV_MW[1])]
            fit = fits[1]
            rows[f"{name}_{system}"] = {
                "fit": fit,
                "beta": fit.params["beta"],
                "beta_err": fit.errors["beta"],
                "beta_sys_ebv": 0.5 * abs(fits[2].params["beta"] - fits[0].params["beta"]),
                "alpha": fit.params["alpha"],
                "chi2": fit.chi2,
                "dof": fit.dof,
                "n_points": len(sub),
                "offsets_mag": fit.offsets,
            }
    joint = rows["joint_natural"]
    posterior = sample_posterior(window, joint["fit"], sys_err=0.02, n_walkers=40, n_steps=16000, burn=4000)
    alpha, beta = posterior.interval("alpha"), posterior.interval("beta")
    beta_err = 0.5 * (beta[2] - beta[1])
    sys_system = {
        k: 0.5 * abs(rows[f"{k}_catalog"]["beta"] - rows[f"{k}_natural"]["beta"]) for k in ("liverpool", "joint")
    }
    beta_sys = float(np.hypot(joint["beta_sys_ebv"], sys_system["joint"]))
    closure = evaluate_closure_relations(alpha[0], 0.5 * (alpha[2] - alpha[1]), beta[0], np.hypot(beta_err, beta_sys))
    tensions = {
        key: abs(beta[0] - value) / np.sqrt(beta_err**2 + beta_sys**2 + err**2)
        for key, (value, err) in (("oconnor", LITERATURE_BETA), ("laskar", LASKAR_BETA))
    }
    for key, row in rows.items():
        print(
            f"{key:<18} beta = {row['beta']:.3f} ± {row['beta_err']:.3f} ± {row['beta_sys_ebv']:.3f} (E(B-V)) "
            f"alpha = {row['alpha']:.3f}  chi2/dof = {row['chi2']:.1f}/{row['dof']}"
        )
    print(
        f"Joint optical + NIR, {lo:g}-{hi:g} d (MCMC): beta = {beta[0]:.2f} ± {beta_err:.2f} (stat + calibration) "
        f"± {joint['beta_sys_ebv']:.2f} (E(B-V)) ± {sys_system['joint']:.2f} (photometric system); "
        f"O'Connor et al.: {LITERATURE_BETA[0]} ± {LITERATURE_BETA[1]} ({tensions['oconnor']:.1f}σ), "
        f"Laskar et al.: {LASKAR_BETA[0]} ± {LASKAR_BETA[1]} ({tensions['laskar']:.1f}σ)"
    )
    for test in closure[:3]:
        print(
            f"  closure {test.relation.label:<32} alpha predicted {test.alpha_predicted:.2f} ± "
            f"{test.alpha_predicted_err:.2f} vs {alpha[0]:.2f} ({test.n_sigma:.1f}σ)"
        )

    def clean(row):
        return {k: v for k, v in row.items() if k != "fit"}

    return {
        "window": window,
        "fit": joint["fit"],
        "posterior": posterior,
        "summary": {
            "window_days": list(SED_WINDOW_DAYS),
            "ebv_mw": list(EBV_MW),
            "extinction_law": "Cardelli, Clayton & Mathis (1989), R_V = 3.1, integrated over each filter curve",
            "calibration_prior_mag": CALIBRATION_PRIOR_MAG,
            "systematic_floor": 0.02,
            "filters": {b: round(joint["fit"].spectral.grid(b).effective_nm, 1) for b in window.bands},
            "fits": {k: clean(v) for k, v in rows.items()},
            "alpha": {"median": alpha[0], "lower": alpha[1], "upper": alpha[2]},
            "beta": {
                "median": beta[0],
                "lower": beta[1],
                "upper": beta[2],
                "stat_err": beta_err,
                "sys_err_ebv": joint["beta_sys_ebv"],
                "sys_err_photometric_system": sys_system["joint"],
                "sys_err": beta_sys,
            },
            "liverpool_only_sys_err_photometric_system": sys_system["liverpool"],
            "literature_beta": [
                {
                    "value": LITERATURE_BETA[0],
                    "error": LITERATURE_BETA[1],
                    "source": "O'Connor et al. 2023, Science Advances 9, eadi1405 (optical/IR after 1.7 d)",
                    "tension_sigma": float(tensions["oconnor"]),
                },
                {
                    "value": LASKAR_BETA[0],
                    "error": LASKAR_BETA[1],
                    "source": "Laskar et al. 2023, ApJL 946, L23 (NIR to optical at about 4.4 d)",
                    "tension_sigma": float(tensions["laskar"]),
                },
            ],
            "closure": [
                {"relation": c.relation.label, "p": c.p, "p_err": c.p_err, "n_sigma": c.n_sigma} for c in closure[:3]
            ],
        },
    }


def measure_color_term(data: Path, catalog: Table, manifest: Table) -> tuple[float, float]:
    """The ZTF r colour term against Pan-STARRS1 g - r: median of per-frame fits over the good r frames.

    A colour term is a property of the instrument, so one value, measured from
    every frame's calibration stars, is far better constrained than a fit to the
    few dozen stars of a single frame.
    """
    fitted = PipelineConfig(**SETTINGS, photometry="psf", color_terms={"zr": ("g", "r")})
    slopes = []
    for entry in manifest:
        name = str(entry["file"])
        if entry["filtercode"] != "zr" or name in REJECTED:
            continue
        try:
            slopes.append(reduce(data / "ztf" / name, catalog, "zr", config=fitted).zeropoint.color_term)
        except RuntimeError:
            continue
    slopes = np.asarray(slopes, dtype=float)
    return float(np.median(slopes)), float(1.2533 * np.std(slopes, ddof=1) / np.sqrt(slopes.size))


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

    # The afterglow's spectral index sets the colour term of the ZTF zero points: behind this much dust it is
    # far redder than the Pan-STARRS1 stars that calibrate it.
    sed = measure_spectral_index(args.data)
    target_gr = power_law_color(sed["summary"]["beta"]["median"], "ps1:g", "ps1:r", ebv_mw=EBV_MW[0])
    slope, slope_err = measure_color_term(args.data, catalog, manifest)
    config = PipelineConfig(
        **SETTINGS,
        photometry="psf",
        color_terms={"zr": ("g", "r", slope, slope_err)},
        target_color=target_gr,
    )
    print(
        f"ZTF r colour term {slope:.3f} ± {slope_err:.3f} per mag of Pan-STARRS1 g - r, applied at the afterglow's "
        f"g - r = {target_gr:.2f}"
    )

    rows, frames, references, templates = [], {}, {}, {}
    for band in ("zr", "zg"):
        path = args.data / "ztf" / f"ztf_ref_{band}_refimg.fits"
        templates[band] = Template.from_fits(path, fwhm_guess=config.fwhm_guess)
        ref = reduce(path, catalog, band, config=APERTURE)
        references[band] = ref
        print(
            f"{band} reference: FWHM {templates[band].fwhm:.2f} px; {ref.flux_ujy:6.2f} ± {ref.flux_err_ujy:4.2f} µJy "
            f"in the aperture at the burst position (ZP from {ref.zeropoint.n_stars} stars)"
        )

    for entry in manifest:
        name, band = str(entry["file"]), str(entry["filtercode"])
        status = REJECTED.get(name, "ok")
        try:
            frame = reduce(args.data / "ztf" / name, catalog, band, config=config, template=templates[band])
            simple = reduce(args.data / "ztf" / name, catalog, band, config=APERTURE)
        except RuntimeError as exc:
            print(f"  {name}: skipped ({exc})")
            rows.append((name, band, float(entry["obsmjd"] - TRIGGER_MJD), *([np.nan] * 10), "no zero point"))
            continue
        frames[name] = frame
        ref = references[band]
        net = simple.flux_ujy - ref.flux_ujy
        err = float(np.hypot(simple.flux_err_ujy, ref.flux_err_ujy))
        rows.append(
            (
                name,
                band,
                frame.time / SECONDS_PER_DAY,
                frame.fwhm,
                frame.difference.fwhm,
                frame.zeropoint.value,
                frame.zeropoint.error,
                frame.zeropoint.n_stars,
                frame.flux_ujy,
                frame.flux_err_ujy,
                frame.snr,
                net,
                err,
                status,
            )
        )
        print(
            f"  {name}  {band}  t = {frame.time / SECONDS_PER_DAY:6.2f} d  "
            f"F = {frame.flux_ujy:7.2f} ± {frame.flux_err_ujy:5.2f} µJy ({frame.snr:+5.1f}σ; difference image, PSF)  "
            f"[aperture − reference: {net:7.2f} ± {err:5.2f}]  {status}"
        )

    table = Table(
        rows=rows,
        names=(
            "file",
            "band",
            "time_days",
            "fwhm_px",
            "diff_fwhm_px",
            "zeropoint",
            "zeropoint_err",
            "n_zp_stars",
            "flux_ujy",
            "flux_err_ujy",
            "snr",
            "flux_refsub_aperture_ujy",
            "flux_refsub_aperture_err_ujy",
            "status",
        ),
    )
    for col in table.colnames[2:-1]:
        if col != "n_zp_stars":
            table[col].info.format = ".4g"
    table.write(args.results / "ztf_photometry.csv", format="ascii.csv", overwrite=True)

    good = table[(table["status"] == "ok") & (table["band"] == "zr")]
    times = np.asarray(good["time_days"]) * SECONDS_PER_DAY
    ztf_r = LightCurve(times, good["flux_ujy"], good["flux_err_ujy"], "zr", {"source": "ZTF, this pipeline"})
    ztf_r.save_csv(args.results / "lightcurve_ztf_r.csv")
    simple_r = LightCurve(
        times, good["flux_refsub_aperture_ujy"], good["flux_refsub_aperture_err_ujy"], "zr", {"source": "ZTF, v0.2"}
    )
    lt_r = load_literature(args.data / "laskar2023_liverpool.csv")

    late = ztf_r.subset(ztf_r.time_days >= FIT_TMIN_DAYS)
    fit_ztf = fit_single_power_law(late.time, late.flux, late.flux_err)
    late_simple = simple_r.subset(simple_r.time_days >= FIT_TMIN_DAYS)
    fit_simple = fit_single_power_law(late_simple.time, late_simple.flux, late_simple.flux_err)
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

    fits = (
        ("ZTF r only", fit_ztf),
        ("ZTF r only, v0.2", fit_simple),
        ("Liverpool r only", fit_lt),
        ("ZTF + Liverpool", fit_joint),
    )
    for label, fit in fits:
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
        "method": "reference image subtracted (Bramich kernel), forced PSF photometry on the difference image",
        "reference_fwhm_px": {b: t.fwhm for b, t in templates.items()},
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
            for name, f in (
                ("ztf_r", fit_ztf),
                ("ztf_r_aperture_minus_reference", fit_simple),
                ("liverpool_r", fit_lt),
                ("joint", fit_joint),
            )
        },
        "literature_alpha": {
            "value": LITERATURE_ALPHA[0],
            "error": LITERATURE_ALPHA[1],
            "source": "Laskar et al. 2023, ApJL, r'-band decay after 0.40 d",
        },
        "spectral_index": sed["summary"],
        "color_term": {
            "zr_slope_per_mag_g_minus_r": slope,
            "zr_slope_err": slope_err,
            "afterglow_g_minus_r": target_gr,
        },
    }
    with open(args.results / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    make_figures(args, frames, references, ztf_r, lt_r, simple_r, fit_joint, table, sed)
    return 0


def make_figures(args, frames, references, ztf_r, lt_r, simple_r, fit, table, sed) -> None:
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

    first = frames[r_names[0]]  # the first detection, for the subtraction figure
    first_img, _ = load_fits(first.path)
    d = first.difference
    matched_label = "Science, PSF-matched" if d.psf_from == "template" else "Science"
    template_label = "Reference" if d.psf_from == "template" else "Reference, PSF-matched"

    for theme in ("light", "dark"):
        fig, _ = plot_lightcurve(
            ztf_r,
            fit,
            extra=[
                (lt_r, "Liverpool Telescope r', Laskar et al. 2023"),
                (simple_r, "ZTF r, aperture − reference flux (v0.2)"),
            ],
            theme=theme,
            band_labels={"zr": "ZTF r, difference imaging + PSF fit"},
            model_range=(FIT_TMIN_DAYS, 60.0),
            title="GRB 221009A in public ZTF images",
            subtitle=(
                "Reference image subtracted, PSF photometry · "
                f"power law fitted to t ≥ {FIT_TMIN_DAYS:g} d (ZTF + Liverpool)"
            ),
            annotate=True,
            savepath=args.figures / f"lightcurve_{theme}.png",
        )
        plt.close(fig)
        fig, _ = plot_subtraction(
            d.as_difference_psf(first_img),
            d.matched_template,
            d.data,
            (first.target.x, first.target.y),
            r_aper=first.r_aper * d.fwhm / first.fwhm,
            half=14,
            labels=(matched_label, template_label, "Difference"),
            title=f"Image subtraction at {first.time / SECONDS_PER_DAY:.1f} d (ZTF r, 29 arcsec across)",
            theme=theme,
            savepath=args.figures / f"subtraction_{theme}.png",
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
        beta = sed["summary"]["beta"]
        fig, _ = plot_spectrum(
            sed["window"],
            sed["fit"],
            SED_REFERENCE_DAYS,
            group_labels={
                "lt": "Liverpool IO:O (Laskar et al.)",
                "lmi": "Lowell LMI (O'Connor et al.)",
                "gmos-s": "Gemini GMOS-S",
                "f2": "Gemini Flamingos-2",
            },
            fit_label=f"β = {beta['median']:.2f} ± {beta['stat_err']:.2f} ± {beta['sys_err']:.2f} (sys)",
            title="GRB 221009A: optical to near-infrared spectrum behind the Galactic plane",
            subtitle=(
                f"{SED_WINDOW_DAYS[0]:g} to {SED_WINDOW_DAYS[1]:g} d, scaled to {SED_REFERENCE_DAYS:g} d · "
                f"E(B−V) = {EBV_MW[0]} forward-modelled through each filter curve"
            ),
            theme=theme,
            savepath=args.figures / f"sed_{theme}.png",
        )
        plt.close(fig)
        fig, _ = plot_posterior(
            sed["posterior"],
            ["alpha", "beta"],
            title="Posterior of the decay and spectral indices (optical + NIR)",
            theme=theme,
            savepath=args.figures / f"posterior_{theme}.png",
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
