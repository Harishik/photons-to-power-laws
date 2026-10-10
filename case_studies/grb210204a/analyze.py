#!/usr/bin/env python3
"""GRB 210204A: a three-band afterglow with a jet break and late flares, from public ZTF images.

ZTF discovered this afterglow itself, 38 minutes after the Fermi/GBM trigger,
and kept observing the field in g, r and i for a month. The script

1. measures the ZTF colour terms against Pan-STARRS1 once per band, from all
   frames, and applies them at the afterglow's own colour;
2. subtracts the deep ZTF reference image from every epoch and measures the
   afterglow on the difference images with PSF photometry;
3. removes the (small) Galactic extinction and fits a smoothly broken power
   law jointly to g, r and i before the late flares, with a shared spectral
   index, then computes the posterior on a grid of break time and early decay
   index (with a single epoch before the break, MCMC does not converge);
4. compares the break, the decay and the late flaring with Kumar et al.
   (2022), whose light curve combines ZTF with several other telescopes.

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

from grb_afterglow import LightCurve, PipelineConfig, compare_models, evaluate_closure_relations, fit_lightcurve
from grb_afterglow.calibration import load_fits
from grb_afterglow.extinction import deredden, power_law_color
from grb_afterglow.lightcurve import SECONDS_PER_DAY
from grb_afterglow.mcmc import grid_posterior
from grb_afterglow.pipeline import FrameResult, Template, header_time_since_trigger, reduce_frame
from grb_afterglow.plotting import plot_filmstrip, plot_lightcurve, plot_posterior, plot_subtraction

HERE = Path(__file__).resolve().parent
TARGET = SkyCoord(117.08054 * u.deg, 11.40949 * u.deg)  # ZTF21aagwbjr, GCN Circular 29405
TRIGGER_MJD = 59249.27042824  # Fermi/GBM, 2021-02-04 06:29:25 UT
REDSHIFT = 0.876  # Kumar et al. (2022)
EBV_MW = 0.028  # Schlafly & Finkbeiner (2011) at the burst position (IRSA dust service)
FIT_TMAX_DAYS = 7.0  # the power-law fit stops before the flares
# Pivot wavelengths of the ZTF system bandpasses (SVO Filter Profile Service, Palomar/ZTF.g, .r, .i).
ZTF_WAVELENGTHS_NM = {"zg": 478.35, "zr": 641.71, "zi": 786.74}
# Kumar et al. (2022), joint r/R light curve: alpha1 ~ 0.33, t_b = 0.37 +- 0.30 d, alpha2 = 1.18 +- 0.03;
# beta ~ 0.75 from the optical-to-X-ray flux ratio; flares at 7.6e5 s and 1.1e6 s.
LITERATURE = {"t_break_days": (0.37, 0.30), "alpha2": (1.18, 0.03), "beta_ox": 0.75, "flares_days": (8.8, 12.7)}
COLORS = {"zg": ("g", "r"), "zr": ("g", "r"), "zi": ("r", "i")}

SETTINGS = {
    "fwhm_guess": 2.0,
    "aperture_radius_fwhm": 1.0,
    "annulus_fwhm": (4.0, 8.0),
    "min_zp_snr": 10.0,
    "catalog_bands": {"zg": "g", "zr": "r", "zi": "i"},
}


def reduce(path: Path, catalog: Table, band: str, config: PipelineConfig, template: Template | None = None):
    image, header = load_fits(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        wcs = WCS(header)
    return reduce_frame(
        image,
        wcs,
        TARGET,
        catalog,
        band=band,
        time=header_time_since_trigger(header, TRIGGER_MJD),
        exptime=float(header.get("EXPTIME", 1.0)),
        gain=header.get("GAIN"),
        saturation=header.get("SATURATE"),
        config=config,
        template=template,
        path=str(path),
    )


def lightcurve(frames: list[FrameResult]) -> LightCurve:
    return LightCurve(
        [f.time for f in frames],
        [f.flux_ujy for f in frames],
        [f.flux_err_ujy for f in frames],
        np.array([f.band for f in frames], dtype=object),
        {"object": "GRB 210204A", "source": "ZTF, this pipeline"},
    )


def fit_early(lc: LightCurve):
    """Joint g/r/i fits before the flares, after removing the Galactic extinction."""
    corrected = deredden(lc, ebv_mw=EBV_MW, wavelengths_nm=ZTF_WAVELENGTHS_NM)
    early = corrected.subset(corrected.time_days <= FIT_TMAX_DAYS)
    kwargs = {"reference_band": "zr", "wavelengths_nm": ZTF_WAVELENGTHS_NM, "sys_err": 0.02}
    return corrected, early, fit_lightcurve(early, "single", **kwargs), fit_lightcurve(early, "broken", **kwargs)


def break_posterior(early: LightCurve, broken):
    """Posterior of the jet-break fit, integrated on a grid of break time and early decay index.

    The survey saw the afterglow once (0.6 h) before it next visited at 0.94 d. A break between the two leaves
    the early decay index constrained by a single point, so the posterior has a long, thin tail towards early
    breaks whose weight is set by the prior range of alpha1. Ensemble MCMC explores it very slowly (its
    autocorrelation time keeps growing with chain length); a grid integrates it exactly. The alpha1 grid is
    refined around the best fit, where the posterior is narrow; a grid twice as fine gives the same intervals.
    """
    t = early.time[early.time > 0]
    a1 = broken.params["alpha1"]
    coarse, fine = np.arange(-3.0, 6.0 + 1e-9, 0.1), np.arange(a1 - 0.5, a1 + 0.5, 0.01)
    alpha1 = np.unique(np.round(np.concatenate([coarse, fine]), 6))
    grid = {"t_break": np.geomspace(t.min(), t.max(), 60), "alpha1": alpha1}
    return grid_posterior(early, broken, grid, sys_err=0.02, seed=1), {k: int(v.size) for k, v in grid.items()}


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
    paths = {str(r["file"]): (args.data / "ztf" / str(r["file"]), str(r["filtercode"])) for r in manifest}
    templates = {
        band: Template.from_fits(args.data / "ztf" / f"ztf_ref_{band}_refimg.fits", fwhm_guess=SETTINGS["fwhm_guess"])
        for band in COLORS
    }

    # Pass 1: colour terms (median of per-frame fits) and a first light curve for the afterglow's colour.
    fitted = PipelineConfig(**SETTINGS, photometry="psf", color_terms=COLORS)
    first, slopes = [], {band: [] for band in COLORS}
    for path, band in paths.values():
        try:
            frame = reduce(path, catalog, band, fitted, templates[band])
        except RuntimeError as exc:
            print(f"  {path.name}: skipped ({exc})")
            continue
        first.append(frame)
        slopes[band].append(frame.zeropoint.color_term)
    color_terms = {}
    for band, values in slopes.items():
        values = np.asarray(values, dtype=float)
        error = 1.2533 * np.std(values, ddof=1) / np.sqrt(values.size)  # standard error of a median
        color_terms[band] = (*COLORS[band], float(np.median(values)), float(error))
    beta0 = fit_early(lightcurve(first))[3].params["beta"]
    target = {band: power_law_color(beta0, blue, red, ebv_mw=EBV_MW) for band, (blue, red, *_) in color_terms.items()}
    for band, (blue, red, slope, err) in color_terms.items():
        print(f"{band} colour term {slope:+.3f} ± {err:.3f} per mag of {blue} - {red}, applied at {target[band]:.2f}")

    # Pass 2: the final photometry, with fixed colour terms.
    config = PipelineConfig(**SETTINGS, photometry="psf", color_terms=color_terms, target_color=target)
    frames = []
    for path, band in paths.values():
        try:
            frames.append(reduce(path, catalog, band, config, templates[band]))
        except RuntimeError:
            continue
    lc = lightcurve(frames)
    lc.save_csv(args.results / "lightcurve_ztf.csv")
    columns = {
        "file": [Path(f.path).name for f in frames],
        "band": [f.band for f in frames],
        "time_days": [f.time / SECONDS_PER_DAY for f in frames],
        "fwhm_px": [f.fwhm for f in frames],
        "zeropoint": [f.zeropoint.value for f in frames],
        "zeropoint_err": [f.zeropoint.error for f in frames],
        "n_zp_stars": [f.zeropoint.n_stars for f in frames],
        "flux_ujy": [f.flux_ujy for f in frames],
        "flux_err_ujy": [f.flux_err_ujy for f in frames],
        "snr": [f.snr for f in frames],
    }
    names = tuple(columns)
    table = Table(columns)
    table.sort("time_days")
    for col in names[2:]:
        if col != "n_zp_stars":
            table[col].info.format = ".4g"
    table.write(args.results / "ztf_photometry.csv", format="ascii.csv", overwrite=True)
    for f in sorted(frames, key=lambda f: f.time):
        print(
            f"  {Path(f.path).name}  {f.band}  t = {f.time / SECONDS_PER_DAY:6.2f} d  "
            f"F = {f.flux_ujy:8.2f} ± {f.flux_err_ujy:6.2f} µJy ({f.snr:+6.1f}σ)"
        )

    corrected, early, single, broken = fit_early(lc)
    print(compare_models(single, broken).summary())
    posterior, grid_size = break_posterior(early, broken)
    print(posterior.summary())
    a2, beta = posterior.interval("alpha2"), posterior.interval("beta")
    tb = posterior.interval("t_break_days")
    closure = evaluate_closure_relations(a2[0], 0.5 * (a2[2] - a2[1]), beta[0], 0.5 * (beta[2] - beta[1]))
    for test in closure[:3]:
        print(f"  after the break: {test.relation.label:<32} p = {test.p:.2f} ± {test.p_err:.2f} ({test.n_sigma:.1f}σ)")

    # The flares: the ZTF points in the window of the flares Kumar et al. report, against the power law.
    window = corrected.subset((corrected.time_days > 8.0) & (corrected.time_days < 14.0))
    nu = np.array([299_792_458.0 / (ZTF_WAVELENGTHS_NM[str(b)] * 1e-9) for b in window.band])
    resid = window.flux - broken.predict(window.time, nu)
    w = 1.0 / window.flux_err**2
    flare_excess = float(np.sum(w * resid) / np.sum(w))
    flare_excess_err = float(np.sum(w) ** -0.5)
    print(
        f"8-14 d (the flares of Kumar et al.): {len(window)} ZTF points, mean excess over the power law "
        f"{flare_excess:+.1f} ± {flare_excess_err:.1f} µJy ({flare_excess / flare_excess_err:+.1f}σ)"
    )

    summary = {
        "target": "GRB 210204A (ZTF21aagwbjr)",
        "position_deg": [TARGET.ra.deg, TARGET.dec.deg],
        "trigger_mjd": TRIGGER_MJD,
        "redshift": REDSHIFT,
        "ebv_mw": EBV_MW,
        "n_frames": len(frames),
        "n_detections_3sigma": int(np.sum(lc.snr > 3)),
        "color_terms": {b: {"color": f"{c[0]} - {c[1]}", "slope": c[2], "err": c[3]} for b, c in color_terms.items()},
        "afterglow_colors": target,
        "fit_tmax_days": FIT_TMAX_DAYS,
        "least_squares": {
            name: {"params": f.params, "errors": f.errors, "chi2": f.chi2, "dof": f.dof, "bic": f.bic}
            for name, f in (("single", single), ("broken", broken))
        },
        "posterior": {
            name: dict(zip(("median", "lower_68", "upper_68"), posterior.interval(name), strict=True))
            for name in ("t_break_days", "alpha1", "alpha2", "beta")
        },
        "prob_t_break_before_half_day": float(np.mean(posterior["t_break_days"] < 0.5)),
        "posterior_method": {
            "grid": grid_size,
            "note": "grid over t_break (log-uniform prior) and alpha1 (uniform in [-3, 6]); amplitude, alpha2 and "
            "beta marginalised in the Laplace approximation (grid_posterior)",
        },
        "closure_after_break": [
            {"relation": c.relation.label, "p": c.p, "p_err": c.p_err, "n_sigma": c.n_sigma} for c in closure[:3]
        ],
        "flare_window_8_14_days": {
            "n_points": len(window),
            "mean_excess_ujy": flare_excess,
            "mean_excess_err_ujy": flare_excess_err,
        },
        "literature": {
            "source": "Kumar et al. 2022, MNRAS, arXiv:2204.07587",
            "t_break_days": list(LITERATURE["t_break_days"]),
            "alpha2": list(LITERATURE["alpha2"]),
            "beta_optical_xray": LITERATURE["beta_ox"],
            "flares_days": list(LITERATURE["flares_days"]),
        },
    }
    with open(args.results / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(
        f"alpha2 = {a2[0]:.2f} (+{a2[2] - a2[0]:.2f} -{a2[0] - a2[1]:.2f}), Kumar et al.: {LITERATURE['alpha2'][0]} ± "
        f"{LITERATURE['alpha2'][1]}; t_b = {tb[0]:.2f} d (+{tb[2] - tb[0]:.2f} -{tb[0] - tb[1]:.2f}), "
        f"Kumar et al.: {LITERATURE['t_break_days'][0]} ± {LITERATURE['t_break_days'][1]} d; "
        f"beta = {beta[0]:.2f} ± {0.5 * (beta[2] - beta[1]):.2f}"
    )

    make_figures(args, frames, corrected, broken, posterior)
    return 0


def make_figures(args, frames, lc, fit, posterior) -> None:
    r_frames = sorted((f for f in frames if f.band == "zr" and f.time > 0), key=lambda f: f.time)
    first = r_frames[0]
    first_img, _ = load_fits(first.path)
    d = first.difference
    picks = [r_frames[i] for i in np.unique(np.linspace(0, len(r_frames) - 1, 9).round().astype(int))]
    ref_img, _ = load_fits(args.data / "ztf" / "ztf_ref_zr_refimg.fits")
    panels = [{"image": ref_img, "x": 60.0, "y": 60.0, "r_aper": first.r_aper, "label": "reference"}]
    for f in picks:
        img, _ = load_fits(f.path)
        t = f.time / SECONDS_PER_DAY
        label = f"{t * 24:.1f} h" if t < 1 else f"{t:.1f} d"
        panels.append({"image": img, "x": f.target.x, "y": f.target.y, "r_aper": f.r_aper, "label": label})
    ref_wcs = WCS(load_fits(args.data / "ztf" / "ztf_ref_zr_refimg.fits")[1])
    panels[0]["x"], panels[0]["y"] = (float(v) for v in ref_wcs.world_to_pixel(TARGET))
    band_labels = {"zg": "ZTF g", "zr": "ZTF r", "zi": "ZTF i"}
    after = lc.subset(lc.time > 0)  # the pre-burst frames cannot be shown on a logarithmic time axis
    for theme in ("light", "dark"):
        fig, _ = plot_lightcurve(
            after,
            fit,
            band_labels=band_labels,
            model_range=(after.time_days.min(), FIT_TMAX_DAYS),
            wavelengths_nm=ZTF_WAVELENGTHS_NM,
            title="GRB 210204A in public ZTF images",
            subtitle=(
                f"Difference imaging + PSF photometry, corrected for E(B−V) = {EBV_MW} · "
                f"jet-break fit to g, r, i before {FIT_TMAX_DAYS:g} d"
            ),
            theme=theme,
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
            labels=("Science" if d.psf_from == "science" else "Science, PSF-matched", "Reference", "Difference"),
            title=f"Image subtraction {first.time / 3600:.1f} h after the burst (ZTF r, 29 arcsec across)",
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
        fig, _ = plot_posterior(
            posterior,
            derived={"p": posterior.p("jet_slow")},
            title="Posterior of the jet-break fit",
            theme=theme,
            savepath=args.figures / f"posterior_{theme}.png",
        )
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
