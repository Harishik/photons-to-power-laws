"""End-to-end demonstration on a synthetic campaign with known ground truth."""

from __future__ import annotations

import json
from collections.abc import Callable
from os import PathLike
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from .calibration import build_master_bias, build_master_flat, calibrate_frame, load_fits
from .closure import evaluate_closure_relations
from .detection import estimate_background
from .fitting import compare_models, fit_lightcurve
from .lightcurve import AB_MAG_1UJY, SECONDS_PER_DAY
from .pipeline import FrameResult, PipelineResult, run_pipeline
from .synthetic import generate_dataset


def _recovery_row(name: str, injected: float, value: float, error: float) -> dict:
    n_sigma = abs(value - injected) / error if error > 0 else float("inf")
    return {
        "parameter": name,
        "injected": injected,
        "recovered": value,
        "error": error,
        "n_sigma": n_sigma,
        "pass": bool(n_sigma < 3.0),
    }


def run_demo(
    out_dir: str | PathLike[str] = "p2pl_demo",
    *,
    seed: int = 42,
    figures_dir: str | PathLike[str] | None = None,
    themes: tuple[str, ...] = ("light", "dark"),
    animate: bool = False,
    log: Callable[[str], None] = print,
) -> dict:
    """Simulate, reduce, fit, and validate a synthetic multi-band afterglow.

    Writes ``lightcurve.csv``, ``frames.csv``, and ``summary.json`` to
    ``out_dir`` and figures to ``figures_dir`` (default ``out_dir/figures``).

    Returns
    -------
    dict
        The summary, including injected versus recovered parameters and a
        ``status`` of ``"PASS"`` when every parameter is within 3 sigma.
    """
    out = Path(out_dir)
    data_dir = out / "data"
    fig_dir = Path(figures_dir) if figures_dir else out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    log("[1/6] Simulating a g/r/i follow-up campaign ...")
    truth = generate_dataset(data_dir, seed=seed)
    raw = data_dir / "raw"
    bands = sorted({f["band"] for f in truth["frames"]}, key="gri".index)

    log("[2/6] Building master bias and per-band flats ...")
    master_bias = build_master_bias([load_fits(p)[0] for p in sorted(raw.glob("bias_*.fits"))])
    master_flats = {
        b: build_master_flat([load_fits(p)[0] for p in sorted(raw.glob(f"flat_{b}_*.fits"))], master_bias=master_bias)
        for b in bands
    }

    log("[3/6] Reducing science frames ...")
    catalog = Table.read(data_dir / "catalog.csv", format="ascii.csv")
    target = SkyCoord(truth["afterglow"]["ra"] * u.deg, truth["afterglow"]["dec"] * u.deg)

    def report(f: FrameResult) -> None:
        log(
            f"      {f.band}  t = {f.time / SECONDS_PER_DAY:7.4f} d  FWHM {f.fwhm:4.2f} px  "
            f"ZP {f.zeropoint.value:6.3f} ({f.zeropoint.n_stars:2d} stars)  "
            f"F = {f.flux_ujy:9.2f} ± {f.flux_err_ujy:6.2f} µJy"
        )

    result = run_pipeline(
        sorted(raw.glob("sci_*.fits")),
        target,
        catalog,
        master_bias=master_bias,
        master_flats=master_flats,
        on_frame=report,
    )
    lc = result.lightcurve
    lc.meta["object"] = "SYNTHETIC-GRB"
    lc.save_csv(out / "lightcurve.csv")
    result.summary_table().write(out / "frames.csv", format="ascii.csv", overwrite=True)

    log("[4/6] Fitting single and broken power laws jointly across bands ...")
    single = fit_lightcurve(lc, "single", reference_band="r")
    broken = fit_lightcurve(lc, "broken", reference_band="r")
    comparison = compare_models(single, broken)
    log(comparison.summary())
    best = comparison.best

    log("[5/6] Testing closure relations ...")
    p, e = broken.params, broken.errors
    pre = evaluate_closure_relations(p["alpha1"], e["alpha1"], p["beta"], e["beta"])
    post = evaluate_closure_relations(p["alpha2"], e["alpha2"], p["beta"], e["beta"])
    for when, best_rel in (("before break", pre[0]), ("after break", post[0])):
        log(
            f"      {when + ':':<14}{best_rel.relation.label}  "
            f"(p = {best_rel.p:.2f} ± {best_rel.p_err:.2f}, {best_rel.n_sigma:.1f}σ)"
        )

    ag = truth["afterglow"]
    rows = [
        _recovery_row("alpha1", ag["alpha1"], p["alpha1"], e["alpha1"]),
        _recovery_row("alpha2", ag["alpha2"], p["alpha2"], e["alpha2"]),
        _recovery_row("beta", ag["beta"], p["beta"], e["beta"]),
        _recovery_row(
            "t_break_days",
            ag["t_break_s"] / SECONDS_PER_DAY,
            p["t_break"] / SECONDS_PER_DAY,
            e["t_break"] / SECONDS_PER_DAY,
        ),
        _recovery_row("p", ag["p"], post[0].p, post[0].p_err),
    ]
    pulls = np.array(
        [
            (f.flux_ujy - fr["afterglow_flux_ujy"]) / f.flux_err_ujy
            for f, fr in zip(result.frames, _truth_for(result, truth), strict=True)
        ]
    )
    status = (
        "PASS"
        if (
            all(r["pass"] for r in rows)
            and best is broken
            and pre[0].relation.key == "ism_slow"
            and post[0].relation.key == "jet_slow"
        )
        else "CHECK"
    )
    summary = {
        "status": status,
        "seed": seed,
        "n_frames": len(result.frames),
        "bands": bands,
        "recovery": rows,
        "flux_pulls": {"mean": float(np.mean(pulls)), "std": float(np.std(pulls))},
        "model_comparison": {
            r.model_name: {"chi2": r.chi2, "dof": r.dof, "aic": r.aic, "bic": r.bic} for r in comparison.results
        },
        "preferred_model": best.model_name,
        "closure": {
            "pre_break": {
                "relation": pre[0].relation.label,
                "p": pre[0].p,
                "p_err": pre[0].p_err,
                "n_sigma": pre[0].n_sigma,
            },
            "post_break": {
                "relation": post[0].relation.label,
                "p": post[0].p,
                "p_err": post[0].p_err,
                "n_sigma": post[0].n_sigma,
            },
        },
        "injected_regime": ag["regime"],
    }
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    log("[6/6] Drawing figures ...")
    make_demo_figures(result, truth, broken, single, fig_dir, data_dir, master_bias, master_flats, themes=themes)
    if animate:
        _make_animation(result, broken, fig_dir / "hero.gif", master_bias, master_flats)
    for r in rows:
        log(
            f"      {r['parameter']:<13} injected {r['injected']:7.3f}  "
            f"recovered {r['recovered']:7.3f} ± {r['error']:.3f}  ({r['n_sigma']:.1f}σ)"
        )
    log(f"Validation: {status}. Outputs in {out}/, figures in {fig_dir}/")
    return summary


def _truth_for(result: PipelineResult, truth: dict) -> list[dict]:
    by_name = {f["file"]: f for f in truth["frames"]}
    return [by_name[Path(f.path).name] for f in result.frames]


def _calibrated(path: str, master_bias, master_flats, band):
    raw, _ = load_fits(path)
    return raw, calibrate_frame(raw, master_bias=master_bias, master_flat=master_flats[band])


def make_demo_figures(
    result, truth, broken, single, fig_dir, data_dir, master_bias, master_flats, *, themes=("light", "dark")
):
    """Write the README figures for the synthetic demo."""
    from .plotting import plot_closure_relations, plot_filmstrip, plot_lightcurve, plot_pipeline_stages, plot_zeropoints

    lc = result.lightcurve
    p, e = broken.params, broken.errors
    best_post = evaluate_closure_relations(p["alpha2"], e["alpha2"], p["beta"], e["beta"])[0]
    post_p = (best_post.p, best_post.p_err)
    truth_zp = {f["file"]: f["zeropoint"] for f in truth["frames"]}
    r_frames = [f for f in result.frames if f.band == "r"]
    stage = r_frames[len(r_frames) // 2]
    raw, cal = _calibrated(stage.path, master_bias, master_flats, "r")
    picks = np.unique(np.linspace(0, len(r_frames) - 1, 8).round().astype(int))
    film = []
    for f in (r_frames[i] for i in picks):
        _, img = _calibrated(f.path, master_bias, master_flats, "r")
        t = f.time / SECONDS_PER_DAY
        film.append(
            {
                "image": img,
                "x": f.target.x,
                "y": f.target.y,
                "r_aper": f.r_aper,
                "label": f"{t * 24:.1f} h" if t < 1 else f"{t:.1f} d",
            }
        )

    for theme in themes:
        sfx = f"_{theme}"
        fig, _ = plot_lightcurve(
            lc,
            broken,
            alternative=single,
            theme=theme,
            title="Synthetic afterglow: jet break recovered from raw pixels",
            subtitle=(
                f"{len(result.frames)} frames in g, r, i · joint fit with shared temporal decay and spectral index β"
            ),
            savepath=fig_dir / f"lightcurve{sfx}.png",
        )
        plt.close(fig)
        fig, _ = plot_zeropoints(
            result.frames, truth_zeropoints=truth_zp, theme=theme, savepath=fig_dir / f"zeropoints{sfx}.png"
        )
        plt.close(fig)
        fig, _ = plot_closure_relations(
            [
                ("before break", p["alpha1"], e["alpha1"], p["beta"], e["beta"]),
                ("after break", p["alpha2"], e["alpha2"], p["beta"], e["beta"]),
            ],
            note=f"inferred p = {post_p[0]:.2f} ± {post_p[1]:.2f}  (injected {truth['afterglow']['p']:.1f})",
            theme=theme,
            savepath=fig_dir / f"closure{sfx}.png",
        )
        plt.close(fig)
        fig, _ = plot_pipeline_stages(
            raw,
            cal,
            stage.sources_xy,
            stage.matched_xy,
            (stage.target.x, stage.target.y),
            stage.r_aper,
            stage.annulus,
            theme=theme,
            savepath=fig_dir / f"pipeline_stages{sfx}.png",
        )
        plt.close(fig)
        fig, _ = plot_filmstrip(
            film,
            half=int(4 * stage.fwhm + 4),
            theme=theme,
            title="The afterglow fading in r band (aperture sized to each frame's seeing)",
            savepath=fig_dir / f"filmstrip{sfx}.png",
        )
        plt.close(fig)


def _make_animation(result: PipelineResult, fit, savepath: Path, master_bias, master_flats) -> None:
    from .animation import animate_afterglow

    r_frames = [f for f in result.frames if f.band == "r"]
    half = 22
    cutouts, times = [], []
    for f in r_frames:
        _, img = _calibrated(f.path, master_bias, master_flats, "r")
        img = img - estimate_background(img).background
        scale = 10 ** (-0.4 * (f.zeropoint.value - AB_MAG_1UJY)) / f.exptime
        ix, iy = int(np.rint(f.target.x)), int(np.rint(f.target.y))
        cutouts.append(img[iy - half : iy + half + 1, ix - half : ix + half + 1] * scale)
        times.append(f.time / SECONDS_PER_DAY)
    r_aper = float(np.median([f.r_aper for f in r_frames])) * 1.4
    animate_afterglow(cutouts, times, result.lightcurve, fit, savepath, r_aper=r_aper)
