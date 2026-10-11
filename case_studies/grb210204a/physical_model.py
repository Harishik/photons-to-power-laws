#!/usr/bin/env python3
"""GRB 210204A: a physical blast-wave model fitted to the ZTF light curve (needs afterglowpy).

Kumar et al. (2022) fitted a top-hat jet with afterglowpy to radio, X-ray and
optical data of this burst. This script asks what the ZTF g, r, i light curve
measured by this pipeline says about the same model:

1. Kumar et al.'s published posterior medians, evaluated against the ZTF
   photometry: does their model describe our data?
2. The electron index, the jet opening angle and the energy fitted to ZTF
   alone, with the density, epsilon_B and viewing angle held at Kumar et al.'s
   values, and the posterior sampled with MCMC.
3. The degeneracy that optical data cannot break: energy, opening angle,
   electron index and epsilon_B fitted with the density fixed at Kumar et al.'s
   10^-5.67 cm^-3, at 10^-4 (about 50 times higher) and at 10^-2.

Reads the light curve written by ``analyze.py`` (``results/lightcurve_ztf.csv``).

    pip install "grb-afterglow[models]"
    python case_studies/grb210204a/physical_model.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from grb_afterglow import LightCurve
from grb_afterglow.physical import AfterglowFit, AfterglowModel, fit_afterglow, sample_afterglow
from grb_afterglow.plotting import plot_lightcurve, plot_posterior

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze import EBV_MW, REDSHIFT, ZTF_WAVELENGTHS_NM

FIT_TMAX_DAYS = 8.0  # Kumar et al. fit the optical data before 8 days, before the late flares
# Kumar et al. (2022), Table 2: posterior medians and 68 % intervals of their top-hat jet fit to radio, X-ray and
# optical data, with epsilon_e = 0.1 and xi_N = 1 fixed.
KUMAR_2022 = {
    "theta_obs": (0.010, 0.002, 0.002),
    "log10_E0": (54.06, 0.03, 0.03),
    "theta_core": (0.024, 0.002, 0.003),
    "log10_n0": (-5.67, 0.16, 0.13),
    "p": (2.18, 0.026, 0.026),
    "log10_eps_B": (-0.86, 0.15, 0.17),
}
FIXED = {"log10_eps_e": -1.0, "xi_N": 1.0}
DENSITIES = (-5.67, -4.0, -2.0)  # log10 n0 (cm^-3) for the degeneracy test
# Starting points for the degeneracy fits: physical models have local minima.
STARTS = [
    {"log10_E0": e0, "theta_core": th, "p": p, "log10_eps_B": eb}
    for e0 in (53.0, 54.5)
    for th in (0.02, 0.06)
    for p in (2.2, 2.6)
    for eb in (-3.0, -1.0)
]


def kumar_params() -> dict:
    return {**FIXED, **{k: v[0] for k, v in KUMAR_2022.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lightcurve", type=Path, default=HERE / "results" / "lightcurve_ztf.csv")
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--figures", type=Path, default=HERE / "figures")
    parser.add_argument("--steps", type=int, default=8000, help="MCMC steps per walker")
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)

    lc = LightCurve.read_csv(args.lightcurve)
    window = lc.subset((lc.time > 0) & (lc.time_days <= FIT_TMAX_DAYS))
    model = AfterglowModel(REDSHIFT, "tophat", ebv_mw=EBV_MW, wavelengths_nm=ZTF_WAVELENGTHS_NM)
    print(f"{len(window)} ZTF points before {FIT_TMAX_DAYS:g} d; d_L = {model.luminosity_distance_cm:.4e} cm")

    # 1. Kumar et al.'s model against the ZTF photometry, and the single flux scale that best matches it.
    t, y = window.time, window.flux
    e = np.hypot(window.flux_err, 0.02 * np.abs(y))
    kumar = model.predict_band(t, window.band, kumar_params())
    chi2_kumar = float(np.sum(((y - kumar) / e) ** 2))
    w = kumar**2 / e**2
    scale = float(np.sum(w * y / kumar) / np.sum(w))
    chi2_scaled = float(np.sum(((y - scale * kumar) / e) ** 2))
    print(
        f"Kumar et al. median model: chi2 = {chi2_kumar:.1f} for {len(window)} points; "
        f"best scale {scale:.3f} ({2.5 * np.log10(scale):+.3f} mag), chi2 = {chi2_scaled:.1f}"
    )

    # 2. p, theta_core and E0 from ZTF alone; density, epsilon_B and viewing angle at Kumar et al.'s values.
    held = {**FIXED, **{k: KUMAR_2022[k][0] for k in ("log10_n0", "log10_eps_B", "theta_obs")}}
    free = ["log10_E0", "theta_core", "p"]
    start = {k: KUMAR_2022[k][0] for k in free}
    fit = fit_afterglow(window, model, free, fixed=held, start=start, sys_err=0.02)
    print(fit.summary())
    posterior = sample_afterglow(window, fit, n_walkers=16, n_steps=args.steps, burn=args.steps // 4, sys_err=0.02)
    print(posterior.summary())

    # 3. The degeneracy: the density fixed at very different values, energy, angle, p and epsilon_B free.
    degenerate = []
    for log_n0 in DENSITIES:
        d = fit_afterglow(
            window,
            model,
            ["log10_E0", "theta_core", "p", "log10_eps_B"],
            fixed={**FIXED, "log10_n0": log_n0, "theta_obs": KUMAR_2022["theta_obs"][0]},
            start=[{**start, "log10_eps_B": KUMAR_2022["log10_eps_B"][0]}, *STARTS],
            sys_err=0.02,
        )
        degenerate.append(d)
        print(f"  n0 = 1e{log_n0:g}: " + d.summary())

    def interval(name):
        med, lo, hi = posterior.interval(name)
        return {"median": med, "lower_68": lo, "upper_68": hi}

    summary = {
        "data": {"n_points": len(window), "t_max_days": FIT_TMAX_DAYS, "sys_err": 0.02, "ebv_mw": EBV_MW},
        "model": {
            "code": "afterglowpy",
            "jet": "tophat",
            "redshift": REDSHIFT,
            "luminosity_distance_cm": model.luminosity_distance_cm,
            "cosmology": "Planck18",
        },
        "kumar_2022_median_model": {
            "params": kumar_params(),
            "chi2": chi2_kumar,
            "best_flux_scale": scale,
            "best_flux_scale_mag": 2.5 * np.log10(scale),
            "chi2_scaled": chi2_scaled,
        },
        "fit_with_kumar_density_and_eps_B": {
            "fixed": held,
            "least_squares": {"params": {n: fit.params[n] for n in free}, "errors": fit.errors},
            "chi2": fit.chi2,
            "dof": fit.dof,
            "posterior": {n: interval(n) for n in free},
            "acceptance": posterior.acceptance_fraction,
        },
        "degeneracy": [
            {
                "log10_n0": float(n0),
                "params": {k: d.params[k] for k in d.free},
                "errors": d.errors,
                "chi2": d.chi2,
                "dof": d.dof,
            }
            for n0, d in zip(DENSITIES, degenerate, strict=True)
        ],
        "kumar_2022": {
            "source": "Kumar et al. 2022, MNRAS, arXiv:2204.07587, Table 2",
            "posterior": {k: {"median": v[0], "minus": v[1], "plus": v[2]} for k, v in KUMAR_2022.items()},
            "fixed": FIXED,
        },
    }
    with open(args.results / "physical_model.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    kumar_fit = AfterglowFit(model, kumar_params(), [], {}, chi2_kumar, len(window), np.zeros((0, 0)))
    for theme in ("light", "dark"):
        fig, _ = plot_lightcurve(
            lc.subset(lc.time > 0),
            fit,
            alternative=kumar_fit,
            alternative_label="Kumar et al. (2022) model, r band",
            alternative_band="zr",
            theme=theme,
            band_labels={"zg": "ZTF g", "zr": "ZTF r", "zi": "ZTF i"},
            model_range=(window.time_days.min(), FIT_TMAX_DAYS),
            annotate=False,
            title="GRB 210204A: a top-hat jet fitted to the ZTF light curve",
            subtitle=(
                f"afterglowpy, t ≤ {FIT_TMAX_DAYS:g} d · p = {fit.params['p']:.2f}, "
                f"θ_core = {fit.params['theta_core']:.3f} rad, log E0 = {fit.params['log10_E0']:.2f} "
                f"(n0, ε_B, ε_e, θ_obs as Kumar et al.) · χ²/dof = {fit.chi2:.1f}/{fit.dof}"
            ),
            savepath=args.figures / f"physical_model_{theme}.png",
        )
        plt.close(fig)
        fig, _ = plot_posterior(
            posterior,
            free,
            truths={k: KUMAR_2022[k][0] for k in free},
            truths_label="Kumar et al. (2022)",
            title="Top-hat jet, ZTF alone",
            theme=theme,
            savepath=args.figures / f"physical_posterior_{theme}.png",
        )
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
