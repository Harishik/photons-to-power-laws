#!/usr/bin/env python3
"""GRB 210204A: the optical, X-ray and radio afterglow in one physical jet model (needs afterglowpy and nautilus).

``physical_model.py`` shows that the ZTF light curve alone cannot separate the
energy, the density and the magnetic field of the blast wave: fits with the
density fixed 50 times apart describe it equally well. X-rays and radio waves
see other parts of the synchrotron spectrum, whose fluxes depend on those
parameters in other combinations. This script adds them:

- Swift/XRT: 19 count-rate bins at 1.9-3.5 d, converted to the unabsorbed flux
  density at 3 keV by ``fetch_xray.py``. That conversion's uncertainty
  (absorption, spectral shape) is common to every point, so it enters as one
  calibration offset with a Gaussian prior of 0.1 mag (about 10 %).
- uGMRT: the three radio detections of Kumar et al. (2022) at 1.25 and
  0.65 GHz, 16-33 d after the burst, with a 15 % error floor for the flux
  scale and interstellar scintillation, which their errors may leave out
  (an assumption; the paper does not say).

The model is afterglowpy's top-hat jet, with ``epsilon_e = 0.1`` and
``xi_N = 1`` fixed as in Kumar et al. and self-absorption from Granot & Sari
(2002), which afterglowpy's spectrum leaves out. Six parameters are free:
``E0``, ``theta_core``, ``theta_obs``, ``n0``, ``p`` and ``epsilon_B``, with the
priors of :mod:`grb_afterglow.physical` (``theta_core`` uniform from 0.01 to
pi/2 rad, ``theta_obs`` with density ``sin theta_obs`` up to 0.8 rad).

Two families of jets fit these data: a narrow jet seen almost on its axis, as
Kumar et al. found, and jets 0.3 to 0.8 rad wide (up to the bound of the
``theta_obs`` prior) seen just inside their edge.
Nested sampling gives the posterior and the evidence of each:

1. the narrow family (``theta_core`` up to 0.1 rad), from the optical data
   alone and from all three wavebands: what X-ray and radio data add;
2. all three wavebands under the full prior, which the wide family dominates;
3. the Bayes factor between the two families, from the two evidences.

Reads ``results/lightcurve_ztf.csv`` (from ``analyze.py``), ``data/swift_xrt.csv``
(from ``fetch_xray.py``) and ``data/kumar2022_radio.csv``.

    pip install "grb-afterglow[models,nested]"
    python case_studies/grb210204a/broadband_model.py --n-jobs 12   # about 80 minutes of sampling
    python case_studies/grb210204a/broadband_model.py --quick    # a short, rough run, for testing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

from grb_afterglow import LightCurve
from grb_afterglow.nested import Prior, nested_posterior
from grb_afterglow.physical import PRIORS, AfterglowFit, AfterglowModel, fit_afterglow, floor_fractions
from grb_afterglow.plotting import plot_broadband, plot_posterior

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze import EBV_MW, REDSHIFT, ZTF_WAVELENGTHS_NM
from physical_model import FIT_TMAX_DAYS, FIXED, KUMAR_2022

FREE = ["log10_E0", "theta_core", "theta_obs", "log10_n0", "p", "log10_eps_B"]
SYS_ERR = {"default": 0.02, "xrt": 0.0, "gmrt": 0.15}  # ZTF 2 %, XRT statistical only, uGMRT 15 %
XRT_OFFSET_PRIOR_MAG = 0.1
NARROW_MAX_RAD = 0.1  # the narrow family: theta_core up to 0.1 rad (about 6 degrees)
BAND_LABELS = {
    "xrt:3keV": "Swift/XRT, 3 keV",
    "zg": "ZTF g",
    "zr": "ZTF r",
    "zi": "ZTF i",
    "gmrt:1254.6MHz": "uGMRT, 1.25 GHz",
    "gmrt:647.8MHz": "uGMRT, 0.65 GHz",
}
INSTRUMENTS = {"ZTF": ("zg", "zr", "zi"), "XRT": ("xrt:3keV",), "uGMRT": ("gmrt:1254.6MHz", "gmrt:647.8MHz")}


def read_table(path: Path) -> LightCurve:
    table = Table.read(path, format="ascii.csv", comment="#")
    return LightCurve(
        np.asarray(table["time_s"], dtype=float),
        np.asarray(table["flux_ujy"], dtype=float),
        np.asarray(table["flux_err_ujy"], dtype=float),
        np.asarray(table["band"], dtype=object),
    )


def concatenate(*curves: LightCurve) -> LightCurve:
    return LightCurve(
        np.concatenate([c.time for c in curves]),
        np.concatenate([c.flux for c in curves]),
        np.concatenate([c.flux_err for c in curves]),
        np.concatenate([np.asarray(c.band, dtype=object) for c in curves]),
    )


def load(lightcurve: Path) -> tuple[LightCurve, LightCurve, LightCurve]:
    """(ZTF before FIT_TMAX_DAYS, ZTF after it, ZTF + XRT + uGMRT)."""
    ztf = LightCurve.read_csv(lightcurve)
    ztf = ztf.subset(ztf.time > 0)
    early = ztf.subset(ztf.time_days <= FIT_TMAX_DAYS)
    late = ztf.subset(ztf.time_days > FIT_TMAX_DAYS)
    xrt = read_table(HERE / "data" / "swift_xrt.csv")
    radio = read_table(HERE / "data" / "kumar2022_radio.csv")
    return early, late, concatenate(early, xrt, radio)


def chi2_by_instrument(lc: LightCurve, fit: AfterglowFit) -> dict:
    """Chi-squared of each instrument's points, with the error floors of the fit."""
    e = np.hypot(lc.flux_err, floor_fractions(lc.band, SYS_ERR) * np.abs(lc.flux))
    resid = (lc.flux - fit.predict_band(lc.time, lc.band)) / e
    bands = np.asarray(lc.band, dtype=object).astype(str)
    out = {}
    for name, members in INSTRUMENTS.items():
        keep = np.isin(bands, members)
        if keep.any():
            out[name] = {"chi2": float(np.sum(resid[keep] ** 2)), "n_points": int(keep.sum())}
    return out


def best_fit(lc: LightCurve, first: AfterglowFit, posterior, starts, bounds, rng) -> AfterglowFit:
    """Least squares started from ``starts`` and from the best of a few hundred posterior draws."""
    pick = rng.choice(len(posterior.samples), size=min(200, len(posterior.samples)), replace=False)
    evaluate = first.evaluator(lc.band)
    e = np.hypot(lc.flux_err, floor_fractions(lc.band, SYS_ERR) * np.abs(lc.flux))

    def chi2(row):
        params = dict(zip(posterior.names, row, strict=True))
        return float(np.sum(((lc.flux - evaluate(lc.time, params)) / e) ** 2))

    draws = posterior.samples[pick]
    best = dict(zip(posterior.names, draws[int(np.argmin([chi2(r) for r in draws]))], strict=True))
    return fit_afterglow(
        lc,
        first.model,
        FREE,
        fixed=FIXED,
        start=[best, *starts],
        bounds=bounds,
        sys_err=SYS_ERR,
        calibration_offsets=first.priors or None,
    )


def intervals(posterior) -> dict:
    out = {}
    for name in posterior.names:
        q = np.percentile(posterior[name], [2.275, 15.865, 50.0, 84.135, 97.725])
        out[name] = dict(zip(("lower_95", "lower_68", "median", "upper_68", "upper_95"), map(float, q), strict=True))
    return out


def collimated_energy(params) -> float:
    """Kinetic energy of the two jets, ``E0 (1 - cos theta_core)`` (erg)."""
    return float(10.0 ** params["log10_E0"] * (1.0 - np.cos(params["theta_core"])))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lightcurve", type=Path, default=HERE / "results" / "lightcurve_ztf.csv")
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--figures", type=Path, default=HERE / "figures")
    parser.add_argument("--n-live", type=int, default=2000, help="live points per nested bound")
    parser.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 1), help="worker processes")
    parser.add_argument("--quick", action="store_true", help="a short, rough run for testing")
    parser.add_argument("--verbose", action="store_true", help="print the sampler's progress")
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    settings = {"n_live": args.n_live, "n_eff": 10000, "n_networks": 4, "seed": 1, "n_jobs": args.n_jobs}
    if args.quick:
        settings.update(n_live=200, n_eff=500, n_networks=1, max_calls=3000, discard_exploration=False)

    early, late, broadband = load(args.lightcurve)
    model = AfterglowModel(REDSHIFT, "tophat", ebv_mw=EBV_MW, wavelengths_nm=ZTF_WAVELENGTHS_NM, self_absorption=True)
    counts = {name: int(np.isin(np.asarray(broadband.band).astype(str), b).sum()) for name, b in INSTRUMENTS.items()}
    print(f"{counts} points; d_L = {model.luminosity_distance_cm:.4e} cm")
    kumar = {k: KUMAR_2022[k][0] for k in FREE}
    narrow_bounds = {"theta_core": (PRIORS["theta_core"][0], NARROW_MAX_RAD)}
    offsets = {"xrt": XRT_OFFSET_PRIOR_MAG}
    fits = {
        "optical": fit_afterglow(early, model, FREE, fixed=FIXED, start=kumar, bounds=narrow_bounds, sys_err=SYS_ERR),
        "broadband": fit_afterglow(
            broadband, model, FREE, fixed=FIXED, start=kumar, sys_err=SYS_ERR, calibration_offsets=offsets
        ),
    }
    narrow = {"theta_core": Prior("uniform", PRIORS["theta_core"][0], NARROW_MAX_RAD)}
    runs = {  # name: (light curve, fit, prior overrides)
        "optical_narrow": (early, fits["optical"], narrow),
        "broadband_narrow": (broadband, fits["broadband"], narrow),
        "broadband_full": (broadband, fits["broadband"], None),
    }
    posteriors, minutes = {}, {}
    for name, (lc, fit, priors) in runs.items():
        start = time.perf_counter()
        posteriors[name] = nested_posterior(lc, fit, priors=priors, verbose=args.verbose, **settings)
        minutes[name] = (time.perf_counter() - start) / 60
        print(f"{name}: {posteriors[name].summary()} ({minutes[name]:.1f} min)")

    # The full prior is the narrow and the wide family side by side, so Z_full = f Z_narrow + (1 - f) Z_wide,
    # with f the narrow family's share of the theta_core prior.
    lo, hi = PRIORS["theta_core"]
    share = (NARROW_MAX_RAD - lo) / (hi - lo)
    ln_z_narrow, ln_z_full = posteriors["broadband_narrow"].log_evidence, posteriors["broadband_full"].log_evidence
    prob_narrow = float(share * np.exp(ln_z_narrow - ln_z_full))
    ln_z_wide = ln_z_full + float(np.log((1.0 - prob_narrow) / (1.0 - share))) if prob_narrow < 1 else float("nan")
    ln_b = ln_z_narrow - ln_z_wide
    ln_b_err = float(np.hypot(*(posteriors[k].log_evidence_err for k in ("broadband_narrow", "broadband_full"))))
    print(f"ln Z narrow {ln_z_narrow:.2f}, wide {ln_z_wide:.2f}: ln B (narrow vs wide) = {ln_b:.2f} ± {ln_b_err:.2f}")
    print(f"P(narrow family | data, full prior) = {prob_narrow:.2e}")

    rng = np.random.default_rng(2)
    best = {
        "narrow": best_fit(broadband, fits["broadband"], posteriors["broadband_narrow"], [kumar], narrow_bounds, rng),
        "wide": best_fit(broadband, fits["broadband"], posteriors["broadband_full"], [], None, rng),
    }
    for name, fit in best.items():
        print(f"{name} best fit: {fit.summary()}; E0 (1 - cos theta_core) = {collimated_energy(fit.params):.2e} erg")
    post = posteriors["broadband_narrow"]
    nu_sa = model.self_absorption_frequency({**FIXED, **{n: post[n] for n in FREE}})
    nu_sa_q = np.percentile(nu_sa, [2.5, 50.0, 97.5])
    print(f"nu_sa (narrow family): median {nu_sa_q[1]:.3g} Hz, 95 % range {nu_sa_q[0]:.3g}-{nu_sa_q[2]:.3g} Hz")
    for n in FREE:
        o, b = (intervals(posteriors[k])[n] for k in ("optical_narrow", "broadband_narrow"))
        print(
            f"  {n:12s} optical {o['median']:.3f} [{o['lower_68']:.3f}, {o['upper_68']:.3f}]   "
            f"broadband {b['median']:.3f} [{b['lower_68']:.3f}, {b['upper_68']:.3f}]   Kumar {KUMAR_2022[n][0]}"
        )

    summary = {
        "data": {
            "n_points": counts,
            "ztf_t_max_days": FIT_TMAX_DAYS,
            "sys_err": SYS_ERR,
            "xrt_offset_prior_mag": XRT_OFFSET_PRIOR_MAG,
            "xrt": "data/swift_xrt.csv, unabsorbed flux density at 3 keV (fetch_xray.py)",
            "radio": "data/kumar2022_radio.csv, Kumar et al. 2022 Table A4",
            "ebv_mw": EBV_MW,
        },
        "model": {
            "code": "afterglowpy",
            "jet": "tophat",
            "free": FREE,
            "fixed": FIXED,
            "self_absorption": "Granot & Sari 2002, Table 2, break 1 (multiplicative factor)",
            "redshift": REDSHIFT,
            "luminosity_distance_cm": model.luminosity_distance_cm,
            "priors": {
                **{n: list(PRIORS[n]) for n in FREE},
                "shapes": "uniform; theta_obs with density sin(theta_obs); offset_xrt Gaussian, sigma 0.1 mag",
                "narrow_family_theta_core_max": NARROW_MAX_RAD,
            },
        },
        "sampler": {"code": "nautilus", "quick": args.quick, **settings},
        "posterior": {
            k: {
                **intervals(posteriors[k]),
                "log_evidence": posteriors[k].log_evidence,
                "log_evidence_err": posteriors[k].log_evidence_err,
                "likelihood_calls": posteriors[k].n_calls,
                "minutes": minutes[k],
            }
            for k in posteriors
        },
        "geometry": {
            "narrow_share_of_theta_core_prior": share,
            "log_evidence_narrow": ln_z_narrow,
            "log_evidence_wide": ln_z_wide,
            "ln_bayes_factor_narrow_vs_wide": ln_b,
            "ln_bayes_factor_err": ln_b_err,
            "prob_narrow_under_full_prior": prob_narrow,
        },
        "best_fit": {
            name: {
                "params": {n: fit.params[n] for n in fit.free},
                "errors": fit.errors,
                "chi2": fit.chi2,
                "dof": fit.dof,
                "chi2_by_instrument": chi2_by_instrument(broadband, fit),
                "collimated_energy_erg": collimated_energy(fit.params),
            }
            for name, fit in best.items()
        },
        "self_absorption_frequency_hz_narrow": {
            "median": float(nu_sa_q[1]),
            "lower_95": float(nu_sa_q[0]),
            "upper_95": float(nu_sa_q[2]),
        },
        "kumar_2022": {
            "source": "Kumar et al. 2022, MNRAS 513, 2777, Table 2",
            "posterior": {k: {"median": v[0], "minus": v[1], "plus": v[2]} for k, v in KUMAR_2022.items()},
        },
    }
    with open(args.results / "broadband_model.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    for theme in ("light", "dark"):
        n, w = best["narrow"], best["wide"]
        fig, _ = plot_broadband(
            broadband,
            n,
            alternative=w,
            alternative_label=f"wide jet seen near its edge, χ² = {w.chi2:.1f}",
            unfitted=late,
            band_labels=BAND_LABELS,
            model_range=(early.time_days.min() / 1.5, 60.0),
            title="GRB 210204A from X-rays to radio: one jet, two geometries",
            subtitle=(
                f"afterglowpy top-hat jet with self-absorption · narrow jet: θ_core = {n.params['theta_core']:.3f}, "
                f"log E0 = {n.params['log10_E0']:.2f}, log n0 = {n.params['log10_n0']:.1f}, p = {n.params['p']:.2f}"
                f" · χ²/dof = {n.chi2:.1f}/{n.dof}"
            ),
            theme=theme,
            savepath=args.figures / f"broadband_lightcurve_{theme}.png",
        )
        plt.close(fig)
        fig, _ = plot_posterior(
            posteriors["broadband_narrow"],
            FREE,
            compare=posteriors["optical_narrow"],
            labels=("ZTF + XRT + uGMRT", "ZTF alone"),
            truths=kumar,
            truths_label="Kumar et al. (2022)",
            title="Narrow top-hat jet: what X-ray and radio data add",
            theme=theme,
            savepath=args.figures / f"broadband_posterior_{theme}.png",
        )
        plt.close(fig)
        fig, _ = plot_posterior(
            posteriors["broadband_full"],
            ["theta_core", "theta_obs", "log10_E0", "log10_n0", "p", "log10_eps_B"],
            compare=posteriors["broadband_narrow"],
            labels=("full prior (wide jets dominate)", f"θ_core ≤ {NARROW_MAX_RAD:g} rad"),
            title="Two jet geometries fit the optical, X-ray and radio data",
            theme=theme,
            savepath=args.figures / f"broadband_geometry_{theme}.png",
        )
        plt.close(fig)
    print(f"done in {sum(minutes.values()):.1f} min of sampling")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
