#!/usr/bin/env python3
"""GRB 210204A: the jet-break posterior by nested sampling, and the evidence for the break (needs nautilus).

The survey saw this afterglow once, 0.6 h after the burst, before its next
visit at 0.94 d. A break anywhere between the two leaves the early decay
index set by that one point, so the posterior of the break time has a long
tail towards early breaks next to its main peak near 1 d. ``analyze.py``
integrates it on a grid because ensemble MCMC crawls along that tail. Nested
sampling explores it without a grid, and also gives the evidence of each
model, so this script

1. samples the broken power-law posterior with nested sampling and compares
   it with the grid of ``analyze.py``;
2. computes the Bayes factor between the broken and the single power law.

Reads the light curve written by ``analyze.py`` (``results/lightcurve_ztf.csv``).

    pip install "grb-afterglow[nested]"
    python case_studies/grb210204a/nested_break.py            # 10-20 minutes on a laptop
    python case_studies/grb210204a/nested_break.py --quick    # a short run, for testing
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

from grb_afterglow import LightCurve
from grb_afterglow.nested import bayes_factor, nested_posterior
from grb_afterglow.plotting import plot_posterior

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze import FIT_TMAX_DAYS, break_posterior, fit_early

PARAMS = ["t_break_days", "alpha1", "alpha2", "beta"]


def describe(posterior) -> dict:
    out = {}
    for name in PARAMS:
        x = posterior[name]
        out[name] = {
            "median": float(np.median(x)),
            "lower_68": float(np.percentile(x, 15.865)),
            "upper_68": float(np.percentile(x, 84.135)),
            "lower_95": float(np.percentile(x, 2.275)),
            "upper_95": float(np.percentile(x, 97.725)),
        }
    out["prob_t_break_before_half_day"] = float(np.mean(posterior["t_break_days"] < 0.5))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lightcurve", type=Path, default=HERE / "results" / "lightcurve_ztf.csv")
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--figures", type=Path, default=HERE / "figures")
    parser.add_argument("--n-live", type=int, default=2000, help="live points per nested bound")
    parser.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 1), help="worker processes")
    parser.add_argument("--quick", action="store_true", help="few live points and networks, for testing")
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    settings = {"n_live": args.n_live, "n_eff": 10000, "n_networks": 4, "seed": 1, "n_jobs": args.n_jobs}
    if args.quick:
        settings.update(n_live=300, n_eff=2000, n_networks=1)

    lc = LightCurve.read_csv(args.lightcurve)
    _, early, single, broken = fit_early(lc)
    grid, grid_size = break_posterior(early, broken)

    runs, seconds = {}, {}
    for name, fit in (("broken", broken), ("single", single)):
        start = time.perf_counter()
        runs[name] = nested_posterior(early, fit, sys_err=0.02, **settings)
        seconds[name] = time.perf_counter() - start
        print(f"{name}: {runs[name].summary()} ({seconds[name]:.0f} s)")
    ln_b, ln_b_err = bayes_factor(runs["broken"], runs["single"])
    print(f"ln B (broken vs single) = {ln_b:.2f} ± {ln_b_err:.2f}, log10 B = {ln_b / np.log(10):.2f}")

    nested, gridded = describe(runs["broken"]), describe(grid)
    for name in PARAMS:
        n, g = nested[name], gridded[name]
        print(
            f"  {name:13s} nested {n['median']:.3f} [{n['lower_68']:.3f}, {n['upper_68']:.3f}]"
            f"   grid {g['median']:.3f} [{g['lower_68']:.3f}, {g['upper_68']:.3f}]"
        )
    summary = {
        "data": {
            "n_points": int(np.sum(early.time > 0)),
            "sys_err": 0.02,
            "source": f"results/lightcurve_ztf.csv, 0 < t <= {FIT_TMAX_DAYS:g} d, Galactic extinction removed",
        },
        "sampler": {"code": "nautilus", "quick": args.quick, **settings},
        "broken": {
            "nested": nested,
            "grid": {**gridded, "grid_size": grid_size, "method": "grid_posterior, as in analyze.py"},
            "log_evidence": runs["broken"].log_evidence,
            "log_evidence_err": runs["broken"].log_evidence_err,
            "likelihood_calls": runs["broken"].n_calls,
            "seconds": seconds["broken"],
        },
        "single": {
            "log_evidence": runs["single"].log_evidence,
            "log_evidence_err": runs["single"].log_evidence_err,
            "likelihood_calls": runs["single"].n_calls,
            "seconds": seconds["single"],
        },
        "ln_bayes_factor_broken_vs_single": ln_b,
        "ln_bayes_factor_err": ln_b_err,
        "priors": (
            "log10 amplitude uniform within 2 dex of the least-squares value; log10 t_break uniform over the "
            "observed times; alpha1, alpha2 uniform in [-3, 6]; beta uniform in [-3, 5]"
        ),
    }
    with open(args.results / "nested_break.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    for theme in ("light", "dark"):
        fig, _ = plot_posterior(
            runs["broken"],
            PARAMS,
            compare=grid,
            labels=("nested sampling (nautilus)", "grid integration"),
            title="GRB 210204A jet break: nested sampling against the grid",
            theme=theme,
            savepath=args.figures / f"nested_break_{theme}.png",
        )
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
