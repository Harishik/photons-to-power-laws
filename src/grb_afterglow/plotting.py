"""Plotting helpers for light curves and zero-point diagnostics."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .fitting import FitResult, single_power_law
from .lightcurve import LightCurve
from .zeropoint import ZeroPoint


def plot_lightcurve(
    lc: LightCurve,
    fit: FitResult | None = None,
    *,
    title: str = "GRB optical afterglow light curve",
    subtitle: str | None = None,
    source: str | None = None,
    savepath: str | None = None,
):
    """Plot flux density versus time on logarithmic axes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(lc.time, lc.flux, yerr=lc.flux_err, fmt="o", capsize=3, label="photometry")
    if fit is not None and "alpha" in fit.params:
        tt = np.geomspace(lc.time.min(), lc.time.max(), 300)
        ff = single_power_law(tt, fit.params["amplitude"], fit.params["alpha"])
        ax.plot(tt, ff, label=f"fit alpha={fit.params['alpha']:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time since burst (s)")
    ax.set_ylabel("Flux density (microJy)")
    ax.set_title(title if subtitle is None else f"{title}\n{subtitle}")
    if source:
        ax.text(0.01, 0.02, source, transform=ax.transAxes, fontsize=8, alpha=0.7)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=180)
    return fig, ax


def plot_calibration(inst_mags, catalog_mags, zp: ZeroPoint, *, savepath: str | None = None):
    """Plot catalog magnitude versus instrumental magnitude plus zero point."""
    inst = np.asarray(inst_mags, dtype=float)
    cat = np.asarray(catalog_mags, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(inst, cat, s=20)
    x = np.linspace(np.nanmin(inst), np.nanmax(inst), 100)
    ax.plot(x, x + zp.value, label=f"ZP={zp.value:.3f} ± {zp.error:.3f}")
    ax.set_xlabel("Instrumental magnitude")
    ax.set_ylabel("Catalog magnitude")
    ax.set_title("Photometric zero-point solution")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=180)
    return fig, ax
