"""Power-law fitting for GRB afterglow light curves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


@dataclass
class FitResult:
    model_name: str
    params: dict[str, float]
    errors: dict[str, float]
    chi2: float
    dof: int

    @property
    def reduced_chi2(self) -> float:
        if self.dof <= 0:
            return float("nan")
        return float(self.chi2 / self.dof)

    def summary(self) -> str:
        text = []
        for key, value in self.params.items():
            err = self.errors.get(key, float("nan"))
            text.append(f"{key}={value:.4g}±{err:.2g}")
        return f"{self.model_name}: " + ", ".join(text) + f"; reduced chi2={self.reduced_chi2:.3g}"


def single_power_law(t, amplitude: float, alpha: float, t_ref: float = 1000.0):
    t = np.asarray(t, dtype=float)
    return amplitude * (t / t_ref) ** (-alpha)


def broken_power_law(t, amplitude: float, t_break: float, alpha1: float, alpha2: float, smooth: float = 3.0, t_ref: float = 1000.0):
    t = np.asarray(t, dtype=float)
    x = np.maximum(t / t_break, 1e-12)
    shape = x ** (-alpha1) * (1 + x ** smooth) ** (-(alpha2 - alpha1) / smooth)
    xr = max(t_ref / t_break, 1e-12)
    norm = xr ** (-alpha1) * (1 + xr ** smooth) ** (-(alpha2 - alpha1) / smooth)
    return amplitude * shape / norm


def _clean(t, flux, flux_err):
    t = np.asarray(t, dtype=float)
    flux = np.asarray(flux, dtype=float)
    err = np.asarray(flux_err, dtype=float)
    keep = np.isfinite(t) & np.isfinite(flux) & np.isfinite(err) & (t > 0) & (flux > 0) & (err > 0)
    if keep.sum() < 3:
        raise ValueError("Need at least three finite positive points to fit.")
    return t[keep], flux[keep], err[keep]


def fit_single_power_law(t, flux, flux_err, *, t_ref: float = 1000.0) -> FitResult:
    t, y, e = _clean(t, flux, flux_err)

    def model(tt, amp, alpha):
        return single_power_law(tt, amp, alpha, t_ref=t_ref)

    p0 = [float(np.median(y)), 1.0]
    popt, pcov = curve_fit(model, t, y, sigma=e, p0=p0, absolute_sigma=True, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    resid = (y - model(t, *popt)) / e
    return FitResult(
        "single_power_law",
        {"amplitude": float(popt[0]), "alpha": float(popt[1])},
        {"amplitude": float(perr[0]), "alpha": float(perr[1])},
        float(np.sum(resid ** 2)),
        int(len(t) - len(popt)),
    )


def fit_broken_power_law(t, flux, flux_err, *, smooth: float = 3.0, t_ref: float = 1000.0) -> FitResult:
    t, y, e = _clean(t, flux, flux_err)

    def model(tt, amp, tb, a1, a2):
        return broken_power_law(tt, amp, tb, a1, a2, smooth=smooth, t_ref=t_ref)

    p0 = [float(np.median(y)), float(np.median(t)), 1.0, 2.0]
    popt, pcov = curve_fit(
        model,
        t,
        y,
        sigma=e,
        p0=p0,
        bounds=([0, min(t), 0, 0], [np.inf, max(t), 5, 5]),
        absolute_sigma=True,
        maxfev=30000,
    )
    perr = np.sqrt(np.diag(pcov))
    resid = (y - model(t, *popt)) / e
    keys = ["amplitude", "t_break", "alpha1", "alpha2"]
    return FitResult(
        "broken_power_law",
        {k: float(v) for k, v in zip(keys, popt)},
        {k: float(v) for k, v in zip(keys, perr)},
        float(np.sum(resid ** 2)),
        int(len(t) - len(popt)),
    )
