"""Power-law models and fitting for GRB afterglow light curves.

Afterglows are modelled as

    F_nu(t) = A (t / t_ref)^(-alpha) (nu / nu_ref)^(-beta) [10^(-0.4 A_V,host k(nu))]

with an optional smooth temporal break (a jet break) where the decay index
steepens from ``alpha1`` to ``alpha2``, and optional dust extinction in the host
galaxy (SMC curve ``k`` at the rest-frame wavelength). Fits are weighted least
squares in flux space, so faint and even negative forced-photometry points are
used correctly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit

from .lightcurve import LightCurve

SPEED_OF_LIGHT_M_S = 299_792_458.0

EFFECTIVE_WAVELENGTH_NM: dict[str, float] = {
    "g": 481.0,
    "r": 617.0,
    "i": 752.0,
    "z": 866.0,
    "y": 962.0,
}
"""Pan-STARRS1 effective wavelengths (Tonry et al. 2012), used as band defaults."""


def band_frequency_hz(band: str, wavelengths_nm: dict[str, float] | None = None) -> float:
    """Effective frequency of a band in Hz."""
    table = wavelengths_nm or EFFECTIVE_WAVELENGTH_NM
    key = band if band in table else band[-1:]
    if key not in table:
        raise KeyError(f"No effective wavelength known for band {band!r}.")
    return SPEED_OF_LIGHT_M_S / (table[key] * 1e-9)


def single_power_law(t, amplitude: float, alpha: float, t_ref: float = 1000.0):
    """``amplitude * (t / t_ref) ** -alpha``."""
    t = np.asarray(t, dtype=float)
    return amplitude * (t / t_ref) ** (-alpha)


def broken_power_law(
    t,
    amplitude: float,
    t_break: float,
    alpha1: float,
    alpha2: float,
    smooth: float = 3.0,
    t_ref: float = 1000.0,
):
    """Smoothly broken power law, normalised to ``amplitude`` at ``t_ref``.

    ``F ∝ x^-alpha1 (1 + x^s)^(-(alpha2 - alpha1)/s)`` with ``x = t / t_break``,
    which decays as ``t^-alpha1`` well before the break and ``t^-alpha2`` well
    after it. Larger ``smooth`` (``s``) gives a sharper break.
    """
    t = np.asarray(t, dtype=float)

    def shape(x):
        return x ** (-alpha1) * (1.0 + x**smooth) ** (-(alpha2 - alpha1) / smooth)

    x = np.maximum(t / t_break, 1e-12)
    xr = max(t_ref / t_break, 1e-12)
    return amplitude * shape(x) / shape(xr)


@dataclass
class FitResult:
    """Best-fit parameters, uncertainties, and goodness of fit."""

    model_name: str
    params: dict[str, float]
    errors: dict[str, float]
    chi2: float
    dof: int
    t_ref: float
    smooth: float | None = None
    nu_ref: float | None = None
    covariance: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)), repr=False)
    redshift: float | None = None

    @property
    def n_params(self) -> int:
        return len(self.params)

    @property
    def n_points(self) -> int:
        return self.dof + self.n_params

    @property
    def reduced_chi2(self) -> float:
        return float(self.chi2 / self.dof) if self.dof > 0 else float("nan")

    @property
    def aic(self) -> float:
        """Akaike information criterion, ``chi2 + 2k``."""
        return float(self.chi2 + 2 * self.n_params)

    @property
    def bic(self) -> float:
        """Bayesian information criterion, ``chi2 + k ln n``."""
        return float(self.chi2 + self.n_params * np.log(max(self.n_points, 1)))

    def predict(self, t, nu: float | np.ndarray | None = None) -> np.ndarray:
        """Evaluate the best-fit model at times ``t`` (s) and frequency ``nu`` (Hz)."""
        p = self.params
        if self.model_name == "broken_power_law":
            flux = broken_power_law(
                t,
                p["amplitude"],
                p["t_break"],
                p["alpha1"],
                p["alpha2"],
                smooth=self.smooth or 3.0,
                t_ref=self.t_ref,
            )
        else:
            flux = single_power_law(t, p["amplitude"], p["alpha"], t_ref=self.t_ref)
        if "beta" in p and nu is not None and self.nu_ref:
            flux = flux * (np.asarray(nu, dtype=float) / self.nu_ref) ** (-p["beta"])
        if "host_av" in p and nu is not None:
            flux = flux * 10 ** (-0.4 * p["host_av"] * _host_curve(np.asarray(nu, dtype=float), self.redshift or 0.0))
        return np.asarray(flux)

    def summary(self) -> str:
        parts = []
        for key, value in self.params.items():
            err = self.errors.get(key, float("nan"))
            parts.append(f"{key}={value:.4g}±{err:.2g}")
        return (
            f"{self.model_name}: "
            + ", ".join(parts)
            + f"; chi2/dof={self.chi2:.1f}/{self.dof} ({self.reduced_chi2:.2f}), BIC={self.bic:.1f}"
        )


@dataclass
class ModelComparison:
    """Information-criterion comparison of competing fits to the same data."""

    results: list[FitResult]

    @property
    def best(self) -> FitResult:
        return min(self.results, key=lambda r: r.bic)

    def delta_bic(self, result: FitResult) -> float:
        return result.bic - self.best.bic

    def summary(self) -> str:
        lines = [f"{'model':<20}{'chi2/dof':>14}{'AIC':>10}{'BIC':>10}{'ΔBIC':>8}"]
        for r in sorted(self.results, key=lambda r: r.bic):
            lines.append(
                f"{r.model_name:<20}{r.chi2:>8.1f}/{r.dof:<5d}{r.aic:>10.1f}{r.bic:>10.1f}{self.delta_bic(r):>8.1f}"
            )
        return "\n".join(lines)


def _host_curve(nu_hz: np.ndarray, redshift: float) -> np.ndarray:
    """SMC extinction ``A_lambda / A_V`` at the rest-frame wavelength of observed frequencies."""
    from .extinction import pei92_smc

    return pei92_smc(SPEED_OF_LIGHT_M_S / nu_hz * 1e9 / (1.0 + redshift))


def compare_models(*results: FitResult) -> ModelComparison:
    """Rank fits by BIC. A ΔBIC above about 10 is strong evidence against a model."""
    if not results:
        raise ValueError("Need at least one fit to compare.")
    return ModelComparison(list(results))


def _clean(t, flux, flux_err, nu=None, sys_err: float = 0.0):
    t = np.asarray(t, dtype=float)
    flux = np.asarray(flux, dtype=float)
    err = np.asarray(flux_err, dtype=float)
    nu = np.ones_like(t) if nu is None else np.asarray(nu, dtype=float)
    keep = np.isfinite(t) & np.isfinite(flux) & np.isfinite(err) & np.isfinite(nu) & (t > 0) & (err > 0)
    if keep.sum() < 3:
        raise ValueError("Need at least three finite points with positive time and error to fit.")
    t, flux, err, nu = t[keep], flux[keep], err[keep], nu[keep]
    if sys_err > 0:
        err = np.hypot(err, sys_err * np.abs(flux))
    return t, flux, err, nu


def _initial_slope(t, flux, err) -> tuple[float, float]:
    """Weighted log-log regression on the significant points: (log10 F at t=1, alpha)."""
    good = flux > 2 * err
    if good.sum() < 2:
        good = flux > 0
    if good.sum() < 2:
        return float(np.log10(np.max(np.abs(flux)) + 1e-30)), 1.0
    x, y = np.log10(t[good]), np.log10(flux[good])
    w = (flux[good] / err[good]) ** 2
    slope, intercept = np.polyfit(x, y, 1, w=np.sqrt(w))
    return float(intercept), float(-slope)


def _run_fit(
    model: Callable,
    xdata,
    y: np.ndarray,
    e: np.ndarray,
    starts: Sequence[Sequence[float]],
    bounds: tuple[Sequence[float], Sequence[float]],
):
    best = None
    for p0 in starts:
        p0 = np.clip(np.asarray(p0, float), np.asarray(bounds[0]) + 1e-9, np.asarray(bounds[1]) - 1e-9)
        try:
            popt, pcov = curve_fit(model, xdata, y, p0=p0, sigma=e, absolute_sigma=True, bounds=bounds, maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        chi2 = float(np.sum(((y - model(xdata, *popt)) / e) ** 2))
        if best is None or chi2 < best[2]:
            best = (popt, pcov, chi2)
    if best is None:
        raise RuntimeError("Power-law fit did not converge from any starting point.")
    return best


def _package(name, keys, popt, pcov, chi2, n, t_ref, smooth=None, nu_ref=None, redshift=None) -> FitResult:
    with np.errstate(invalid="ignore"):
        perr = np.sqrt(np.abs(np.diag(pcov)))
    return FitResult(
        name,
        {k: float(v) for k, v in zip(keys, popt, strict=True)},
        {k: float(v) for k, v in zip(keys, perr, strict=True)},
        float(chi2),
        int(n - len(popt)),
        float(t_ref),
        smooth,
        nu_ref,
        np.asarray(pcov),
        redshift,
    )


def _fit(
    model_name: str,
    t,
    flux,
    flux_err,
    nu=None,
    *,
    nu_ref: float | None = None,
    smooth: float = 3.0,
    t_ref: float | None = None,
    sys_err: float = 0.0,
    host_redshift: float | None = None,
) -> FitResult:
    t, y, e, nu = _clean(t, flux, flux_err, nu, sys_err)
    multiband = nu_ref is not None and np.unique(nu).size > 1
    t_ref = float(t_ref if t_ref is not None else np.exp(np.mean(np.log(t))))
    log_a0, alpha0 = _initial_slope(t, y, e)
    amp0 = max(10**log_a0 * t_ref ** (-alpha0), 1e-12)
    nu_norm = nu / nu_ref if multiband else np.ones_like(nu)

    if model_name == "single_power_law":
        keys = ["amplitude", "alpha"]
        lo, hi = [0.0, -3.0], [np.inf, 6.0]
        starts = [[amp0, alpha0], [amp0, 1.0]]

        def temporal(tt, p):
            return single_power_law(tt, p[0], p[1], t_ref=t_ref)
    elif model_name == "broken_power_law":
        keys = ["amplitude", "t_break", "alpha1", "alpha2"]
        lo, hi = [0.0, t.min(), -3.0, -3.0], [np.inf, t.max(), 6.0, 6.0]
        starts = [
            [amp0, tb, a1, a1 + da]
            for tb in np.geomspace(t.min(), t.max(), 9)[1:-1]
            for a1 in (max(alpha0 - 0.5, 0.2), alpha0)
            for da in (0.5, 1.2)
        ]

        def temporal(tt, p):
            return broken_power_law(tt, p[0], p[1], p[2], p[3], smooth=smooth, t_ref=t_ref)
    else:
        raise ValueError(f"Unknown model {model_name!r}.")

    host_k = np.zeros_like(t)
    if host_redshift is not None:
        if not multiband:
            raise ValueError("Host extinction can only be fitted to multi-band data.")
        host_k = _host_curve(nu, host_redshift)
    if multiband and host_redshift is not None:
        keys = [*keys, "beta", "host_av"]
        lo, hi = [*lo, -3.0, 0.0], [*hi, 5.0, 10.0]
        starts = [[*s, b, av] for s in starts for b in (0.5, 1.0) for av in (0.0, 0.5)]

        def model(x, *p):
            return temporal(x[0], p[:-2]) * x[1] ** (-p[-2]) * 10 ** (-0.4 * p[-1] * x[2])
    elif multiband:
        keys = [*keys, "beta"]
        lo, hi = [*lo, -3.0], [*hi, 5.0]
        starts = [[*s, b] for s in starts for b in (0.5, 1.0)]

        def model(x, *p):
            return temporal(x[0], p[:-1]) * x[1] ** (-p[-1])
    else:

        def model(x, *p):
            return temporal(x[0], p)

    xdata = np.vstack([t, nu_norm, host_k])
    popt, pcov, chi2 = _run_fit(model, xdata, y, e, starts, (lo, hi))
    return _package(
        model_name,
        keys,
        popt,
        pcov,
        chi2,
        t.size,
        t_ref,
        smooth if model_name == "broken_power_law" else None,
        nu_ref if multiband else None,
        host_redshift,
    )


def fit_single_power_law(t, flux, flux_err, *, t_ref: float | None = None, sys_err: float = 0.0) -> FitResult:
    """Fit ``F = A (t / t_ref)^-alpha`` to a single-band light curve.

    Parameters
    ----------
    t, flux, flux_err
        Times (s), flux densities, and 1-sigma errors.
    t_ref
        Normalisation time. Defaults to the geometric mean of ``t``, which
        minimises the correlation between ``A`` and ``alpha``.
    sys_err
        Fractional systematic error added in quadrature (e.g. 0.02 for a
        2% calibration floor).
    """
    return _fit("single_power_law", t, flux, flux_err, t_ref=t_ref, sys_err=sys_err)


def fit_broken_power_law(
    t, flux, flux_err, *, smooth: float = 3.0, t_ref: float | None = None, sys_err: float = 0.0
) -> FitResult:
    """Fit a smoothly broken power law (e.g. a jet break) to a single-band light curve.

    The fit is started from a grid of break times to avoid local minima, and
    the solution with the lowest chi-squared is returned.
    """
    return _fit("broken_power_law", t, flux, flux_err, smooth=smooth, t_ref=t_ref, sys_err=sys_err)


def fit_lightcurve(
    lc: LightCurve,
    model: str = "single",
    *,
    reference_band: str | None = None,
    wavelengths_nm: dict[str, float] | None = None,
    smooth: float = 3.0,
    t_ref: float | None = None,
    sys_err: float = 0.0,
    host_redshift: float | None = None,
) -> FitResult:
    """Fit a :class:`LightCurve`, jointly across bands if it has more than one.

    For multi-band data the temporal model is shared (afterglow breaks are
    achromatic) and a spectral index ``beta`` in ``F ∝ nu^-beta`` is fitted
    alongside, so the fit measures both slopes needed for closure relations.

    Parameters
    ----------
    lc
        The light curve.
    model
        ``"single"`` or ``"broken"``.
    reference_band
        Band whose flux ``amplitude`` refers to. Defaults to the first band.
    wavelengths_nm
        Effective wavelengths per band; defaults to Pan-STARRS1 values.
    host_redshift
        If given, the host galaxy's dust extinction ``host_av`` (``A_V`` of an
        SMC-like curve, Pei 1992, at this redshift) is fitted too, so ``beta``
        is the intrinsic spectral index. Milky Way extinction should be
        removed first (:func:`grb_afterglow.extinction.deredden`).
    """
    name = {"single": "single_power_law", "broken": "broken_power_law"}.get(model, model)
    bands = lc.bands
    if len(bands) == 1:
        if host_redshift is not None:
            raise ValueError("Host extinction can only be fitted to multi-band data.")
        return _fit(name, lc.time, lc.flux, lc.flux_err, smooth=smooth, t_ref=t_ref, sys_err=sys_err)
    nu = np.array([band_frequency_hz(str(b), wavelengths_nm) for b in lc.band])
    nu_ref = band_frequency_hz(reference_band or bands[0], wavelengths_nm)
    return _fit(
        name,
        lc.time,
        lc.flux,
        lc.flux_err,
        nu,
        nu_ref=nu_ref,
        smooth=smooth,
        t_ref=t_ref,
        sys_err=sys_err,
        host_redshift=host_redshift,
    )
