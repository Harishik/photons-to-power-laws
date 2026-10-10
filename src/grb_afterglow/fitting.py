"""Power-law models and fitting for GRB afterglow light curves.

Afterglows are modelled as

    F(t, band) = A (t / t_ref)^(-alpha) S_band(beta, A_V,host) 10^(-0.4 delta_instrument)

with an optional smooth temporal break (a jet break) where the decay index
steepens from ``alpha1`` to ``alpha2``. ``S_band`` is the spectrum
``F_nu ∝ nu^-beta`` behind Milky Way and (optionally fitted) host-galaxy dust,
integrated over the band's filter curve where one is known
(:mod:`grb_afterglow.spectral`), and ``delta`` is an optional calibration
offset per instrument for light curves that combine several telescopes. Fits
are weighted least squares in flux space, so faint and even negative
forced-photometry points are used correctly; calibration offsets can carry
Gaussian priors.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import svd
from scipy.optimize import least_squares

from .filters import EFFECTIVE_WAVELENGTH_NM, SPEED_OF_LIGHT_M_S, Bandpass, band_frequency_hz, instrument_of
from .lightcurve import LightCurve
from .spectral import SpectralModel

__all__ = [
    "EFFECTIVE_WAVELENGTH_NM",
    "SPEED_OF_LIGHT_M_S",
    "FitResult",
    "ModelComparison",
    "band_frequency_hz",
    "broken_power_law",
    "compare_models",
    "fit_broken_power_law",
    "fit_lightcurve",
    "fit_single_power_law",
    "single_power_law",
]

OFFSET_PREFIX = "offset_"
"""Calibration offsets are reported as parameters named ``offset_<instrument or band>`` (magnitudes)."""


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


def offset_key(band: str, keys) -> str | None:
    """The calibration offset that applies to ``band``: its own name if listed, else its instrument's."""
    band = str(band)
    if band in keys:
        return band
    inst = instrument_of(band)
    return inst if inst is not None and inst in keys else None


@dataclass
class FitResult:
    """Best-fit parameters, uncertainties, and goodness of fit.

    ``chi2`` is the data's chi-squared; prior terms on calibration offsets are
    not included in it. ``spectral`` describes how each band sees the
    spectrum (filter curves, Galactic dust) in multi-band fits, and
    ``priors`` holds the Gaussian prior widths (mag) of calibration offsets.
    """

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
    spectral: SpectralModel | None = field(default=None, repr=False)
    priors: dict[str, float] = field(default_factory=dict)

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

    @property
    def offsets(self) -> dict[str, float]:
        """Fitted calibration offsets (mag) by instrument or band; the model is fainter by this much."""
        return {k[len(OFFSET_PREFIX) :]: v for k, v in self.params.items() if k.startswith(OFFSET_PREFIX)}

    def temporal(self, t) -> np.ndarray:
        """The temporal model alone: the intrinsic flux density at the reference wavelength."""
        p = self.params
        if self.model_name == "broken_power_law":
            return broken_power_law(
                t, p["amplitude"], p["t_break"], p["alpha1"], p["alpha2"], smooth=self.smooth or 3.0, t_ref=self.t_ref
            )
        return single_power_law(t, p["amplitude"], p["alpha"], t_ref=self.t_ref)

    def predict(self, t, nu: float | np.ndarray | None = None) -> np.ndarray:
        """Evaluate the best-fit model at times ``t`` (s) and single frequencies ``nu`` (Hz).

        The spectrum is evaluated at ``nu`` itself, including any dust in the
        model; use :meth:`predict_band` for the flux measured through a filter.
        """
        p = self.params
        flux = self.temporal(t)
        if nu is None or "beta" not in p:
            return np.asarray(flux)
        if self.spectral is not None:
            return np.asarray(flux * self.spectral.monochromatic(nu, p["beta"], p.get("host_av", 0.0)))
        if self.nu_ref:  # fits made before the spectral model existed
            flux = flux * (np.asarray(nu, dtype=float) / self.nu_ref) ** (-p["beta"])
        if "host_av" in p:
            flux = flux * 10 ** (-0.4 * p["host_av"] * _host_curve(np.asarray(nu, dtype=float), self.redshift or 0.0))
        return np.asarray(flux)

    def evaluator(self, bands) -> Callable[[np.ndarray, Mapping[str, float]], np.ndarray]:
        """A fast ``f(t, params)`` of the model for measurements in ``bands`` (one per point, or one for all).

        The band quadrature and offset lookup are precomputed once, which is what
        repeated evaluations (MCMC) need.
        """
        bands = np.atleast_1d(np.asarray(bands, dtype=object)).astype(str)
        unique, index = np.unique(bands, return_inverse=True)
        stack = self.spectral.stack(list(unique)) if self.spectral is not None else None
        offset_names = [OFFSET_PREFIX + k if (k := offset_key(b, self.offsets)) else None for b in unique]
        smooth, t_ref, name = self.smooth or 3.0, self.t_ref, self.model_name

        def evaluate(t, p):
            if name == "broken_power_law":
                flux = broken_power_law(t, p["amplitude"], p["t_break"], p["alpha1"], p["alpha2"], smooth, t_ref)
            else:
                flux = single_power_law(t, p["amplitude"], p["alpha"], t_ref)
            factor = np.ones(unique.size)
            if stack is not None and "beta" in p:
                factor = stack.factors(p["beta"], p.get("host_av", 0.0))
            dm = np.array([p[k] if k else 0.0 for k in offset_names])
            return flux * (factor * 10 ** (-0.4 * dm))[index]

        return evaluate

    def predict_band(self, t, band) -> np.ndarray:
        """The model of measurements at times ``t`` (s) in ``band`` (one name, or one per time).

        This is what the fit compares with the data: the spectrum integrated
        over the band's filter curve (where known), behind the model's dust,
        and shifted by the band's calibration offset.
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        bands = np.broadcast_to(np.asarray(band, dtype=object), t.shape)
        return self.evaluator(bands)(t, self.params)

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


def _clean(t, flux, flux_err, bands=None, sys_err: float = 0.0):
    t = np.asarray(t, dtype=float)
    flux = np.asarray(flux, dtype=float)
    err = np.asarray(flux_err, dtype=float)
    keep = np.isfinite(t) & np.isfinite(flux) & np.isfinite(err) & (t > 0) & (err > 0)
    if keep.sum() < 3:
        raise ValueError("Need at least three finite points with positive time and error to fit.")
    t, flux, err = t[keep], flux[keep], err[keep]
    if bands is not None:
        bands = np.broadcast_to(np.asarray(bands, dtype=object), keep.shape)[keep].astype(str)
    if sys_err > 0:
        err = np.hypot(err, sys_err * np.abs(flux))
    return t, flux, err, bands


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
    residuals: Callable[[np.ndarray], np.ndarray],
    starts: Sequence[Sequence[float]],
    bounds: tuple[Sequence[float], Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Bounded least squares from several starting points; the best solution and its covariance."""
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    best = None
    for p0 in starts:
        p0 = np.clip(np.asarray(p0, float), lo + 1e-9, hi - 1e-9)
        try:
            res = least_squares(residuals, p0, bounds=(lo, hi), method="trf", max_nfev=20000)
        except (RuntimeError, ValueError):
            continue
        if res.success and (best is None or res.cost < best.cost):
            best = res
    if best is None:
        raise RuntimeError("Power-law fit did not converge from any starting point.")
    # Covariance from the Jacobian, as scipy's curve_fit does: a pseudo-inverse of J^T J.
    _, s, vt = svd(best.jac, full_matrices=False)
    keep = s > np.finfo(float).eps * max(best.jac.shape) * s[0]
    s, vt = s[keep], vt[keep]
    return best.x, (vt.T / s**2) @ vt


def _fit(
    model_name: str,
    t,
    flux,
    flux_err,
    bands=None,
    *,
    spectral: SpectralModel | None = None,
    smooth: float = 3.0,
    t_ref: float | None = None,
    sys_err: float = 0.0,
    fit_host: bool = False,
    calibration_offsets: Mapping[str, float | None] | None = None,
) -> FitResult:
    t, y, e, bands = _clean(t, flux, flux_err, bands, sys_err)
    unique, band_index = (None, None) if bands is None else np.unique(bands, return_inverse=True)
    multiband = spectral is not None and unique is not None and unique.size > 1
    if fit_host and not multiband:
        raise ValueError("Host extinction can only be fitted to multi-band data.")
    t_ref = float(t_ref if t_ref is not None else np.exp(np.mean(np.log(t))))
    log_a0, alpha0 = _initial_slope(t, y, e)
    amp0 = max(10**log_a0 * t_ref ** (-alpha0), 1e-12)

    if model_name == "single_power_law":
        keys = ["amplitude", "alpha"]
        lo, hi = [0.0, -3.0], [np.inf, 6.0]
        starts = [[amp0, alpha0], [amp0, 1.0]]

        def temporal(p):
            return single_power_law(t, p[0], p[1], t_ref=t_ref)
    elif model_name == "broken_power_law":
        keys = ["amplitude", "t_break", "alpha1", "alpha2"]
        lo, hi = [0.0, t.min(), -3.0, -3.0], [np.inf, t.max(), 6.0, 6.0]
        starts = [
            [amp0, tb, a1, a1 + da]
            for tb in np.geomspace(t.min(), t.max(), 9)[1:-1]
            for a1 in (max(alpha0 - 0.5, 0.2), alpha0)
            for da in (0.5, 1.2)
        ]

        def temporal(p):
            return broken_power_law(t, p[0], p[1], p[2], p[3], smooth=smooth, t_ref=t_ref)
    else:
        raise ValueError(f"Unknown model {model_name!r}.")

    n_temporal = len(keys)
    stack = None
    if multiband:
        stack = spectral.stack(list(unique))
        keys, lo, hi = [*keys, "beta"], [*lo, -3.0], [*hi, 5.0]
        starts = [[*s, b] for s in starts for b in (0.5, 1.0)]
        if fit_host:
            keys, lo, hi = [*keys, "host_av"], [*lo, 0.0], [*hi, 10.0]
            starts = [[*s, av] for s in starts for av in (0.0, 0.5)]

    # Calibration offsets: one per instrument (or band) named in calibration_offsets that has data.
    priors = dict(calibration_offsets or {})
    point_keys = [offset_key(b, priors) for b in bands] if (priors and bands is not None) else []
    offset_names = [k for k in priors if k in set(point_keys)]
    offset_index = np.array([offset_names.index(k) if k else -1 for k in point_keys], dtype=int)
    sigma = np.array([np.inf if priors[k] is None else float(priors[k]) for k in offset_names])
    if offset_names:
        if np.any(sigma <= 0):
            raise ValueError("Calibration offset priors must be positive (or None for no prior).")
        if np.all(offset_index >= 0) and np.all(np.isinf(sigma)):
            raise ValueError(
                "Every point has a free calibration offset, which is degenerate with the amplitude: "
                "leave the reference instrument out of calibration_offsets or give its offset a prior."
            )
        keys = [*keys, *(OFFSET_PREFIX + k for k in offset_names)]
        lo, hi = [*lo, *([-5.0] * len(offset_names))], [*hi, *([5.0] * len(offset_names))]
        starts = [[*s, *([0.0] * len(offset_names))] for s in starts]
    n_offsets = len(offset_names)
    constrained = np.isfinite(sigma)

    def model(p):
        flux = temporal(p)
        if stack is not None:
            factors = stack.factors(p[n_temporal], p[n_temporal + 1] if fit_host else 0.0)
            flux = flux * factors[band_index]
        if n_offsets:
            dm = np.append(p[len(p) - n_offsets :], 0.0)[offset_index]  # index -1 picks the trailing 0
            flux = flux * 10 ** (-0.4 * dm)
        return flux

    def residuals(p):
        r = (y - model(p)) / e
        if constrained.any():
            r = np.concatenate([r, p[len(p) - n_offsets :][constrained] / sigma[constrained]])
        return r

    popt, pcov = _run_fit(residuals, starts, (lo, hi))
    chi2 = float(np.sum(((y - model(popt)) / e) ** 2))
    with np.errstate(invalid="ignore"):
        perr = np.sqrt(np.abs(np.diag(pcov)))
    return FitResult(
        model_name,
        {k: float(v) for k, v in zip(keys, popt, strict=True)},
        {k: float(v) for k, v in zip(keys, perr, strict=True)},
        chi2,
        int(t.size - len(popt)),
        t_ref,
        smooth if model_name == "broken_power_law" else None,
        spectral.nu_ref if multiband else None,
        np.asarray(pcov),
        spectral.redshift if fit_host else None,
        spectral if multiband else None,
        {k: float(s) for k, s in zip(offset_names, sigma, strict=True) if np.isfinite(s)},
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
    wavelengths_nm: Mapping[str, float] | None = None,
    smooth: float = 3.0,
    t_ref: float | None = None,
    sys_err: float = 0.0,
    host_redshift: float | None = None,
    ebv_mw: float = 0.0,
    r_v: float = 3.1,
    bandpasses: Mapping[str, Bandpass] | bool = True,
    calibration_offsets: Mapping[str, float | None] | None = None,
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
        Band whose wavelength the ``amplitude`` refers to (the intrinsic flux
        density there, before any dust). Defaults to the first band.
    wavelengths_nm
        Effective wavelengths of bands without a filter curve; defaults to
        Pan-STARRS1 values for plain names such as ``"r"``.
    host_redshift
        If given, the host galaxy's dust extinction ``host_av`` (``A_V`` of an
        SMC-like curve, Pei 1992, at this redshift) is fitted too, so ``beta``
        is the intrinsic spectral index.
    ebv_mw, r_v
        Galactic dust (CCM89), applied to the model rather than removed from
        the data. Integrated over each filter curve this is exact, where
        dereddening each band by one number is not. Leave at 0 for light curves
        that were already corrected (:func:`grb_afterglow.extinction.deredden`).
    bandpasses
        True (default) integrates the model over the registry's transmission
        curve for bands named ``"instrument:filter"`` (see
        :func:`grb_afterglow.filters.available_filters`); a mapping supplies
        curves by band name; False evaluates every band at one wavelength.
    calibration_offsets
        Light curves combining several telescopes: ``{instrument: prior}``
        fits a zero-point offset (mag) for each listed instrument (the part of
        ``"instrument:filter"`` band names before the colon, or a full band
        name), with a Gaussian prior of that width in magnitudes, or None for
        no prior. Unlisted instruments define the flux scale. Offsets appear as
        ``offset_<instrument>`` parameters; a positive offset means that
        instrument measures the source fainter than the model.
    """
    name = {"single": "single_power_law", "broken": "broken_power_law"}.get(model, model)
    bands = lc.bands
    if len(bands) == 1:
        if host_redshift is not None:
            raise ValueError("Host extinction can only be fitted to multi-band data.")
        if ebv_mw:
            raise ValueError("Galactic dust is forward-modelled only in multi-band fits; deredden single bands.")
        return _fit(
            name,
            lc.time,
            lc.flux,
            lc.flux_err,
            lc.band,
            smooth=smooth,
            t_ref=t_ref,
            sys_err=sys_err,
            calibration_offsets=calibration_offsets,
        )
    spectral = SpectralModel.for_reference_band(
        str(reference_band or bands[0]),
        ebv_mw=ebv_mw,
        r_v=r_v,
        redshift=host_redshift,
        wavelengths_nm=wavelengths_nm,
        bandpasses=bandpasses,
    )
    return _fit(
        name,
        lc.time,
        lc.flux,
        lc.flux_err,
        lc.band,
        spectral=spectral,
        smooth=smooth,
        t_ref=t_ref,
        sys_err=sys_err,
        fit_host=host_redshift is not None,
        calibration_offsets=calibration_offsets,
    )
