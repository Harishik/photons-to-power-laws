"""Physical afterglow models: the synchrotron emission of a relativistic jet, fitted to a light curve.

Power laws measure slopes. A physical model of the blast wave connects them to
the burst itself: the isotropic-equivalent kinetic energy ``E0`` along the jet axis, the density ``n0`` of the medium
around it, the jet's half-opening angle ``theta_core`` and the observer's
viewing angle ``theta_obs``, the electron index ``p``, and the fractions of
the shock energy in electrons and magnetic field, ``epsilon_e`` and
``epsilon_B``. This module evaluates such models with ``afterglowpy`` (Ryan et
al. 2020, ApJ 896, 166), an optional dependency
(``pip install "grb-afterglow[models]"``), and fits them to a
:class:`~grb_afterglow.lightcurve.LightCurve` by least squares and MCMC.

Priors follow common practice (e.g. Kumar et al. 2022): uniform in
``log10 E0``, ``log10 n0``, ``log10 epsilon_e`` and ``log10 epsilon_B``,
uniform in ``p`` and ``theta_core``, and ``p(theta_obs) ∝ sin(theta_obs)`` for
randomly oriented jets.

A word of caution, demonstrated by the GRB 210204A case study: optical data
alone measure the spectral and temporal slopes and the time of the jet break,
which is three constraints on seven parameters. Energy, density and
``epsilon_B`` then trade off against each other along a ridge, and the
posterior can have separate modes. X-ray and radio data constrain other
combinations of them. Bands named after their frequency (``"xrt:3keV"``,
``"gmrt:1.25GHz"``, see :func:`grb_afterglow.filters.tagged_frequency_hz`) put
such measurements in the same light curve; ``calibration_offsets`` gives an
instrument's common flux-scale uncertainty (an X-ray absorption correction,
say) a prior; ``sys_err`` takes an error floor per instrument; and
``AfterglowModel(self_absorption=True)`` adds the synchrotron self-absorption
that matters at radio frequencies. For posteriors with several modes, sample
with :func:`grb_afterglow.nested.nested_posterior`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .fitting import OFFSET_PREFIX, offset_key
from .lightcurve import LightCurve
from .spectral import SpectralModel

__all__ = [
    "DEFAULTS",
    "PRIORS",
    "AfterglowFit",
    "AfterglowModel",
    "fit_afterglow",
    "floor_fractions",
    "sample_afterglow",
]

# Parameter name -> afterglowpy keyword; names starting with "log10_" are passed as 10 ** value.
_KEYWORDS = {
    "log10_E0": "E0",
    "theta_core": "thetaCore",
    "theta_wing": "thetaWing",
    "theta_obs": "thetaObs",
    "log10_n0": "n0",
    "p": "p",
    "log10_eps_e": "epsilon_e",
    "log10_eps_B": "epsilon_B",
    "xi_N": "xi_N",
}

PRIORS: dict[str, tuple[float, float]] = {
    "log10_E0": (48.0, 56.0),
    "theta_core": (0.01, 0.5 * np.pi),
    "theta_wing": (0.01, 0.5 * np.pi),
    "theta_obs": (0.0, 0.8),
    "log10_n0": (-6.0, 2.0),
    "p": (2.0001, 4.0),
    "log10_eps_e": (-6.0, 0.0),
    "log10_eps_B": (-6.0, 0.0),
    "xi_N": (1e-3, 1.0),
}
"""Default prior bounds of every parameter (uniform within them; ``theta_obs`` is weighted by ``sin``)."""

DEFAULTS: dict[str, float] = {
    "log10_E0": 53.0,
    "theta_core": 0.1,
    "theta_wing": 0.4,
    "theta_obs": 0.0,
    "log10_n0": 0.0,
    "p": 2.3,
    "log10_eps_e": -1.0,
    "log10_eps_B": -3.0,
    "xi_N": 1.0,
}
"""Values of the parameters that are neither fitted nor given (``epsilon_e = 0.1``, ``xi_N = 1``, on axis)."""


def _afterglowpy():
    try:
        import afterglowpy
    except ImportError as exc:  # pragma: no cover - exercised only without the optional dependency
        raise ImportError('Physical afterglow models need afterglowpy: pip install "grb-afterglow[models]"') from exc
    return afterglowpy


@dataclass
class AfterglowModel:
    """A jet seen from Earth: its structure, redshift and distance, and the dust in front of it.

    Parameters
    ----------
    redshift
        Source redshift.
    jet
        ``"tophat"`` (uniform jet with sharp edges) or ``"gaussian"`` (energy
        falling off as a Gaussian in angle, truncated at ``theta_wing``).
    luminosity_distance_cm
        Defaults to the distance at ``redshift`` in the Planck 2018 cosmology.
    ebv_mw, r_v
        Galactic extinction applied to the model at each band's effective
        wavelength (CCM89).
    wavelengths_nm, bandpasses
        How band names map to frequencies, as in
        :func:`~grb_afterglow.fitting.fit_lightcurve`: the model is evaluated at
        each band's pivot (registry bands) or effective wavelength.
    self_absorption
        Multiply the flux by the synchrotron self-absorption factor
        ``(1 - exp(-tau)) / tau``, ``tau = (nu / nu_sa) ** (-5/3)``, which turns
        the ``nu^(1/3)`` spectrum below ``nu_m`` into ``nu^2`` below ``nu_sa``.
        afterglowpy's documented spectra leave self-absorption out, which only
        matters at radio frequencies; ``nu_sa`` is that of Granot & Sari (2002,
        ApJ 568, 820, Table 2, break 1) for slow cooling with ``nu_sa < nu_m``
        in a uniform medium, see :meth:`self_absorption_frequency`.
    """

    redshift: float
    jet: str = "tophat"
    luminosity_distance_cm: float | None = None
    ebv_mw: float = 0.0
    r_v: float = 3.1
    wavelengths_nm: Mapping[str, float] | None = None
    bandpasses: object = True
    self_absorption: bool = False
    _spectral: SpectralModel | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.jet not in ("tophat", "gaussian"):
            raise ValueError("jet must be 'tophat' or 'gaussian'.")
        if self.luminosity_distance_cm is None:
            from astropy.cosmology import Planck18

            self.luminosity_distance_cm = float(Planck18.luminosity_distance(self.redshift).to("cm").value)
        self._spectral = SpectralModel(
            1.0, ebv_mw=self.ebv_mw, r_v=self.r_v, wavelengths_nm=self.wavelengths_nm, bandpasses=self.bandpasses
        )

    def afterglowpy_kwargs(self, params: Mapping[str, float]) -> dict:
        """The keyword arguments of ``afterglowpy.fluxDensity`` for these parameters."""
        grb = _afterglowpy()
        values = {**DEFAULTS, **params}
        kwargs = {
            "jetType": grb.jet.TopHat if self.jet == "tophat" else grb.jet.Gaussian,
            "specType": grb.jet.SimpleSpec,
            "d_L": self.luminosity_distance_cm,
            "z": self.redshift,
        }
        for name, key in _KEYWORDS.items():
            if name == "theta_wing" and self.jet == "tophat":
                continue
            value = float(values[name])
            kwargs[key] = 10.0**value if name.startswith("log10_") else value
        return kwargs

    def flux_density(self, t_s, nu_hz, params: Mapping[str, float]) -> np.ndarray:
        """Flux density in microjansky at observer times ``t_s`` (s) and frequencies ``nu_hz``, before dust."""
        grb = _afterglowpy()
        t, nu = np.broadcast_arrays(np.atleast_1d(np.asarray(t_s, dtype=float)), np.asarray(nu_hz, dtype=float))
        flux = 1e3 * grb.fluxDensity(t, nu, **self.afterglowpy_kwargs(params))
        if self.self_absorption:
            tau = (nu / self.self_absorption_frequency(params)) ** (-5.0 / 3.0)
            flux = flux * np.where(tau > 1e-8, -np.expm1(-tau) / np.maximum(tau, 1e-300), 1.0)
        return flux

    def self_absorption_frequency(self, params: Mapping[str, float]):
        """Observed self-absorption frequency ``nu_sa`` (Hz) of Granot & Sari (2002), Table 2, break 1.

        Parameter values may be arrays (posterior samples), giving an array.

        For slow cooling with ``nu_sa < nu_m`` in a uniform medium,
        ``nu_sa = 1.24e9 (p-1)^(3/5) (3p+2)^(-3/5) (1+z)^-1 epsbar_e^-1
        eps_B^(1/5) n0^(3/5) E52^(1/5)`` Hz, constant in time, with
        ``epsbar_e = eps_e (p-2)/(p-1)``. With only a fraction ``xi_N`` of the
        electrons accelerated, the parameters enter as ``eps_e / xi_N``,
        ``eps_B / xi_N``, ``n0 xi_N`` and ``E0 xi_N`` (the scaling under which
        afterglowpy's flux is unchanged; Eichler & Waxman 2005, ApJ 627, 861), which
        multiplies ``nu_sa`` by ``xi_N^(8/5)``. It is an estimate: afterglowpy's
        cooling frequency is not that of Granot & Sari, and the expression does
        not hold after the jet break or when ``nu_sa > nu_m``.
        """
        v = {k: np.asarray(x, dtype=float) for k, x in {**DEFAULTS, **params}.items()}
        p = v["p"]
        eps_bar = 10.0 ** v["log10_eps_e"] * (p - 2.0) / (p - 1.0)
        nu = (
            1.24e9
            * ((p - 1.0) / (3.0 * p + 2.0)) ** 0.6
            / (1.0 + self.redshift)
            / eps_bar
            * 10.0 ** (0.2 * v["log10_eps_B"] + 0.6 * v["log10_n0"] + 0.2 * (v["log10_E0"] - 52.0))
            * v["xi_N"] ** 1.6
        )
        return float(nu) if nu.ndim == 0 else nu

    def band_wavelength_nm(self, band: str) -> float:
        return self._spectral.grid(str(band)).effective_nm

    def evaluator(self, bands):
        """A fast ``f(t, params)`` for measurements in ``bands``: frequencies and dust precomputed."""
        bands = np.atleast_1d(np.asarray(bands, dtype=object)).astype(str)
        unique, index = np.unique(bands, return_inverse=True)
        lam = np.array([self.band_wavelength_nm(b) for b in unique])
        nu = (299_792_458.0 / (lam * 1e-9))[index]
        dust = (10 ** (-0.4 * self._spectral.galactic_extinction(lam)))[index]

        def evaluate(t, params):
            return self.flux_density(np.broadcast_to(t, nu.shape), nu, params) * dust

        return evaluate

    def predict_band(self, t_s, band, params: Mapping[str, float]) -> np.ndarray:
        """Model flux density (µJy) of measurements in ``band`` at times ``t_s``, behind the Galactic dust."""
        t = np.atleast_1d(np.asarray(t_s, dtype=float))
        return self.evaluator(np.broadcast_to(np.asarray(band, dtype=object), t.shape))(t, params)


@dataclass
class AfterglowFit:
    """A physical model fitted to a light curve.

    ``params`` holds every parameter, fitted or fixed; ``free`` names the fitted
    ones, which have ``errors`` (from the Jacobian, in the parameters' own units).
    """

    model: AfterglowModel
    params: dict[str, float]
    free: list[str]
    errors: dict[str, float]
    chi2: float
    dof: int
    covariance: np.ndarray = field(repr=False)
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict, repr=False)
    priors: dict[str, float] = field(default_factory=dict)
    sys_err: float | Mapping[str, float] = 0.0

    spectral = None  # plotting treats the fit like a power-law fit with its own band model
    nu_ref = None

    @property
    def model_name(self) -> str:
        return f"afterglowpy {self.model.jet} jet"

    @property
    def reduced_chi2(self) -> float:
        return float(self.chi2 / self.dof) if self.dof > 0 else float("nan")

    @property
    def offsets(self) -> dict[str, float]:
        """Calibration offsets (mag) by instrument or band; positive means that instrument measures fainter."""
        return {k[len(OFFSET_PREFIX) :]: v for k, v in self.params.items() if k.startswith(OFFSET_PREFIX)}

    def predict_band(self, t, band) -> np.ndarray:
        """The model of measurements in ``band`` at times ``t`` (s), including the band's calibration offset."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return self.evaluator(np.broadcast_to(np.asarray(band, dtype=object), t.shape))(t, self.params)

    def evaluator(self, bands):
        """A fast ``f(t, params)`` for measurements in ``bands``, with calibration offsets applied."""
        bands = np.atleast_1d(np.asarray(bands, dtype=object)).astype(str)
        model = self.model.evaluator(bands)
        keys = [offset_key(b, self.offsets) for b in bands]
        names = [OFFSET_PREFIX + k if k else None for k in keys]
        fixed = dict(self.params)

        def evaluate(t, params):
            p = {**fixed, **params}
            dm = np.array([p[n] if n else 0.0 for n in names])
            return model(t, p) * 10 ** (-0.4 * dm)

        return evaluate

    def summary(self) -> str:
        parts = [f"{n}={self.params[n]:.4g}±{self.errors.get(n, float('nan')):.2g}" for n in self.free]
        fixed = [f"{n}={v:.4g}" for n, v in self.params.items() if n not in self.free]
        return (
            f"{self.model_name}: " + ", ".join(parts) + f" (fixed: {', '.join(fixed)}); "
            f"chi2/dof={self.chi2:.1f}/{self.dof} ({self.reduced_chi2:.2f})"
        )


def floor_fractions(bands, sys_err: float | Mapping[str, float]) -> np.ndarray:
    """Fractional systematic error per measurement: one value, or by band name, then instrument, then "default"."""
    bands = np.atleast_1d(np.asarray(bands, dtype=object)).astype(str)
    if not isinstance(sys_err, Mapping):
        return np.full(bands.size, float(sys_err))
    default = float(sys_err.get("default", 0.0))

    def lookup(band):
        key = offset_key(band, sys_err)
        return float(sys_err[key]) if key else default

    return np.array([lookup(b) for b in bands])


def _data(lc: LightCurve, sys_err: float | Mapping[str, float]):
    t, y, e = (np.asarray(a, dtype=float) for a in (lc.time, lc.flux, lc.flux_err))
    bands = np.asarray(lc.band, dtype=object).astype(str)
    keep = np.isfinite(t) & np.isfinite(y) & np.isfinite(e) & (t > 0) & (e > 0)
    floor = floor_fractions(bands, sys_err)
    e = np.where(floor > 0, np.hypot(e, floor * np.abs(y)), e)
    return t[keep], y[keep], e[keep], bands[keep]


def _check(model: AfterglowModel, free: Sequence[str], fixed: Mapping[str, float]) -> None:
    allowed = set(_KEYWORDS) - ({"theta_wing"} if model.jet == "tophat" else set())
    unknown = (set(free) | set(fixed)) - allowed
    if unknown:
        raise ValueError(f"Unknown parameters {sorted(unknown)}; known: {sorted(allowed)}.")
    if not free:
        raise ValueError("Nothing to fit: name at least one free parameter.")


def _covariance(residuals, x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Parameter covariance from a central-difference Jacobian with steps matched to the uncertainties.

    afterglowpy integrates numerically, so its output carries noise at the 1e-4 level. A Jacobian from steps
    much smaller than the parameter uncertainties (what least squares uses to find the minimum) is noisy
    enough to shrink the covariance of nearly degenerate parameters several-fold. Two passes: steps of 1e-3 of
    the prior range, then a third of the uncertainty that gives.
    """
    from scipy.linalg import pinvh

    width = hi - lo
    steps = 1e-3 * width
    cov = np.zeros((x.size, x.size))
    for _ in range(2):
        columns = []
        for k in range(x.size):
            up, down = x.copy(), x.copy()
            up[k] = min(x[k] + steps[k], hi[k])
            down[k] = max(x[k] - steps[k], lo[k])
            columns.append((residuals(up) - residuals(down)) / (up[k] - down[k]))
        jac = np.column_stack(columns)
        cov = pinvh(jac.T @ jac)
        steps = np.clip(0.3 * np.sqrt(np.abs(np.diag(cov))), 1e-6 * width, 0.05 * width)
    return cov


def fit_afterglow(
    lc: LightCurve,
    model: AfterglowModel,
    free: Sequence[str],
    *,
    fixed: Mapping[str, float] | None = None,
    start: Mapping[str, float] | Sequence[Mapping[str, float]] | None = None,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    sys_err: float | Mapping[str, float] = 0.0,
    calibration_offsets: Mapping[str, float | None] | None = None,
) -> AfterglowFit:
    """Least-squares fit of a physical afterglow model.

    Parameters
    ----------
    lc
        The light curve, in flux density (µJy), with band names the model can
        map to frequencies.
    model
        The :class:`AfterglowModel`.
    free
        Parameters to fit, from ``log10_E0``, ``theta_core``, ``theta_obs``,
        ``log10_n0``, ``p``, ``log10_eps_e``, ``log10_eps_B``, ``xi_N`` and, for a
        Gaussian jet, ``theta_wing``.
    fixed
        Values of other parameters; the rest take :data:`DEFAULTS`.
    start
        Starting values of free parameters (default: ``fixed``, then
        :data:`DEFAULTS`), or a list of them: the fit is started from each and
        the lowest chi-squared kept. Physical models have local minima, so give
        several starts when the solution may lie far from the first guess.
    bounds
        Bounds overriding :data:`PRIORS`.
    sys_err
        Fractional systematic error added in quadrature: one value, or a
        mapping by band name or instrument (``{"ztf": 0.02, "gmrt": 0.15}``, with
        an optional ``"default"``). Use it for errors independent from point to
        point; a calibration shared by all of an instrument's points belongs in
        ``calibration_offsets``.
    calibration_offsets
        ``{instrument: prior}``: a zero-point offset (mag) for each listed
        instrument (or band), fitted with a Gaussian prior of that width in
        magnitudes, or None for a flat prior within ±1 mag. The offsets are
        parameters ``offset_<instrument>``, as in
        :func:`~grb_afterglow.fitting.fit_lightcurve`; a positive offset means the
        instrument measures the source fainter than the model, as when an X-ray
        flux scale depends on an uncertain absorption correction.
    """
    from scipy.optimize import least_squares

    fixed = dict(fixed or {})
    free = list(free)
    _check(model, free, fixed)
    t, y, e, bands = _data(lc, sys_err)
    priors = dict(calibration_offsets or {})
    used = {offset_key(b, priors) for b in bands} - {None}
    offset_names = [OFFSET_PREFIX + k for k in priors if k in used]
    limits = {**PRIORS, **{n: (-1.0, 1.0) for n in offset_names}, **(bounds or {})}
    free = [*free, *offset_names]
    sigma = {OFFSET_PREFIX + k: float(v) for k, v in priors.items() if v is not None and OFFSET_PREFIX + k in free}
    if any(v <= 0 for v in sigma.values()):
        raise ValueError("Calibration offset priors must be positive (or None for no prior).")
    if t.size <= len(free):
        raise ValueError("More free parameters than data points.")
    fit_shell = AfterglowFit(model, {**DEFAULTS, **fixed, **dict.fromkeys(offset_names, 0.0)}, free, {}, 0.0, 0, None)
    evaluate = fit_shell.evaluator(bands)
    base = {**DEFAULTS, **fixed}
    lo = np.array([limits[n][0] for n in free])
    hi = np.array([limits[n][1] for n in free])
    starts = [start] if start is None or isinstance(start, Mapping) else list(start)
    prior_index = np.array([free.index(n) for n in sigma], dtype=int)
    prior_sigma = np.array(list(sigma.values()))

    def data_residuals(x):
        return (y - evaluate(t, {**base, **dict(zip(free, x, strict=True))})) / e

    def residuals(x):
        r = data_residuals(x)
        return np.concatenate([r, x[prior_index] / prior_sigma]) if prior_index.size else r

    res = None
    for guess in starts:
        x0 = np.array([{**base, **dict.fromkeys(offset_names, 0.0), **(guess or {})}[n] for n in free], dtype=float)
        x0 = np.clip(x0, lo + 1e-6 * (hi - lo), hi - 1e-6 * (hi - lo))
        # afterglowpy integrates numerically: finite differences need steps well above its tolerance.
        trial = least_squares(
            residuals, x0, bounds=(lo, hi), method="trf", jac="3-point", diff_step=1e-4, x_scale="jac", max_nfev=2000
        )
        if res is None or trial.cost < res.cost:
            res = trial
    cov = _covariance(residuals, res.x, lo, hi)
    with np.errstate(invalid="ignore"):
        err = np.sqrt(np.abs(np.diag(cov)))
    params = {**base, **{n: float(v) for n, v in zip(free, res.x, strict=True)}}
    return AfterglowFit(
        model,
        params,
        free,
        {n: float(v) for n, v in zip(free, err, strict=True)},
        float(np.sum(data_residuals(res.x) ** 2)),
        int(t.size - len(free)),
        cov,
        {n: tuple(limits[n]) for n in free},
        {n[len(OFFSET_PREFIX) :]: v for n, v in sigma.items()},
        sys_err,
    )


def sample_afterglow(
    lc: LightCurve,
    fit: AfterglowFit,
    *,
    n_walkers: int = 32,
    n_steps: int = 3000,
    burn: int = 1000,
    thin: int = 5,
    sys_err: float | Mapping[str, float] | None = None,
    seed: int = 0,
):
    """Sample the posterior of a physical model's free parameters with ``emcee``.

    The likelihood is the fit's Gaussian likelihood in flux density (with the
    fit's error floors unless ``sys_err`` is given), the priors are uniform
    within the fit's bounds (in the parameters' own, mostly logarithmic,
    units), ``∝ sin(theta_obs)`` for the viewing angle, and the fit's Gaussian
    priors on calibration offsets. Walkers start around the least-squares
    solution. Returns a :class:`~grb_afterglow.mcmc.Posterior`. For multimodal
    posteriors and the evidence, see :func:`grb_afterglow.nested.nested_posterior`.
    """
    import emcee

    from .mcmc import Posterior

    t, y, e, bands = _data(lc, fit.sys_err if sys_err is None else sys_err)
    evaluate = fit.evaluator(bands)
    names = list(fit.free)
    lo = np.array([fit.bounds[n][0] for n in names])
    hi = np.array([fit.bounds[n][1] for n in names])
    j_obs = names.index("theta_obs") if "theta_obs" in names else None
    gaussian = [(names.index(OFFSET_PREFIX + k), s) for k, s in fit.priors.items()]

    def log_prob(x):
        if np.any(x < lo) or np.any(x > hi):
            return -np.inf
        prior = -0.5 * sum((x[j] / s) ** 2 for j, s in gaussian)
        if j_obs is not None:
            if x[j_obs] <= 0:
                return -np.inf
            prior += float(np.log(np.sin(x[j_obs])))
        model = evaluate(t, dict(zip(names, x, strict=True)))
        if not np.all(np.isfinite(model)):
            return -np.inf
        return prior - 0.5 * float(np.sum(((y - model) / e) ** 2))

    rng = np.random.default_rng(seed)
    center = np.array([fit.params[n] for n in names])
    scale = np.array([max(fit.errors.get(n, 0.0), 1e-3 * (h - lo_)) for n, lo_, h in zip(names, lo, hi, strict=True)])
    scale = np.minimum(scale, 0.05 * (hi - lo))
    p0 = center + scale * rng.standard_normal((n_walkers, len(names)))
    p0 = np.clip(p0, lo + 1e-6 * (hi - lo), hi - 1e-6 * (hi - lo))
    sampler = emcee.EnsembleSampler(n_walkers, len(names), log_prob)
    sampler.random_state = np.random.RandomState(seed).get_state()
    sampler.run_mcmc(p0, n_steps, progress=False)
    chain = sampler.get_chain(discard=burn, thin=thin, flat=True)
    try:
        tau = sampler.get_autocorr_time(discard=burn, quiet=True)
    except emcee.autocorr.AutocorrError:  # pragma: no cover - quiet=True only warns
        tau = None
    return Posterior(fit, names, chain, float(np.mean(sampler.acceptance_fraction)), tau)
