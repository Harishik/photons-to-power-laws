"""Bayesian posteriors for light-curve fits.

Least squares gives a best fit and a covariance matrix, which describes the
uncertainties well only when the likelihood is close to Gaussian in the
parameters. For a jet break that is often not the case: the break time is
poorly constrained when few points lie on one side of it, its uncertainty is
asymmetric, and it is correlated with both decay indices. Sampling the
posterior with Markov chain Monte Carlo gives the full distribution of every
parameter, of their correlations, and of derived quantities such as the
electron index ``p``.

The sampler is the affine-invariant ensemble of Goodman & Weare (2010) as
implemented in ``emcee`` (Foreman-Mackey et al. 2013). The likelihood is the
same Gaussian, flux-space likelihood that the least-squares fit maximises, so
the two agree where the posterior is Gaussian. Priors are uniform in
``log10 amplitude``, ``log10 t_break`` (within the observed time range), the
decay and spectral indices, and the host ``A_V`` (non-negative).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .closure import CLOSURE_RELATIONS
from .fitting import FitResult, band_frequency_hz
from .lightcurve import SECONDS_PER_DAY, LightCurve

_BOUNDS = {
    "alpha": (-3.0, 6.0),
    "alpha1": (-3.0, 6.0),
    "alpha2": (-3.0, 6.0),
    "beta": (-3.0, 5.0),
    "host_av": (0.0, 10.0),
}


@dataclass
class Posterior:
    """Posterior samples of a light-curve model.

    ``samples`` has one column per name in ``names``, in the parameters' natural
    units (``t_break`` in seconds, ``amplitude`` in the light curve's flux unit).
    """

    fit: FitResult
    names: list[str]
    samples: np.ndarray = field(repr=False)
    acceptance_fraction: float
    autocorr_time: np.ndarray | None = field(default=None, repr=False)

    def __getitem__(self, name: str) -> np.ndarray:
        if name == "t_break_days":
            return self["t_break"] / SECONDS_PER_DAY
        return self.samples[:, self.names.index(name)]

    def interval(self, name: str, level: float = 0.6827) -> tuple[float, float, float]:
        """Median and central credible interval ``(median, lower, upper)``."""
        q = 50.0 * (1.0 - level), 50.0, 50.0 * (1.0 + level)
        lo, med, hi = np.percentile(self[name], q)
        return float(med), float(lo), float(hi)

    def p(self, relation: str) -> np.ndarray:
        """Electron-index samples implied by ``beta`` under a closure relation (e.g. ``"jet_slow"``)."""
        rel = next(r for r in CLOSURE_RELATIONS if r.key == relation)
        return np.vectorize(rel.p_of_beta)(self["beta"])

    def summary(self) -> str:
        head = f"{self.fit.model_name} posterior: {len(self.samples)} samples"
        lines = [f"{head}, acceptance {self.acceptance_fraction:.2f}"]
        for name in self.names:
            key = "t_break_days" if name == "t_break" else name
            med, lo, hi = self.interval(key)
            lines.append(f"  {key:<13} {med:.4g}  -{med - lo:.2g} +{hi - med:.2g}")
        return "\n".join(lines)


def _data(lc: LightCurve, fit: FitResult, sys_err: float, wavelengths_nm):
    t, y, e = (np.asarray(a, dtype=float) for a in (lc.time, lc.flux, lc.flux_err))
    nu = None
    if fit.nu_ref is not None:
        nu = np.array([band_frequency_hz(str(b), wavelengths_nm) for b in lc.band])
    keep = np.isfinite(t) & np.isfinite(y) & np.isfinite(e) & (t > 0) & (e > 0)
    if sys_err > 0:
        e = np.hypot(e, sys_err * np.abs(y))
    return t[keep], y[keep], e[keep], None if nu is None else nu[keep]


def sample_posterior(
    lc: LightCurve,
    fit: FitResult,
    *,
    n_walkers: int = 32,
    n_steps: int = 4000,
    burn: int = 1000,
    thin: int = 5,
    sys_err: float = 0.0,
    wavelengths_nm: dict[str, float] | None = None,
    seed: int = 0,
) -> Posterior:
    """Sample the posterior of the model in ``fit`` given the light curve it was fitted to.

    Parameters
    ----------
    lc
        The light curve (the same data, cuts and bands as the fit).
    fit
        A least-squares :class:`~grb_afterglow.fitting.FitResult`; its model,
        reference time, reference frequency, break smoothness and redshift are
        reused, and the walkers start in a small ball around its best fit.
    n_walkers, n_steps, burn, thin
        Ensemble size, chain length, discarded initial steps, and thinning.
    sys_err
        Fractional systematic error floor, as in the fit.
    wavelengths_nm
        Effective wavelengths per band, as in the fit.
    seed
        Seed of the random number generator, for reproducible chains.
    """
    import emcee

    t, y, e, nu = _data(lc, fit, sys_err, wavelengths_nm)
    names = list(fit.params)
    log_names = {"amplitude", "t_break"}
    t_lo, t_hi = np.log10(t.min()), np.log10(t.max())

    def to_params(theta):
        return {n: (10.0**v if n in log_names else float(v)) for n, v in zip(names, theta, strict=True)}

    def log_prob(theta):
        for n, v in zip(names, theta, strict=True):
            lo, hi = (t_lo, t_hi) if n == "t_break" else _BOUNDS.get(n, (-np.inf, np.inf))
            if not lo <= v <= hi:
                return -np.inf
        model = replace(fit, params=to_params(theta)).predict(t, nu)
        return -0.5 * float(np.sum(((y - model) / e) ** 2))

    start = np.array([np.log10(fit.params[n]) if n in log_names else fit.params[n] for n in names])

    # Initial spread of the walkers: the least-squares errors, in the sampled (log) units.
    def spread(n):
        err = fit.errors.get(n, 0.0)
        return max(err / (fit.params[n] * np.log(10)) if n in log_names else err, 1e-4)

    scale = np.array([spread(n) for n in names])
    rng = np.random.default_rng(seed)
    p0 = start + 0.1 * scale * rng.standard_normal((n_walkers, len(names)))
    if "host_av" in names:
        j = names.index("host_av")
        p0[:, j] = np.abs(p0[:, j])
    if "t_break" in names:
        j = names.index("t_break")
        p0[:, j] = np.clip(p0[:, j], t_lo + 1e-6, t_hi - 1e-6)
    sampler = emcee.EnsembleSampler(n_walkers, len(names), log_prob)
    sampler.random_state = np.random.RandomState(seed).get_state()
    sampler.run_mcmc(p0, n_steps, progress=False)
    chain = sampler.get_chain(discard=burn, thin=thin, flat=True)
    try:
        tau = sampler.get_autocorr_time(discard=burn, quiet=True)
    except emcee.autocorr.AutocorrError:  # pragma: no cover - quiet=True only warns
        tau = None
    samples = chain.copy()
    for j, n in enumerate(names):
        if n in log_names:
            samples[:, j] = 10.0 ** samples[:, j]
    return Posterior(fit, names, samples, float(np.mean(sampler.acceptance_fraction)), tau)
