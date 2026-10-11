"""Nested sampling: posteriors that MCMC cannot explore, and the Bayesian evidence.

Ensemble MCMC moves walkers through the posterior by small steps, so it can
stay in one mode of a multimodal posterior, or crawl along a narrow funnel,
without any sign of trouble other than a growing autocorrelation time.
Nested sampling (Skilling 2004) instead shrinks a population of live points
from the whole prior towards the likelihood peaks, so separate modes are found
and weighted by their mass. It also returns the evidence ``Z``, the integral
of likelihood times prior, whose ratio between two models (the Bayes factor)
compares them with an automatic penalty for complexity.

The sampler is ``nautilus`` (Lange 2023, MNRAS 525, 3181), an optional
dependency (``pip install "grb-afterglow[nested]"``). It learns the shape of
the likelihood with small neural networks and importance-samples within the
nested bounds it draws from them, so it needs far fewer likelihood calls than
samplers that step through the prior, which matters for the physical models
(milliseconds per call). On the funnel-shaped jet-break posterior of
GRB 210204A it reproduced an independent importance-sampling evidence to 0.01
in ``ln Z``. Of the alternatives tried there, ``dynesty``'s default sampler
had not finished after five hours, ``UltraNest``'s needed about 14 times more
likelihood calls, and their faster settings gave break times that moved with
the random seed.

Priors are proper, as the evidence requires, and match those of
:mod:`grb_afterglow.mcmc` and :func:`grb_afterglow.physical.sample_afterglow`
where those are proper.

Power-law fits:

- uniform in ``log10`` amplitude within ``AMPLITUDE_DEX`` of the best fit;
- uniform in ``log10`` break time over the observed time range;
- uniform decay and spectral indices, and a uniform host ``A_V``, within the
  bounds of :mod:`grb_afterglow.mcmc`;
- Gaussian calibration offsets with the fit's prior widths, and uniform within
  the ``±5`` mag of :mod:`grb_afterglow.mcmc` for offsets without one.

Physical fits: uniform within the fit's bounds, ``p(theta_obs) ∝ sin
theta_obs``, and the fit's Gaussian priors on calibration offsets, truncated
to the fit's bounds as in :func:`~grb_afterglow.physical.sample_afterglow`.
Any prior can be replaced through ``priors``.

The likelihood is Gaussian in flux density, with its normalisation, so the
evidence of the same model with different error floors can be compared too.
The evidence depends on the prior volume: Bayes factors between models that
share parameters with the same priors (a single and a broken power law, say)
are meaningful; absolute values are not.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, ndtri

from .fitting import OFFSET_PREFIX, FitResult
from .lightcurve import LightCurve
from .mcmc import _BOUNDS, _OFFSET_BOUNDS, Posterior, _sampled

__all__ = ["AMPLITUDE_DEX", "Prior", "bayes_factor", "nested_posterior"]

AMPLITUDE_DEX = 2.0
"""Half-width, in dex, of the uniform prior on ``log10`` amplitude of a power-law fit."""

_EXCLUDED = -1e300  # log-likelihood of a parameter set the model cannot evaluate


@dataclass(frozen=True)
class Prior:
    """A one-dimensional prior, defined by its transform from the unit interval.

    ``kind`` is ``"uniform"`` on ``[a, b]``, ``"normal"`` with mean ``a`` and
    standard deviation ``b`` (truncated to ``[lower, upper]`` if those are
    given), or ``"sine"`` on ``[a, b]`` (density ``∝ sin x``, for a viewing
    angle).
    """

    kind: str
    a: float
    b: float
    lower: float = -np.inf
    upper: float = np.inf

    def __post_init__(self) -> None:
        if self.kind not in ("uniform", "normal", "sine"):
            raise ValueError(f"Unknown prior kind {self.kind!r}.")
        if self.kind != "normal" and not self.b > self.a:
            raise ValueError("A uniform or sine prior needs a < b.")
        if self.kind == "normal" and not self.b > 0:
            raise ValueError("A normal prior needs a positive standard deviation.")
        if not self.upper > self.lower:
            raise ValueError("A truncated prior needs lower < upper.")

    def transform(self, u: float) -> float:
        if self.kind == "uniform":
            return self.a + u * (self.b - self.a)
        if self.kind == "normal":
            lo, hi = ndtr((self.lower - self.a) / self.b), ndtr((self.upper - self.a) / self.b)
            q = min(max(lo + u * (hi - lo), 1e-300), 1.0 - 1e-16)
            return float(np.clip(self.a + self.b * ndtri(q), self.lower, self.upper))
        ca, cb = np.cos(self.a), np.cos(self.b)
        return float(np.arccos(ca - u * (ca - cb)))


class _Likelihood:
    """Gaussian log-likelihood of a fit's model, picklable for worker processes.

    The model evaluator is a closure, so it is built lazily in each process
    rather than pickled.
    """

    def __init__(self, fit, data, names, log_names):
        self.fit, self.data, self.names, self.log_names = fit, data, names, log_names
        self._evaluate = None

    def __getstate__(self):
        return {**self.__dict__, "_evaluate": None}

    def __call__(self, theta) -> float:
        t, y, e, bands = self.data
        if self._evaluate is None:
            self._evaluate = self.fit.evaluator(bands)
        params = {n: (10.0**v if n in self.log_names else float(v)) for n, v in zip(self.names, theta, strict=True)}
        model = self._evaluate(t, params)
        if not np.all(np.isfinite(model)):
            return _EXCLUDED
        return -0.5 * float(np.sum(((y - model) / e) ** 2))


def _power_law_problem(lc: LightCurve, fit: FitResult, sys_err):
    from .mcmc import _data
    from .physical import floor_fractions

    t, y, e, bands = _data(lc, 0.0)
    floor = floor_fractions(bands, sys_err)
    data = (t, y, np.where(floor > 0, np.hypot(e, floor * np.abs(y)), e), bands)
    names = list(fit.params)
    log_t = np.log10(data[0])

    def default(name):
        if name == "amplitude":
            centre = np.log10(fit.params["amplitude"])
            return Prior("uniform", centre - AMPLITUDE_DEX, centre + AMPLITUDE_DEX)
        if name == "t_break":
            return Prior("uniform", float(log_t.min()), float(log_t.max()))
        if name.startswith(OFFSET_PREFIX):
            sigma = fit.priors.get(name[len(OFFSET_PREFIX) :])
            return Prior("normal", 0.0, sigma) if sigma else Prior("uniform", *_OFFSET_BOUNDS)
        return Prior("uniform", *_BOUNDS[name])

    log_names = {n for n in names if _sampled(n)}
    return names, default, _Likelihood(fit, data, names, log_names), log_names


def _afterglow_problem(lc: LightCurve, fit, sys_err):
    from .physical import _data

    data = _data(lc, fit.sys_err if sys_err is None else sys_err)
    names = list(fit.free)

    def default(name):
        lo, hi = fit.bounds[name]
        if name.startswith(OFFSET_PREFIX) and name[len(OFFSET_PREFIX) :] in fit.priors:
            return Prior("normal", 0.0, fit.priors[name[len(OFFSET_PREFIX) :]], lower=lo, upper=hi)
        if name == "theta_obs":
            return Prior("sine", max(lo, 1e-6), hi)
        return Prior("uniform", lo, hi)

    return names, default, _Likelihood(fit, data, names, set()), set()


def nested_posterior(
    lc: LightCurve,
    fit,
    *,
    priors: Mapping[str, Prior] | None = None,
    sys_err: float | Mapping[str, float] | None = None,
    n_live: int = 2000,
    n_eff: int = 10000,
    seed: int = 0,
    n_jobs: int = 1,
    n_networks: int = 4,
    max_calls: int | None = None,
    discard_exploration: bool = True,
    verbose: bool = False,
) -> Posterior:
    """Sample the posterior of a fit with nested sampling, and compute its evidence.

    Parameters
    ----------
    lc
        The light curve the fit was made to.
    fit
        A power-law :class:`~grb_afterglow.fitting.FitResult` or a physical
        :class:`~grb_afterglow.physical.AfterglowFit`. Its model, fixed
        parameters, bands and calibration priors are reused; the least-squares
        values only centre the amplitude prior of a power-law fit.
    priors
        :class:`Prior` per parameter name, replacing the defaults (see the
        module docstring). For power laws, amplitude and break time are in
        ``log10``.
    sys_err
        Fractional systematic error floor: one value, or a mapping by band name,
        instrument and ``"default"`` as in
        :func:`~grb_afterglow.physical.floor_fractions`. For a physical fit it
        defaults to the fit's.
    n_live
        Live points per nested bound. More find small or distant modes more
        reliably, at the cost of more likelihood calls.
    n_eff
        Effective sample size at which sampling stops. The statistical error of
        ``ln Z`` is about ``1 / sqrt(n_eff)``.
    seed
        Seed of the random number generator, for reproducible runs.
    n_jobs
        Worker processes for likelihood calls and network training. Above 1,
        a script that calls this must guard its entry point with
        ``if __name__ == "__main__":``.
    n_networks
        Neural networks averaged to draw each bound. Fewer train faster, at the
        cost of more likelihood calls; one is enough for quick checks of cheap
        models.
    max_calls
        Stop after this many likelihood calls (for quick checks). A run stopped
        before it converged warns, and its evidence error is infinite.
    discard_exploration
        Drop the points of the initial exploration, which carry a small bias,
        as the ``nautilus`` documentation recommends for published results.
    verbose
        Print the sampler's progress, one line per bound.

    Returns
    -------
    Posterior
        Equal-weight samples in natural units, with ``log_evidence`` and
        ``log_evidence_err`` (natural logarithm; the error is the statistical
        part, ``1 / sqrt(n_eff)``) and the number of likelihood calls in
        ``n_calls``.

    Raises
    ------
    RuntimeError
        If the sampler drew no points (``max_calls`` too small).
    """
    import nautilus

    if isinstance(fit, FitResult):
        names, default, log_likelihood, log_names = _power_law_problem(lc, fit, 0.0 if sys_err is None else sys_err)
    else:
        names, default, log_likelihood, log_names = _afterglow_problem(lc, fit, sys_err)
    unknown = set(priors or {}) - set(names)
    if unknown:
        raise ValueError(f"Priors given for parameters the fit does not have: {sorted(unknown)}.")
    chosen = {n: (priors or {}).get(n) or default(n) for n in names}
    transforms: list[Callable[[float], float]] = [chosen[n].transform for n in names]

    def prior_transform(u):
        return np.array([f(x) for f, x in zip(transforms, u, strict=True)])

    sampler = nautilus.Sampler(
        prior_transform,
        log_likelihood,
        n_dim=len(names),
        n_live=n_live,
        n_networks=n_networks,
        seed=seed,
        pool=None if n_jobs <= 1 else (n_jobs, n_jobs),
    )
    try:
        converged = sampler.run(
            n_eff=n_eff,
            n_like_max=np.inf if max_calls is None else max_calls,
            discard_exploration=discard_exploration,
            verbose=verbose,
        )
    finally:  # nautilus leaves its worker processes running
        for pool in (sampler.pool_l, sampler.pool_s):
            if pool is not None:
                pool.pool.terminate()
    if sampler.log_z is None or sampler.n_like == 0:
        raise RuntimeError("Nested sampling drew no points; raise max_calls.")
    if not converged:
        warnings.warn(
            f"Nested sampling stopped at max_calls={max_calls} before it converged: the posterior is rough and the "
            "evidence can be far off.",
            RuntimeWarning,
            stacklevel=2,
        )
    samples, _, _ = sampler.posterior(equal_weight=True, equal_weight_boost=10.0)
    samples = np.array(samples, dtype=float)
    for j, name in enumerate(names):
        if name in log_names:
            samples[:, j] = 10.0 ** samples[:, j]
    effective = float(sampler.n_eff)
    e = log_likelihood.data[2]
    normalisation = -float(np.sum(np.log(e))) - 0.5 * e.size * np.log(2.0 * np.pi)  # the likelihood omits it
    return Posterior(
        fit,
        names,
        samples,
        float("nan"),
        None,
        log_evidence=float(sampler.log_z) + normalisation,
        log_evidence_err=1.0 / np.sqrt(effective) if converged and effective > 0 else float("inf"),
        n_calls=int(sampler.n_like),
    )


def bayes_factor(first: Posterior, second: Posterior) -> tuple[float, float]:
    """``ln B = ln Z_first - ln Z_second`` and its uncertainty, from two nested-sampling posteriors.

    ``ln B`` above about 5 is usually read as strong evidence for the first model.
    """
    if first.log_evidence is None or second.log_evidence is None:
        raise ValueError("Both posteriors need an evidence (from nested_posterior).")
    return first.log_evidence - second.log_evidence, float(np.hypot(first.log_evidence_err, second.log_evidence_err))
