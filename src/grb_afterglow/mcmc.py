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
implemented in ``emcee`` (Foreman-Mackey et al. 2013). Where it fails, for
example when a break lies so close to the only early epoch that the early
decay index is unconstrained and the posterior becomes a long, thin funnel,
:func:`grid_posterior` integrates the difficult parameters on a grid instead. The likelihood is the
same Gaussian, flux-space likelihood that the least-squares fit maximises, so
the two agree where the posterior is Gaussian. Priors are uniform in
``log10 amplitude``, ``log10 t_break`` (within the observed time range), the
decay and spectral indices, and the host ``A_V`` (non-negative); calibration
offsets between instruments keep the Gaussian priors they had in the fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .closure import CLOSURE_RELATIONS
from .fitting import OFFSET_PREFIX, FitResult
from .lightcurve import SECONDS_PER_DAY, LightCurve

_BOUNDS = {
    "alpha": (-3.0, 6.0),
    "alpha1": (-3.0, 6.0),
    "alpha2": (-3.0, 6.0),
    "beta": (-3.0, 5.0),
    "host_av": (0.0, 10.0),
}
_OFFSET_BOUNDS = (-5.0, 5.0)


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
        how = "grid" if np.isnan(self.acceptance_fraction) else f"acceptance {self.acceptance_fraction:.2f}"
        lines = [f"{head}, {how}"]
        for name in self.names:
            key = "t_break_days" if name == "t_break" else name
            med, lo, hi = self.interval(key)
            lines.append(f"  {key:<13} {med:.4g}  -{med - lo:.2g} +{hi - med:.2g}")
        return "\n".join(lines)


def _data(lc: LightCurve, sys_err: float):
    t, y, e = (np.asarray(a, dtype=float) for a in (lc.time, lc.flux, lc.flux_err))
    keep = np.isfinite(t) & np.isfinite(y) & np.isfinite(e) & (t > 0) & (e > 0)
    if sys_err > 0:
        e = np.hypot(e, sys_err * np.abs(y))
    return t[keep], y[keep], e[keep], np.asarray(lc.band, dtype=object)[keep].astype(str)


def _bounds(name: str) -> tuple[float, float]:
    if name.startswith(OFFSET_PREFIX):
        return _OFFSET_BOUNDS
    return _BOUNDS.get(name, (-np.inf, np.inf))


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
    spread_break: bool = False,
    moves=None,
) -> Posterior:
    """Sample the posterior of the model in ``fit`` given the light curve it was fitted to.

    Parameters
    ----------
    lc
        The light curve (the same data, cuts and bands as the fit).
    fit
        A least-squares :class:`~grb_afterglow.fitting.FitResult`; its model,
        reference time, spectral model (bands, filter curves and dust), break
        smoothness and calibration priors are reused, and the walkers start in
        a small ball around its best fit.
    n_walkers, n_steps, burn, thin
        Ensemble size, chain length, discarded initial steps, and thinning.
    sys_err
        Fractional systematic error floor, as in the fit.
    wavelengths_nm
        Unused: the fit carries its own band definitions. Kept for backward
        compatibility.
    seed
        Seed of the random number generator, for reproducible chains.
    spread_break
        Start the walkers' break times spread uniformly in ``log t`` over the
        observed range (and the decay indices over a wide range) instead of
        around the best fit. With few epochs on one side of the break the
        posterior can have a long tail that walkers started at the best fit
        take very long to find; check the autocorrelation time either way.
    moves
        ``emcee`` proposal moves (default: the affine-invariant stretch move).
    """
    import emcee

    del wavelengths_nm
    t, y, e, bands = _data(lc, sys_err)
    evaluate = fit.evaluator(bands)
    names = list(fit.params)
    log_names = {"amplitude", "t_break"}
    t_lo, t_hi = np.log10(t.min()), np.log10(t.max())

    def to_params(theta):
        return {n: (10.0**v if n in log_names else float(v)) for n, v in zip(names, theta, strict=True)}

    prior_index = [(names.index(OFFSET_PREFIX + k), s) for k, s in fit.priors.items()]

    def log_prob(theta):
        for n, v in zip(names, theta, strict=True):
            lo, hi = (t_lo, t_hi) if n == "t_break" else _bounds(n)
            if not lo <= v <= hi:
                return -np.inf
        model = evaluate(t, to_params(theta))
        prior_chi2 = sum((theta[j] / s) ** 2 for j, s in prior_index)
        return -0.5 * (float(np.sum(((y - model) / e) ** 2)) + prior_chi2)

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
        if spread_break:
            p0[:, j] = rng.uniform(t_lo, t_hi, n_walkers)
            for n in ("alpha1", "alpha2"):
                k = names.index(n)
                p0[:, k] = fit.params[n] + rng.uniform(-1.0, 1.0, n_walkers)
        p0[:, j] = np.clip(p0[:, j], t_lo + 1e-6, t_hi - 1e-6)
    sampler = emcee.EnsembleSampler(n_walkers, len(names), log_prob, moves=moves)
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


def _sampled(name: str) -> bool:
    """Whether a parameter is sampled in log10 (amplitude and break time)."""
    return name in ("amplitude", "t_break")


def _log_interval(a: float, b: float) -> float:
    """``log(Phi(b) - Phi(a))`` for a standard normal, accurate far into either tail."""
    from scipy.special import log_ndtr

    if a > 0:  # both limits in the upper tail: use the symmetric lower tail
        a, b = -b, -a
    hi, lo = log_ndtr(b), log_ndtr(a)
    return float(hi + np.log1p(-np.exp(min(lo - hi, 0.0)))) if b > a else -np.inf


def _truncated_draws(mean, cov, j, lo, hi, n, rng) -> np.ndarray:
    """Draws from a Gaussian whose parameter ``j`` is truncated to ``[lo[j], hi[j]]`` (exact, any truncation)."""
    from scipy.stats import truncnorm

    sd = np.sqrt(cov[j, j])
    a, b = (lo[j] - mean[j]) / sd, (hi[j] - mean[j]) / sd
    xj = mean[j] + sd * truncnorm.rvs(a, b, size=n, random_state=rng)
    rest = np.flatnonzero(np.arange(len(mean)) != j)
    out = np.empty((n, len(mean)))
    out[:, j] = xj
    if rest.size:
        gain = cov[rest, j] / cov[j, j]
        cond = cov[np.ix_(rest, rest)] - np.outer(gain, cov[j, rest])
        chol = np.linalg.cholesky(cond + 1e-15 * np.eye(rest.size) * np.trace(cond))
        out[:, rest] = mean[rest] + np.outer(xj - mean[j], gain) + rng.standard_normal((n, rest.size)) @ chol.T
    return out


def grid_posterior(
    lc: LightCurve,
    fit: FitResult,
    grid: dict[str, np.ndarray],
    *,
    sys_err: float = 0.0,
    n_samples: int = 40000,
    seed: int = 0,
) -> Posterior:
    """Posterior from a grid over one or two parameters, the others marginalised as Gaussians.

    At every grid point the remaining parameters are fitted by least squares
    and integrated in the Laplace approximation (a Gaussian with the inverse
    of ``J^T J`` as covariance), which is accurate when the data constrain them
    well at fixed grid values. The grid parameters are integrated exactly,
    whatever shape their posterior has. Priors are those of
    :func:`sample_posterior`: uniform in ``log10`` of the amplitude and break
    time, uniform in the indices, and the fit's Gaussian priors on calibration
    offsets. Samples are drawn from the result, so it plugs into everything
    that takes a :class:`Posterior`.

    Parameters
    ----------
    lc, fit
        The light curve and a least-squares fit to it, as for :func:`sample_posterior`.
    grid
        Grid values per parameter in natural units, e.g.
        ``{"t_break": np.geomspace(t0, t1, 60), "alpha1": np.linspace(-3, 6, 91)}``.
        Values are sorted (and duplicates dropped); ``t_break`` is in seconds.
        The grid should cover the prior range wherever the posterior is not
        negligible, and resolve it where it is narrow (spacing may be uneven);
        check that a finer grid gives the same result. A single value fixes
        that parameter.
    sys_err
        Fractional systematic error floor, as in the fit.
    n_samples, seed
        Number of samples drawn from the gridded posterior, and the random seed.
    """
    import itertools

    from scipy.optimize import least_squares

    t, y, e, bands = _data(lc, sys_err)
    evaluate = fit.evaluator(bands)
    names = list(fit.params)
    fixed = list(grid)
    if not 1 <= len(fixed) <= 2 or any(n not in names for n in fixed):
        raise ValueError("grid must name one or two of the fit's parameters.")
    free = [n for n in names if n not in fixed]
    t_lo, t_hi = np.log10(t.min()), np.log10(t.max())

    log_amp = np.log10(fit.params["amplitude"])

    def bounds(n):
        if n == "t_break":
            return t_lo, t_hi
        return (log_amp - 6.0, log_amp + 6.0) if n == "amplitude" else _bounds(n)

    def to_natural(n, v):
        return 10.0**v if _sampled(n) else v

    # Grid in sampled units, with the prior volume of each cell (a single value is a fixed parameter).
    def axis(n):
        values = np.asarray(grid[n], dtype=float)
        return np.unique(np.log10(values) if _sampled(n) else values)

    axes = [axis(n) for n in fixed]
    widths = [np.gradient(a) if a.size > 1 else np.zeros(1) for a in axes]
    grid_sigma = {
        k: fit.priors[n[len(OFFSET_PREFIX) :]]
        for k, n in enumerate(fixed)
        if n.startswith(OFFSET_PREFIX) and n[len(OFFSET_PREFIX) :] in fit.priors
    }
    lo = np.array([bounds(n)[0] for n in free])
    hi = np.array([bounds(n)[1] for n in free])
    prior_sigma = np.array(
        [fit.priors.get(n[len(OFFSET_PREFIX) :], np.inf) if n.startswith(OFFSET_PREFIX) else np.inf for n in free]
    )
    has_prior = np.isfinite(prior_sigma)
    start = np.array([np.log10(fit.params[n]) if _sampled(n) else fit.params[n] for n in free])

    nodes, log_mass, means, covs, pins = [], [], [], [], []
    x0 = start
    for index in itertools.product(*(range(a.size) for a in axes)):
        values = {n: to_natural(n, axes[k][i]) for k, (n, i) in enumerate(zip(fixed, index, strict=True))}

        def residuals(q, values=values):
            params = {**values, **{n: to_natural(n, v) for n, v in zip(free, q, strict=True)}}
            r = (y - evaluate(t, params)) / e
            return np.concatenate([r, q[has_prior] / prior_sigma[has_prior]]) if has_prior.any() else r

        # Start from the neighbouring grid point's solution or the best fit, whichever fits better here.
        guesses = [np.clip(g, lo + 1e-9, hi - 1e-9) for g in (x0, start)]
        guess = min(guesses, key=lambda g: float(np.sum(residuals(g) ** 2)))
        best = least_squares(residuals, guess, bounds=(lo, hi), method="trf")
        x0 = best.x
        hess = best.jac.T @ best.jac
        sign, logdet = np.linalg.slogdet(hess)
        if sign <= 0:
            continue
        cov = np.linalg.inv(hess)
        cost, centre, pinned, log_in = best.cost, best.x, -1, 0.0
        active = np.flatnonzero(best.active_mask)
        if active.size == 1:
            # One parameter held at its bound by the data: the quadratic approximation there is a Gaussian centred
            # beyond the bound and truncated by it. Its mass inside the bounds is that of the full Gaussian times
            # the probability of the parameter's marginal lying within its bounds; it is sampled exactly below.
            grad = best.jac.T @ best.fun
            shift = np.linalg.solve(hess, grad)
            cost, centre, pinned = best.cost - 0.5 * float(grad @ shift), best.x - shift, int(active[0])
            sd = np.sqrt(cov[pinned, pinned])
            log_in = _log_interval((lo[pinned] - centre[pinned]) / sd, (hi[pinned] - centre[pinned]) / sd)
        volume = np.prod([w[i] if w[i] > 0 else 1.0 for w, i in zip(widths, index, strict=True)])
        prior = sum((axes[k][index[k]] / sigma) ** 2 for k, sigma in grid_sigma.items())  # gridded offsets
        nodes.append([a[i] for a, i in zip(axes, index, strict=True)])
        log_mass.append(-cost - 0.5 * prior - 0.5 * logdet + np.log(volume) + log_in)
        means.append(centre)
        covs.append(cov)
        pins.append(pinned)
    if not nodes:
        raise RuntimeError("No grid point gave a usable fit.")

    nodes, means, covs, pins = np.asarray(nodes), np.asarray(means), np.asarray(covs), np.asarray(pins)
    weights = np.exp(np.asarray(log_mass) - np.max(log_mass))
    for k, name in enumerate(fixed):  # cells wholly outside the prior range carry no weight
        b = bounds(name)
        half = 0.5 * widths[k][np.searchsorted(axes[k], nodes[:, k])]
        weights *= (nodes[:, k] + half >= b[0]) & (nodes[:, k] - half <= b[1])
    if not weights.sum() > 0:
        raise ValueError("The grid lies outside the prior range (t_break is in seconds, within the data's time span).")
    weights /= weights.sum()
    rng = np.random.default_rng(seed)
    samples = np.empty((n_samples, len(names)))
    filled, passes = 0, 0
    while filled < n_samples:
        passes += 1
        if passes > 1000 and filled < 0.01 * n_samples:
            raise RuntimeError("Almost every draw falls outside the prior bounds; grid the bounded parameter instead.")
        n = n_samples - filled
        cell = rng.choice(len(weights), size=n, p=weights)
        # Spread grid values uniformly within their cells, and draw the others from their Gaussians.
        grid_part = nodes[cell] + np.column_stack(
            [(rng.random(n) - 0.5) * widths[k][np.searchsorted(axes[k], nodes[cell, k])] for k in range(len(fixed))]
        )
        chol = np.linalg.cholesky(covs[cell])
        free_part = means[cell] + np.einsum("nij,nj->ni", chol, rng.standard_normal((n, len(free))))
        for c in np.unique(cell[pins[cell] >= 0]):  # truncated draws for cells with a pinned parameter
            sel = np.flatnonzero(cell == c)
            free_part[sel] = _truncated_draws(means[c], covs[c], pins[c], lo, hi, sel.size, rng)
        ok = np.all((free_part >= lo) & (free_part <= hi), axis=1)
        for k, name in enumerate(fixed):
            b = bounds(name)
            ok &= (grid_part[:, k] >= b[0]) & (grid_part[:, k] <= b[1])
        m = int(ok.sum())
        block = np.empty((m, len(names)))
        for k, name in enumerate(fixed):
            block[:, names.index(name)] = grid_part[ok, k]
        for k, name in enumerate(free):
            block[:, names.index(name)] = free_part[ok, k]
        samples[filled : filled + m] = block
        filled += m
    for j, name in enumerate(names):
        if _sampled(name):
            samples[:, j] = 10.0 ** samples[:, j]
    return Posterior(fit, names, samples, float("nan"), None)
