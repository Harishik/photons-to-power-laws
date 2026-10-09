"""Closure relations: from measured slopes to afterglow physics.

In the standard synchrotron forward-shock model the temporal index ``alpha`` and
spectral index ``beta`` (``F ∝ t^-alpha nu^-beta``) are both fixed by the
power-law index ``p`` of the shocked electrons' energy distribution. Which
relation holds depends on the circumburst medium (constant-density ISM or a
``rho ∝ r^-2`` stellar wind), on where the observing band sits relative to the
cooling frequency ``nu_c``, and on whether the jet break has happened.

Comparing a measured ``(alpha, beta)`` pair with every relation is therefore a
direct test of the emission regime. Relations below assume slow cooling and
``p > 2`` and follow Sari, Piran & Narayan (1998), Chevalier & Li (2000), and
Sari, Piran & Halpern (1999), as tabulated by Zhang & Mészáros (2004) and
Racusin et al. (2009).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClosureRelation:
    """One closure relation of the standard afterglow model."""

    key: str
    label: str
    alpha_of_p: Callable[[float], float]
    beta_of_p: Callable[[float], float]
    p_of_beta: Callable[[float], float]

    def alpha_of_beta(self, beta: float) -> float:
        """Temporal index predicted for a measured spectral index."""
        return float(self.alpha_of_p(self.p_of_beta(beta)))


CLOSURE_RELATIONS: tuple[ClosureRelation, ...] = (
    ClosureRelation(
        "ism_slow",
        "ISM, ν_m < ν < ν_c",
        lambda p: 3 * (p - 1) / 4,
        lambda p: (p - 1) / 2,
        lambda b: 2 * b + 1,
    ),
    ClosureRelation(
        "ism_fast",
        "ISM, ν > ν_c",
        lambda p: (3 * p - 2) / 4,
        lambda p: p / 2,
        lambda b: 2 * b,
    ),
    ClosureRelation(
        "wind_slow",
        "Wind, ν_m < ν < ν_c",
        lambda p: (3 * p - 1) / 4,
        lambda p: (p - 1) / 2,
        lambda b: 2 * b + 1,
    ),
    ClosureRelation(
        "wind_fast",
        "Wind, ν > ν_c",
        lambda p: (3 * p - 2) / 4,
        lambda p: p / 2,
        lambda b: 2 * b,
    ),
    ClosureRelation(
        "jet_slow",
        "Post-jet-break, ν_m < ν < ν_c",
        lambda p: p,
        lambda p: (p - 1) / 2,
        lambda b: 2 * b + 1,
    ),
    ClosureRelation(
        "jet_fast",
        "Post-jet-break, ν > ν_c",
        lambda p: p,
        lambda p: p / 2,
        lambda b: 2 * b,
    ),
)


@dataclass(frozen=True)
class ClosureTest:
    """How well one closure relation matches a measured ``(alpha, beta)``."""

    relation: ClosureRelation
    alpha_predicted: float
    alpha_predicted_err: float
    p: float
    p_err: float
    n_sigma: float

    @property
    def consistent(self) -> bool:
        """Agrees within 2 sigma and implies an electron index compatible with ``p > 2``."""
        return self.n_sigma <= 2.0 and self.p + 2.0 * self.p_err > 2.0


def evaluate_closure_relations(alpha: float, alpha_err: float, beta: float, beta_err: float) -> list[ClosureTest]:
    """Compare a measured temporal and spectral index with every closure relation.

    For each relation, ``p`` is inferred from ``beta`` and used to predict
    ``alpha``. The tension is the difference between measured and predicted
    ``alpha`` divided by their combined uncertainty.

    Returns
    -------
    list of ClosureTest
        Sorted from best to worst agreement.
    """
    results = []
    for rel in CLOSURE_RELATIONS:
        p = rel.p_of_beta(beta)
        dp_dbeta = rel.p_of_beta(beta + 1.0) - p
        p_err = abs(dp_dbeta) * beta_err
        dalpha_dp = rel.alpha_of_p(p + 1.0) - rel.alpha_of_p(p)
        a_pred = rel.alpha_of_p(p)
        a_pred_err = abs(dalpha_dp) * p_err
        sigma = float(np.hypot(alpha_err, a_pred_err)) or 1e-12
        results.append(ClosureTest(rel, a_pred, a_pred_err, p, p_err, abs(alpha - a_pred) / sigma))
    return sorted(results, key=lambda r: r.n_sigma)
