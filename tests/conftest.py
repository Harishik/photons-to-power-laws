"""Shared fixtures: synthetic star images and a small simulated campaign."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from grb_afterglow.synthetic import AfterglowModel, SyntheticConfig, generate_dataset


def gaussian_stars(positions, *, flux=5000.0, fwhm=3.0, sky=100.0, shape=(64, 64), seed=1, noise=True):
    """Poisson-noise image of Gaussian stars on a flat sky."""
    rng = np.random.default_rng(seed)
    img = np.full(shape, sky, dtype=float)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    fluxes = np.broadcast_to(np.asarray(flux, dtype=float), (len(positions),))
    for (x, y), f in zip(positions, fluxes, strict=True):
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
        img += f * g / (2 * np.pi * sigma**2)
    return rng.poisson(img).astype(float) if noise else img


@pytest.fixture
def star_image():
    return gaussian_stars


@pytest.fixture(scope="session")
def small_campaign(tmp_path_factory):
    """A fast two-band campaign with a jet break, written once per test session."""
    out = tmp_path_factory.mktemp("campaign")
    config = SyntheticConfig(
        shape=(160, 160),
        bands=("g", "r"),
        n_epochs=7,
        n_stars=45,
        t_last=3.0 * 86400.0,
        afterglow=AfterglowModel(p=2.4, t_break=0.3 * 86400.0, mag_ref=14.8),
    )
    truth = generate_dataset(out, config=config, seed=7)
    return out, truth
