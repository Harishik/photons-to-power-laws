"""Synthetic GRB field generator used by the demo and tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from astropy.io import fits


def _gaussian(shape, x0, y0, flux, fwhm):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    g = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))
    return flux * g / g.sum()


def generate_dataset(
    out_dir: str,
    *,
    alpha: float = 1.10,
    seed: int = 42,
    n_epochs: int = 10,
    shape: tuple[int, int] = (128, 128),
) -> dict:
    """Generate a small synthetic GRB afterglow FITS dataset.

    The output intentionally stays lightweight, but it contains realistic enough
    bias, flat, sky, reference stars, and a fading afterglow to validate the
    reduction pipeline end to end.
    """
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    gain = 1.4
    read_noise = 5.0
    exptime = 60.0
    sky = 180.0
    bias_level = 500.0
    fwhm = 3.2

    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    true_flat = 1.0 + 0.03 * (xx - shape[1] / 2) / shape[1] + 0.02 * (yy - shape[0] / 2) / shape[0]
    true_flat += rng.normal(0, 0.003, shape)

    n_bias = 5
    n_flat = 5
    for i in range(n_bias):
        frame = bias_level + rng.normal(0, read_noise, shape)
        fits.PrimaryHDU(frame).writeto(raw / f"bias_{i:02d}.fits", overwrite=True)

    for i in range(n_flat):
        illumination = 20000.0 * true_flat
        frame = bias_level + rng.poisson(np.maximum(illumination, 1)).astype(float)
        fits.PrimaryHDU(frame).writeto(raw / f"flat_{i:02d}.fits", overwrite=True)

    n_ref = 18
    margin = 15
    ref_x = rng.uniform(margin, shape[1] - margin, n_ref)
    ref_y = rng.uniform(margin, shape[0] - margin, n_ref)
    ref_mag = rng.uniform(15.0, 19.0, n_ref)
    zp = 25.0
    ref_flux = exptime * 10 ** (-0.4 * (ref_mag - zp))

    ag_x = shape[1] * 0.53
    ag_y = shape[0] * 0.48
    times = np.geomspace(300.0, 35000.0, n_epochs)
    ag_mag0 = 15.6
    ag_flux0 = exptime * 10 ** (-0.4 * (ag_mag0 - zp))
    science_files = []

    for j, t in enumerate(times):
        scene = np.full(shape, sky, dtype=float)
        for x, y, flux in zip(ref_x, ref_y, ref_flux):
            scene += _gaussian(shape, x, y, flux, fwhm)
        ag_flux = ag_flux0 * (t / times[0]) ** (-alpha)
        scene += _gaussian(shape, ag_x, ag_y, ag_flux, fwhm)
        electrons = np.maximum(scene * true_flat, 1)
        raw_frame = bias_level + rng.poisson(electrons).astype(float) + rng.normal(0, read_noise, shape)
        hdr = fits.Header()
        hdr["TSINCE"] = float(t)
        name = f"science_{j:02d}.fits"
        fits.PrimaryHDU(raw_frame, hdr).writeto(raw / name, overwrite=True)
        science_files.append(name)

    truth = {
        "detector": {"gain": gain, "read_noise": read_noise},
        "exptime": exptime,
        "n_bias": n_bias,
        "n_flat": n_flat,
        "science_files": science_files,
        "afterglow": {"x": ag_x, "y": ag_y, "alpha": alpha},
        "reference_stars": {"x": ref_x.tolist(), "y": ref_y.tolist(), "catalog_mag": ref_mag.tolist()},
    }
    with open(out / "truth.json", "w") as fh:
        json.dump(truth, fh, indent=2)
    return truth


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--alpha", type=float, default=1.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(args.out_dir, alpha=args.alpha, seed=args.seed)
