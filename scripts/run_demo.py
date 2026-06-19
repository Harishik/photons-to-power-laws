#!/usr/bin/env python3
"""End-to-end demo: synthetic raw frames to calibrated GRB afterglow light curve."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from astropy.table import Table

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from grb_afterglow import (  # noqa: E402
    LightCurve,
    aperture_photometry_at,
    aperture_photometry_table,
    build_master_bias,
    build_master_flat,
    calibrate_frame,
    calibrate_magnitudes,
    compute_zeropoint,
    detect_sources,
    estimate_background,
    fit_single_power_law,
)
from grb_afterglow.calibration import load_fits  # noqa: E402
from grb_afterglow.detection import estimate_fwhm  # noqa: E402
from grb_afterglow.plotting import plot_calibration, plot_lightcurve  # noqa: E402
from grb_afterglow.synthetic import generate_dataset  # noqa: E402
from grb_afterglow.zeropoint import match_xy  # noqa: E402


def main() -> int:
    data_dir = os.path.join(ROOT, "data", "synthetic", "run")
    out_dir = os.path.join(ROOT, "examples")
    os.makedirs(out_dir, exist_ok=True)

    print("[1/7] Generating synthetic GRB field ...")
    truth = generate_dataset(data_dir, alpha=1.10, seed=42)
    raw_dir = os.path.join(data_dir, "raw")
    gain = truth["detector"]["gain"]
    read_noise = truth["detector"]["read_noise"]
    exptime = truth["exptime"]
    ax, ay = truth["afterglow"]["x"], truth["afterglow"]["y"]

    print("[2/7] Building master calibration frames ...")
    bias_frames = [load_fits(os.path.join(raw_dir, f"bias_{i:02d}.fits"))[0] for i in range(truth["n_bias"])]
    flat_frames = [load_fits(os.path.join(raw_dir, f"flat_{i:02d}.fits"))[0] for i in range(truth["n_flat"])]
    master_bias = build_master_bias(bias_frames)
    master_flat = build_master_flat(flat_frames, master_bias=master_bias)

    ref_truth = Table()
    ref_truth["x"] = truth["reference_stars"]["x"]
    ref_truth["y"] = truth["reference_stars"]["y"]
    ref_truth["catalog_mag"] = truth["reference_stars"]["catalog_mag"]

    times, mags, mag_errs = [], [], []
    saved_calib_plot = False

    print("[3/7] Reducing science frames and measuring photometry ...")
    for sci_name in truth["science_files"]:
        raw, hdr = load_fits(os.path.join(raw_dir, sci_name))
        t = float(hdr["TSINCE"])
        cal = calibrate_frame(raw, master_bias=master_bias, master_flat=master_flat)
        bkg = estimate_background(cal)
        sources = detect_sources(cal, fwhm=3.5, threshold_sigma=5.0, background=bkg)
        if len(sources) == 0:
            continue
        fwhm = estimate_fwhm(cal, sources)
        r_aper = max(2.5, 1.5 * fwhm)
        phot = aperture_photometry_table(cal, sources, r_aper=r_aper, gain=gain, read_noise=read_noise, exptime=exptime)
        idx_m, idx_r, _ = match_xy(phot, ref_truth, max_sep=3.0)
        if len(idx_m) < 5:
            continue
        inst = np.asarray(phot["instrumental_mag"])[idx_m]
        inst_err = np.asarray(phot["instrumental_mag_err"])[idx_m]
        cat = np.asarray(ref_truth["catalog_mag"])[idx_r]
        zp = compute_zeropoint(inst, cat, instrumental_errs=inst_err)
        if not saved_calib_plot:
            plot_calibration(inst, cat, zp, savepath=os.path.join(out_dir, "zeropoint_solution.png"))
            saved_calib_plot = True
        ag = aperture_photometry_at(cal, ax, ay, r_aper=r_aper, gain=gain, read_noise=read_noise, exptime=exptime)
        cal_mag, cal_mag_err = calibrate_magnitudes(np.array([ag.instrumental_mag]), zp, instrumental_errs=np.array([ag.instrumental_mag_err]))
        if np.isfinite(cal_mag[0]) and ag.snr > 3:
            times.append(t)
            mags.append(float(cal_mag[0]))
            mag_errs.append(float(cal_mag_err[0]))
            print(f"      t={t:8.0f}s  mag={cal_mag[0]:6.3f} +/- {cal_mag_err[0]:.3f}")

    print("[4/7] Building light curve ...")
    lc = LightCurve(np.array(times), np.array(mags), np.array(mag_errs)).finite()
    lc.meta["object"] = "SYNTHETIC-GRB"
    lc.save_csv(os.path.join(out_dir, "lightcurve.csv"))

    print("[5/7] Fitting single power law ...")
    fit = fit_single_power_law(lc.time, lc.flux, lc.flux_err)
    print(fit.summary())

    print("[6/7] Validating against truth ...")
    recovered = fit.params["alpha"]
    injected = truth["afterglow"]["alpha"]
    dev = abs(recovered - injected)
    n_sigma = dev / fit.errors["alpha"] if fit.errors["alpha"] > 0 else float("inf")
    status = "PASS" if (dev < 0.1 or n_sigma < 3) else "CHECK"

    print("[7/7] Saving figures ...")
    plot_lightcurve(
        lc,
        fit,
        title=f"Synthetic GRB afterglow: recovered alpha = {recovered:.2f} (injected {injected:.2f})",
        subtitle="Single power-law fit",
        source="Photons to Power Laws demo / synthetic data",
        savepath=os.path.join(out_dir, "lightcurve.png"),
    )

    summary = {
        "injected_alpha": injected,
        "recovered_alpha": recovered,
        "recovered_alpha_err": fit.errors["alpha"],
        "deviation": dev,
        "n_sigma": n_sigma,
        "reduced_chi2": fit.reduced_chi2,
        "n_epochs": len(lc),
        "status": status,
    }
    with open(os.path.join(out_dir, "demo_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Done. Figures and data written to {out_dir}/")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
