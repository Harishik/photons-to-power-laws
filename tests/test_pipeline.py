"""End-to-end: raw synthetic frames in, injected physics out."""

import json

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from grb_afterglow import (
    build_master_bias,
    build_master_flat,
    evaluate_closure_relations,
    fit_lightcurve,
    run_pipeline,
)
from grb_afterglow.calibration import load_fits
from grb_afterglow.pipeline import header_time_since_trigger


@pytest.fixture(scope="module")
def reduced(small_campaign):
    out, truth = small_campaign
    raw = out / "raw"
    bias = build_master_bias([load_fits(p)[0] for p in sorted(raw.glob("bias_*.fits"))])
    flats = {
        b: build_master_flat([load_fits(p)[0] for p in sorted(raw.glob(f"flat_{b}_*.fits"))], master_bias=bias)
        for b in ("g", "r")
    }
    catalog = Table.read(out / "catalog.csv", format="ascii.csv")
    target = SkyCoord(truth["afterglow"]["ra"] * u.deg, truth["afterglow"]["dec"] * u.deg)
    result = run_pipeline(sorted(raw.glob("sci_*.fits")), target, catalog, master_bias=bias, master_flats=flats)
    return result, truth


def test_every_frame_is_calibrated(reduced):
    result, truth = reduced
    assert len(result.frames) == len(truth["frames"])
    assert all(f.zeropoint.n_stars >= 5 for f in result.frames)


def test_fluxes_are_unbiased_with_honest_errors(reduced):
    result, truth = reduced
    by_name = {f["file"]: f for f in truth["frames"]}
    pulls = np.array(
        [(f.flux_ujy - by_name[f.path.split("/")[-1]]["afterglow_flux_ujy"]) / f.flux_err_ujy for f in result.frames]
    )
    assert abs(pulls.mean()) < 0.7
    assert pulls.std() < 1.8


def test_seeing_is_measured(reduced):
    result, truth = reduced
    by_name = {f["file"]: f for f in truth["frames"]}
    scale = truth["config"]["pixel_scale"]
    ratio = [f.fwhm / (by_name[f.path.split("/")[-1]]["seeing_arcsec"] / scale) for f in result.frames]
    # A Gaussian fitted to a Moffat(beta=3) core reads ~10% wide; it must be stable.
    assert 1.0 < np.median(ratio) < 1.25 and np.std(ratio) < 0.05


def test_physics_is_recovered(reduced):
    result, truth = reduced
    ag = truth["afterglow"]
    fit = fit_lightcurve(result.lightcurve, "broken", reference_band="r")
    p, e = fit.params, fit.errors
    assert p["alpha1"] == pytest.approx(ag["alpha1"], abs=max(4 * e["alpha1"], 0.03))
    assert p["alpha2"] == pytest.approx(ag["alpha2"], abs=max(4 * e["alpha2"], 0.1))
    assert p["beta"] == pytest.approx(ag["beta"], abs=max(4 * e["beta"], 0.05))
    best = evaluate_closure_relations(p["alpha2"], e["alpha2"], p["beta"], e["beta"])[0]
    assert best.relation.key == "jet_slow"


def test_summary_table_has_one_row_per_frame(reduced):
    result, _ = reduced
    table = result.summary_table()
    assert len(table) == len(result.frames)
    assert {"zeropoint", "flux_ujy", "snr"} <= set(table.colnames)


def test_truth_file_is_written(small_campaign):
    out, truth = small_campaign
    assert json.loads((out / "truth.json").read_text())["afterglow"]["p"] == truth["afterglow"]["p"]


def test_header_time_since_trigger():
    from astropy.io import fits

    assert header_time_since_trigger(fits.Header({"TSINCE": 42.0})) == 42.0
    hdr = fits.Header({"OBSMJD": 60000.5, "EXPTIME": 30.0})
    assert header_time_since_trigger(hdr, 60000.0) == pytest.approx(43215.0)
    with pytest.raises(ValueError):
        header_time_since_trigger(hdr)
