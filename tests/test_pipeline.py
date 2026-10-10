"""End-to-end: raw synthetic frames in, injected physics out."""

import json

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from grb_afterglow import (
    PipelineConfig,
    build_master_bias,
    build_master_flat,
    evaluate_closure_relations,
    fit_lightcurve,
    run_pipeline,
)
from grb_afterglow.calibration import calibrate_frame, load_fits
from grb_afterglow.pipeline import Template, header_time_since_trigger, reduce_frame


@pytest.fixture(scope="module")
def calibration(small_campaign):
    out, truth = small_campaign
    raw = out / "raw"
    bias = build_master_bias([load_fits(p)[0] for p in sorted(raw.glob("bias_*.fits"))])
    flats = {
        b: build_master_flat([load_fits(p)[0] for p in sorted(raw.glob(f"flat_{b}_*.fits"))], master_bias=bias)
        for b in ("g", "r")
    }
    catalog = Table.read(out / "catalog.csv", format="ascii.csv")
    target = SkyCoord(truth["afterglow"]["ra"] * u.deg, truth["afterglow"]["dec"] * u.deg)
    templates = {b: out / info["file"] for b, info in truth["templates"].items()}
    return raw, bias, flats, catalog, target, templates


@pytest.fixture(scope="module")
def reduced(small_campaign, calibration):
    """The recommended reduction: template subtraction and PSF photometry."""
    _, truth = small_campaign
    raw, bias, flats, catalog, target, templates = calibration
    result = run_pipeline(
        sorted(raw.glob("sci_*.fits")),
        target,
        catalog,
        master_bias=bias,
        master_flats=flats,
        config=PipelineConfig(photometry="psf"),
        templates=templates,
    )
    return result, truth


def _late_r_frames(raw, truth, n=3):
    frames = sorted((f for f in truth["frames"] if f["band"] == "r"), key=lambda f: f["time_s"])[-n:]
    return [raw / f["file"] for f in frames], {f["file"]: f for f in frames}


def test_every_frame_is_calibrated(reduced):
    result, truth = reduced
    assert len(result.frames) == len(truth["frames"])
    assert all(f.zeropoint.n_stars >= 5 for f in result.frames)
    assert all(f.subtracted and f.photometry == "psf" and f.psf is not None for f in result.frames)
    # The templates are sharper than every science frame, so they are the image that gets convolved.
    assert all(f.difference.psf_from == "science" for f in result.frames)


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


def test_simulator_writes_templates_and_host(small_campaign):
    from astropy.io import fits

    out, truth = small_campaign
    assert truth["host"] is not None and truth["host"]["mag_r"] > 0
    for band, info in truth["templates"].items():
        header = fits.getheader(out / info["file"])
        assert header["IMAGETYP"] == "TEMPLATE" and header["FILTER"] == band
        assert header["EXPTIME"] > 0 and "CTYPE1" in header


def test_template_subtraction_removes_host_light(small_campaign, calibration):
    """Aperture photometry of the late r frames, with and without the template."""
    _, truth = small_campaign
    raw, bias, flats, catalog, target, templates = calibration
    files, by_name = _late_r_frames(raw, truth)
    host_ujy = 10 ** (-0.4 * (truth["host"]["mag_r"] - 23.9))

    def offsets(result):
        assert len(result.frames) == len(files)
        return np.array([f.flux_ujy - by_name[f.path.split("/")[-1]]["afterglow_flux_ujy"] for f in result.frames])

    plain = run_pipeline(files, target, catalog, master_bias=bias, master_flats=flats)
    subtracted = run_pipeline(files, target, catalog, master_bias=bias, master_flats=flats, templates=templates)
    assert all(f.photometry == "aperture" and f.subtracted for f in subtracted.frames)
    # Without the template most of the host's light is measured as afterglow; with it, little remains.
    assert offsets(plain).mean() > 0.5 * host_ujy
    assert abs(offsets(subtracted).mean()) < 0.3 * host_ujy


@pytest.mark.parametrize("photometry", ["aperture", "psf"])
def test_template_blurrier_than_science(small_campaign, calibration, photometry):
    """The science image is convolved instead (a deliberately extreme case: template FWHM ~1.7x the science)."""
    from scipy.ndimage import gaussian_filter

    out, truth = small_campaign
    raw, bias, flats, catalog, target, _ = calibration
    sharp = Template.from_fits(out / truth["templates"]["r"]["file"])
    blurred = gaussian_filter(sharp.image, 2.6)  # FWHM ~ 6.6 px, wider than every science frame
    template = Template(blurred, sharp.wcs, float(np.hypot(sharp.fwhm, 2.355 * 2.6)))
    frames = sorted((f for f in truth["frames"] if f["band"] == "r"), key=lambda f: f["time_s"])[1:3]
    for info in frames:
        image_raw, header = load_fits(raw / info["file"])
        image = calibrate_frame(image_raw, master_bias=bias, master_flat=flats["r"])
        from astropy.wcs import WCS

        result = reduce_frame(
            image,
            WCS(header),
            target,
            catalog,
            band="r",
            time=header["TSINCE"],
            exptime=header["EXPTIME"],
            gain=header["GAIN"],
            config=PipelineConfig(photometry=photometry),
            template=template,
        )
        assert result.difference.psf_from == "template"
        assert result.difference.fwhm > result.fwhm
        if photometry == "psf":  # the science PSF through the kernel is exact
            assert result.flux_ujy == pytest.approx(info["afterglow_flux_ujy"], abs=4 * result.flux_err_ujy)
        else:  # transferring the aperture scale through convolved stars is good to a few per cent
            assert result.flux_ujy == pytest.approx(info["afterglow_flux_ujy"], rel=0.05)


def test_photometry_option_is_validated():
    with pytest.raises(ValueError):
        PipelineConfig(photometry="optimal")
