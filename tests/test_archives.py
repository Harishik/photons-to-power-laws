"""Archive helpers, tested offline with canned responses."""

import numpy as np

from grb_afterglow import archives


def test_ztf_science_url():
    row = {"filefracday": 20221011129144, "field": 590, "filtercode": "zr", "ccdid": 11, "imgtypecode": "o", "qid": 4}
    assert archives.ztf_science_url(row) == (
        "https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/2022/1011/129144/"
        "ztf_20221011129144_000590_zr_c11_o_q4_sciimg.fits"
    )


def test_ztf_reference_url():
    assert archives.ztf_reference_url(590, "zg", 11, 4).endswith(
        "/ref/000/field000590/zg/ccd11/q4/ztf_000590_zg_c11_q4_refimg.fits"
    )


def test_cutout_url_encodes_position():
    url = archives.cutout_url("https://x/y.fits", 288.26, 19.77, 120)
    assert url.startswith("https://x/y.fits?center=288.26%2C19.77") and "size=120arcsec" in url


def test_query_panstarrs_parses_and_masks(monkeypatch):
    csv = (
        b"objID,raMean,decMean,nDetections,gMeanPSFMag,gMeanPSFMagErr,rMeanPSFMag,rMeanPSFMagErr\n"
        b"1,288.1,19.7,10,18.5,0.01,18.0,0.01\n"
        b"2,288.2,19.8,7,-999.0,-999.0,20.0,0.03\n"
    )
    monkeypatch.setattr(archives, "_get", lambda url, **kw: csv)
    table = archives.query_panstarrs(288.15, 19.75, 1.0, bands="gr")
    assert table["r"].tolist() == [18.0, 20.0]
    assert np.isnan(table["g"][1])


def test_query_ztf_adds_mjd(monkeypatch):
    csv = b"obsjd,filtercode\n2459863.6291667,zr\n2459861.6824537,zg\n"
    monkeypatch.setattr(archives, "_get", lambda url, **kw: csv)
    table = archives.query_ztf_science_images(288.26, 19.77, 59860.0, 59900.0)
    assert table["filtercode"].tolist() == ["zg", "zr"]
    assert table["obsmjd"][0] == np.float64(2459861.6824537) - 2400000.5
