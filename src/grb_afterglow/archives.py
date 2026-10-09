"""Access to public archives: ZTF images (IRSA) and Pan-STARRS1 photometry (MAST).

Only the standard library is used for HTTP, so no extra dependency is needed.
Every function that touches the network has a ``timeout`` and raises
:class:`ArchiveError` with the failing URL if a request does not succeed.
"""

from __future__ import annotations

import io
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from os import PathLike
from pathlib import Path

import numpy as np
from astropy.table import Table

IRSA_ZTF_SEARCH_URL = "https://irsa.ipac.caltech.edu/ibe/search/ztf/products/sci"
IRSA_ZTF_DATA_URL = "https://irsa.ipac.caltech.edu/ibe/data/ztf/products"
MAST_PS1_URL = "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr1/mean.csv"

JD_TO_MJD = 2400000.5
_USER_AGENT = "photons-to-power-laws (https://github.com/Harishik/photons-to-power-laws)"


class ArchiveError(RuntimeError):
    """A request to a remote archive failed."""


def _get(url: str, *, timeout: float = 120.0, retries: int = 3) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise ArchiveError(f"Not found: {url}") from exc
            last = exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
        time.sleep(2.0 * (attempt + 1))
    raise ArchiveError(f"Request failed after {retries} attempts: {url} ({last})")


def download(url: str, path: str | PathLike[str], *, timeout: float = 120.0, retries: int = 3) -> Path:
    """Download ``url`` to ``path`` (parent directories are created)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(_get(url, timeout=timeout, retries=retries))
    return out


def query_ztf_science_images(
    ra: float, dec: float, mjd_start: float, mjd_end: float, *, timeout: float = 120.0
) -> Table:
    """List ZTF single-epoch science images covering a position.

    Returns the IRSA metadata table (one row per CCD quadrant image) sorted by
    time, with an added ``obsmjd`` column.
    """
    where = f"obsjd>{mjd_start + JD_TO_MJD} AND obsjd<{mjd_end + JD_TO_MJD}"
    query = urllib.parse.urlencode({"POS": f"{ra},{dec}", "ct": "csv", "WHERE": where})
    table = Table.read(io.BytesIO(_get(f"{IRSA_ZTF_SEARCH_URL}?{query}", timeout=timeout)), format="ascii.csv")
    if len(table):
        table["obsmjd"] = np.asarray(table["obsjd"], dtype=float) - JD_TO_MJD
        table.sort("obsmjd")
    return table


def ztf_science_url(row: Mapping, product: str = "sciimg.fits") -> str:
    """URL of a single-epoch ZTF product described by a metadata row."""
    ffd = str(row["filefracday"])
    year, monthday, fracday = ffd[:4], ffd[4:8], ffd[8:]
    name = (
        f"ztf_{ffd}_{int(row['field']):06d}_{row['filtercode']}_c{int(row['ccdid']):02d}"
        f"_{row['imgtypecode']}_q{int(row['qid'])}_{product}"
    )
    return f"{IRSA_ZTF_DATA_URL}/sci/{year}/{monthday}/{fracday}/{name}"


def ztf_reference_url(field: int, filtercode: str, ccdid: int, qid: int, product: str = "refimg.fits") -> str:
    """URL of the deep ZTF reference (template) image for a field, filter, and quadrant."""
    fff = f"{int(field):06d}"
    return (
        f"{IRSA_ZTF_DATA_URL}/ref/{fff[:3]}/field{fff}/{filtercode}/ccd{int(ccdid):02d}/q{int(qid)}/"
        f"ztf_{fff}_{filtercode}_c{int(ccdid):02d}_q{int(qid)}_{product}"
    )


def cutout_url(url: str, ra: float, dec: float, size_arcsec: float) -> str:
    """Ask IRSA for a square cutout of an image instead of the full quadrant."""
    params = urllib.parse.urlencode({"center": f"{ra},{dec}", "size": f"{size_arcsec}arcsec", "gzip": "false"})
    return f"{url}?{params}"


def query_panstarrs(
    ra: float,
    dec: float,
    radius_arcmin: float,
    *,
    bands: str = "grizy",
    min_detections: int = 5,
    timeout: float = 120.0,
) -> Table:
    """Cone search of the Pan-STARRS1 DR1 mean-photometry catalog at MAST.

    Returns a table with ``ra``, ``dec`` (deg) and, per band, PSF magnitudes
    ``g``, ``g_err``, ... (AB). Missing photometry (``-999`` in the archive)
    becomes NaN.
    """
    columns = ["objID", "raMean", "decMean", "nDetections"]
    for b in bands:
        columns += [f"{b}MeanPSFMag", f"{b}MeanPSFMagErr"]
    query = urllib.parse.urlencode(
        {
            "ra": ra,
            "dec": dec,
            "radius": radius_arcmin / 60.0,
            "nDetections.gte": min_detections,
            "columns": "[" + ",".join(columns) + "]",
        }
    )
    raw = Table.read(io.BytesIO(_get(f"{MAST_PS1_URL}?{query}", timeout=timeout)), format="ascii.csv")
    out = Table()
    out["objID"] = raw["objID"]
    out["ra"] = np.asarray(raw["raMean"], dtype=float)
    out["dec"] = np.asarray(raw["decMean"], dtype=float)
    out["n_detections"] = raw["nDetections"]
    for b in bands:
        mag = np.asarray(raw[f"{b}MeanPSFMag"], dtype=float)
        err = np.asarray(raw[f"{b}MeanPSFMagErr"], dtype=float)
        bad = (mag < -900) | (err < -900)
        out[b] = np.where(bad, np.nan, mag)
        out[f"{b}_err"] = np.where(bad, np.nan, err)
    return out
