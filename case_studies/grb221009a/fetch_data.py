#!/usr/bin/env python3
"""Download the public data used by the GRB 221009A case study.

* ZTF single-epoch science images (IRSA) covering the burst from one day
  before to 40 days after the trigger, as 120" cutouts;
* the deep ZTF reference images for the same field, filters, and quadrant;
* the Pan-STARRS1 DR1 catalog (MAST) around the burst, for the zero points.

The files are already committed under ``data/``; run this only to rebuild them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from astropy.table import Table

from grb_afterglow.archives import (
    ArchiveError,
    cutout_url,
    download,
    query_panstarrs,
    query_ztf_science_images,
    ztf_reference_url,
    ztf_science_url,
)

HERE = Path(__file__).resolve().parent
RA, DEC = 288.26452, 19.77350  # Swift/UVOT position, GCN 32632
TRIGGER_MJD = 59861.553460648  # Fermi/GBM trigger 2022-10-09 13:16:59 UT, GCN 32636
CUTOUT_ARCSEC = 120.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=HERE / "data")
    args = parser.parse_args()
    out: Path = args.out
    (out / "ztf").mkdir(parents=True, exist_ok=True)

    print("Querying IRSA for ZTF images ...")
    meta = query_ztf_science_images(RA, DEC, TRIGGER_MJD - 1.0, TRIGGER_MJD + 40.0)
    status = []
    for row in meta:
        name = f"ztf_{row['filefracday']}_{row['filtercode']}_sciimg.fits"
        try:
            download(cutout_url(ztf_science_url(row), RA, DEC, CUTOUT_ARCSEC), out / "ztf" / name)
            status.append("ok")
            print(f"  {name}")
        except ArchiveError as exc:
            status.append("unavailable")
            print(f"  {name}: {exc}")
    meta["file"] = [f"ztf_{r['filefracday']}_{r['filtercode']}_sciimg.fits" for r in meta]
    meta["download"] = status
    keep = [
        "file",
        "download",
        "obsmjd",
        "filtercode",
        "field",
        "ccdid",
        "qid",
        "exptime",
        "seeing",
        "airmass",
        "maglimit",
        "infobits",
    ]
    meta[keep].write(out / "ztf_manifest.csv", format="ascii.csv", overwrite=True)

    first = meta[0]
    for filtercode in sorted(set(meta["filtercode"])):
        url = ztf_reference_url(first["field"], filtercode, first["ccdid"], first["qid"])
        download(cutout_url(url, RA, DEC, CUTOUT_ARCSEC), out / "ztf" / f"ztf_ref_{filtercode}_refimg.fits")
        print(f"  reference image, {filtercode}")

    print("Querying MAST for Pan-STARRS1 stars ...")
    catalog: Table = query_panstarrs(RA, DEC, radius_arcmin=1.6, bands="gri")
    catalog.write(out / "panstarrs_dr1.csv", format="ascii.csv", overwrite=True)
    print(f"  {len(catalog)} stars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
