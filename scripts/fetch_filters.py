#!/usr/bin/env python3
"""Download the filter transmission curves of grb_afterglow.filters from the SVO Filter Profile Service.

The curves are committed under ``src/grb_afterglow/data/filters``; run this only to rebuild them.
SVO: Rodrigo et al. 2012 (IVOA Working Draft), Rodrigo & Solano 2020 (XIV.0 Scientific Meeting of the SEA),
Rodrigo et al. 2024 (A&A 689, A93).
"""

from __future__ import annotations

import csv
import re
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "grb_afterglow" / "data" / "filters"
SVO = "https://svo2.cab.inta-csic.es/theory/fps/getdata.php?format=ascii&id="
SVO_META = "https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID="

ACKNOWLEDGEMENT = (
    'This research has made use of the SVO Filter Profile Service "Carlos Rodrigo", funded by '
    "MCIN/AEI/10.13039/501100011033/ through grant PID2023-146210NB-I00 (Rodrigo et al. 2012, 2020, 2024)."
)

# name in grb_afterglow -> (SVO filter ID, note)
FILTERS = {
    "ztf:g": ("Palomar/ZTF.g", "ZTF system throughput"),
    "ztf:r": ("Palomar/ZTF.r", "ZTF system throughput"),
    "ztf:i": ("Palomar/ZTF.i", "ZTF system throughput"),
    "ps1:g": ("PAN-STARRS/PS1.g", "Pan-STARRS1 system (Tonry et al. 2012)"),
    "ps1:r": ("PAN-STARRS/PS1.r", "Pan-STARRS1 system (Tonry et al. 2012)"),
    "ps1:i": ("PAN-STARRS/PS1.i", "Pan-STARRS1 system (Tonry et al. 2012)"),
    "ps1:z": ("PAN-STARRS/PS1.z", "Pan-STARRS1 system (Tonry et al. 2012)"),
    "ps1:y": ("PAN-STARRS/PS1.y", "Pan-STARRS1 system (Tonry et al. 2012)"),
    "sdss:u": ("SLOAN/SDSS.u", "SDSS 2.5 m system"),
    "sdss:g": ("SLOAN/SDSS.g", "SDSS 2.5 m system"),
    "sdss:r": ("SLOAN/SDSS.r", "SDSS 2.5 m system"),
    "sdss:i": ("SLOAN/SDSS.i", "SDSS 2.5 m system"),
    "sdss:z": ("SLOAN/SDSS.z", "SDSS 2.5 m system"),
    "lt:g": ("Liverpool/IOO.SDSS-g", "Liverpool Telescope IO:O system"),
    "lt:r": ("Liverpool/IOO.SDSS-r", "Liverpool Telescope IO:O system"),
    "lt:i": ("Liverpool/IOO.SDSS-i", "Liverpool Telescope IO:O system"),
    "lt:z": ("Liverpool/IOO.SDSS-z", "Liverpool Telescope IO:O system"),
    "gmos-s:g": ("Gemini/GMOS-S.g", "Gemini South GMOS-S"),
    "gmos-s:r": ("Gemini/GMOS-S.r", "Gemini South GMOS-S"),
    "gmos-s:i": ("Gemini/GMOS-S.i", "Gemini South GMOS-S"),
    "gmos-s:z": ("Gemini/GMOS-S.z", "Gemini South GMOS-S"),
    "f2:J": ("Gemini/Flamingos2.J", "Gemini South Flamingos-2"),
    "f2:H": ("Gemini/Flamingos2.H", "Gemini South Flamingos-2"),
    "f2:Ks": ("Gemini/Flamingos2.Ks", "Gemini South Flamingos-2"),
    "wirc:J": ("P200/WIRC.J_ori", "Palomar 200-inch WIRC (original filter data)"),
    "wirc:H": ("P200/WIRC.H_ori", "Palomar 200-inch WIRC (original filter data)"),
    "wirc:Ks": ("P200/WIRC.Ks", "Palomar 200-inch WIRC"),
    "2mass:J": ("2MASS/2MASS.J", "2MASS (Cohen et al. 2003)"),
    "2mass:H": ("2MASS/2MASS.H", "2MASS (Cohen et al. 2003)"),
    "2mass:Ks": ("2MASS/2MASS.Ks", "2MASS (Cohen et al. 2003)"),
    "wfc3:F625W": ("HST/WFC3_UVIS2.F625W", "HST WFC3/UVIS chip 2"),
    "wfc3:F775W": ("HST/WFC3_UVIS2.F775W", "HST WFC3/UVIS chip 2"),
    "wfc3:F098M": ("HST/WFC3_IR.F098M", "HST WFC3/IR"),
    "wfc3:F125W": ("HST/WFC3_IR.F125W", "HST WFC3/IR"),
    "wfc3:F160W": ("HST/WFC3_IR.F160W", "HST WFC3/IR"),
}


def file_name(svo_id: str) -> str:
    return svo_id.replace("/", "__") + ".dat"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (svo_id, note) in FILTERS.items():
        path = OUT / file_name(svo_id)
        with urllib.request.urlopen(SVO + svo_id, timeout=60) as response:
            data = response.read().decode()
        if not data.strip():
            raise RuntimeError(f"SVO returned nothing for {svo_id}")
        path.write_text(data)
        with urllib.request.urlopen(SVO_META + svo_id, timeout=60) as response:
            meta = response.read().decode()
        # SVO DetectorType: 1 = photon counter, 0 = energy counter (the curve is then an energy response).
        detector = {"1": "photon", "0": "energy"}[re.search(r'name="DetectorType"[^>]*value="(\d)"', meta).group(1)]
        pivot_nm = float(re.search(r'name="WavelengthPivot"[^>]*value="([^"]*)"', meta).group(1)) / 10.0
        row = {"name": name, "svo_id": svo_id, "file": path.name, "detector": detector}
        rows.append({**row, "svo_pivot_nm": f"{pivot_nm:.2f}", "note": note})
        print(f"{name:<12} {svo_id:<26} {len(data.splitlines()):4d} points  {detector:<6} pivot {pivot_nm:7.1f} nm")
    with open(OUT / "index.csv", "w", newline="") as fh:
        fh.write(f"# Filter curves from the SVO Filter Profile Service, retrieved {date.today().isoformat()}.\n")
        fh.write("# Wavelengths in Angstrom; transmission as given by SVO (system throughput where available).\n")
        fh.write(f"# {ACKNOWLEDGEMENT}\n")
        writer = csv.DictWriter(fh, fieldnames=["name", "svo_id", "file", "detector", "svo_pivot_nm", "note"])
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
