#!/usr/bin/env python3
"""Download the Swift/XRT afterglow of GRB 210204A and convert it to flux densities at 3 keV.

Swift observed GRB 210204A as a target of opportunity (target ID 00014055,
GCN Circular 29412). The UK Swift Science Data Centre (Evans et al. 2007,
2009) provides the 0.3-10 keV light curve in count rates and a time-averaged
absorbed power-law fit to the spectrum. This script

1. downloads the light curve (PCCURVE.qdp, with flux.qdp for its time zero)
   and the spectrum products (interval0.tar.gz);
2. reads the photon index, the unabsorbed 0.3-10 keV flux and the count rate
   of the spectral fit;
3. converts every count rate to an unabsorbed flux density at 3 keV with that
   spectrum.

3 keV is where this conversion depends least on the uncertain absorption: the
intrinsic column density and the photon index are correlated in the fit, and
along that correlation the factor changes by less than 1 % at 3 keV (against
13 % at 1.65 keV and 23 % at 10 keV over the 90 % range of the column). The
flux densities that the UKSSDC Burst Analyser gives at 10 keV use a photon
index per bin from hardness ratios, whose uncertainty is not in their error
bars. The common uncertainty of the conversion (absorption, spectral shape,
archive reprocessing) is left to a calibration offset in the fit.

    python case_studies/grb210204a/fetch_xray.py
"""

from __future__ import annotations

import argparse
import io
import re
import tarfile
from pathlib import Path

from astropy.time import Time
from scipy.integrate import quad

from grb_afterglow.archives import _get

HERE = Path(__file__).resolve().parent
TARGET_ID = "00014055"
TRIGGER_MJD = 59249.27042824  # Fermi/GBM, 2021-02-04 06:29:25 UT, as in analyze.py
ENERGY_KEV = 3.0
KEV_ERG = 1.602176634e-9
H_KEV_S = 4.135667696e-18
BAND_KEV = (0.3, 10.0)
URLS = {
    "curve": f"https://www.swift.ac.uk/xrt_curves/{TARGET_ID}/PCCURVE.qdp",
    "flux": f"https://www.swift.ac.uk/xrt_curves/{TARGET_ID}/flux.qdp",
    "spectrum": f"https://www.swift.ac.uk/xrt_spectra/{TARGET_ID}/interval0.tar.gz",
}


def read_qdp_rows(text: str) -> list[list[float]]:
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if not parts or parts[0].startswith(("!", "READ", "NO", "@")):
            continue
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            continue
    return rows


def spectral_fit(tar_bytes: bytes) -> dict:
    """Photon index, unabsorbed flux and corrected count rate of the time-averaged PC spectrum."""
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("pc_fit.fit"))
        text = tar.extractfile(member).read().decode()

    def number(label):
        return float(re.search(rf"{label}\s*:?\s*([0-9.eE+-]+)", text).group(1))

    fit = {
        "gamma": number("gamma"),
        "unabsorbed_flux_cgs": number("Unabs flux"),
        "count_rate": number("File count rate") * number("Corr factor"),
        "nh_galactic_1e22": number("Gal_nh"),
        "nh_intrinsic_1e22": number("excess_nh"),
        "text": text,
    }
    return fit


def density_per_count(gamma: float, unabsorbed_flux: float, count_rate: float, energy_kev: float) -> float:
    """Unabsorbed flux density (Jy) at ``energy_kev`` per count per second, for a power-law spectrum."""
    norm = unabsorbed_flux / (KEV_ERG * quad(lambda e: e ** (1.0 - gamma), *BAND_KEV)[0])  # photons/cm2/s/keV at 1 keV
    return norm * energy_kev ** (1.0 - gamma) * KEV_ERG * H_KEV_S * 1e23 / count_rate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=HERE / "data" / "swift_xrt.csv")
    args = parser.parse_args()

    curve = _get(URLS["curve"]).decode()
    flux_header = _get(URLS["flux"]).decode()
    met = re.search(r"T0.*?MET\s*=\s*([0-9.]+)", flux_header)
    t0_met = float(met.group(1))
    t0_utc = re.search(r"=\s*(\d{4} \w{3} \d{2}) at (\d{2}:\d{2}:\d{2}\.\d+)", flux_header)
    t0 = Time.strptime(f"{t0_utc.group(1)} {t0_utc.group(2)}", "%Y %b %d %H:%M:%S.%f", scale="utc")
    offset_s = float((t0 - Time(TRIGGER_MJD, format="mjd", scale="utc")).sec)
    fit = spectral_fit(_get(URLS["spectrum"]))
    cf = density_per_count(fit["gamma"], fit["unabsorbed_flux_cgs"], fit["count_rate"], ENERGY_KEV)
    rows = read_qdp_rows(curve)
    nu = ENERGY_KEV / H_KEV_S

    header = [
        f"# Swift/XRT (PC mode) afterglow of GRB 210204A, target ID {TARGET_ID}, from the UK Swift Science Data Centre",
        f"# (Evans et al. 2007, 2009): light curve {URLS['curve']},",
        f"# spectrum {URLS['spectrum']}. Built by fetch_xray.py.",
        f"# time_s: since the Fermi/GBM trigger (MJD {TRIGGER_MJD}); the UKSSDC time zero, Swift MET {t0_met}",
        f"# ({t0.isot} UTC), is {offset_s:+.3f} s from it. Bin times are mean photon arrival times.",
        f"# flux_ujy: unabsorbed flux density at {ENERGY_KEV:g} keV ({nu:.6e} Hz) = count rate x {cf:.5e} Jy/(ct/s).",
        f"# Conversion from the time-averaged spectrum: photon index {fit['gamma']:.4f}, unabsorbed 0.3-10 keV flux",
        f"# {fit['unabsorbed_flux_cgs']:.4e} erg/cm2/s at {fit['count_rate']:.5f} ct/s, N_H "
        f"{fit['nh_galactic_1e22']:.4f}e22 (Galactic) and {fit['nh_intrinsic_1e22']:.4f}e22 cm-2 at z = 0.876.",
        "# Errors: 1-sigma statistical, from the count rate. The conversion's own uncertainty is common to all points.",
        "time_s,time_err_minus_s,time_err_plus_s,band,rate_cps,rate_err_cps,flux_ujy,flux_err_ujy,counts,exposure_s",
    ]
    lines = []
    for r in rows:
        time, t_plus, t_minus, rate, rate_err = r[0], r[1], r[2], r[3], r[4]
        counts, exposure = r[10], r[12]
        lines.append(
            f"{time + offset_s:.3f},{-t_minus:.3f},{t_plus:.3f},xrt:{ENERGY_KEV:g}keV,{rate:.6e},{abs(rate_err):.6e},"
            f"{rate * cf * 1e6:.6g},{abs(rate_err) * cf * 1e6:.6g},{int(counts)},{exposure:.3f}"
        )
    args.out.write_text("\n".join(header + lines) + "\n")
    print(f"{len(lines)} XRT points written to {args.out}; {cf:.5e} Jy per ct/s at {ENERGY_KEV:g} keV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
