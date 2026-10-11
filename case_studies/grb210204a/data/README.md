# Data for the GRB 210204A case study

All files here are public data, committed so the analysis runs offline. `../fetch_data.py` rebuilds
the ZTF and Pan-STARRS1 files and `../fetch_xray.py` the Swift/XRT one.

| File | Content | Source |
|---|---|---|
| `ztf/ztf_*_sciimg.fits` | 120″ cutouts of all 47 ZTF single-epoch science images covering the position from 2021-02-02 to 2021-03-07 (field 515, CCD 5, quadrant 1), in g, r and i | [IRSA ZTF archive](https://irsa.ipac.caltech.edu/Missions/ztf.html) |
| `ztf/ztf_ref_*_refimg.fits` | Cutouts of the deep ZTF reference images of the same quadrant, one per filter | IRSA |
| `ztf_manifest.csv` | IRSA metadata for every image | IRSA |
| `panstarrs_dr1.csv` | Pan-STARRS1 DR1 mean PSF photometry (g, r, i) within 1.6′ | [MAST](https://catalogs.mast.stsci.edu/panstarrs/) |
| `swift_xrt.csv` | Swift/XRT 0.3–10 keV light curve (PC mode, 19 bins at 1.9–3.5 d; target 00014055) and the unabsorbed flux density at 3 keV, converted with the time-averaged spectral fit (photon index 1.72) | [UK Swift Science Data Centre](https://www.swift.ac.uk/xrt_curves/00014055/) |
| `kumar2022_radio.csv` | The three uGMRT detections at 1.25 and 0.65 GHz, 16–33 d, transcribed from Table A4 of Kumar et al. (2022, MNRAS 513, 2777), with the caveats of the transcription in its header | [arXiv:2204.07587](https://arxiv.org/abs/2204.07587) |

Coordinates: RA 117.08054°, Dec +11.40949° (ZTF21aagwbjr, GCN Circular 29405).
Trigger: Fermi/GBM, 2021-02-04 06:29:25 UT (Kumar et al. 2022).
Galactic extinction: E(B−V) = 0.028 mag (Schlafly & Finkbeiner 2011, from the IRSA dust service).

## Acknowledgements

The ZTF images are public survey data obtained with the Samuel Oschin 48-inch Telescope at Palomar
Observatory as part of the Zwicky Transient Facility project, supported by the U.S. National Science
Foundation and a collaboration of partner institutions, with operations conducted by COO, IPAC, and
UW. They were retrieved from the NASA/IPAC Infrared Science Archive. Work that publishes results from
these data should include the
[official ZTF acknowledgement](https://irsa.ipac.caltech.edu/data/ZTF/docs/releases/dr09/src/dr09.html)
and cite Masci et al. 2019 (PASP 131, 018003).

This work made use of data supplied by the UK Swift Science Data Centre at the University of
Leicester (Evans et al. 2007, A&A 469, 379; Evans et al. 2009, MNRAS 397, 1177).

Pan-STARRS1 data come from the PS1 public science archive at the Mikulski Archive for Space Telescopes
(Chambers et al. 2016, arXiv:1612.05560; Flewelling et al. 2020, ApJS 251, 7).
