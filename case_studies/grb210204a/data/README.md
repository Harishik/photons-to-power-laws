# Data for the GRB 210204A case study

All files here are public data, committed so the analysis runs offline. `../fetch_data.py` rebuilds
them.

| File | Content | Source |
|---|---|---|
| `ztf/ztf_*_sciimg.fits` | 120″ cutouts of all 47 ZTF single-epoch science images covering the position from 2021-02-02 to 2021-03-07 (field 515, CCD 5, quadrant 1), in g, r and i | [IRSA ZTF archive](https://irsa.ipac.caltech.edu/Missions/ztf.html) |
| `ztf/ztf_ref_*_refimg.fits` | Cutouts of the deep ZTF reference images of the same quadrant, one per filter | IRSA |
| `ztf_manifest.csv` | IRSA metadata for every image | IRSA |
| `panstarrs_dr1.csv` | Pan-STARRS1 DR1 mean PSF photometry (g, r, i) within 1.6′ | [MAST](https://catalogs.mast.stsci.edu/panstarrs/) |

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

Pan-STARRS1 data come from the PS1 public science archive at the Mikulski Archive for Space Telescopes
(Chambers et al. 2016, arXiv:1612.05560; Flewelling et al. 2020, ApJS 251, 7).
