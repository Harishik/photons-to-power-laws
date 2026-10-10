# Data for the GRB 221009A case study

All files here are public data, committed so the analysis runs offline and in
CI. `../fetch_data.py` rebuilds everything except the literature tables.

| File | Content | Source |
|---|---|---|
| `ztf/ztf_*_sciimg.fits` | 120″ cutouts of ZTF single-epoch science images (field 590, CCD 11, quadrant 4), g and r, 2022-10-09 to 2022-11-12 | [IRSA ZTF archive](https://irsa.ipac.caltech.edu/Missions/ztf.html) |
| `ztf/ztf_ref_*_refimg.fits` | Cutouts of the deep pre-burst ZTF reference images for the same quadrant | IRSA |
| `ztf_manifest.csv` | IRSA metadata for every image covering the position, including five late epochs IRSA does not serve | IRSA |
| `panstarrs_dr1.csv` | Pan-STARRS1 DR1 mean PSF photometry (g, r, i) within 1.6′ | [MAST](https://catalogs.mast.stsci.edu/panstarrs/) |
| `laskar2023_liverpool.csv` | Liverpool Telescope griz photometry, transcribed from Table 2 of Laskar et al. (2023) | [doi:10.3847/2041-8213/acbfad](https://doi.org/10.3847/2041-8213/acbfad) |
| `oconnor2023_oir.csv` | Lowell LMI, Gemini GMOS-S and Flamingos-2, Palomar WIRC and HST WFC3 photometry, transcribed from the table of optical and infrared observations of O'Connor et al. (2023), with the registry filter curve that models each band | [doi:10.1126/sciadv.adi1405](https://doi.org/10.1126/sciadv.adi1405) |

Coordinates: RA 288.26452°, Dec +19.77350° (Swift/UVOT, GCN Circular 32632).
Trigger: Fermi/GBM, 2022-10-09 13:16:59 UT (GCN Circular 32636).

## Acknowledgements

The ZTF images are public Phase-II survey data obtained with the Samuel Oschin
48-inch Telescope at Palomar Observatory as part of the Zwicky Transient
Facility project, supported by the U.S. National Science Foundation and a
collaboration of partner institutions, with operations conducted by COO, IPAC,
and UW. They were retrieved from the NASA/IPAC Infrared Science Archive. Work
that publishes results from these data should include the
[official ZTF acknowledgement](https://irsa.ipac.caltech.edu/data/ZTF/docs/releases/dr09/src/dr09.html)
and cite Masci et al. 2019 (PASP 131, 018003).

Pan-STARRS1 data come from the PS1 public science archive at the Mikulski
Archive for Space Telescopes (Chambers et al. 2016, arXiv:1612.05560;
Flewelling et al. 2020, ApJS 251, 7); see the
[PS1 acknowledgement](https://panstarrs.stsci.edu/) for the full text.
