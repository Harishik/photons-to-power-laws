"""Baseline: plain aperture photometry (r = 1 FWHM, masked sky annulus), no aperture correction."""

import numpy as np

from grb_afterglow.detection import estimate_background, make_source_mask
from grb_afterglow.photometry import forced_photometry


def psf_photometry(image, positions, *, fwhm, source_xy, gain=None, var=None):
    bkg = estimate_background(image)
    mask = make_source_mask(image, bkg, dilate=max(2, int(np.rint(1.5 * fwhm))))
    t = forced_photometry(image, positions, r_aper=fwhm, r_in=3 * fwhm, r_out=5 * fwhm, gain=gain, sky_mask=mask)
    return {"flux": np.asarray(t["net_counts"]), "flux_err": np.asarray(t["net_counts_err"])}
