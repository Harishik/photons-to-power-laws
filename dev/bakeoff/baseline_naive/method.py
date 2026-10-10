"""Baseline: resample the reference, match its flux scale and background, and subtract (no PSF matching)."""

import numpy as np
from scipy import ndimage


def subtract(sci, sci_wcs, ref, ref_wcs, *, sci_var=None, ref_var=None, sci_fwhm=None, ref_fwhm=None):
    ny, nx = sci.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    ra, dec = sci_wcs.pixel_to_world_values(xx.ravel(), yy.ravel())
    rx, ry = ref_wcs.world_to_pixel_values(ra, dec)
    r = ndimage.map_coordinates(ref, [ry, rx], order=3, mode="nearest").reshape(sci.shape)
    a = np.vstack([r.ravel(), np.ones(r.size)]).T
    good = np.isfinite(sci.ravel()) & np.isfinite(r.ravel())
    coef, *_ = np.linalg.lstsq(a[good], sci.ravel()[good], rcond=None)
    return {"diff": sci - (coef[0] * r + coef[1])}
