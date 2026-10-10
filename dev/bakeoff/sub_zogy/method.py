"""ZOGY proper image subtraction (Zackay, Ofek & Gal-Yam 2016, ApJ 830, 27).

Given a new (science) image ``N`` and a reference ``R`` of the same field, the
proper difference image is, in Fourier space (ZOGY eq. 13),

.. math::

    \\hat D = \\frac{F_r \\hat P_r \\hat N - F_n \\hat P_n \\hat R}
                  {\\sqrt{\\sigma_n^2 F_r^2 |\\hat P_r|^2 + \\sigma_r^2 F_n^2 |\\hat P_n|^2}},

where ``P_n``, ``P_r`` are the PSFs, ``sigma_n``, ``sigma_r`` the background
noise levels and ``F_n``, ``F_r`` the photometric scales of the two images.
``D`` has white noise and the PSF ``P_D`` of ZOGY eq. 14. It is the optimal
statistic for transient detection and photometry when the noise is dominated by
the background, and it never deconvolves either image, so it is stable whichever
image has the better seeing.

Steps:

1. resample the reference onto the science grid through the WCS (flux-conserving
   quintic spline, which preserves the PSF far better than a cubic one);
2. measure each image's sky (a constant: a mesh background would eat star wings
   and extended hosts differently in the two images) and white noise;
3. fit a pixel-integrated elliptical Moffat PSF jointly to bright stars of each
   image, fitting all detected neighbours at the same time and adding the
   neighbours revealed by the fit residuals, and rejecting stars that fit badly;
4. measure the flux ratio ``F_r / F_n`` and a residual translation from bright
   stars on the cross-convolved images ``N * P_r`` and ``R * P_n``, which share
   one PSF, so the ratio needs no aperture correction;
5. form ``D`` with zero-padded FFTs, rescaled so a transient of ``F`` science
   counts has total flux ``F`` with PSF ``P_D``.
"""

from __future__ import annotations

import numpy as np
from astropy.stats import sigma_clipped_stats
from scipy import fft, ndimage
from scipy.optimize import least_squares
from scipy.signal import fftconvolve

from grb_afterglow.detection import detect_sources, estimate_background, make_source_mask

_SUB = 3  # sub-pixel samples per axis for pixel integration of the PSF model


# --------------------------------------------------------------------------- geometry and noise


def _ref_coordinates(sci_wcs, ref_wcs, shape):
    """Reference pixel coordinates of every science pixel centre, and the area Jacobian."""
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    world = sci_wcs.pixel_to_world_values(xx.ravel(), yy.ravel())
    rx, ry = (np.asarray(c).reshape(shape) for c in ref_wcs.world_to_pixel_values(*world))
    jac = np.abs(np.gradient(rx, axis=1) * np.gradient(ry, axis=0) - np.gradient(rx, axis=0) * np.gradient(ry, axis=1))
    return rx, ry, jac  # jac = reference pixels per science pixel


def _resample(ref, rx, ry, jac, shift=(0.0, 0.0)):
    """Flux-conserving quintic-spline resampling of ``ref`` onto the science grid.

    ``shift`` is a residual translation of the reference (science pixels).
    Returns the resampled image (zero outside the footprint) and its coverage mask.
    """
    sx, sy = shift
    cx = rx - np.gradient(rx, axis=1) * sx - np.gradient(rx, axis=0) * sy
    cy = ry - np.gradient(ry, axis=1) * sx - np.gradient(ry, axis=0) * sy
    ny_r, nx_r = ref.shape
    valid = (cx >= 2) & (cx <= nx_r - 3) & (cy >= 2) & (cy <= ny_r - 3)
    out = ndimage.map_coordinates(np.nan_to_num(ref), [cy, cx], order=5, mode="nearest") * jac
    return np.where(valid, out, 0.0), valid


def _noise_model(image, var=None):
    """Constant sky level, white background noise ``sigma`` and inverse gain of an image.

    ``var`` (per-pixel variance), when given, sets ``sigma`` and the shot-noise
    term; otherwise both come from the image, with the inverse gain approximated
    by ``sigma**2 / sky`` (Poisson-limited sky).
    """
    image = np.nan_to_num(np.asarray(image, float))
    mask = make_source_mask(image, estimate_background(image), dilate=2)
    _, sky, sigma = sigma_clipped_stats(image[~mask], sigma=3.0)
    if var is not None:
        var = np.asarray(var, float)
        sigma = np.sqrt(np.median(var[~mask]))
        bright = image - sky > 30 * sigma
        inv_gain = np.median((var[bright] - sigma**2) / (image - sky)[bright]) if bright.sum() > 20 else 0.0
    else:
        inv_gain = sigma**2 / sky if sky > sigma**2 else 0.0
    return float(sky), float(sigma), max(float(inv_gain), 0.0)


# --------------------------------------------------------------------------- PSF model


def _moffat(par, dx, dy, half, sub=_SUB):
    """Pixel-integrated elliptical Moffat with unit integral, centred at (dx, dy) on a (2h+1)^2 grid.

    ``par = (fwhm, beta, g1, g2)``; ``g1, g2`` shear the circular profile at
    constant area, so ``fwhm`` is the geometric-mean FWHM.
    """
    fwhm, beta, g1, g2 = par
    alpha = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    norm = np.sqrt(max(1.0 - g1 * g1 - g2 * g2, 1e-6))
    off = (np.arange(sub) + 0.5) / sub - 0.5
    grid = np.arange(-half, half + 1)
    yy = (grid[:, None, None, None] + off[None, None, :, None]) - dy
    xx = (grid[None, :, None, None] + off[None, None, None, :]) - dx
    q = ((1 - g1) * xx * xx - 2 * g2 * xx * yy + (1 + g1) * yy * yy) / (norm * alpha**2)
    return ((beta - 1.0) / (np.pi * alpha**2) * (1.0 + q) ** (-beta)).mean(axis=(2, 3))


def _detect(image, fwhm):
    """DAOFIND sources of a sky-subtracted image: ``(xy, flux, peak)``, brightest first."""
    src = detect_sources(image, fwhm=fwhm, threshold_sigma=5.0, background=np.zeros_like(image))
    if len(src) == 0:
        return np.empty((0, 2)), np.empty(0), np.empty(0)
    xy = np.column_stack([np.asarray(src["x_centroid"]), np.asarray(src["y_centroid"])])
    return xy, np.asarray(src["flux"], float), np.asarray(src["peak"], float)


def _stamp_fit(par, stamp, w, centre, nbr, half):
    """Variable-projection model of one stamp: Moffat star + neighbours + constant.

    Positions are fixed; the amplitudes and the constant are solved by weighted
    linear least squares. Returns the weighted residual vector.
    """
    cols = [_moffat(par, dx, dy, half).ravel() for dx, dy in [centre, *nbr]]
    a = np.column_stack([*cols, np.ones(stamp.size)]) * w[:, None]
    coef, *_ = np.linalg.lstsq(a, stamp.ravel() * w, rcond=None)
    return stamp.ravel() * w - a @ coef


def _new_neighbours(par, star, half, fwhm, nsig=5.0):
    """Positions (stamp offsets) of significant positive residual peaks not yet modelled."""
    stamp, _, centre, nbr = star
    res = _stamp_fit(par, *star, half).reshape(stamp.shape)
    psf = _moffat(par, 0.0, 0.0, max(int(np.ceil(1.5 * fwhm)), 2))
    snr = ndimage.correlate(res, psf, mode="constant") / np.sqrt(np.sum(psf**2))
    peaks = (snr == ndimage.maximum_filter(snr, size=3)) & (snr > nsig)
    new = np.argwhere(peaks)[:, ::-1] - half
    known = np.array([centre, *nbr])
    far = [np.min(np.hypot(*(known - p).T)) > fwhm and np.hypot(*(p - centre)) > 2 * fwhm for p in new]
    return [p.astype(float) for p, f in zip(new, far, strict=True) if f]


def _fit_psf(image, valid, sigma, inv_gain, fwhm_guess, max_stars=25):
    """Fit a pixel-integrated elliptical Moffat PSF to the bright stars of an image.

    Bright, unblended, fully covered stars are fitted with one shared shape.
    Every detected source within reach of a stamp is fitted at the same time
    (fixed position, free amplitude), and neighbours too faint or too close to
    have been detected are added from the significant peaks of the fit
    residuals. Star centres and the shared shape are refined alternately, and
    stars that still fit badly (saturated, blended, variable) are dropped.

    Returns
    -------
    numpy.ndarray
        ``(fwhm, beta, g1, g2)``; a circular ``beta = 3`` Moffat of the guessed
        FWHM if fewer than two stars are usable.
    """
    par = np.array([fwhm_guess, 3.0, 0.0, 0.0])
    xy, flux, peak = _detect(image, fwhm_guess)
    half = max(int(np.ceil(3.5 * fwhm_guess)), 6)
    cover = ndimage.binary_erosion(np.pad(valid, 1), iterations=half + 1)[1:-1, 1:-1]
    stars = []
    for i, (x, y) in enumerate(xy):
        ix, iy = int(np.rint(x)), int(np.rint(y))
        d = np.hypot(*(xy - xy[i]).T)
        blended = np.any((d > 0.5) & (d < 1.5 * fwhm_guess) & (flux > 0.1 * flux[i]))
        if cover[iy, ix] and peak[i] > 20 * sigma and not blended:
            stamp = image[iy - half : iy + half + 1, ix - half : ix + half + 1]
            w = 1.0 / np.sqrt(sigma**2 + inv_gain * np.maximum(stamp, 0.0)).ravel()
            near = (d > 0.5) & (d < half + 2 * fwhm_guess)
            stars.append([stamp, w, np.array([x - ix, y - iy]), list(xy[near] - [ix, iy])])
        if len(stars) == max_stars:
            break
    if len(stars) < 2:
        return par
    lo, hi = [0.5, 1.2, -0.5, -0.5], [4.0 * fwhm_guess + 2.0, 15.0, 0.5, 0.5]
    for it in range(4):
        for st in stars:  # centres, with the shape fixed
            st[2] = least_squares(
                lambda c, par=par, st=st: _stamp_fit(par, st[0], st[1], c, st[3], half),
                st[2],
                bounds=([-1.5, -1.5], [1.5, 1.5]),
                x_scale=0.1,
            ).x
        par = least_squares(
            lambda p, stars=stars: np.concatenate([_stamp_fit(p, *st, half) for st in stars]),
            np.clip(par, np.add(lo, 1e-3), np.subtract(hi, 1e-3)),
            bounds=(lo, hi),
            x_scale=[0.1, 0.3, 0.02, 0.02],
        ).x
        if it == 3:
            break
        for st in stars:
            st[3] = st[3] + _new_neighbours(par, st, half, par[0])
        chi2 = np.array([np.mean(_stamp_fit(par, *st, half) ** 2) for st in stars])
        good = chi2 < 3.0 * np.median(chi2)
        if 2 <= good.sum() < len(stars):
            stars = [st for st, g in zip(stars, good, strict=True) if g]
    return par


def _psf_fwhm(psf):
    """FWHM (geometric mean) of a centred PSF stamp, from a pixel-integrated Moffat fit."""
    half = psf.shape[0] // 2
    flat = psf.ravel()

    def resid(p):
        m = _moffat(p, 0.0, 0.0, half).ravel()
        return flat - m * (m @ flat) / (m @ m)

    guess = max(2.0 * np.sqrt(np.sum(psf >= 0.5 * psf.max()) / np.pi), 1.0)
    return float(least_squares(resid, [guess, 3.0, 0.0, 0.0], bounds=([0.3, 1.2, -0.5, -0.5], [50, 15, 0.5, 0.5])).x[0])


# --------------------------------------------------------------------------- flux ratio and alignment


def _embed(stamp, pad):
    """Zero-padded copy of a centred, odd-sized kernel with its centre moved to index (0, 0)."""
    out = np.zeros(pad)
    h = stamp.shape[0] // 2
    out[: 2 * h + 1, : 2 * h + 1] = stamp
    return np.roll(out, (-h, -h), axis=(0, 1))


def _conv(image, kernel, pad):
    """Linear (zero-padded) convolution of ``image`` with a centred kernel, cropped to ``image``."""
    out = fft.irfft2(fft.rfft2(image, s=pad) * fft.rfft2(_embed(kernel, pad)), s=pad)
    return out[: image.shape[0], : image.shape[1]]


def _weighted_clipped_mean(val, err, nsig=3.0):
    """Inverse-variance weighted mean with iterative ``nsig`` clipping; returns (mean, kept)."""
    ok = np.ones(val.size, bool)
    for _ in range(10):
        w = 1.0 / err[ok] ** 2
        mean = np.sum(w * val[ok]) / w.sum()
        new = np.abs(val - mean) < nsig * err
        if new.sum() < max(2, 0.5 * val.size) or (new == ok).all():
            break
        ok = new
    w = 1.0 / err[ok] ** 2
    return float(np.sum(w * val[ok]) / w.sum()), ok


def _ratio_and_shift(n_img, r_img, var_n, var_r, p_n, p_r, xy, fwhm_c, valid, pad):
    """Flux ratio ``F_r / F_n`` and residual translation of ``R`` from bright stars.

    Both are measured on the cross-convolved images ``N' = N * P_r`` and
    ``R' = R * P_n``, which share the PSF ``P_n * P_r`` whatever the seeing, so
    the ratio is free of aperture corrections. On a stamp around each star,
    ``R' = beta N' + s . grad(N') + c`` is solved by linear least squares, and
    the noise of ``beta`` is propagated exactly from both variance maps through
    the convolutions. Per-star ratios are combined by a weighted, 3-sigma
    clipped mean (0.3 % per-star floor), so a transient, a variable star or a
    residual blend cannot bias it; the translation ``s`` (science pixels) is
    the weighted mean over the retained stars.
    """
    n_c, r_c = _conv(n_img, p_r, pad), _conv(r_img, p_n, pad)
    gy, gx = np.gradient(n_c)
    hs = int(np.ceil(1.5 * fwhm_c))
    hp = p_n.shape[0] // 2
    reach = hs + int(np.ceil(0.5 * fwhm_c))
    cover = ndimage.binary_erosion(np.pad(valid, 1), iterations=reach + 1)[1:-1, 1:-1]
    vn_pad, vr_pad = np.pad(var_n, hp), np.pad(var_r, hp)
    rows = []
    for x, y in np.rint(xy).astype(int):
        if not (0 <= x < valid.shape[1] and 0 <= y < valid.shape[0] and cover[y, x]):
            continue
        sl = np.s_[y - hs : y + hs + 1, x - hs : x + hs + 1]
        a = np.column_stack([n_c[sl].ravel(), gx[sl].ravel(), gy[sl].ravel(), np.ones(n_c[sl].size)])
        u = np.linalg.pinv(a)
        coef = u @ r_c[sl].ravel()
        if coef[0] <= 0:
            continue
        # beta = u0 . (R * P_n): noise reaches R through (u0 * P_n) and N through beta (u0 * P_r)
        u0 = u[0].reshape(2 * hs + 1, 2 * hs + 1)
        big = np.s_[y - hs : y + hs + 2 * hp + 1, x - hs : x + hs + 2 * hp + 1]
        var = np.sum(fftconvolve(u0, p_n) ** 2 * vr_pad[big]) + coef[0] ** 2 * np.sum(
            fftconvolve(u0, p_r) ** 2 * vn_pad[big]
        )
        if coef[0] / np.sqrt(var) > 30:
            rows.append((coef[0], coef[1] / coef[0], coef[2] / coef[0], np.sqrt(var)))
    if len(rows) < 2:
        return None, (0.0, 0.0)
    beta, sx, sy, err = np.array(rows).T
    mean, ok = _weighted_clipped_mean(beta, np.hypot(err, 0.003 * beta))
    w = 1.0 / err[ok] ** 2
    return mean, (float(np.sum(w * sx[ok]) / w.sum()), float(np.sum(w * sy[ok]) / w.sum()))


# --------------------------------------------------------------------------- main entry point


def subtract(sci, sci_wcs, ref, ref_wcs, *, sci_var=None, ref_var=None, sci_fwhm=None, ref_fwhm=None):
    """ZOGY proper image subtraction of ``ref`` from ``sci``.

    Parameters
    ----------
    sci, ref : numpy.ndarray
        Science and reference images (sky included), on any pixel grids.
    sci_wcs, ref_wcs : astropy.wcs.WCS
        Their world coordinate systems.
    sci_var, ref_var : numpy.ndarray or None
        Per-pixel variances. Estimated from the images when ``None``.
    sci_fwhm, ref_fwhm : float or None
        Seeing FWHM of each image in its own pixels, as measured by the caller;
        starting guesses for the PSF fits and the scale of the returned ``fwhm``.

    Returns
    -------
    dict
        ``diff``: proper difference image on the science grid in science counts
        (a transient of ``F`` science counts has total ``F``, with PSF ``P_D``),
        zero where the reference does not cover the science image;
        ``var``: its per-pixel background variance (``inf`` where not covered);
        ``fwhm``: FWHM of ``P_D`` in science pixels, on the scale of the
        supplied FWHMs (the model FWHM of ``P_D`` times the ratio of supplied to
        model FWHM of the input image whose PSF it is closest to), so that it
        compares directly with FWHMs from the caller's own estimator;
        diagnostics ``flux_ratio`` (``F_r / F_n``), ``shift`` (residual
        translation of the reference, px), ``psf_sci``/``psf_ref`` (Moffat
        ``fwhm, beta, g1, g2``) and ``psf_diff``/``psf`` (``P_D`` stamp);
        ``kernel`` (``K_n``, the filter applied to the science image, in
        ``scipy.ndimage.correlate`` orientation) and ``convolved = "science"``.
    """
    sci = np.asarray(sci, float)
    ref = np.asarray(ref, float)
    shape = sci.shape
    rx, ry, jac = _ref_coordinates(sci_wcs, ref_wcs, shape)
    area = float(np.median(jac))  # reference pixels per science pixel
    fn_guess = float(sci_fwhm or 3.0)
    fr_guess = float(ref_fwhm or 3.0) / np.sqrt(area)

    # Sky and noise; the reference noise is measured on its native grid, where it
    # is white, and converted to science-pixel units.
    sky_n, sig_n, ig_n = _noise_model(sci, sci_var)
    sky_r, sig_r, ig_r = _noise_model(ref, ref_var)
    sig_r *= np.sqrt(area)
    ref0 = np.nan_to_num(ref - sky_r)
    r_img, valid = _resample(ref0, rx, ry, jac)
    n_img = np.where(valid, np.nan_to_num(sci - sky_n), 0.0)
    var_n = sig_n**2 + ig_n * np.maximum(n_img, 0)

    shape_n = _fit_psf(n_img, valid, sig_n, ig_n, fn_guess)
    shape_r = _fit_psf(r_img, valid, sig_r, ig_r, fr_guess)
    half = int(np.ceil(4.5 * max(shape_n[0], shape_r[0]))) + 2
    p_n, p_r = _moffat(shape_n, 0.0, 0.0, half), _moffat(shape_r, 0.0, 0.0, half)
    p_n, p_r = p_n / p_n.sum(), p_r / p_r.sum()
    fwhm_c = float(np.hypot(shape_n[0], shape_r[0]))
    pad = tuple(fft.next_fast_len(s + 2 * half + 2 * int(np.ceil(3 * fwhm_c)) + 2) for s in shape)

    # Flux ratio and residual translation, iterated with re-resampling of R.
    xy, _, _ = _detect(n_img, fn_guess)
    shift = np.zeros(2)
    beta = None
    for _ in range(4):
        var_r = sig_r**2 + ig_r * np.maximum(r_img, 0)
        beta, ds = _ratio_and_shift(n_img, r_img, var_n, var_r, p_n, p_r, xy, fwhm_c, valid, pad)
        if beta is None or np.hypot(*ds) < 0.01:
            break
        shift += ds
        r_img, valid = _resample(ref0, rx, ry, jac, shift=tuple(shift))
        n_img = np.where(valid, np.nan_to_num(sci - sky_n), 0.0)
    if beta is None:  # no usable stars: fall back to the ratio of total fluxes
        beta = float(np.sum(r_img) / np.sum(n_img))

    # Proper difference image (ZOGY eq. 13), rescaled to science counts.
    pn_hat, pr_hat = fft.rfft2(_embed(p_n, pad)), fft.rfft2(_embed(p_r, pad))
    n_hat, r_hat = fft.rfft2(n_img, s=pad), fft.rfft2(r_img, s=pad)
    a2, b2 = sig_n**2 * beta**2, sig_r**2
    denom = np.sqrt(a2 * np.abs(pr_hat) ** 2 + b2 * np.abs(pn_hat) ** 2) + 1e-30
    scale = np.sqrt(a2 + b2) / beta  # F_n / F_D for F_n = 1, F_r = beta
    diff = fft.irfft2(scale * (beta * pr_hat * n_hat - pn_hat * r_hat) / denom, s=pad)[: shape[0], : shape[1]]
    p_d = np.roll(fft.irfft2(scale * beta * pr_hat * pn_hat / denom, s=pad), (half, half), axis=(0, 1))
    p_d = p_d[: 2 * half + 1, : 2 * half + 1]
    # The kernel applied to the science image, D = K_n * N - K_r * R (unit sum: fluxes are preserved).
    k_n = np.roll(fft.irfft2(scale * beta * pr_hat / denom, s=pad), (half, half), axis=(0, 1))
    k_n = k_n[: 2 * half + 1, : 2 * half + 1]
    _, med, _ = sigma_clipped_stats(diff[valid], sigma=3.0)
    diff = np.where(valid, diff - med, 0.0)

    # FWHM of P_D on the caller's FWHM scale, anchored to the closer input PSF.
    f_d = _psf_fwhm(p_d)
    near_sci = abs(f_d - shape_n[0]) <= abs(f_d - shape_r[0])
    given, model = (sci_fwhm, shape_n[0]) if near_sci else (ref_fwhm and ref_fwhm / np.sqrt(area), shape_r[0])
    fwhm_d = f_d * given / model if given else f_d
    return {
        "diff": diff,
        "var": np.where(valid, sig_n**2 + sig_r**2 / beta**2, np.inf),
        "fwhm": float(fwhm_d),
        "flux_ratio": float(beta),
        "shift": (float(shift[0]), float(shift[1])),
        "psf_sci": shape_n,
        "psf_ref": shape_r,
        "psf_diff": p_d,
        "psf": p_d,
        # for the harness: the science-side kernel in scipy.ndimage.correlate orientation
        "kernel": k_n[::-1, ::-1],
        "convolved": "science",
    }
