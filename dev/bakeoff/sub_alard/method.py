"""Image subtraction with the Alard & Lupton (1998) Gaussian-basis convolution kernel.

Alard, C. & Lupton, R. H. 1998, ApJ, 503, 325: "A method for optimal image subtraction".

The reference (template) image is resampled onto the science pixel grid through
the two WCSs. The image with the sharper PSF is then convolved with a kernel

.. math::

    K(u, v) = \\sum_n a_n \\, e^{-(u^2 + v^2) / 2\\sigma_n^2} \\, u^{p_n} v^{q_n}

(a few Gaussians, each multiplied by polynomials up to a given degree) so that
it matches the blurrier image (if the two PSFs differ by less than a 0.5 px
Gaussian the deeper reference is convolved), and a low-order differential
background is fitted at the same time. Both are linear in their coefficients
and are solved by weighted least squares over every unmasked pixel of the
overlap, iterating the noise model and sigma-clipping outliers (the transient,
variable or saturated stars, cosmic rays).

Two refinements over plain least squares matter for unbiased photometry:

* **Errors in variables.** The convolved image is noisy, so ordinary least
  squares shrinks the kernel towards zero (regression dilution) by an amount
  that depends on the noise-to-signal ratio, i.e. a few per cent of photometric
  scale. The expected noise contribution to the normal matrix is computed
  from the variance map, the weights and the kernel basis, and subtracted
  ("corrected least squares"); in directions of the kernel space where the
  data barely exceed the noise the correction is capped instead of amplified.
* **Resampled noise.** Spline resampling lowers and correlates the noise
  of the reference. Its variance factor and autocorrelation are propagated by
  resampling a fixed-seed white-noise field through the same WCS mapping, and
  used both in the weights and in the errors-in-variables correction.

The difference image is always returned on the science grid and in science
counts, so a transient of F science counts has total flux F in it:

* reference convolved: ``D = S - (K * R + B)`` has the science PSF;
* science convolved: ``D = (K * S + B - R) / sum(K)`` has the reference PSF,
  where ``sum(K)`` is the fitted reference-to-science photometric ratio.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg, ndimage

from grb_afterglow.detection import estimate_background
from grb_afterglow.subtraction import resample_to_grid

FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))
# Quintic-spline resampling: cubic splines broaden an undersampled (FWHM ~2 px) PSF by ~1.5%,
# quintic by ~0.4%, which keeps the difference-image PSF equal to the reference PSF.
RESAMPLE_ORDER = 5


def estimate_variance(image: np.ndarray) -> np.ndarray:
    """Per-pixel variance of an image without a noise map.

    Sky noise comes from the local background RMS; source shot noise is added
    with the effective gain ``sky / rms**2`` implied by a sky-limited image
    (it is dropped if the sky level is not positive, e.g. for sky-subtracted data).
    """
    bkg = estimate_background(image)
    rms2 = np.maximum(bkg.rms, 1e-12) ** 2
    sky = float(np.median(bkg.background))
    gain = sky / float(np.median(rms2))
    if not (np.isfinite(gain) and gain > 0):
        return rms2
    source = np.clip(ndimage.uniform_filter(image, 3) - bkg.background, 0.0, None)
    return rms2 + source / gain


def resampling_noise(src_shape, src_wcs, dst_wcs, dst_shape, *, order=5, max_lag=2, seed=0):
    """Variance factor and noise autocorrelation imprinted by spline resampling.

    A fixed-seed unit white-noise field is resampled with the same mapping as the
    data. Returns a smooth (quadratic) map of the resampled variance and the
    normalised autocorrelation for lags up to ``max_lag`` pixels.
    """
    noise = np.random.default_rng(seed).standard_normal(src_shape)
    out, valid = resample_to_grid(noise, src_wcs, dst_wcs, dst_shape, order=order)
    P = poly_terms(dst_shape, 2)
    coef = np.linalg.lstsq(P[:, valid].T, out[valid] ** 2, rcond=None)[0]
    factor = np.clip(np.tensordot(coef, P, axes=1), 0.1, 1.5)
    n = 2 * max_lag + 1
    acf = np.zeros((n, n))
    ny, nx = dst_shape
    core = out[max_lag : ny - max_lag, max_lag : nx - max_lag]
    for dy in range(-max_lag, max_lag + 1):
        for dx in range(-max_lag, max_lag + 1):
            shifted = out[max_lag + dy : ny - max_lag + dy, max_lag + dx : nx - max_lag + dx]
            acf[max_lag + dy, max_lag + dx] = np.mean(core * shifted)
    return factor, acf / acf[max_lag, max_lag]


def poly_terms(shape, order):
    """Polynomial images ``x**p * y**q`` (``p + q <= order``) on [-1, 1] coordinates."""
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    x = (xx - 0.5 * (nx - 1)) / (0.5 * nx)
    y = (yy - 0.5 * (ny - 1)) / (0.5 * ny)
    return np.array([x**p * y**q for p in range(order + 1) for q in range(order + 1 - p)])


class KernelBasis:
    """Alard-Lupton basis: Gaussians of widths ``sigmas`` times ``u**p v**q`` with ``p + q <= degree``.

    The raw functions are separable, which makes convolving an image with all of
    them cheap; they are strongly non-orthogonal, so the fit uses the orthonormal
    combinations ``T @ raw`` (numerically degenerate directions are dropped).
    """

    def __init__(self, sigmas, degrees, half):
        self.half = int(half)
        u = np.arange(-self.half, self.half + 1, dtype=float)
        self.factors, self.terms = [], []
        for gi, (s, d) in enumerate(zip(sigmas, degrees, strict=True)):
            g = np.exp(-0.5 * (u / s) ** 2)
            self.factors.append([g * (u / s) ** p for p in range(d + 1)])
            self.terms += [(gi, p, q) for p in range(d + 1) for q in range(d + 1 - p)]
        raw = np.array([np.outer(self.factors[g][q], self.factors[g][p]) for g, p, q in self.terms])
        flat = raw.reshape(len(raw), -1)
        U, S, _ = np.linalg.svd(flat, full_matrices=False)
        keep = S > 1e-8 * S[0]
        self.T = (U[:, keep] / S[keep]).T
        self.kernels = (self.T @ flat).reshape(-1, 2 * self.half + 1, 2 * self.half + 1)

    def convolve(self, image):
        """Return the image convolved with every (orthonormalised) basis kernel."""
        xpass, out = {}, np.empty((len(self.terms), *image.shape))
        for k, (g, p, q) in enumerate(self.terms):
            if (g, p) not in xpass:
                xpass[(g, p)] = ndimage.convolve1d(image, self.factors[g][p], axis=1, mode="nearest")
            out[k] = ndimage.convolve1d(xpass[(g, p)], self.factors[g][q], axis=0, mode="nearest")
        return np.tensordot(self.T, out, axes=1)

    def noise_matrix(self, weight, var, acf):
        """Expected noise part of the normal matrix, ``E[sum_p w (B_i * n)(B_j * n)]``.

        ``var`` is the per-pixel variance of the convolved image and ``acf`` its
        normalised noise autocorrelation (``[[1.0]]`` for white noise).
        """
        h, ny, nx = self.half, *weight.shape
        size = (2 * ny, 2 * nx)
        # R(u) = sum_p w(p) var(p - u) for kernel offsets |u| <= half (FFT cross-correlation)
        cc = np.fft.irfft2(np.fft.rfft2(weight, s=size) * np.conj(np.fft.rfft2(var, s=size)), s=size)
        R = np.roll(cc, (h, h), axis=(0, 1))[: 2 * h + 1, : 2 * h + 1]
        nk, m = len(self.kernels), acf.shape[0] // 2
        left = (self.kernels * R).reshape(nk, -1)
        N = np.zeros((nk, nk))
        pad = np.pad(self.kernels, ((0, 0), (m, m), (m, m)))
        for dy in range(-m, m + 1):
            for dx in range(-m, m + 1):
                if acf[m + dy, m + dx] != 0.0:
                    shifted = pad[:, m + dy : m + dy + 2 * h + 1, m + dx : m + dx + 2 * h + 1]
                    N += acf[m + dy, m + dx] * left @ shifted.reshape(nk, -1).T
        return 0.5 * (N + N.T)


def fit_kernel(
    target,
    conv,
    var_target,
    var_conv,
    acf_conv,
    fit_mask,
    basis,
    *,
    bg_order=1,
    kappa=5.0,
    grow=1,
    max_iter=8,
    eiv=True,
):
    """Fit ``target ~ K * conv + B`` by iterated, sigma-clipped weighted least squares.

    Returns ``(kernel, model, background, var_model, used_mask, chi_scale)`` where
    ``model = K * conv + B``, ``var_model`` the per-pixel variance of
    ``target - model`` and ``chi_scale`` the robust RMS of the normalised residuals.
    """
    C = basis.convolve(conv)
    P = poly_terms(target.shape, bg_order)
    nk = len(C)
    design = np.concatenate([C, P])
    use = fit_mask.copy()
    var_model = var_target + var_conv
    for _ in range(max_iter):
        w = np.where(use, 1.0 / var_model, 0.0)
        A = design[:, use]
        Aw = A * w[use]
        M, b = Aw @ A.T, Aw @ target[use]
        # eliminate the background terms (Schur complement), solve for the kernel
        Mgg_inv = np.linalg.inv(M[nk:, nk:])
        S = M[:nk, :nk] - M[:nk, nk:] @ Mgg_inv @ M[nk:, :nk]
        r = b[:nk] - M[:nk, nk:] @ Mgg_inv @ b[nk:]
        if eiv:
            # S = signal + noise; in the N-orthonormal eigenbasis divide by (lambda - 1)
            # where the signal dominates, and keep the least-squares 1/lambda-like value (>= 1) elsewhere.
            try:
                lam, V = linalg.eigh(S, basis.noise_matrix(w, var_conv, acf_conv))
                a = V @ ((V.T @ r) / np.maximum(lam - 1.0, 1.0))
            except linalg.LinAlgError:  # noise matrix not positive definite: plain least squares
                a = np.linalg.solve(S, r)
        else:
            a = np.linalg.solve(S, r)
        g = Mgg_inv @ (b[nk:] - M[nk:, :nk] @ a)
        kernel = np.tensordot(a, basis.kernels, axes=1)
        background = np.tensordot(g, P, axes=1)
        model = np.tensordot(a, C, axes=1) + background
        var_model = var_target + ndimage.convolve(var_conv, kernel**2, mode="nearest")
        z = (target - model) / np.sqrt(var_model)
        zu = z[use]
        chi = 1.4826 * float(np.median(np.abs(zu - np.median(zu))))
        new_use = fit_mask & ~ndimage.binary_dilation(np.abs(z) > kappa * max(chi, 1.0), iterations=grow)
        if np.array_equal(new_use, use):
            break
        use = new_use
    return kernel, model, background, var_model, use, chi


def alard_lupton_subtract(
    sci,
    sci_wcs,
    ref,
    ref_wcs,
    *,
    sci_fwhm,
    ref_fwhm,
    sci_var=None,
    ref_var=None,
    convolve="auto",
    sigma_factors=(0.5, 1.0, 2.0, 4.0),
    degrees=(6, 4, 2, 2),
    half=None,
    bg_order=1,
    kappa=5.0,
    clip_grow=1.0,
    eiv=True,
):
    """Subtract a reference image from a science image with an Alard-Lupton kernel.

    Parameters
    ----------
    sci, ref : numpy.ndarray
        Science and reference images (background included, any photometric scale).
    sci_wcs, ref_wcs : astropy.wcs.WCS
        Their celestial WCSs; the reference is resampled onto the science grid.
    sci_fwhm, ref_fwhm : float
        PSF FWHMs in pixels of each image's own grid (pixel scales are assumed equal).
    sci_var, ref_var : numpy.ndarray, optional
        Per-pixel variances; estimated from the images if omitted.
    convolve : {"auto", "science", "reference"}
        Which image is convolved. ``"auto"`` convolves the sharper one; when the
        matching kernel would be unresolved (Gaussian-equivalent width below
        0.5 px) the deeper reference is convolved, so ``D`` keeps the science PSF.
    sigma_factors, degrees : sequence
        Basis Gaussian widths in units of the matching width
        ``sqrt(|sigma_ref**2 - sigma_sci**2|)`` (floored at 0.7 px), and the
        polynomial degree attached to each Gaussian. The first three follow
        Alard & Lupton (6, 4, 2); the fourth, wide one carries the kernel wings
        needed between PSFs with extended (Moffat-like) wings, without which
        the kernel sum (photometric ratio) comes out ~0.5% low.
    half : int, optional
        Kernel half-size in pixels; default ``max(3 * widest sigma, 2 * FWHM)``.
    bg_order : int
        Polynomial order of the differential background.
    kappa : float
        Sigma-clipping threshold of the iterative fit.
    clip_grow : float
        Clipped pixels are grown by this many FWHM (of the blurrier image), so
        the wings of a clipped transient or badly matched star leave the fit too.
    eiv : bool
        Apply the errors-in-variables correction for noise in the convolved image.

    Returns
    -------
    dict
        ``diff`` (science grid, science counts), ``var`` (per-pixel variance of
        ``diff``, ignoring correlations), ``fwhm`` (PSF FWHM of ``diff``),
        ``kernel``, ``flux_ratio`` (reference counts per science count),
        ``convolved`` (which image), ``valid`` (pixels covered by both images and
        away from kernel edges), ``n_fit`` and ``chi_scale`` (fit diagnostics).
    """
    sci = np.asarray(sci, dtype=float)
    ref = np.asarray(ref, dtype=float)
    shape = sci.shape
    sci_bad = ~np.isfinite(sci)
    sci = np.where(sci_bad, np.nanmedian(sci), sci)
    ref_filled = np.where(np.isfinite(ref), ref, np.nanmedian(ref))
    sci_var = estimate_variance(sci) if sci_var is None else np.asarray(sci_var, dtype=float)
    ref_var = estimate_variance(ref_filled) if ref_var is None else np.asarray(ref_var, dtype=float)

    ref_r, inside = resample_to_grid(ref, ref_wcs, sci_wcs, shape, order=RESAMPLE_ORDER)
    factor, ref_acf = resampling_noise(ref.shape, ref_wcs, sci_wcs, shape, order=RESAMPLE_ORDER)
    ref_var_r = resample_to_grid(
        np.where(np.isfinite(ref_var), ref_var, np.nanmedian(ref_var)), ref_wcs, sci_wcs, shape, order=1
    )[0]
    ref_var_r = np.maximum(ref_var_r * factor, 1e-12)
    sci_var = np.maximum(np.where(np.isfinite(sci_var), sci_var, np.nanmax(sci_var)), 1e-12)

    s_sci, s_ref = sci_fwhm / FWHM_PER_SIGMA, ref_fwhm / FWHM_PER_SIGMA
    s_match = np.sqrt(abs(s_ref**2 - s_sci**2))
    if convolve == "auto":
        convolve = "science" if (s_sci < s_ref and s_match >= 0.5) else "reference"
    if convolve not in ("science", "reference"):
        raise ValueError(f"convolve must be 'auto', 'science' or 'reference', not {convolve!r}")
    sigmas = [f * max(s_match, 0.7) for f in sigma_factors]
    if half is None:
        half = int(np.ceil(max(3.0 * max(sigmas), 2.0 * max(sci_fwhm, ref_fwhm))))
    basis = KernelBasis(sigmas, degrees, half)

    valid = inside & ~sci_bad
    away = np.zeros(shape, dtype=bool)
    away[half:-half, half:-half] = True
    fit_mask = valid & away
    grow = max(1, round(clip_grow * max(sci_fwhm, ref_fwhm)))
    kw = dict(bg_order=bg_order, kappa=kappa, grow=grow, eiv=eiv)
    if convolve == "science":
        white = np.ones((1, 1))
        kern, model, _, var_m, use, chi = fit_kernel(ref_r, sci, ref_var_r, sci_var, white, fit_mask, basis, **kw)
        ratio = float(kern.sum())
        diff, var, fwhm = (model - ref_r) / ratio, var_m / ratio**2, ref_fwhm
    else:
        kern, model, _, var_m, use, chi = fit_kernel(sci, ref_r, sci_var, ref_var_r, ref_acf, fit_mask, basis, **kw)
        ratio = 1.0 / float(kern.sum())
        diff, var, fwhm = sci - model, var_m, sci_fwhm
    return {
        "diff": diff,
        "var": var,
        "fwhm": float(fwhm),
        "kernel": kern,
        "flux_ratio": ratio,
        "convolved": convolve,
        "valid": fit_mask,
        "n_fit": int(use.sum()),
        "chi_scale": chi,
    }


def subtract(sci, sci_wcs, ref, ref_wcs, *, sci_var, ref_var, sci_fwhm, ref_fwhm):
    """Bake-off entry point: Alard-Lupton subtraction with default settings.

    See :func:`alard_lupton_subtract`. Returns a dict with ``diff`` (science grid,
    science counts), ``var`` and ``fwhm`` (PSF FWHM of ``diff`` in pixels).
    """
    res = alard_lupton_subtract(
        sci, sci_wcs, ref, ref_wcs, sci_fwhm=float(sci_fwhm), ref_fwhm=float(ref_fwhm), sci_var=sci_var, ref_var=ref_var
    )
    # The harness applies kernels with scipy.ndimage.correlate; this kernel was fitted for convolution.
    res["kernel"] = res["kernel"][::-1, ::-1]
    return res
