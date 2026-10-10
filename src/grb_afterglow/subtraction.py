"""Difference imaging: subtract a deep reference image to remove host and static light.

A transient on top of a galaxy, or blended with field stars, cannot be measured
cleanly in the science image alone. Image subtraction removes everything that
did not change: the reference (template) image is resampled onto the science
pixel grid, matched to the science image's PSF and photometric scale with a
convolution kernel, and subtracted. Only the transient survives.

The kernel uses the delta-function basis of Bramich (2008, MNRAS 386, L77):
every kernel pixel is a free parameter, fitted together with a smooth
differential background by weighted linear least squares,

.. math::

    I_{\\rm blurrier}(x) = \\sum_k K_k\\, I_{\\rm sharper}(x + o_k) + \\sum_j b_j P_j(x),

so no assumption about the PSF shapes is needed. The sharper image is the one
that is convolved, because a kernel can only broaden; within
``FWHM_TOLERANCE`` of each other the deeper template is convolved. Four
refinements keep the fit unbiased when the convolved image is noisy:

* an errors-in-variables correction (Fuller 1987, *Measurement Error Models*):
  noise in the convolved image biases a least-squares kernel towards zero,
  which would leave residuals of every static source;
* Laplacian regularisation of the kernel (Becker et al. 2012, MNRAS 425, 1341)
  where the data carry no information;
* iterative sigma clipping of the transient, variable stars and cosmic rays;
* a refit of the photometric scale on bright pixels only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
from scipy import linalg, ndimage

FWHM_TOLERANCE = 0.05  # relative FWHM difference treated as matched seeing
_RHO_HALF = 2  # half-size of the stored noise autocorrelation of a resampled image


@dataclass
class DifferenceImage:
    """The result of subtracting a template from a science image.

    ``data = as_difference_psf(science) - matched_template`` holds in both
    matching directions.

    Attributes
    ----------
    data
        Difference image on the science pixel grid, in science counts: a
        transient of total flux F (science counts) has total flux F here.
    fwhm
        PSF FWHM of ``data`` in pixels.
    psf_from
        ``"science"`` if the template was convolved to the science PSF (so
        ``data`` has the science PSF), ``"template"`` if the science image was
        convolved to the template PSF.
    matched_template
        What was subtracted: the template resampled, PSF-matched where it was
        the convolved image, background-matched, and scaled to science counts.
    valid
        Pixels covered by both images and unaffected by kernel edges.
    kernel
        The fitted convolution kernel (``scipy.ndimage.correlate`` orientation),
        in units of the target image's counts per convolved-image count.
    variance
        Per-pixel variance of ``data``, neglecting the pixel-to-pixel noise
        correlation that convolution introduces.
    info
        Method diagnostics: ``flux_ratio`` (template counts per science count),
        ``kernel_half``, ``n_fit`` (pixels in the kernel fit) and ``reg_scale``.
    """

    data: np.ndarray
    fwhm: float
    psf_from: str
    matched_template: np.ndarray
    valid: np.ndarray
    kernel: np.ndarray | None = None
    variance: np.ndarray | None = None
    info: dict = field(default_factory=dict)

    def as_difference_psf(self, image: np.ndarray) -> np.ndarray:
        """Give a science-grid image the PSF of the difference image.

        Returns ``image`` unchanged if the template was matched to the science
        image, and ``image`` convolved with the normalised kernel otherwise. Stars
        in the result are measured exactly as the transient is in ``data``,
        which ties a measurement on the difference image to the science
        image's zero point without assumptions about the PSF wings.
        """
        image = np.asarray(image, dtype=float)
        if self.psf_from == "science" or self.kernel is None:
            return image
        return ndimage.correlate(image, self.kernel / self.kernel.sum(), mode="nearest")


def _source_coords(wcs_from: WCS, wcs_to: WCS, shape: tuple[int, int]) -> np.ndarray:
    """``(2, ny, nx)`` (row, column) position in ``wcs_from`` pixels of every ``wcs_to`` pixel."""
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    px, py = wcs_from.world_to_pixel_values(*wcs_to.pixel_to_world_values(xx.ravel(), yy.ravel()))
    return np.array([np.reshape(py, shape), np.reshape(px, shape)], dtype=float)


def _interpolate(image: np.ndarray, coords: np.ndarray, *, order: int, margin: float) -> tuple[np.ndarray, np.ndarray]:
    src = np.asarray(image, dtype=float)
    finite = np.isfinite(src)
    filled = np.where(finite, src, np.nanmedian(src)) if not finite.all() else src
    out = ndimage.map_coordinates(filled, coords, order=order, mode="nearest")
    h, w = src.shape
    inside = (coords[1] >= margin) & (coords[1] <= w - 1 - margin)
    inside &= (coords[0] >= margin) & (coords[0] <= h - 1 - margin)
    bad = ndimage.map_coordinates((~finite).astype(float), coords, order=1, mode="constant", cval=1.0)
    return out, inside & (bad < 1e-3)


def resample_to_grid(
    image: np.ndarray, wcs_from: WCS, wcs_to: WCS, shape: tuple[int, int], *, order: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Resample ``image`` (with WCS ``wcs_from``) onto the pixel grid of ``wcs_to``.

    Uses cubic-spline interpolation of the input at the sky position of every
    output pixel. Returns the resampled image and a boolean map of output
    pixels whose interpolation stencil lies fully inside finite input data.
    """
    return _interpolate(image, _source_coords(wcs_from, wcs_to, shape), order=order, margin=2.0)


def _resampling_noise(in_shape: tuple[int, int], coords: np.ndarray, order: int = 3, n_side: int = 3):
    """Variance factor and normalised autocorrelation of resampled white noise.

    Interpolation is linear, so its effect on white noise follows from its
    impulse response: unit impulses on a sparse grid of input pixels are
    resampled and the autocorrelation of the response is averaged over them.
    """
    h = _RHO_HALF
    ny, nx = coords.shape[1:]
    impulses = np.zeros(in_shape)
    for oy in np.linspace(0.25 * ny, 0.75 * ny, n_side).astype(int):
        for ox in np.linspace(0.25 * nx, 0.75 * nx, n_side).astype(int):
            iy, ix = int(np.rint(coords[0, oy, ox])), int(np.rint(coords[1, oy, ox]))
            if 0 <= iy < in_shape[0] and 0 <= ix < in_shape[1]:
                impulses[iy, ix] = 1.0
    resp = ndimage.map_coordinates(impulses, coords, order=order, mode="constant")
    core = resp[h : ny - h, h : nx - h]
    acf = np.array(
        [
            [np.sum(core * resp[h + dy : ny - h + dy, h + dx : nx - h + dx]) for dx in range(-h, h + 1)]
            for dy in range(-h, h + 1)
        ]
    )
    return acf[h, h] / max(impulses.sum(), 1.0), acf / acf[h, h]


def variance_model(image: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Per-pixel noise variance of an image that still contains its sky.

    ``sigma_sky**2 + max(I_smooth - sky, 0) / g_eff``, with a robust sky noise
    ``sigma_sky`` and the effective gain implied by Poisson sky noise,
    ``g_eff = sky / sigma_sky**2`` (an upper bound on the source shot noise when
    read noise contributes; omitted if the image looks sky-subtracted).
    ``I_smooth`` is a 3x3 box average, so the weights do not correlate with the
    pixel noise.
    """
    image = np.asarray(image, dtype=float)
    sel = np.isfinite(image) if valid is None else (valid & np.isfinite(image))
    _, sky, std = sigma_clipped_stats(image[sel], sigma=3.0, stdfunc="mad_std")
    inv_gain = std**2 / sky if sky > 10.0 * std else 0.0
    smooth = ndimage.uniform_filter(np.where(sel, image, sky), 3)
    return std**2 + inv_gain * np.maximum(smooth - sky, 0.0)


def _poly_terms(shape: tuple[int, int], degree: int) -> list[np.ndarray]:
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    u = (xx - (nx - 1) / 2) / max((nx - 1) / 2, 1)
    v = (yy - (ny - 1) / 2) / max((ny - 1) / 2, 1)
    return [u**i * v**j for i in range(degree + 1) for j in range(degree + 1 - i)]


def _laplacian_penalty(half: int) -> np.ndarray:
    """``L.T @ L`` for the 5-point Laplacian ``L`` of a (2h+1)^2 kernel that is zero outside."""
    n = 2 * half + 1
    lap = -4.0 * np.eye(n * n)
    for i in range(n):
        for j in range(n):
            for ii, jj in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= ii < n and 0 <= jj < n:
                    lap[i * n + j, ii * n + jj] = 1.0
    return lap.T @ lap


def _correlated_pairs(offsets, rho):
    """Index pairs of kernel pixels whose regressor noise is correlated, and that correlation."""
    h = rho.shape[0] // 2
    pairs = [
        (k, m, rho[ly - ky + h, lx - kx + h])
        for k, (ky, kx) in enumerate(offsets)
        for m, (ly, lx) in enumerate(offsets)
        if abs(ly - ky) <= h and abs(lx - kx) <= h and rho[ly - ky + h, lx - kx + h] != 0.0
    ]
    k, m, r = zip(*pairs, strict=True)
    return np.array(k), np.array(m), np.array(r)


def fit_kernel(
    conv: np.ndarray,
    target: np.ndarray,
    conv_var: np.ndarray,
    target_var: np.ndarray,
    fit_mask: np.ndarray,
    half: int,
    *,
    rho: np.ndarray | None = None,
    degree: int = 1,
    lam: float = 0.01,
    clip: float = 4.0,
    n_iter: int = 6,
    eiv: bool = True,
) -> dict:
    """Fit a delta-function-basis kernel and a polynomial differential background.

    Solves ``target ~ correlate(conv, K) + sum_j b_j P_j`` by weighted least
    squares over ``fit_mask`` (weights ``1 / (target_var + correlate(conv_var,
    K**2))``), iterating the weights and the sigma clipping.

    Parameters
    ----------
    conv, target
        Image to convolve (sky-subtracted, zero where invalid) and image to match.
    conv_var, target_var
        Per-pixel noise variance of ``conv`` and ``target``.
    fit_mask
        Pixels that may enter the fit (kernel footprint inside valid data).
    half
        Kernel half-width; the kernel has ``(2 half + 1)**2`` free pixels.
    rho
        ``(5, 5)`` normalised noise autocorrelation of ``conv`` (white if omitted).
    degree
        Degree of the 2-D polynomial background.
    lam
        Laplacian regularisation in units of the mean per-pixel noise term of
        the normal matrix, so it only acts where noise dominates; it is raised
        tenfold until the noise-corrected matrix is positive definite.
    clip
        Sigma-clipping threshold for the normalised residuals.
    n_iter
        Maximum number of reweighting and clipping iterations.
    eiv
        Apply the errors-in-variables correction for the noise in ``conv``.

    Returns
    -------
    dict
        ``kernel``, ``background`` (image), ``mask`` (pixels in the final fit),
        ``n_fit`` and ``reg_scale`` (factor by which ``lam`` had to be raised).
    """
    n = 2 * half + 1
    if rho is None:
        rho = np.zeros((2 * _RHO_HALF + 1,) * 2)
        rho[_RHO_HALF, _RHO_HALF] = 1.0
    offsets = [(dy, dx) for dy in range(-half, half + 1) for dx in range(-half, half + 1)]
    nk = len(offsets)
    pad, vpad = np.pad(conv, half), np.pad(conv_var, half)
    polys = _poly_terms(conv.shape, degree)
    penalty = _laplacian_penalty(half)
    pk, pm, pr = _correlated_pairs(offsets, rho)

    kernel = np.zeros((n, n))
    kernel[half, half] = 1.0
    mask = fit_mask.copy()
    reg_scale = 1.0
    for it in range(n_iter):
        var_tot = target_var + ndimage.correlate(conv_var, kernel**2, mode="constant")
        yy, xx = np.nonzero(mask)
        w = 1.0 / var_tot[yy, xx]
        a = np.empty((yy.size, nk + len(polys)))
        s_noise = np.empty(nk)  # sum_i w_i Var(conv[i + o_k])
        for k, (dy, dx) in enumerate(offsets):
            a[:, k] = pad[yy + half + dy, xx + half + dx]
            s_noise[k] = w @ vpad[yy + half + dy, xx + half + dx]
        for j, p in enumerate(polys):
            a[:, nk + j] = p[yy, xx]
        aw = a * w[:, None]
        normal, rhs = a.T @ aw, aw.T @ target[yy, xx]
        del a, aw
        noise = np.zeros((nk, nk))
        if eiv:
            np.add.at(noise, (pk, pm), pr * 0.5 * (s_noise[pk] + s_noise[pm]))
        reg = lam * s_noise.mean() * penalty
        for reg_scale in 10.0 ** np.arange(5):
            mat = normal.copy()
            mat[:nk, :nk] += reg_scale * reg - noise
            try:
                sol = linalg.cho_solve(linalg.cho_factor(mat), rhs)
                break
            except linalg.LinAlgError:
                continue
        else:  # no positive-definite correction found: plain (attenuated) least squares
            mat = normal.copy()
            mat[:nk, :nk] += reg
            sol, reg_scale = linalg.solve(mat, rhs, assume_a="sym"), np.nan
        kernel = sol[:nk].reshape(n, n)
        background = sum(c * p for c, p in zip(sol[nk:], polys, strict=True))
        model = ndimage.correlate(conv, kernel, mode="constant") + background
        resid = (target - model) / np.sqrt(target_var + ndimage.correlate(conv_var, kernel**2, mode="constant"))
        new = fit_mask & ~ndimage.binary_dilation(fit_mask & (np.abs(resid) > clip))
        if it > 0 and np.array_equal(new, mask):
            break
        mask = new
    return {
        "kernel": kernel,
        "background": background,
        "mask": mask,
        "n_fit": int(mask.sum()),
        "reg_scale": float(reg_scale),
    }


def refine_scale(
    conv: np.ndarray,
    target: np.ndarray,
    kernel: np.ndarray,
    mask: np.ndarray,
    deep: np.ndarray,
    deep_var: np.ndarray,
    *,
    degree: int = 1,
    target_var: np.ndarray | None = None,
    nsig: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-fit the kernel normalisation (photometric ratio) on bright-source pixels.

    With the kernel shape fixed, ``target - B = s * correlate(conv, K / sum(K))``
    holds pixel by pixel, so the ratio of the sums of both sides over bright
    pixels estimates ``s`` without the attenuation that noise causes in a
    least-squares fit and without relying on the noise model. Bright pixels are
    selected on ``deep`` (the background-free, deeper side); the polynomial
    background is re-fitted alongside. The kernel is returned unchanged if fewer
    than 50 bright pixels are found.
    """
    shape = ndimage.correlate(conv, kernel / kernel.sum(), mode="constant")
    core = ndimage.uniform_filter(deep, 3) > nsig * np.sqrt(np.maximum(deep_var, 0.0) / 9.0)
    sel = ndimage.binary_dilation(core, iterations=2) & mask
    polys = _poly_terms(conv.shape, degree)
    sw = np.ones(int(mask.sum())) if target_var is None else 1.0 / np.sqrt(target_var[mask])
    design = np.array([p[mask] for p in polys]).T * sw[:, None]

    def fit_background(scale):
        coef = np.linalg.lstsq(design, (target - scale * shape)[mask] * sw, rcond=None)[0]
        return sum(c * p for c, p in zip(coef, polys, strict=True))

    scale = float(kernel.sum())
    if sel.sum() >= 50:
        for _ in range(3):  # scale and background are coupled; alternating converges in a few steps
            scale = float(np.sum((target - fit_background(scale))[sel]) / np.sum(shape[sel]))
    return kernel * scale / kernel.sum(), fit_background(scale)


def kernel_half_width(science_fwhm: float, template_fwhm: float) -> int:
    """Kernel half-width (3-7 px): 1.5 times the larger FWHM covers Moffat-like wings."""
    return int(np.clip(np.ceil(1.5 * max(science_fwhm, template_fwhm)), 3, 7))


def subtract_images(
    science: np.ndarray,
    science_wcs: WCS,
    template: np.ndarray,
    template_wcs: WCS,
    *,
    science_fwhm: float,
    template_fwhm: float,
    science_var: np.ndarray | None = None,
    template_var: np.ndarray | None = None,
    exclude: np.ndarray | None = None,
    half: int | None = None,
    degree: int = 1,
    lam: float = 0.01,
    clip: float = 4.0,
) -> DifferenceImage:
    """Subtract a PSF- and flux-matched template from a science image.

    Parameters
    ----------
    science, template
        Calibrated images in their own counts, sky included. Non-finite pixels
        are allowed and excluded.
    science_wcs, template_wcs
        Their celestial WCS; the template is resampled onto the science grid
        (flux-conserving cubic-spline interpolation).
    science_fwhm, template_fwhm
        PSF FWHM of each image in its own pixels. The sharper image is
        convolved; within ``FWHM_TOLERANCE`` of each other the (deeper)
        template is.
    science_var, template_var
        Per-pixel variances from a noise model; estimated from the images with
        :func:`variance_model` when omitted.
    exclude
        Boolean map on the science grid of pixels to keep out of the kernel
        fit, such as the surroundings of the transient: its faint wings
        survive the sigma clipping and would otherwise be partly absorbed by
        the kernel, over-subtracting the transient.
    half
        Kernel half-width; by default from the FWHMs (:func:`kernel_half_width`).
    degree
        Degree of the polynomial differential background.
    lam, clip
        Regularisation strength and clipping threshold, see :func:`fit_kernel`.
    """
    sci = np.asarray(science, dtype=float)
    shape = sci.shape
    tpl = np.asarray(template, dtype=float)
    coords = _source_coords(template_wcs, science_wcs, shape)
    area = proj_plane_pixel_area(science_wcs) / proj_plane_pixel_area(template_wcs)
    tpl_r, valid_t = _interpolate(tpl, coords, order=3, margin=1.0)
    tpl_r *= area
    factor, rho_t = _resampling_noise(tpl.shape, coords)
    tvar_native = variance_model(tpl) if template_var is None else np.asarray(template_var, dtype=float)
    var_t = ndimage.map_coordinates(tvar_native, coords, order=1, mode="nearest") * area**2 * factor
    valid_s = np.isfinite(sci)
    sci_f = np.where(valid_s, sci, 0.0)
    var_s = variance_model(sci_f, valid_s) if science_var is None else np.asarray(science_var, dtype=float)
    half = kernel_half_width(science_fwhm, template_fwhm) if half is None else int(half)

    convolve_science = science_fwhm < (1.0 - FWHM_TOLERANCE) * template_fwhm
    if convolve_science:
        conv, cvar, cvalid, rho, target, tvar, tvalid = sci_f, var_s, valid_s, None, tpl_r, var_t, valid_t
    else:
        conv, cvar, cvalid, rho, target, tvar, tvalid = tpl_r, var_t, valid_t, rho_t, sci_f, var_s, valid_s
    _, sky, _ = sigma_clipped_stats(conv[cvalid], sigma=3.0, stdfunc="mad_std")
    conv0 = np.where(cvalid, conv - sky, 0.0)
    cvar = np.where(cvalid, cvar, 0.0)
    good = ndimage.binary_erosion(cvalid, np.ones((2 * half + 1,) * 2, bool), border_value=0) & tvalid
    usable = good if exclude is None else good & ~np.asarray(exclude, dtype=bool)
    fit = fit_kernel(conv0, target, cvar, tvar, usable, half, rho=rho, degree=degree, lam=lam, clip=clip)
    kernel, background = fit["kernel"], fit["background"]
    if convolve_science:
        deep, deep_var = target - background, tvar
    else:
        deep, deep_var = ndimage.correlate(conv0, kernel, mode="constant"), cvar * kernel.sum() ** 2
    kernel, background = refine_scale(
        conv0, target, kernel, fit["mask"], deep, deep_var, degree=degree, target_var=tvar
    )
    matched = ndimage.correlate(conv0, kernel, mode="constant") + background
    corr = 1.0  # noise correlation of the resampled template raises the variance of its convolution
    if rho is not None:
        acf = ndimage.correlate(np.pad(kernel, _RHO_HALF), kernel, mode="constant")
        c = _RHO_HALF + half
        window = acf[c - _RHO_HALF : c + _RHO_HALF + 1, c - _RHO_HALF : c + _RHO_HALF + 1]
        corr = float(np.sum(window * rho) / np.sum(kernel**2))
    var_m = ndimage.correlate(cvar, kernel**2, mode="constant") * corr
    scale = float(kernel.sum())
    if convolve_science:  # ``matched`` is the science image in template counts with the template PSF
        diff = (matched - target) / scale
        var = (var_m + tvar) / scale**2
        subtracted = (target - background) / scale + sky
        fwhm, psf_from, flux_ratio = float(template_fwhm), "template", scale
    else:
        diff = target - matched
        var = var_m + tvar
        subtracted = matched
        fwhm, psf_from, flux_ratio = float(science_fwhm), "science", 1.0 / scale
    return DifferenceImage(
        data=np.where(good, diff, np.nan),
        fwhm=fwhm,
        psf_from=psf_from,
        matched_template=subtracted,
        valid=good,
        kernel=kernel,
        variance=np.where(good, var, np.nan),
        info={
            "flux_ratio": flux_ratio,
            "kernel_half": half,
            "n_fit": fit["n_fit"],
            "reg_scale": fit["reg_scale"],
        },
    )
