"""Publication-style figures for light curves, calibration, and closure relations.

Every figure function takes ``theme="light"`` or ``theme="dark"`` so README
figures can follow the viewer's colour scheme. Colours come from one validated
palette: bands are ordered by wavelength (bluer band, bluer hue), each band also
has its own marker shape, so identity never rests on colour alone.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from os import PathLike

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.ticker import FixedLocator, FormatStrFormatter, FuncFormatter, LogLocator, NullFormatter, NullLocator

from .closure import CLOSURE_RELATIONS
from .fitting import FitResult, band_frequency_hz
from .lightcurve import AB_ZEROPOINT_UJY, SECONDS_PER_DAY, LightCurve

# Font lookups for italic mathtext scan every installed font; keep that quiet.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


@dataclass(frozen=True)
class Theme:
    surface: str
    ink: str
    ink_secondary: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, ...]
    image_cmap: str = "gray"


THEMES = {
    "light": Theme(
        surface="#fcfcfb",
        ink="#0b0b0b",
        ink_secondary="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        series=("#2a78d6", "#1baf7a", "#eb6834", "#4a3aa7"),
    ),
    "dark": Theme(
        surface="#1a1a19",
        ink="#ffffff",
        ink_secondary="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        series=("#3987e5", "#199e70", "#d95926", "#9085e9"),
    ),
}

BAND_ORDER = ("u", "g", "r", "i", "z", "y")
BAND_MARKERS = {"g": "o", "r": "s", "i": "D", "z": "^", "y": "v", "u": "p"}
_FONT_CANDIDATES = ("Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans")


def _font() -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in _FONT_CANDIDATES if f in available), "DejaVu Sans")


def band_style(band: str, theme: str = "light", index: int = 0) -> tuple[str, str]:
    """Colour and marker for a band, ordered by wavelength."""
    pal = THEMES[theme].series
    key = str(band)[-1:]
    if key in ("g", "r", "i"):
        return pal[("g", "r", "i").index(key)], BAND_MARKERS[key]
    return pal[index % len(pal)], BAND_MARKERS.get(key, "o")


@contextmanager
def themed(theme: str = "light") -> Iterator[Theme]:
    """Matplotlib style context for a theme."""
    t = THEMES[theme]
    font = _font()
    rc = {
        "figure.facecolor": t.surface,
        "axes.facecolor": t.surface,
        "savefig.facecolor": t.surface,
        "axes.edgecolor": t.axis,
        "axes.labelcolor": t.ink_secondary,
        "axes.titlecolor": t.ink,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.titlelocation": "left",
        "axes.labelsize": 10.5,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": t.grid,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "xtick.color": t.muted,
        "ytick.color": t.muted,
        "xtick.labelcolor": t.ink_secondary,
        "ytick.labelcolor": t.ink_secondary,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": t.ink_secondary,
        "text.color": t.ink,
        "font.family": "sans-serif",
        "font.sans-serif": [font, "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": font,
        "mathtext.it": f"{font}:italic",
        "mathtext.bf": f"{font}:bold",
        "lines.linewidth": 1.8,
        "lines.solid_capstyle": "round",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }
    with plt.rc_context(rc):
        yield t


def format_log_time_axis(ax) -> None:
    """Label a logarithmic time axis with plain numbers (0.01, 0.1, 1, 10 ...).

    Spans shorter than about two decades also get 2 and 5 labels so the axis
    is never left with a single tick.
    """
    lo, hi = ax.get_xlim()
    subs = (1.0,) if np.log10(hi / lo) > 2.5 else (1.0, 2.0, 5.0)
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=subs))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 1.0))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _save(fig, savepath: str | PathLike[str] | None) -> None:
    if savepath:
        fig.savefig(savepath)


def _flux_to_mag(f):
    with np.errstate(divide="ignore", invalid="ignore"):
        return -2.5 * np.log10(np.clip(f, 1e-30, None) / AB_ZEROPOINT_UJY)


def _mag_to_flux(m):
    return AB_ZEROPOINT_UJY * 10 ** (-0.4 * np.asarray(m))


def _sorted_bands(lc: LightCurve) -> list[str]:
    def key(b):
        short = str(b)[-1:]
        return BAND_ORDER.index(short) if short in BAND_ORDER else len(BAND_ORDER)

    return sorted(lc.bands, key=key)


def plot_lightcurve(
    lc: LightCurve,
    fit: FitResult | None = None,
    *,
    alternative: FitResult | None = None,
    extra: Sequence[tuple[LightCurve, str]] = (),
    band_labels: dict[str, str] | None = None,
    model_range: tuple[float, float] | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    theme: str = "light",
    upper_limit_sigma: float = 3.0,
    show_residuals: bool = True,
    annotate: bool = True,
    figsize: tuple[float, float] = (8.0, 5.6),
    savepath: str | PathLike[str] | None = None,
):
    """Plot a light curve with its best-fit model and normalised residuals.

    Detections are markers with 1-sigma error bars; points below
    ``upper_limit_sigma`` are drawn as open downward triangles at the upper
    limit. If ``fit`` has a break, its time and 1-sigma range are marked.

    Parameters
    ----------
    lc
        The light curve to plot.
    fit
        Best-fit model, drawn per band.
    alternative
        A competing model (e.g. a single power law), drawn dashed for comparison.
    extra
        Additional ``(LightCurve, label)`` data sets, e.g. literature photometry,
        drawn as hollow neutral markers for comparison.
    band_labels
        Legend label per band (defaults to the band name).
    model_range
        ``(t_min, t_max)`` in days over which to draw the model, e.g. the
        range that was fitted. Defaults to the full plot.
    """
    with themed(theme) as t:
        if show_residuals and fit is not None:
            fig, (ax, rax) = plt.subplots(
                2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.06}
            )
        else:
            fig, ax = plt.subplots(figsize=figsize)
            rax = None

        bands = _sorted_bands(lc)
        days = lc.time_days
        tmin, tmax = days.min(), days.max()
        for extra_lc, _ in extra:
            tmin, tmax = min(tmin, extra_lc.time_days.min()), max(tmax, extra_lc.time_days.max())
        grid_days = np.geomspace(tmin / 1.4, tmax * 1.4, 400)
        model_days = grid_days
        if model_range is not None:
            model_days = np.geomspace(max(model_range[0], grid_days[0]), min(model_range[1], grid_days[-1]), 400)

        handles = []
        for i, band in enumerate(bands):
            color, marker = band_style(band, theme, i)
            sub = lc.select(band)
            det = sub.detected(upper_limit_sigma)
            d = sub.time_days
            ax.errorbar(
                d[det],
                sub.flux[det],
                yerr=sub.flux_err[det],
                fmt=marker,
                ms=6.5,
                color=color,
                mec=t.surface,
                mew=1.0,
                elinewidth=1.1,
                capsize=0,
                zorder=4,
            )
            if np.any(~det):
                lim = np.maximum(sub.flux[~det], 0) + upper_limit_sigma * sub.flux_err[~det]
                ax.scatter(d[~det], lim, marker="v", s=46, facecolor="none", edgecolor=color, lw=1.2, zorder=4)
            label = (band_labels or {}).get(band, str(band))
            if fit is not None:
                nu = band_frequency_hz(band) if fit.nu_ref else None
                model = fit.predict(model_days * SECONDS_PER_DAY, nu)
                ax.plot(model_days, model, color=color, lw=1.8, alpha=0.9, zorder=3)
                if rax is not None:
                    nu_pts = nu if nu is None else np.full(len(sub), nu)
                    resid = (sub.flux - fit.predict(sub.time, nu_pts)) / sub.flux_err
                    rax.scatter(d, resid, marker=marker, s=26, color=color, edgecolor=t.surface, lw=0.6, zorder=4)
            handles.append(
                Line2D([], [], color=color, marker=marker, ms=6.5, mec=t.surface, lw=1.8 if fit else 0, label=label)
            )

        for j, (extra_lc, extra_label) in enumerate(extra):
            color = t.ink_secondary if j == 0 else t.muted
            det = extra_lc.detected(upper_limit_sigma)
            ax.errorbar(
                extra_lc.time_days[det],
                extra_lc.flux[det],
                yerr=extra_lc.flux_err[det],
                fmt="o",
                ms=5.5,
                mfc="none",
                mec=color,
                color=color,
                mew=1.2,
                elinewidth=1.0,
                capsize=0,
                zorder=3,
            )
            handles.append(Line2D([], [], color=color, marker="o", mfc="none", ms=5.5, lw=0, label=extra_label))

        if np.any(~lc.detected(upper_limit_sigma)):
            handles.append(
                Line2D(
                    [],
                    [],
                    color=t.ink_secondary,
                    marker="v",
                    mfc="none",
                    lw=0,
                    ms=6,
                    label=f"{upper_limit_sigma:g}σ upper limit",
                )
            )

        if alternative is not None:
            ref_band = bands[0]
            nu = band_frequency_hz(ref_band) if alternative.nu_ref else None
            ax.plot(
                grid_days,
                alternative.predict(grid_days * SECONDS_PER_DAY, nu),
                color=t.muted,
                lw=1.3,
                ls=(0, (5, 3)),
                zorder=2,
            )
            handles.append(Line2D([], [], color=t.muted, lw=1.3, ls=(0, (5, 3)), label="single power law"))

        if fit is not None and "t_break" in fit.params:
            tb, tb_err = fit.params["t_break"] / SECONDS_PER_DAY, fit.errors["t_break"] / SECONDS_PER_DAY
            for a in (ax, rax) if rax is not None else (ax,):
                a.axvspan(tb - tb_err, tb + tb_err, color=t.muted, alpha=0.18, lw=0, zorder=1)
                a.axvline(tb, color=t.muted, lw=0.9, zorder=1)
            ax.annotate(
                "jet break",
                xy=(tb, 0),
                xycoords=("data", "axes fraction"),
                xytext=(5, 5),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=9,
                color=t.ink_secondary,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("Flux density (µJy)")
        ax.set_xlim(grid_days[0], grid_days[-1])
        format_log_time_axis(ax)
        all_flux = np.concatenate([lc.flux[lc.detected(upper_limit_sigma)]] + [e.flux for e, _ in extra])
        positive = all_flux[all_flux > 0]
        if positive.size:
            ax.set_ylim(max(positive.min(), 1e-3) / 3, positive.max() * 2.2)
        mag_ax = ax.secondary_yaxis("right", functions=(_flux_to_mag, _mag_to_flux))
        m_faint, m_bright = _flux_to_mag(np.array(ax.get_ylim()))
        step = 1 if (m_faint - m_bright) < 7 else 2
        ticks = np.arange(np.ceil(m_bright / step) * step, m_faint + 1e-9, step)
        mag_ax.yaxis.set_major_locator(FixedLocator(ticks))
        mag_ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
        mag_ax.yaxis.set_minor_locator(NullLocator())
        mag_ax.set_ylabel("AB magnitude")
        mag_ax.tick_params(colors=t.muted, labelcolor=t.ink_secondary)
        mag_ax.spines["right"].set_color(t.axis)

        ax.legend(handles=handles, loc="upper right", ncol=1, handlelength=2.2)
        if title:
            ax.set_title(title, pad=22 if subtitle else 8)
        if subtitle:
            ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9.5, color=t.ink_secondary, va="bottom")

        if annotate and fit is not None:
            ax.text(
                0.02,
                0.04,
                _fit_label(fit),
                transform=ax.transAxes,
                fontsize=9,
                color=t.ink_secondary,
                va="bottom",
                ha="left",
                linespacing=1.5,
            )

        if rax is not None:
            rax.axhspan(-1, 1, color=t.grid, alpha=0.6, lw=0, zorder=0)
            rax.axhline(0, color=t.axis, lw=0.9, zorder=1)
            rax.set_ylabel("Residual (σ)")
            lim = (
                max(3.5, float(np.nanmax(np.abs(rax.collections[-1].get_offsets()[:, 1]))) + 0.5)
                if rax.collections
                else 3.5
            )
            rax.set_ylim(-lim, lim)
            rax.set_xlabel("Time since trigger (days)")
        else:
            ax.set_xlabel("Time since trigger (days)")
        fig.align_ylabels()
        _save(fig, savepath)
        return fig, ax


def _fit_label(fit: FitResult) -> str:
    p, e = fit.params, fit.errors
    rows = []
    if "alpha" in p:
        rows.append(rf"$\alpha$ = {p['alpha']:.3f} ± {e['alpha']:.3f}")
    if "alpha1" in p:
        rows.append(rf"$\alpha_1$ = {p['alpha1']:.3f} ± {e['alpha1']:.3f}")
        rows.append(rf"$\alpha_2$ = {p['alpha2']:.2f} ± {e['alpha2']:.2f}")
        rows.append(rf"$t_b$ = {p['t_break'] / SECONDS_PER_DAY:.3f} ± {e['t_break'] / SECONDS_PER_DAY:.3f} d")
    if "beta" in p:
        rows.append(rf"$\beta$ = {p['beta']:.3f} ± {e['beta']:.3f}")
    rows.append(rf"$\chi^2$/dof = {fit.chi2:.1f}/{fit.dof}")
    return "\n".join(rows)


def plot_zeropoints(
    frames,
    *,
    truth_zeropoints: dict[str, float] | None = None,
    example_index: int | None = None,
    catalog_label: str | None = None,
    trend_title: str = "Zero point tracks the atmosphere",
    theme: str = "light",
    figsize: tuple[float, float] = (10.0, 4.0),
    savepath: str | PathLike[str] | None = None,
):
    """Zero-point diagnostics: one frame's solution, and the zero point over time.

    Parameters
    ----------
    frames
        :class:`~grb_afterglow.pipeline.FrameResult` objects.
    truth_zeropoints
        Optional mapping from frame path name to the true zero point. Because the
        aperture correction is absorbed into the zero point, the comparison is
        made relative to each band's median.
    example_index
        Which frame to show in the left panel (default: the one with most stars).
    catalog_label
        Axis label for the catalog magnitudes, e.g. ``"Pan-STARRS1 r"``.
    trend_title
        Title of the right-hand panel.
    """
    frames = list(frames)
    with themed(theme) as t:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.28})
        ex = frames[example_index] if example_index is not None else max(frames, key=lambda f: f.zeropoint.n_stars)
        color, marker = band_style(ex.band, theme)
        resid = ex.star_catalog_mag - ex.star_instrumental_mag
        used = ex.zeropoint.used if ex.zeropoint.used.size == resid.size else np.ones_like(resid, bool)
        ax0.axhspan(
            ex.zeropoint.value - ex.zeropoint.scatter,
            ex.zeropoint.value + ex.zeropoint.scatter,
            color=t.grid,
            alpha=0.7,
            lw=0,
        )
        ax0.axhline(ex.zeropoint.value, color=t.ink_secondary, lw=1.2)
        ax0.scatter(
            ex.star_catalog_mag[used],
            resid[used],
            s=30,
            marker=marker,
            color=color,
            edgecolor=t.surface,
            lw=0.7,
            zorder=3,
            label="used",
        )
        if np.any(~used):
            ax0.scatter(
                ex.star_catalog_mag[~used],
                resid[~used],
                s=34,
                marker="x",
                color=t.muted,
                lw=1.2,
                zorder=3,
                label="clipped",
            )
        ax0.set_xlabel(f"{catalog_label or f'Catalog {ex.band}'} magnitude (AB)")
        ax0.set_ylabel("Catalog − instrumental (mag)")
        ax0.set_title("Zero-point solution, one frame", pad=24)
        ax0.text(
            0.03,
            0.05,
            f"ZP = {ex.zeropoint.value:.3f} ± {ex.zeropoint.error:.3f}\n"
            f"scatter {ex.zeropoint.scatter:.3f} mag · {ex.zeropoint.n_stars} stars",
            transform=ax0.transAxes,
            fontsize=9,
            color=t.ink_secondary,
            va="bottom",
        )
        span = max(5 * ex.zeropoint.scatter, 0.15)
        ax0.set_ylim(ex.zeropoint.value - span, ex.zeropoint.value + span)
        if np.any(~used):
            ax0.legend(loc="upper left")

        bands = sorted({f.band for f in frames}, key=lambda b: BAND_ORDER.index(b[-1:]) if b[-1:] in BAND_ORDER else 9)
        handles = []
        for i, band in enumerate(bands):
            color, marker = band_style(band, theme, i)
            sub = [f for f in frames if f.band == band]
            tt = np.array([f.time for f in sub]) / SECONDS_PER_DAY
            zp = np.array([f.zeropoint.value for f in sub])
            zpe = np.array([f.zeropoint.error for f in sub])
            ref = np.median(zp)
            ax1.errorbar(
                tt, zp - ref, yerr=zpe, fmt=marker, ms=6, color=color, mec=t.surface, mew=0.8, elinewidth=1, zorder=3
            )
            handles.append(
                Line2D([], [], color=color, marker=marker, lw=0, ms=6, mec=t.surface, label=f"{band} measured")
            )
            if truth_zeropoints:
                tz = np.array([truth_zeropoints.get(str(f.path).split("/")[-1], np.nan) for f in sub])
                ax1.plot(tt, tz - np.nanmedian(tz), color=color, lw=1.2, alpha=0.7, zorder=2)
        if truth_zeropoints:
            handles.append(Line2D([], [], color=t.ink_secondary, lw=1.2, label="true transparency"))
        ax1.axhline(0, color=t.axis, lw=0.9)
        ax1.set_xscale("log")
        lo, hi = ax1.get_xlim()
        ax1.set_xlim(lo / 1.15, hi * 1.15)
        format_log_time_axis(ax1)
        ax1.set_xlabel("Time since trigger (days)")
        ax1.set_ylabel("ZP − band median (mag)")
        ax1.set_title(trend_title, pad=24)
        ax1.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.0),
            ncol=len(handles),
            fontsize=8.5,
            handletextpad=0.3,
            columnspacing=1.0,
            borderaxespad=0.2,
        )
        _save(fig, savepath)
        return fig, (ax0, ax1)


def _math_label(label: str) -> str:
    return label.replace("ν_m < ν < ν_c", r"$\nu_m < \nu < \nu_c$").replace("ν > ν_c", r"$\nu > \nu_c$")


def plot_closure_relations(
    alpha_beta: Sequence[tuple[str, float, float, float, float]],
    *,
    note: str | None = None,
    beta_range: tuple[float, float] = (0.2, 1.4),
    theme: str = "light",
    figsize: tuple[float, float] = (6.6, 5.0),
    savepath: str | PathLike[str] | None = None,
):
    """Plot measured ``(beta, alpha)`` points against the closure relations.

    Parameters
    ----------
    alpha_beta
        Sequence of ``(label, alpha, alpha_err, beta, beta_err)``.
    note
        Optional text for the lower-right corner, e.g. the inferred ``p``.
    """
    with themed(theme) as t:
        fig, ax = plt.subplots(figsize=figsize)
        beta = np.linspace(*beta_range, 100)
        seen = {}
        for rel in CLOSURE_RELATIONS:
            alpha = np.array([rel.alpha_of_beta(b) for b in beta])
            key = tuple(np.round(alpha[[0, -1]], 6))
            if key in seen:
                seen[key] = seen[key].replace("ISM, ", "ISM or wind, ") if "ISM" in seen[key] else seen[key]
                continue
            seen[key] = rel.label
        for rel in CLOSURE_RELATIONS:
            alpha = np.array([rel.alpha_of_beta(b) for b in beta])
            key = tuple(np.round(alpha[[0, -1]], 6))
            if key not in seen:
                continue
            label = seen.pop(key)
            jet = rel.key.startswith("jet")
            ax.plot(
                beta,
                alpha,
                color=t.ink_secondary if jet else t.muted,
                lw=1.2,
                ls="-" if jet else (0, (4, 2.5)),
                zorder=2,
            )
            ax.annotate(
                _math_label(label),
                xy=(beta[-1], alpha[-1]),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=8.5,
                color=t.ink_secondary,
                va="center",
                ha="left",
            )
        point_colors = (THEMES[theme].series[0], THEMES[theme].series[2], THEMES[theme].series[3])
        for i, (label, a, ae, b, be) in enumerate(alpha_beta):
            color = point_colors[i % len(point_colors)]
            ax.errorbar(
                b,
                a,
                xerr=be,
                yerr=ae,
                fmt="o",
                ms=9,
                color=color,
                mec=t.surface,
                mew=1.5,
                elinewidth=1.4,
                capsize=0,
                zorder=5,
            )
            ax.annotate(
                label,
                xy=(b, a),
                xytext=(9, -3),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=9.5,
                color=t.ink,
                weight="bold",
            )
        if note:
            ax.text(
                0.98, 0.04, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=t.ink_secondary
            )
        ax.set_xlim(beta_range[0], beta_range[1] + 0.75)
        ax.set_xticks(np.arange(np.ceil(beta_range[0] * 4) / 4, beta_range[1] + 1e-9, 0.25))
        ax.set_xlabel(r"Spectral index $\beta$  ($F_\nu \propto \nu^{-\beta}$)")
        ax.set_ylabel(r"Temporal index $\alpha$  ($F_\nu \propto t^{-\alpha}$)")
        ax.set_title("Closure relations of the fireball model")
        ax.spines["bottom"].set_bounds(*beta_range)
        _save(fig, savepath)
        return fig, ax


def _cutout(image: np.ndarray, x: float, y: float, half: int) -> tuple[np.ndarray, float, float]:
    ix, iy = int(np.rint(x)), int(np.rint(y))
    y0, x0 = max(iy - half, 0), max(ix - half, 0)
    cut = image[y0 : iy + half + 1, x0 : ix + half + 1]
    return cut, x - x0, y - y0


def show_image(ax, image: np.ndarray, *, theme: str = "light", contrast: float = 0.25, stretch: float = 0.15) -> None:
    """Display an astronomical image with a ZScale + asinh stretch, axes off."""
    finite = image[np.isfinite(image)]
    vmin, vmax = ZScaleInterval(contrast=contrast).get_limits(finite)
    norm = ImageNormalize(vmin=vmin, vmax=vmax + 4 * (vmax - vmin), stretch=AsinhStretch(stretch))
    ax.imshow(image, origin="lower", cmap="gray", norm=norm, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)


def plot_filmstrip(
    panels: Sequence[dict],
    *,
    half: int = 20,
    ncols: int | None = None,
    title: str | None = None,
    theme: str = "light",
    panel_size: float = 1.7,
    savepath: str | PathLike[str] | None = None,
):
    """A row of cutouts around the burst showing it fade.

    Each panel is a dict with ``image``, ``x``, ``y``, ``label``, and optionally
    ``r_aper``, ``annulus`` (``(r_in, r_out)``), and ``flag`` (a short reason
    the epoch was rejected, drawn in place of the aperture).
    """
    n = len(panels)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))
    with themed(theme) as t:
        head = 0.55 if title else 0.3
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(panel_size * ncols * 1.05, (panel_size + 0.3) * nrows + head), squeeze=False
        )
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        accent = THEMES[theme].series[0]
        flag_color = "#d03b3b"
        for ax, p in zip(axes.ravel()[:n], panels, strict=True):
            cut, cx, cy = _cutout(np.asarray(p["image"], float), p["x"], p["y"], half)
            show_image(ax, cut, theme=theme)
            if p.get("flag"):
                ax.add_patch(
                    Circle((cx, cy), p.get("r_aper", 3.0), fill=False, ec=flag_color, lw=1.3, ls=(0, (2, 1.5)))
                )
                ax.text(
                    0.5,
                    0.06,
                    p["flag"],
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="white",
                    bbox={"boxstyle": "round,pad=0.25", "fc": flag_color, "ec": "none", "alpha": 0.9},
                )
            else:
                ax.add_patch(Circle((cx, cy), p.get("r_aper", 3.0), fill=False, ec=accent, lw=1.4))
                if p.get("annulus"):
                    for r in p["annulus"]:
                        ax.add_patch(Circle((cx, cy), r, fill=False, ec=accent, lw=0.6, alpha=0.6))
            ax.set_title(p["label"], fontsize=8.5, color=t.ink_secondary, loc="center", weight="regular", pad=3)
        height = (panel_size + 0.3) * nrows + head
        fig.subplots_adjust(left=0.005, right=0.995, bottom=0.02, top=1 - head / height, wspace=0.05, hspace=0.3)
        if title:
            fig.suptitle(title, x=0.005, y=0.995, ha="left", va="top", fontsize=11.5, color=t.ink, weight="bold")
        _save(fig, savepath)
        return fig, axes


def plot_pipeline_stages(
    raw: np.ndarray,
    calibrated: np.ndarray,
    sources_xy: np.ndarray,
    matched_xy: np.ndarray,
    target_xy: tuple[float, float],
    r_aper: float,
    annulus: tuple[float, float],
    *,
    theme: str = "light",
    savepath: str | PathLike[str] | None = None,
):
    """Four panels showing what each stage does to one frame."""
    with themed(theme) as t:
        fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.5))
        accent, second = THEMES[theme].series[0], THEMES[theme].series[2]
        show_image(axes[0], raw, theme=theme, contrast=0.1)
        axes[0].set_title("1  Raw frame", fontsize=10.5)
        axes[0].text(0.03, 0.03, "bias + vignetted flat", transform=axes[0].transAxes, color="white", fontsize=8)
        show_image(axes[1], calibrated, theme=theme)
        axes[1].set_title("2  (raw − bias) / flat", fontsize=10.5)
        show_image(axes[2], calibrated, theme=theme)
        axes[2].scatter(sources_xy[:, 0], sources_xy[:, 1], s=60, facecolor="none", edgecolor=second, lw=0.9)
        axes[2].scatter(matched_xy[:, 0], matched_xy[:, 1], s=14, color=accent, lw=0)
        axes[2].set_title("3  Detect + match catalog", fontsize=10.5)
        h = [
            Line2D([], [], marker="o", mfc="none", mec=second, lw=0, label="detected"),
            Line2D([], [], marker="o", color=accent, lw=0, ms=4, label="catalog match"),
        ]
        axes[2].legend(
            handles=h,
            loc="lower left",
            fontsize=7.5,
            labelcolor="white",
            handletextpad=0.2,
            borderaxespad=0.4,
            frameon=True,
            facecolor="black",
            edgecolor="none",
            framealpha=0.65,
        )
        half = int(annulus[1] * 1.6)
        cut, cx, cy = _cutout(calibrated, target_xy[0], target_xy[1], half)
        show_image(axes[3], cut, theme=theme)
        axes[3].add_patch(Circle((cx, cy), r_aper, fill=False, ec=accent, lw=1.5))
        for r in annulus:
            axes[3].add_patch(Circle((cx, cy), r, fill=False, ec=accent, lw=0.8, alpha=0.7))
        axes[3].set_title("4  Forced photometry", fontsize=10.5)
        axes[3].text(0.03, 0.03, "aperture + sky annulus", transform=axes[3].transAxes, color="white", fontsize=8)
        for ax in axes:
            ax.title.set_color(t.ink)
        fig.subplots_adjust(wspace=0.05)
        _save(fig, savepath)
        return fig, axes


def plot_subtraction(
    science: np.ndarray,
    template: np.ndarray,
    difference: np.ndarray,
    xy: tuple[float, float],
    *,
    r_aper: float,
    half: int = 30,
    labels: Sequence[str] = ("Science", "Template, PSF-matched", "Difference"),
    title: str | None = None,
    theme: str = "light",
    savepath: str | PathLike[str] | None = None,
):
    """Science, matched template, and difference cutouts around the burst.

    The three panels share one brightness scale (set from the science cutout),
    so static sources visibly cancel and only the transient survives in the
    difference.
    """
    cuts = [_cutout(np.asarray(img, float), xy[0], xy[1], half) for img in (science, template, difference)]
    sci_cut = cuts[0][0]
    finite = sci_cut[np.isfinite(sci_cut)]
    sky = float(np.median(finite))
    vmin, vmax = ZScaleInterval(contrast=0.25).get_limits(finite - sky)
    norm = ImageNormalize(vmin=vmin, vmax=vmax + 4 * (vmax - vmin), stretch=AsinhStretch(0.15))
    with themed(theme) as t:
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6))
        accent = THEMES[theme].series[0]
        for ax, (cut, cx, cy), label, offset in zip(
            axes, cuts, labels, (sky, float(np.median(cuts[1][0])), 0.0), strict=True
        ):
            ax.imshow(cut - offset, origin="lower", cmap="gray", norm=norm, interpolation="nearest")
            ax.add_patch(Circle((cx, cy), r_aper, fill=False, ec=accent, lw=1.4))
            ax.set_title(label, fontsize=10.5, loc="center")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
        for ax, sym in zip(axes[:2], ("−", "="), strict=True):
            ax.text(
                1.04, 0.5, sym, transform=ax.transAxes, fontsize=20, color=t.ink_secondary, ha="center", va="center"
            )
        if title:
            fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, color=t.ink, weight="bold")
        fig.subplots_adjust(wspace=0.12, left=0.01, right=0.99, bottom=0.02, top=0.84 if title else 0.9)
        _save(fig, savepath)
        return fig, axes
