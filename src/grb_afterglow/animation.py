"""Animated GIF of an afterglow fading while its light curve builds up."""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import AsinhStretch, ImageNormalize
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from .fitting import FitResult
from .lightcurve import SECONDS_PER_DAY, LightCurve
from .plotting import _sorted_bands, band_model, band_style, format_log_time_axis, themed


def animate_afterglow(
    cutouts: Sequence[np.ndarray],
    cutout_times_days: Sequence[float],
    lc: LightCurve,
    fit: FitResult | None,
    savepath: str | PathLike[str],
    *,
    r_aper: float = 4.0,
    title: str = "Photons to Power Laws",
    theme: str = "dark",
    fps: float = 5.0,
    hold_frames: int = 12,
    dpi: int = 100,
) -> None:
    """Write a two-panel GIF: the burst fading in the image, and its light curve.

    Parameters
    ----------
    cutouts
        Sky-subtracted cutouts centred on the burst, in a common flux unit
        (e.g. microjansky per pixel) so one brightness scale fits all epochs.
    cutout_times_days
        Time of each cutout since trigger, in days.
    lc
        The measured light curve; points appear as the animation reaches them.
    fit
        Best-fit model, drawn once all data are shown.
    """
    times = np.asarray(cutout_times_days, dtype=float)
    peak = float(np.nanpercentile(cutouts[0], 99.9))
    norm = ImageNormalize(vmin=-0.02 * peak, vmax=peak, stretch=AsinhStretch(0.02))
    bands = _sorted_bands(lc)
    days = lc.time_days
    grid = np.geomspace(days.min() / 1.4, days.max() * 1.4, 300)
    n_steps = 4
    step_times = np.concatenate(
        [np.geomspace(times[i], times[i + 1], n_steps, endpoint=False) for i in range(len(times) - 1)] + [[times[-1]]]
    )
    n_frames = len(step_times) + hold_frames

    with themed(theme) as t:
        fig = plt.figure(figsize=(8.4, 3.7))
        ax_img = fig.add_axes([0.015, 0.06, 0.33, 0.78])
        ax_lc = fig.add_axes([0.45, 0.17, 0.53, 0.67])
        fig.text(0.015, 0.93, title, fontsize=13, weight="semibold", color=t.ink, va="center")
        fig.text(0.985, 0.93, "synthetic GRB · g r i", fontsize=9, color=t.ink_secondary, va="center", ha="right")
        img = ax_img.imshow(cutouts[0], origin="lower", cmap="gray", norm=norm, interpolation="nearest")
        ny, nx = cutouts[0].shape
        ax_img.add_patch(Circle(((nx - 1) / 2, (ny - 1) / 2), r_aper, fill=False, ec=t.series[0], lw=1.3))
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.grid(False)
        for spine in ax_img.spines.values():
            spine.set_visible(False)
        clock = ax_img.text(0.04, 0.05, "", transform=ax_img.transAxes, color="white", fontsize=10, weight="semibold")

        ax_lc.set_xscale("log")
        ax_lc.set_yscale("log")
        ax_lc.set_xlim(grid[0], grid[-1])
        format_log_time_axis(ax_lc)
        pos = lc.flux[lc.flux > 0]
        ax_lc.set_ylim(pos.min() / 4, lc.flux.max() * 2)
        ax_lc.set_xlabel("Time since trigger (days)")
        ax_lc.set_ylabel("Flux density (µJy)")
        cursor = ax_lc.axvline(times[0], color=t.muted, lw=0.8)

        artists = {}
        handles = []
        for i, band in enumerate(bands):
            color, marker = band_style(band, theme, i)
            (pts,) = ax_lc.plot([], [], marker, ms=5.5, color=color, mec=t.surface, mew=0.8, lw=0)
            (model,) = ax_lc.plot([], [], color=color, lw=1.6, alpha=0.0)
            artists[band] = (pts, model, lc.select(band))
            handles.append(Line2D([], [], color=color, marker=marker, lw=0, ms=5.5, label=band))
        ax_lc.legend(handles=handles, loc="upper right", ncol=3, handletextpad=0.1, columnspacing=0.8)
        label = ax_lc.text(0.03, 0.05, "", transform=ax_lc.transAxes, fontsize=9, color=t.ink_secondary)

        def update(frame: int):
            now = step_times[min(frame, len(step_times) - 1)]
            k = int(np.searchsorted(times, now * (1 + 1e-9), side="right") - 1)
            img.set_data(cutouts[max(k, 0)])
            clock.set_text(f"t = {now * 24:.1f} h" if now < 1 else f"t = {now:.2f} d")
            cursor.set_xdata([now, now])
            for band, (pts, model, sub) in artists.items():
                shown = sub.time_days <= now * (1 + 1e-9)
                pts.set_data(sub.time_days[shown], np.clip(sub.flux[shown], pos.min() / 4, None))
                if fit is not None and frame >= len(step_times):
                    model.set_data(grid, band_model(fit, grid * SECONDS_PER_DAY, band))
                    model.set_alpha(min(1.0, 0.25 * (frame - len(step_times) + 1)))
            if fit is not None and frame >= len(step_times) and "alpha1" in fit.params:
                p = fit.params
                label.set_text(
                    f"jet break at {p['t_break'] / SECONDS_PER_DAY:.2f} d · "
                    f"α₁ = {p['alpha1']:.2f}, α₂ = {p['alpha2']:.2f}"
                )
            return []

        anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
        anim.save(str(savepath), writer=PillowWriter(fps=fps), dpi=dpi, savefig_kwargs={"facecolor": t.surface})
        plt.close(fig)
