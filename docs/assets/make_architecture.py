#!/usr/bin/env python3
"""Render the pipeline architecture diagram (light and dark PNGs) from HTML.

The diagram is plain HTML/CSS with inline SVG icons, written by this script and
rendered with a headless Chrome/Chromium:

    python docs/assets/make_architecture.py            # writes architecture_{light,dark}.png
    python docs/assets/make_architecture.py --html-only

Set CHROME=/path/to/chrome if Chrome is not in a standard location.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1840, 1000

# ----------------------------------------------------------------------------- content

STAGES = [
    {
        "title": "Data Inputs",
        "icon": "database",
        "color": "blue",
        "steps": [
            "Raw science frames with WCS",
            "Bias &amp; flat frames",
            "Deep reference image",
            "Pan-STARRS1 catalog (MAST)",
            "ZTF archive (IRSA)",
        ],
        "module": "archives.py",
    },
    {
        "title": "Calibration",
        "icon": "gear",
        "color": "violet",
        "steps": ["Sigma-clipped master bias", "Per-exposure scaled master flat", "(raw − bias) / flat"],
        "module": "calibration.py",
    },
    {
        "title": "Detection &amp; Seeing",
        "icon": "detect",
        "color": "indigo",
        "steps": ["Background + RMS maps", "DAOFIND sources", "Seeing from Gaussian fits", "Source mask for sky"],
        "module": "detection.py",
    },
    {
        "title": "Photometric Calibration",
        "icon": "crosshair",
        "color": "sky",
        "steps": [
            "WCS cross-match, one-to-one",
            "Robust weighted zero point",
            "Colour term at the burst's colour",
            "Solved per frame",
        ],
        "module": "zeropoint.py",
    },
    {
        "title": "Image Subtraction",
        "icon": "subtract",
        "color": "teal",
        "steps": ["Resample template via WCS", "__KERNEL__", "Per tile on full frames", "Noise from blank positions"],
        "module": "subtraction.py",
    },
    {
        "title": "Photometry",
        "icon": "aperture",
        "color": "cyan",
        "steps": ["Aperture, masked sky annulus", "__PSF__", "PSF varying across the frame", "Forced at the burst"],
        "module": "photometry.py · psf.py",
    },
    {
        "title": "Light Curve",
        "icon": "lightcurve",
        "color": "blue",
        "steps": ["Flux density, multi-band", "Upper limits", "CSV import / export"],
        "module": "lightcurve.py",
    },
    {
        "title": "Model Fitting",
        "icon": "powerlaw",
        "color": "orange",
        "steps": [
            "Single &amp; broken power law",
            "Joint fit across instruments, β",
            "Through 35 filter curves",
            "Galactic &amp; host dust",
            "MCMC, grid or nested sampling",
        ],
        "module": "fitting · spectral · mcmc · nested",
    },
    {
        "title": "Physics",
        "icon": "physics",
        "color": "rose",
        "steps": ["Closure relations", "Electron index p", "ISM or wind, jet break", "Jet models, X-ray to radio"],
        "module": "closure · physical",
    },
    {
        "title": "Outputs",
        "icon": "report",
        "color": "green",
        "steps": [
            "Calibrated light curve",
            "α, β, t<sub>b</sub>, p with errors",
            "Light &amp; dark figures, GIF",
            "<code>p2pl</code> command line",
        ],
        "module": "plotting.py · cli.py",
    },
]
REDUCE_SPAN = (2, 5)  # stages covered by pipeline.reduce_frame (0-based, inclusive)

VALIDATION = [
    ("galaxy", "Synthetic g r i campaign", "dithers, seeing, clouds, host galaxy, templates"),
    ("target", "Known ground truth", "α<sub>1</sub>, α<sub>2</sub>, β, t<sub>b</sub>, p"),
    ("play", "<code>p2pl demo</code>", "recovers the injected physics"),
    ("check", "__TESTS__ tests, CI", "Python 3.10 to 3.13"),
]
REAL = [
    ("telescope", "GRB 221009A", "ZTF + near-IR, 4 mag of dust"),
    ("lightcurve", "GRB 210204A", "ZTF + XRT + uGMRT, jet break"),
    ("check", "Matches published results", "Laskar+23, O'Connor+23, Kumar+22"),
]
ROADMAP = [
    ("telescope", "X-ray counts, Poisson likelihood"),
    ("dice", "Structured jets, by evidence"),
]

# ----------------------------------------------------------------------------- icons (24x24, stroke)

ICONS = {
    "database": '<ellipse cx="12" cy="5" rx="7.5" ry="2.8"/><path d="M4.5 5v14c0 1.5 3.4 2.8 7.5 2.8s7.5-1.3 7.5-2.8V5"/><path d="M4.5 12c0 1.5 3.4 2.8 7.5 2.8s7.5-1.3 7.5-2.8"/>',
    "gear": '<circle cx="12" cy="12" r="3.2"/><path d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3M5.3 5.3l2.1 2.1M16.6 16.6l2.1 2.1M5.3 18.7l2.1-2.1M16.6 7.4l2.1-2.1"/><circle cx="12" cy="12" r="6.6"/>',
    "detect": '<circle cx="12" cy="12" r="8.5" stroke-dasharray="2.2 2.4"/><path d="M12 8.2l1.2 2.5 2.7.3-2 1.9.6 2.7L12 14.3l-2.5 1.3.6-2.7-2-1.9 2.7-.3z" fill="currentColor"/>',
    "crosshair": '<circle cx="12" cy="12" r="7"/><path d="M12 2v4M12 18v4M2 12h4M18 12h4"/><path d="M12 9.3l.9 1.8 2 .2-1.5 1.4.4 2-1.8-1-1.8 1 .4-2-1.5-1.4 2-.2z" fill="currentColor"/>',
    "subtract": '<rect x="2.5" y="5" width="8" height="10" rx="1.5"/><rect x="13.5" y="5" width="8" height="10" rx="1.5"/><circle cx="6.5" cy="10" r="1.8" fill="currentColor"/><path d="M9 19.5h6"/><path d="M15.3 10h4.4" opacity=".5"/>',
    "aperture": '<circle cx="12" cy="12" r="9" stroke-dasharray="1.5 2"/><circle cx="12" cy="12" r="5.6"/><circle cx="12" cy="12" r="2.4" fill="currentColor"/>',
    "lightcurve": '<path d="M3 20h18M3 20V4"/><path d="M5.5 16.5l4-4 3.5 2 6.5-8"/><circle cx="5.5" cy="16.5" r="1.4" fill="currentColor"/><circle cx="9.5" cy="12.5" r="1.4" fill="currentColor"/><circle cx="13" cy="14.5" r="1.4" fill="currentColor"/><circle cx="19.5" cy="6.5" r="1.4" fill="currentColor"/>',
    "powerlaw": '<path d="M3 20h18M3 20V4"/><path d="M5 6c3 2 6 4 8.5 6.5S18 17 20.5 18.5"/><circle cx="6.5" cy="7.2" r="1.3" fill="currentColor"/><circle cx="10.5" cy="10.2" r="1.3" fill="currentColor"/><circle cx="14.5" cy="13.7" r="1.3" fill="currentColor"/><circle cx="18.5" cy="17" r="1.3" fill="currentColor"/>',
    "physics": '<ellipse cx="12" cy="12" rx="9" ry="3.6"/><ellipse cx="12" cy="12" rx="9" ry="3.6" transform="rotate(60 12 12)"/><ellipse cx="12" cy="12" rx="9" ry="3.6" transform="rotate(120 12 12)"/><circle cx="12" cy="12" r="1.6" fill="currentColor"/>',
    "report": '<path d="M6 2.8h8.5L19 7.3V21H6z"/><path d="M14.5 2.8v4.5H19"/><path d="M8.5 17l2.6-3 2.2 1.6 3-4"/>',
    "galaxy": '<ellipse cx="12" cy="12" rx="9.5" ry="4.2" transform="rotate(-25 12 12)"/><path d="M7.5 13.5c1-3.5 6-5 8.5-2.5 1.8 1.8.3 4.3-2 4.3-2 0-2.7-1.6-1.6-2.6" /><circle cx="12" cy="12" r="1.3" fill="currentColor"/>',
    "target": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4.2"/><circle cx="12" cy="12" r="1.3" fill="currentColor"/><path d="M12 1.5v4M12 18.5v4M1.5 12h4M18.5 12h4"/>',
    "play": '<rect x="2.5" y="4" width="19" height="14" rx="2"/><path d="M10 8.2v5.6l4.8-2.8z" fill="currentColor"/><path d="M6 21h12"/>',
    "check": '<circle cx="12" cy="12" r="9"/><path d="M7.6 12.4l3 3 5.8-6.4"/>',
    "telescope": '<path d="M3.5 11.5l13-6 1.6 3.4-13 6z"/><path d="M18.1 8.9l2.4-1.1-1.6-3.4-2.4 1.1"/><path d="M10.5 13.8L8 21M12 13l3 8"/>',
    "palette": '<path d="M12 3a9 9 0 1 0 0 18c1.4 0 1.9-1.2 1.2-2.2-.8-1.1 0-2.6 1.4-2.6H17a4 4 0 0 0 4-4C21 6.9 17 3 12 3z"/><circle cx="7.5" cy="11" r="1.2" fill="currentColor"/><circle cx="10" cy="7" r="1.2" fill="currentColor"/><circle cx="14.5" cy="7" r="1.2" fill="currentColor"/>',
    "dust": '<circle cx="6" cy="8" r="1.4" fill="currentColor"/><circle cx="12" cy="5.5" r="1" fill="currentColor"/><circle cx="17.5" cy="9" r="1.6" fill="currentColor"/><circle cx="9" cy="13" r="1.1" fill="currentColor"/><circle cx="15" cy="14.5" r="1.3" fill="currentColor"/><circle cx="6.5" cy="18" r="1" fill="currentColor"/><circle cx="18" cy="19" r="1" fill="currentColor"/>',
    "dice": '<rect x="3.5" y="3.5" width="17" height="17" rx="3"/><circle cx="8.5" cy="8.5" r="1.3" fill="currentColor"/><circle cx="15.5" cy="15.5" r="1.3" fill="currentColor"/><circle cx="12" cy="12" r="1.3" fill="currentColor"/>',
    "flask": '<path d="M9 3h6M10 3v6L4.8 18.5A1.8 1.8 0 0 0 6.4 21h11.2a1.8 1.8 0 0 0 1.6-2.5L14 9V3"/><path d="M7.2 15h9.6"/>',
    "rocket": '<path d="M12 2.5c3.5 2.6 5 6.2 4.5 10.5l-2.3 3H9.8l-2.3-3C7 8.7 8.5 5.1 12 2.5z"/><circle cx="12" cy="9" r="1.7"/><path d="M9.8 16l-2.3 4.5 3-1.6M14.2 16l2.3 4.5-3-1.6"/>',
}


def icon(name: str, size: int = 22, cls: str = "") -> str:
    return (
        f'<svg class="ico {cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>'
    )


# ----------------------------------------------------------------------------- page

CSS = """
:root{
  --bg:#ffffff; --ink:#0f172a; --ink2:#475569; --muted:#94a3b8; --line:#cbd5e1; --card:#f8fafc; --step:#ffffff;
  --blue:#2563eb; --violet:#7c3aed; --indigo:#4f46e5; --sky:#0284c7; --teal:#0d9488; --cyan:#0891b2;
  --orange:#ea580c; --rose:#e11d48; --green:#16a34a; --navy:#1e293b;
}
:root[data-theme=dark]{
  --bg:#0d1117; --ink:#f1f5f9; --ink2:#cbd5e1; --muted:#64748b; --line:#334155; --card:#111827; --step:#0b1220;
  --blue:#60a5fa; --violet:#a78bfa; --indigo:#818cf8; --sky:#38bdf8; --teal:#2dd4bf; --cyan:#22d3ee;
  --orange:#fb923c; --rose:#fb7185; --green:#4ade80; --navy:#cbd5e1;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{width:%(w)dpx;height:%(h)dpx;background:var(--bg);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Helvetica Neue",Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{padding:30px 34px 22px}
header{display:flex;align-items:center;justify-content:center;gap:26px;margin-bottom:6px}
header h1{font-size:54px;letter-spacing:-1.2px;font-weight:800;color:var(--ink)}
.sub{text-align:center;color:var(--ink2);font-size:19px;font-style:italic;margin-bottom:24px}
.comet{width:210px;height:60px}
.row{display:flex;align-items:stretch;gap:0;position:relative}
.stage{flex:1;min-width:0;border:1.6px solid color-mix(in srgb,var(--c) 55%%,transparent);border-radius:14px;
  background:color-mix(in srgb,var(--c) 5%%,var(--card));display:flex;flex-direction:column;position:relative}
.stage .num{width:34px;height:34px;border-radius:50%%;background:var(--c);color:var(--bg);font-weight:800;font-size:18px;
  display:flex;align-items:center;justify-content:center;margin:14px auto 6px}
.stage h3{text-align:center;font-size:16.5px;line-height:1.15;font-weight:750;min-height:40px;padding:0 6px;
  display:flex;align-items:center;justify-content:center}
.stage .big{color:var(--c);display:flex;justify-content:center;margin:6px 0 10px}
.steps{display:flex;flex-direction:column;gap:7px;padding:0 9px 12px;flex:1}
.step{background:var(--step);border:1px solid var(--line);border-radius:8px;padding:7px 6px;font-size:12.6px;
  line-height:1.25;text-align:center;color:var(--ink);min-height:40px;display:flex;align-items:center;justify-content:center}
.step code,.vi code{font-family:"SF Mono",Menlo,monospace;font-size:.92em}
.mod{background:var(--c);color:var(--bg);border-radius:0 0 12px 12px;text-align:center;font-weight:700;font-size:13px;
  padding:7px 4px;font-family:"SF Mono",Menlo,monospace;letter-spacing:-.2px}
.new{position:absolute;top:10px;right:9px;background:var(--c);color:var(--bg);font-size:10px;font-weight:800;
  letter-spacing:.6px;border-radius:999px;padding:2px 7px}
.arrow{width:20px;display:flex;align-items:center;justify-content:center;color:var(--navy)}
.bracket{position:absolute;top:-14px;height:12px;border:2px solid var(--navy);border-bottom:none;border-radius:8px 8px 0 0;opacity:.75}
.bracket span{position:absolute;top:-24px;left:50%%;transform:translateX(-50%%);white-space:nowrap;font-size:13px;
  font-weight:700;color:var(--ink2);background:var(--bg);padding:0 10px;font-family:"SF Mono",Menlo,monospace}
.bottom{display:flex;gap:22px;margin-top:30px}
.box{border-radius:14px;border:1.6px solid var(--line);background:var(--card);display:flex;flex-direction:column}
.box.dashed{border:2px dashed color-mix(in srgb,var(--navy) 55%%,transparent)}
.box h4{display:flex;align-items:center;gap:10px;justify-content:center;font-size:21px;font-weight:780;padding:14px 10px 8px}
.items{display:flex;flex:1;padding:4px 10px 12px}
.vi{flex:1;display:flex;flex-direction:column;align-items:center;text-align:center;padding:6px 8px;gap:6px;font-size:14.5px;line-height:1.25}
.vi + .vi{border-left:1px solid var(--line)}
.vi small{color:var(--ink2);font-size:12.5px}
.vi .ico{color:var(--c,var(--indigo))}
.foot{background:var(--navy);color:var(--bg);text-align:center;font-weight:700;border-radius:0 0 12px 12px;padding:7px;
  font-family:"SF Mono",Menlo,monospace;font-size:13.5px}
.road{padding:6px 18px 14px;display:flex;flex-direction:column;gap:9px}
.road div{display:flex;align-items:center;gap:12px;font-size:15px;color:var(--ink)}
.road .ico{color:var(--ink2)}
.legend{margin-top:22px;border:1px solid var(--line);border-radius:10px;padding:9px 18px;display:flex;justify-content:space-between;
  font-size:13px;color:var(--ink2)}
.legend span{display:flex;align-items:center;gap:7px}
.legend i{width:12px;height:12px;border-radius:50%%;display:inline-block}
.note{text-align:center;color:var(--ink2);font-style:italic;font-size:14px;margin-top:10px}
"""

COMET = """<svg class="comet" viewBox="0 0 210 60" fill="none">
  <defs><linearGradient id="tail" x1="0" x2="1"><stop offset="0" stop-color="var(--violet)"/><stop offset="1" stop-color="var(--sky)" stop-opacity=".15"/></linearGradient></defs>
  <g stroke="var(--violet)" stroke-width="2" stroke-linecap="round">
    <path d="M30 30h-18M30 30h18M30 30v-18M30 30v18M30 30l-12-12M30 30l12 12M30 30l-12 12M30 30l12-12" opacity=".75"/>
  </g>
  <circle cx="30" cy="30" r="6" fill="var(--violet)"/>
  <path d="M42 31c25 6 34-14 58-6s34 18 60 6 26-8 40-2" stroke="url(#tail)" stroke-width="3.2" stroke-linecap="round"/>
  <circle cx="170" cy="26" r="2.6" fill="var(--sky)" opacity=".7"/><circle cx="188" cy="34" r="2" fill="var(--sky)" opacity=".5"/>
  <circle cx="202" cy="22" r="1.6" fill="var(--sky)" opacity=".4"/>
</svg>"""


def build_html(theme: str, kernel: str, psf: str, n_tests: int) -> str:
    stages_html = []
    for i, st in enumerate(STAGES):
        steps = [s.replace("__KERNEL__", kernel).replace("__PSF__", psf) for s in st["steps"]]
        badge = '<div class="new">NEW</div>' if st.get("new") else ""
        stages_html.append(
            f'<div class="stage" style="--c:var(--{st["color"]})">{badge}<div class="num">{i + 1}</div>'
            f'<h3>{st["title"]}</h3><div class="big">{icon(st["icon"], 46)}</div>'
            f'<div class="steps">{"".join(f"<div class=step><span>{s}</span></div>" for s in steps)}</div>'
            f'<div class="mod">{escape(st["module"])}</div></div>'
        )
        if i < len(STAGES) - 1:
            stages_html.append(
                '<div class="arrow"><svg width="16" height="18" viewBox="0 0 16 18"><path d="M2 2l11 7-11 7z" fill="currentColor"/></svg></div>'
            )
    validation = "".join(
        f'<div class="vi" style="--c:var(--indigo)">{icon(ic, 40)}<div>{t.replace("__TESTS__", str(n_tests))}</div><small>{s}</small></div>'
        for ic, t, s in VALIDATION
    )
    real = "".join(
        f'<div class="vi" style="--c:var(--teal)">{icon(ic, 40)}<div>{t}</div><small>{s}</small></div>'
        for ic, t, s in REAL
    )
    road = "".join(f"<div>{icon(ic, 24)}<span>{t}</span></div>" for ic, t in ROADMAP)
    legend_items = [
        ("blue", "Data &amp; light curves"),
        ("violet", "Calibration"),
        ("indigo", "Detection"),
        ("sky", "Zero points"),
        ("teal", "Subtraction"),
        ("cyan", "Photometry"),
        ("orange", "Fitting"),
        ("rose", "Physics"),
        ("green", "Outputs"),
    ]
    legend = "".join(f'<span><i style="background:var(--{c})"></i>{t}</span>' for c, t in legend_items)
    return f"""<!doctype html><html data-theme="{theme}"><head><meta charset="utf-8"><title>Pipeline architecture</title>
<style>{CSS % {"w": WIDTH, "h": HEIGHT}}</style></head><body><div class="wrap">
<header>{COMET}<h1>Photons to Power Laws</h1></header>
<div class="sub">Pipeline architecture: from raw telescope frames to GRB afterglow physics</div>
<div class="row" id="row">{"".join(stages_html)}<div class="bracket" id="bracket"><span>pipeline.reduce_frame · one frame in, one calibrated flux out</span></div></div>
<div class="bottom">
  <div class="box dashed" style="flex:2.15"><h4>{icon("flask", 26)} Validation &amp; Demo</h4><div class="items">{validation}</div>
    <div class="foot">synthetic.py · demo.py · tests/</div></div>
  <div class="box" style="flex:1.55"><h4>{icon("telescope", 26)} Real Data</h4><div class="items">{real}</div>
    <div class="foot">case_studies/</div></div>
  <div class="box" style="flex:1"><h4>{icon("rocket", 26)} Roadmap</h4><div class="road">{road}</div></div>
</div>
<div class="legend">{legend}</div>
<div class="note">Every stage is a module in src/grb_afterglow. Synthetic and real images run through the same code path.</div>
</div>
<script>
  const row = document.getElementById('row'), stages = row.querySelectorAll('.stage'), b = document.getElementById('bracket');
  const a = stages[{REDUCE_SPAN[0]}].getBoundingClientRect(), z = stages[{REDUCE_SPAN[1]}].getBoundingClientRect(), r = row.getBoundingClientRect();
  b.style.left = (a.left - r.left + 6) + 'px'; b.style.width = (z.right - a.left - 12) + 'px';
  row.style.marginTop = "52px";
</script>
</body></html>"""


def find_chrome() -> str | None:
    candidates = [
        os.environ.get("CHROME"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
    ]
    return next((c for c in candidates if c and Path(c).exists()), None)


def render(html: str, png: Path, chrome: str, scale: int = 2) -> None:
    """Screenshot the HTML with headless Chrome, using a throwaway profile (retried once)."""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "diagram.html"
        src.write_text(html, encoding="utf-8")
        cmd = [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={Path(tmp) / 'profile'}",
            f"--force-device-scale-factor={scale}",
            f"--window-size={WIDTH},{HEIGHT}",
            f"--screenshot={png}",
            "--virtual-time-budget=2000",
            src.as_uri(),
        ]
        for _ in range(2):
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and png.exists():
                return
        raise RuntimeError(f"Chrome failed to render {png.name}:\n{result.stderr[-2000:]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kernel", default="Bramich delta-basis kernel")
    parser.add_argument("--psf", default="Empirical PSF, deblended fit")
    parser.add_argument("--tests", type=int, default=134)
    parser.add_argument("--html-only", action="store_true")
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    chrome = None if args.html_only else find_chrome()
    for theme in ("light", "dark"):
        html = build_html(theme, args.kernel, args.psf, args.tests)
        (args.out / f"architecture_{theme}.html").write_text(html, encoding="utf-8") if args.html_only else None
        if chrome:
            render(html, args.out / f"architecture_{theme}.png", chrome)
            print(f"wrote {args.out / f'architecture_{theme}.png'}")
    if not chrome and not args.html_only:
        print("Chrome not found; set CHROME or use --html-only.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
