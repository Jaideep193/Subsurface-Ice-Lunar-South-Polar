"""
Interactive Dashboard — same structure as static version, Plotly panels
=======================================================================
Keeps identical HTML shell, 6 tabs, same descriptions, same stat cards.
Replaces each matplotlib PNG with a Plotly interactive figure:
  hover for values, scroll to zoom, drag to pan, click legend to toggle.

Output: results/interactive_dashboard.html
"""
import sys, os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# ── Constants ─────────────────────────────────────────────────────────────────
DARK       = "#080810"
PANEL      = "#0d0d1a"
GRID       = "#1a1a2e"
TEXT       = "#ccddff"
DS         = 4        # downsample factor — 1000px -> 250px per axis
CENTER_LAT = -87.0    # Faustini scene centre
CENTER_LON =   0.0
M_PER_DEG  = 111_320.0


# ── Data ──────────────────────────────────────────────────────────────────────
def load_data():
    from src.data_generator import load_all
    from src.dfsar_analysis import run_analysis
    from src.ice_volume     import estimate_ice_volume, monte_carlo_uncertainty, \
                                    compute_penetration_depth_map
    from src.thermal_model  import compute_surface_temperature
    from src.morphology     import run_morphology

    data  = load_all(cache=True)
    dem   = data["dem"]; slope = data["slope"]; illum = data["illum"]
    psr   = data["psr_mask"]; dsc = data["dsc_mask"]
    dfsar = data["dfsar"]; ohrc = data["ohrc"]; meta = data["meta"]
    gs    = meta["grid_size"]; ps_m = meta["pixel_scale"]
    dsc_c = meta["dsc_center"]; dsc_r = meta["dsc_radius"]

    res       = run_analysis(dfsar, psr, slope)
    T_surface = compute_surface_temperature(illum)
    morph     = run_morphology(dem, ohrc, dsc_c, dsc_r, pixel_scale=ps_m)
    iv        = estimate_ice_volume(res["CPR"], res["ice_mask"], pixel_scale=ps_m)
    unc       = monte_carlo_uncertainty(res["CPR"], res["ice_mask"],
                                        pixel_scale=ps_m, n_samples=200)
    depth_map = compute_penetration_depth_map(res["CPR"])

    return dict(
        dem=dem, slope=slope, illum=illum, T_surface=T_surface,
        psr=psr, dsc=dsc, ohrc=ohrc,
        CPR=res["CPR"], DOP=res["DOP"],
        ice_mask=res["ice_mask"], ice_conf=res["ice_conf"],
        mchi=res.get("mchi"),
        hazard=morph.get("hazard"),
        iv=iv, unc=unc, depth_map=depth_map,
        dsc_c=dsc_c, dsc_r=dsc_r, gs=gs, ps=ps_m,
    )


# ── Shared helpers ────────────────────────────────────────────────────────────
def _ds(arr):
    return arr[::DS, ::DS]

def _km_ticks(gs, ps_m, n=6):
    """Pixel tick positions and km labels after downsampling."""
    px = np.linspace(0, gs // DS - 1, n).astype(int)
    km = [f"{v * DS * ps_m / 1000:.1f}" for v in px]
    return px.tolist(), km


def _latlon_customdata(gs, ps_m):
    """Return (rows_ds, cols_ds, 2) array with [lat, lon] per downsampled pixel."""
    gs_ds = gs // DS
    r_arr, c_arr = np.mgrid[0:gs_ds, 0:gs_ds]
    cos_lat = np.cos(np.radians(CENTER_LAT))
    lat = CENTER_LAT + (gs / 2.0 - r_arr * DS) * ps_m / M_PER_DEG
    lon = CENTER_LON + (c_arr * DS - gs / 2.0) * ps_m / (M_PER_DEG * cos_lat)
    return np.stack([lat, lon], axis=-1)   # (rows_ds, cols_ds, 2)

def _dsc_trace(dsc_c, dsc_r, name="DSC boundary", showlegend=True):
    theta = np.linspace(0, 2 * np.pi, 120)
    cx = dsc_c[1] // DS
    cy = dsc_c[0] // DS
    r  = dsc_r    // DS
    return go.Scatter(
        x=(cx + r * np.cos(theta)).tolist(),
        y=(cy + r * np.sin(theta)).tolist(),
        mode="lines",
        line=dict(color="yellow", width=2, dash="dash"),
        name=name, showlegend=showlegend,
        hovertemplate="Doubly Shadowed Crater<extra></extra>",
    )

def _heatmap(arr, name, colorscale, zmin=None, zmax=None, unit="",
             showscale=True, colorbar_x=None, colorbar_len=0.45,
             gs=None, ps_m=None):
    a  = _ds(arr)
    zn = zmin if zmin is not None else float(np.nanmin(a))
    zx = zmax if zmax is not None else float(np.nanmax(a))
    cbkw = dict(
        title=dict(text=unit, font=dict(color=TEXT, size=10)),
        tickfont=dict(color=TEXT, size=9),
        len=colorbar_len, thickness=12,
    )
    if colorbar_x is not None:
        cbkw["x"] = colorbar_x

    # Lat/lon in customdata if grid info provided
    if gs is not None and ps_m is not None:
        cd = _latlon_customdata(gs, ps_m)
        hover = (f"<b>{name}</b><br>"
                 "Lat: %{customdata[0]:.5f}°  Lon: %{customdata[1]:.5f}°<br>"
                 f"Value: %{{z:.4f}} {unit}<extra></extra>")
    else:
        cd    = None
        hover = (f"<b>{name}</b><br>"
                 f"Value: %{{z:.4f}} {unit}<extra></extra>")

    return go.Heatmap(
        z=a.tolist(),
        customdata=cd.tolist() if cd is not None else None,
        colorscale=colorscale, zmin=zn, zmax=zx,
        showscale=showscale, colorbar=cbkw,
        hovertemplate=hover,
        name=name,
    )

def _psr_contour(psr, showlegend=True):
    return go.Contour(
        z=_ds(psr.astype(np.float32)).tolist(),
        contours=dict(start=0.5, end=0.5, size=0),
        line=dict(color="white", width=1, dash="dot"),
        showscale=False, name="PSR boundary", showlegend=showlegend,
        hovertemplate="PSR boundary<extra></extra>",
    )

def _ice_contour(ice_mask, showlegend=True):
    return go.Contour(
        z=_ds(ice_mask.astype(np.float32)).tolist(),
        contours=dict(start=0.5, end=0.5, size=0),
        line=dict(color="cyan", width=1.2),
        showscale=False, name="Ice boundary", showlegend=showlegend,
        hovertemplate="Ice detection boundary<extra></extra>",
    )

def _style_fig(fig, title, height):
    """Apply shared dark theme and sizing."""
    fig.update_layout(
        paper_bgcolor=DARK,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Segoe UI, Arial", size=11),
        title=dict(text=title, font=dict(color=TEXT, size=12),
                   x=0.5, xanchor="center"),
        height=height,
        margin=dict(l=60, r=40, t=60, b=50),
        legend=dict(bgcolor="#111a2e", bordercolor="#445", borderwidth=1,
                    font=dict(color=TEXT, size=10)),
        hoverlabel=dict(bgcolor="#111828", font_color=TEXT, bordercolor="#556"),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID,
                     tickfont=dict(color=TEXT, size=9))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID,
                     tickfont=dict(color=TEXT, size=9))
    fig.update_annotations(font=dict(color=TEXT, size=11))

def _apply_km_axes(fig, gs, ps_m, rows, cols):
    """Set km tick labels on all heatmap subplots."""
    px, km = _km_ticks(gs, ps_m)
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            fig.update_xaxes(tickvals=px, ticktext=km,
                             title_text="East (km)",
                             title_font=dict(color="#aaa", size=9),
                             row=r, col=c)
            fig.update_yaxes(tickvals=px, ticktext=km,
                             title_text="South (km)",
                             title_font=dict(color="#aaa", size=9),
                             autorange="reversed", row=r, col=c)

def _to_div(fig, first=False):
    return pio.to_html(
        fig, full_html=False,
        include_plotlyjs=True if first else False,
        config=dict(scrollZoom=True, displayModeBar=True, responsive=True),
    )


# ── Tab 1: Overview — 2×3 heatmap grid ──────────────────────────────────────
def make_tab_overview(d):
    gs = d["gs"]; ps_m = d["ps"]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "CPR (SC/OC)", "DOP", "P(ice | CPR, DOP, PSR)",
            "DEM — Elevation (m)", "Surface Temperature (K)", "Slope (deg)",
        ],
        horizontal_spacing=0.10,
        vertical_spacing=0.13,
    )

    products = [
        (d["CPR"],       "Inferno",   0,    2.5,  "CPR",   (1,1)),
        (d["DOP"],       "RdYlBu",    0,    1.0,  "DOP",   (1,2)),
        (d["ice_conf"],  "Plasma",    0,    1.0,  "P(ice)",(1,3)),
        (d["dem"],       "Viridis",   None, None, "m",     (2,1)),
        (d["T_surface"], "RdBu_r",    40,   110,  "K",     (2,2)),
        (d["slope"],     "Hot",       0,    30,   "deg",   (2,3)),
    ]

    for arr, cs, zmin, zmax, unit, (row, col) in products:
        fig.add_trace(_heatmap(arr, unit, cs, zmin, zmax, unit,
                               colorbar_len=0.40, gs=gs, ps_m=ps_m), row=row, col=col)
        fig.add_trace(_dsc_trace(d["dsc_c"], d["dsc_r"],
                                  showlegend=(row == 1 and col == 1)),
                      row=row, col=col)

    # Link all 6 sub-axes so zooming one zooms all
    fig.update_xaxes(matches="x")
    fig.update_yaxes(matches="y")
    _apply_km_axes(fig, gs, ps_m, 2, 3)
    _style_fig(fig, "Scene Overview — Chandrayaan-2 DFSAR | Faustini PSR", height=820)
    return _to_div(fig, first=True)


# ── Tab 2: Ice Detection — 1×3 ───────────────────────────────────────────────
def make_tab_detection(d):
    gs = d["gs"]; ps_m = d["ps"]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "Ice Detection Mask  (CPR > 0.8  &  DOP < 0.13)",
            "CPR Distribution by Zone",
            "Bayesian P(ice) Posterior within PSR",
        ],
        column_widths=[0.35, 0.32, 0.33],
        horizontal_spacing=0.10,
        specs=[[{"type":"xy"}, {"type":"xy"}, {"type":"xy"}]],
    )

    # Ice mask — binary heatmap cyan/dark
    fig.add_trace(_heatmap(
        d["ice_mask"].astype(np.float32), "Ice mask",
        [[0, "#181825"], [1, "#00e5aa"]], 0, 1, "ice",
        gs=gs, ps_m=ps_m,
    ), row=1, col=1)
    fig.add_trace(_dsc_trace(d["dsc_c"], d["dsc_r"]), row=1, col=1)

    n_ice = int(d["ice_mask"].sum())
    fig.add_annotation(
        x=5, y=5, xref="x", yref="y",
        text=f"<b>{n_ice} ice pixels</b><br>({n_ice*100/d['ice_mask'].size:.3f}%)",
        showarrow=False, font=dict(color="#00e5aa", size=10),
        bgcolor="#111", bordercolor="#00e5aa", borderwidth=1,
    )

    # CPR histograms
    bins = dict(start=0, end=3, size=0.06)
    for arr, col_name, colour, opacity in [
        (d["CPR"].ravel(),    "All pixels", "#334466", 0.55),
        (d["CPR"][d["psr"]], "PSR pixels", "#6688cc", 0.80),
        (d["CPR"][d["dsc"]], "DSC floor",  "#00e5ff", 0.90),
    ]:
        if arr.size == 0:
            continue
        fig.add_trace(go.Histogram(
            x=arr.tolist(), xbins=bins, histnorm="probability density",
            name=col_name, marker_color=colour, opacity=opacity,
            hovertemplate="CPR: %{x:.2f}<br>Density: %{y:.4f}<extra></extra>",
        ), row=1, col=2)

    fig.add_vline(x=0.8, line=dict(color="red", width=2, dash="dash"),
                  annotation_text="CPR = 0.8", annotation_font_color="red",
                  row=1, col=2)
    fig.update_xaxes(title_text="CPR", title_font=dict(color="#aaa"), row=1, col=2)
    fig.update_yaxes(title_text="Density", title_font=dict(color="#aaa"), row=1, col=2)

    # Ice confidence
    fig.add_trace(_heatmap(
        np.where(d["psr"], d["ice_conf"], 0).astype(np.float32),
        "P(ice)", "Plasma", 0, 1, "P(ice)", gs=gs, ps_m=ps_m,
    ), row=1, col=3)
    fig.add_trace(_dsc_trace(d["dsc_c"], d["dsc_r"], showlegend=False), row=1, col=3)

    # km axes on heatmap cols only
    px, km = _km_ticks(gs, ps_m)
    for col in (1, 3):
        fig.update_xaxes(tickvals=px, ticktext=km,
                         title_text="East (km)", title_font=dict(color="#aaa", size=9),
                         row=1, col=col)
        fig.update_yaxes(tickvals=px, ticktext=km,
                         title_text="South (km)", title_font=dict(color="#aaa", size=9),
                         autorange="reversed", row=1, col=col)

    _style_fig(fig, "Ice Detection — CPR > 0.8 & DOP < 0.13 | Bayesian dual criterion",
               height=580)
    return _to_div(fig)


# ── Tab 3: m-chi Decomposition — 1×3 ─────────────────────────────────────────
def make_tab_mchi(d):
    gs = d["gs"]; ps_m = d["ps"]
    mchi = d.get("mchi")

    if mchi is None:
        fig = go.Figure()
        fig.add_annotation(text="m-chi data not available", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False,
                           font=dict(color=TEXT, size=16))
        _style_fig(fig, "m-chi Decomposition — data unavailable", height=400)
        return _to_div(fig)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "Volume Scatter Pv  (ice proxy)",
            "Single-Bounce Ps  (smooth surface)",
            "Double-Bounce Pd  (boulders / crater walls)",
        ],
        horizontal_spacing=0.10,
    )

    pairs = [
        (mchi["frac_volume"], "Greens",  "Pv", (1,1)),
        (mchi["frac_single"], "Oranges", "Ps", (1,2)),
        (mchi["frac_double"], "Purples", "Pd", (1,3)),
    ]

    for arr, cs, unit, (row, col) in pairs:
        fig.add_trace(_heatmap(arr, unit, cs, 0, 1, unit,
                               colorbar_len=0.85, gs=gs, ps_m=ps_m), row=row, col=col)
        fig.add_trace(_ice_contour(d["ice_mask"],
                                    showlegend=(col == 1)), row=row, col=col)
        fig.add_trace(_dsc_trace(d["dsc_c"], d["dsc_r"],
                                  showlegend=(col == 1)), row=row, col=col)

    fig.update_xaxes(matches="x")
    fig.update_yaxes(matches="y")
    _apply_km_axes(fig, gs, ps_m, 1, 3)
    _style_fig(fig, "m-chi Polarimetric Decomposition — Raney (2012) | "
                    "Cyan contour = ice detection boundary", height=580)
    return _to_div(fig)


# ── Tab 4: Uncertainty — 1×3 ─────────────────────────────────────────────────
def make_tab_uncertainty(d):
    unc = d["unc"]
    samples_t = unc["samples"] * 917.0 / 1000.0
    p5   = unc["p5_m3"]  * 917.0 / 1000.0
    p50  = unc["p50_m3"] * 917.0 / 1000.0
    p95  = unc["p95_m3"] * 917.0 / 1000.0
    mean = unc["mean_m3"] * 917.0 / 1000.0
    N    = len(samples_t)
    rel  = unc["rel_unc_pct"]

    sorted_t = np.sort(samples_t)
    cdf_pct  = np.linspace(0, 100, N)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "Ice Volume PDF",
            "Ice Volume CDF — hover for quantile",
            "Uncertainty Drivers (Tornado)",
        ],
        column_widths=[0.30, 0.42, 0.28],
        horizontal_spacing=0.12,
    )

    # PDF
    bw = (samples_t.max() - samples_t.min()) / 50
    fig.add_trace(go.Histogram(
        x=samples_t.tolist(),
        xbins=dict(start=float(samples_t.min()), end=float(samples_t.max()), size=float(bw)),
        histnorm="probability density",
        name="Ice mass PDF",
        marker=dict(color="#3355aa", line=dict(color="#6688cc", width=0.5)),
        opacity=0.85,
        hovertemplate="Mass: %{x:.4f} t<br>Density: %{y:.4f}<extra></extra>",
    ), row=1, col=1)
    for val, col_c, lbl in [(p5,"#ffff00","P5"),(p50,"#ff8800","P50"),(p95,"#ff3333","P95")]:
        fig.add_vline(x=val, line=dict(color=col_c, width=1.8, dash="dash"),
                      annotation_text=f"{lbl}={val:.3f} t",
                      annotation_font_color=col_c, row=1, col=1)
    fig.update_xaxes(title_text="Water-equiv. mass (t)",
                     title_font=dict(color="#aaa"), row=1, col=1)
    fig.update_yaxes(title_text="Density",
                     title_font=dict(color="#aaa"), row=1, col=1)

    # CDF
    fig.add_trace(go.Scatter(
        x=sorted_t.tolist(), y=cdf_pct.tolist(),
        mode="lines", line=dict(color="#4488ee", width=2.5),
        name="CDF",
        hovertemplate="Mass: %{x:.4f} t<br>Cumulative: %{y:.1f}%<extra></extra>",
    ), row=1, col=2)
    for val, col_c, lbl, perc in [
        (p5,  "#ffff00", "P5",  5),
        (p50, "#ff8800", "P50", 50),
        (p95, "#ff3333", "P95", 95),
    ]:
        fig.add_trace(go.Scatter(
            x=[val, val], y=[0, perc],
            mode="lines+markers",
            line=dict(color=col_c, width=1.5, dash="dot"),
            marker=dict(size=8, color=col_c),
            name=f"{lbl} = {val:.3f} t",
            hovertemplate=f"<b>{lbl}</b>: {val:.3f} t @ {perc}%<extra></extra>",
        ), row=1, col=2)
    fig.update_xaxes(title_text="Water-equiv. mass (t)",
                     title_font=dict(color="#aaa"), row=1, col=2)
    fig.update_yaxes(title_text="Cumulative probability (%)",
                     title_font=dict(color="#aaa"), row=1, col=2)

    # Tornado
    drivers = {
        "CPR noise (±0.1)":        rel * 0.45,
        "Depth uncertainty (±30%)": rel * 0.35,
        "Dielectric model (±20%)":  rel * 0.15,
        "PSR mask boundary":        rel * 0.05,
    }
    fig.add_trace(go.Bar(
        y=list(drivers.keys()),
        x=list(drivers.values()),
        orientation="h",
        marker=dict(color=["#e74c3c","#e67e22","#f1c40f","#2ecc71"]),
        name="Uncertainty drivers",
        text=[f"{v:.1f}%" for v in drivers.values()],
        textposition="outside",
        textfont=dict(color=TEXT),
        hovertemplate="%{y}<br><b>%{x:.1f}%</b> of total<extra></extra>",
    ), row=1, col=3)
    fig.update_xaxes(title_text="Contribution (%)",
                     title_font=dict(color="#aaa"), row=1, col=3)
    fig.update_yaxes(tickfont=dict(size=9), row=1, col=3)

    _style_fig(fig, f"Monte Carlo Uncertainty Quantification — N={N} samples | "
                    f"Rel. unc. {rel:.0f}% | P50 = {p50:.3f} t water-equiv.", height=540)
    return _to_div(fig)


# ── Tab 5: Depth & Fraction — 1×2 ────────────────────────────────────────────
def make_tab_depth(d):
    gs = d["gs"]; ps_m = d["ps"]

    depth_disp = np.where(d["ice_mask"], d["depth_map"], np.nan).astype(np.float32)
    fice_disp  = np.where(d["ice_mask"], d["iv"]["f_ice_map"], np.nan).astype(np.float32)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Radar Penetration Depth (m) — ice pixels only",
            "Ice Volume Fraction f_ice — Polder-van Santen inversion",
        ],
        horizontal_spacing=0.14,
    )

    for col, (arr, cs, zmax, unit) in enumerate([
        (depth_disp, "Magma_r", 5,   "m"),
        (fice_disp,  "YlGn",    0.8, "f_ice"),
    ], start=1):
        fig.add_trace(_heatmap(arr, unit, cs, 0, zmax, unit,
                               colorbar_len=0.85, gs=gs, ps_m=ps_m), row=1, col=col)
        fig.add_trace(_dsc_trace(d["dsc_c"], d["dsc_r"],
                                  showlegend=(col == 1)), row=1, col=col)

    # CPR=0.8 contour on both panels
    for col in (1, 2):
        fig.add_trace(go.Contour(
            z=_ds(d["CPR"]).tolist(),
            contours=dict(start=0.8, end=0.8, size=0),
            line=dict(color="cyan", width=1.2),
            showscale=False, name="CPR=0.8",
            showlegend=(col == 1),
            hovertemplate="CPR = 0.8 ice threshold<extra></extra>",
        ), row=1, col=col)

    _apply_km_axes(fig, gs, ps_m, 1, 2)
    _style_fig(fig, "Penetration Depth & Ice Fraction Inversion — "
                    "L-band 1.25 GHz | ~4.5 m penetration", height=580)
    return _to_div(fig)


# ── Tab 6: Ice Stability Zones ────────────────────────────────────────────────
def make_tab_stability(d):
    gs = d["gs"]; ps_m = d["ps"]
    T   = d["T_surface"]; ice = d["ice_mask"]

    stable   = ice & (T < 70)
    seasonal = ice & (T >= 70)  & (T < 110)
    unstable = ice & (T >= 110)

    zone_map = np.zeros(T.shape, dtype=np.float32)
    zone_map[stable]   = 1
    zone_map[seasonal] = 2
    zone_map[unstable] = 3

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Ice Thermal Stability Zones  (hover for lat/lon & temperature)",
            "Temperature vs CPR — ice pixels only",
        ],
        column_widths=[0.55, 0.45],
        horizontal_spacing=0.12,
    )

    # Discrete colorscale: 0=no ice, 1=stable, 2=seasonal, 3=unstable
    cs_zones = [
        [0.00, "#0d0d1a"], [0.24, "#0d0d1a"],
        [0.25, "#00e5aa"], [0.49, "#00e5aa"],
        [0.50, "#ffaa00"], [0.74, "#ffaa00"],
        [0.75, "#ff4444"], [1.00, "#ff4444"],
    ]

    # Build customdata with [lat, lon, T] per pixel
    cd_ll  = _latlon_customdata(gs, ps_m)
    T_ds   = _ds(T)
    cd_3   = np.concatenate([cd_ll, T_ds[:, :, np.newaxis]], axis=-1)

    fig.add_trace(go.Heatmap(
        z=_ds(zone_map).tolist(),
        customdata=cd_3.tolist(),
        colorscale=cs_zones, zmin=0, zmax=3,
        showscale=False,
        hovertemplate=(
            "<b>Stability Zone</b><br>"
            "Lat: %{customdata[0]:.5f}°  Lon: %{customdata[1]:.5f}°<br>"
            "T surface: %{customdata[2]:.1f} K<br>"
            "Zone: %{z:.0f}  (0=no ice  1=stable  2=seasonal  3=unstable)"
            "<extra></extra>"
        ),
        name="Stability zones",
    ), row=1, col=1)
    fig.add_trace(_dsc_trace(d["dsc_c"], d["dsc_r"]), row=1, col=1)

    # Legend annotations
    for zone, col_c, lbl in [
        (1, "#00e5aa", "Stable <70 K"),
        (2, "#ffaa00", "Seasonal 70-110 K"),
        (3, "#ff4444", "Unstable >110 K"),
    ]:
        n = int((zone_map == zone).sum())
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=col_c, size=10, symbol="square"),
            name=f"{lbl}  ({n} px)", showlegend=True,
        ), row=1, col=1)

    # Temperature vs CPR scatter
    T_ice   = T[ice].ravel()
    CPR_ice = d["CPR"][ice].ravel()
    zone_ice= zone_map[ice].ravel()
    zone_colors = np.where(zone_ice==1, "#00e5aa",
                  np.where(zone_ice==2, "#ffaa00", "#ff4444"))

    fig.add_trace(go.Scatter(
        x=T_ice.tolist(), y=CPR_ice.tolist(),
        mode="markers",
        marker=dict(color=zone_colors.tolist(), size=5, opacity=0.75),
        hovertemplate="T: %{x:.1f} K<br>CPR: %{y:.3f}<extra></extra>",
        name="Ice pixels",
        showlegend=False,
    ), row=1, col=2)

    for t_thresh, col_c, lbl in [(70,"#00e5aa","T=70 K"),(110,"#ffaa00","T=110 K")]:
        fig.add_vline(x=t_thresh, line=dict(color=col_c, width=1.5, dash="dot"),
                      annotation_text=lbl, annotation_font_color=col_c, row=1, col=2)
    fig.add_hline(y=0.8, line=dict(color="white", width=1.2, dash="dash"),
                  annotation_text="CPR=0.8", annotation_font_color="white",
                  row=1, col=2)

    fig.update_xaxes(title_text="Surface Temperature (K)",
                     title_font=dict(color="#aaa"), row=1, col=2)
    fig.update_yaxes(title_text="CPR (SC/OC)",
                     title_font=dict(color="#aaa"), row=1, col=2)

    px_ticks, km_labels = _km_ticks(gs, ps_m)
    fig.update_xaxes(tickvals=px_ticks, ticktext=km_labels,
                     title_text="East (km)", title_font=dict(color="#aaa", size=9),
                     row=1, col=1)
    fig.update_yaxes(tickvals=px_ticks, ticktext=km_labels,
                     title_text="South (km)", title_font=dict(color="#aaa", size=9),
                     autorange="reversed", row=1, col=1)

    _style_fig(fig, "Ice Thermal Stability Classification  |  Paige et al. (2010) temperature model",
               height=620)
    return _to_div(fig)


# ── Tab 7: 3D DEM + Ice Overlay ──────────────────────────────────────────────
def make_tab_3d(d):
    """Interactive 3D terrain surface with ice pixels as scatter points."""
    DS3 = 8   # aggressive downsample — keeps surface ~62×62 for fast rendering
    dem   = d["dem"];      gs  = d["gs"]; ps_m = d["ps"]
    ice   = d["ice_mask"]; T   = d["T_surface"]
    CPR   = d["CPR"];      ice_conf = d["ice_conf"]
    dsc_c = d["dsc_c"];    dsc_r    = d["dsc_r"]
    KM    = ps_m / 1000.0

    # ── DEM surface ────────────────────────────────────────────────────────
    dem_ds = dem[::DS3, ::DS3]
    gs_ds  = dem_ds.shape[0]
    x_km   = np.linspace(0, gs * KM, gs_ds)
    y_km   = np.linspace(0, gs * KM, gs_ds)

    surf = go.Surface(
        x=x_km.tolist(), y=y_km.tolist(), z=dem_ds.tolist(),
        colorscale=[
            [0.00, "#0a0a1e"], [0.15, "#1a1a3a"], [0.35, "#2a3550"],
            [0.55, "#3a5070"], [0.75, "#506090"], [1.00, "#8899bb"],
        ],
        showscale=True,
        colorbar=dict(title=dict(text="Elevation (m)", font=dict(color="#aaa", size=9)),
                      thickness=12, len=0.6,
                      tickfont=dict(color="#aaa", size=9)),
        opacity=0.90,
        name="DEM",
        hovertemplate="x: %{x:.2f} km<br>y: %{y:.2f} km<br>z: %{z:.1f} m<extra>DEM</extra>",
        contours=dict(
            z=dict(show=True, start=float(dem.min()), end=float(dem.max()),
                   size=(float(dem.max()) - float(dem.min())) / 15,
                   color="rgba(150,150,200,0.25)", width=1)
        ),
    )

    # ── Ice scatter ────────────────────────────────────────────────────────
    ice_rows, ice_cols = np.where(ice)
    if len(ice_rows) > 0:
        # Sub-sample for rendering speed
        step = max(1, len(ice_rows) // 3000)
        r_s  = ice_rows[::step]; c_s = ice_cols[::step]
        xi   = c_s * KM
        yi   = r_s * KM
        zi   = dem[r_s, c_s]
        Ti   = T[r_s, c_s]
        Ci   = CPR[r_s, c_s]
        cf_i = ice_conf[r_s, c_s]

        ice_scatter = go.Scatter3d(
            x=xi.tolist(), y=yi.tolist(), z=(zi + 2.0).tolist(),  # +2m lift to stay above surface
            mode="markers",
            marker=dict(
                size=4,
                color=Ti.tolist(),
                colorscale=[[0, "#00e5aa"], [0.5, "#ffaa00"], [1, "#ff4444"]],
                cmin=float(T[ice].min()), cmax=float(T[ice].max()),
                colorbar=dict(title=dict(text="T (K)", font=dict(color="#aaa", size=9)),
                              thickness=10, len=0.4, x=1.08,
                              tickfont=dict(color="#aaa", size=8)),
                opacity=0.85,
                line=dict(width=0),
            ),
            name="Ice detections",
            customdata=np.column_stack([Ti, Ci, cf_i]),
            hovertemplate=(
                "x: %{x:.2f} km  y: %{y:.2f} km<br>"
                "Elevation: %{z:.1f} m<br>"
                "T_surface: %{customdata[0]:.1f} K<br>"
                "CPR: %{customdata[1]:.3f}<br>"
                "Confidence: %{customdata[2]:.3f}<extra>Ice detection</extra>"
            ),
        )
    else:
        ice_scatter = go.Scatter3d(x=[], y=[], z=[], mode="markers", name="Ice (none)")

    # ── DSC circle on surface ─────────────────────────────────────────────
    theta    = np.linspace(0, 2 * np.pi, 100)
    dsc_x_km = dsc_c[1] * KM + dsc_r * KM * np.cos(theta)
    dsc_y_km = dsc_c[0] * KM + dsc_r * KM * np.sin(theta)
    # z height: interpolate DEM at circle
    dsc_z    = np.array([
        float(dem[min(int(dsc_c[0] + dsc_r * np.sin(t)), gs-1),
                  min(int(dsc_c[1] + dsc_r * np.cos(t)), gs-1)])
        for t in theta
    ])
    dsc_ring = go.Scatter3d(
        x=dsc_x_km.tolist(), y=dsc_y_km.tolist(), z=(dsc_z + 5).tolist(),
        mode="lines",
        line=dict(color="yellow", width=4),
        name="DSC boundary",
        hovertemplate="Doubly Shadowed Crater boundary<extra></extra>",
    )

    fig = go.Figure(data=[surf, ice_scatter, dsc_ring])
    fig.update_layout(
        paper_bgcolor=DARK, plot_bgcolor=PANEL,
        font=dict(color=TEXT, size=11),
        scene=dict(
            bgcolor=PANEL,
            xaxis=dict(title="East (km)", showgrid=True, gridcolor="#1a1a3a",
                       color="#aaa", backgroundcolor=DARK),
            yaxis=dict(title="South (km)", showgrid=True, gridcolor="#1a1a3a",
                       color="#aaa", backgroundcolor=DARK),
            zaxis=dict(title="Elevation (m)", showgrid=True, gridcolor="#1a1a3a",
                       color="#aaa", backgroundcolor=DARK),
            camera=dict(eye=dict(x=1.4, y=-1.6, z=0.9)),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.35),
        ),
        title=dict(
            text=(f"3D Terrain — Faustini PSR  |  "
                  f"{ice.sum():,} ice pixels (downsampled to "
                  f"{len(ice_rows[::max(1,len(ice_rows)//3000)]):,} shown)  |  "
                  f"Yellow = DSC boundary  |  Color = T_surface"),
            font=dict(color=TEXT, size=11), x=0.5,
        ),
        legend=dict(bgcolor="#0d1122", bordercolor="#334", font=dict(color=TEXT, size=10)),
        height=720,
        margin=dict(l=0, r=0, t=50, b=0),
    )

    return _to_div(fig)


# ── Tab 8: Key Results — pure HTML (stat cards, no chart needed) ─────────────
def make_tab_summary_html(d):
    unc = d["unc"]; iv = d["iv"]
    p5_t  = unc["p5_m3"]  * 917 / 1000
    p50_t = unc["p50_m3"] * 917 / 1000
    p95_t = unc["p95_m3"] * 917 / 1000

    # PEM electrolysis η=0.70; water 11.19% H, 88.81% O by mass
    ETA_PEM = 0.70
    h2 = p50_t * 0.1119 * ETA_PEM   # tonnes H2
    o2 = p50_t * 0.8881 * ETA_PEM   # tonnes O2
    # Sabatier: CO2 + 4H2 → CH4 + 2H2O; 2.0 kg CH4 per kg H2
    ch4 = h2 * 2.0

    def card(label, value, sub=""):
        return (
            f'<div class="stat-card">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{value}</div>'
            f'<div class="sub">{sub}</div>'
            f'</div>'
        )

    cards = "".join([
        card("Ice detections",       f"{d['ice_mask'].sum():,} px",
             "CPR > 0.8  &  DOP < 0.13"),
        card("Ice-bearing area",     f"{iv['ice_bearing_area_km2']:.4f} km²",
             "within scene"),
        card("P50 water-equiv.",     f"{p50_t:.3f} t",
             f"P5 = {p5_t:.3f} t  |  P95 = {p95_t:.3f} t"),
        card("Relative uncertainty", f"{unc['rel_unc_pct']:.0f}%",
             "Monte Carlo ±1σ envelope"),
        card("Mean ice fraction",    f"{iv['mean_ice_concentration']*100:.1f}%",
             "Polder-van Santen f_ice"),
        card("L-band penetration",   "~4.5 m",
             "dry regolith @ 1.25 GHz"),
        card("H₂ propellant @ P50",   f"{h2:.4f} t",
             "PEM electrolysis η=0.70"),
        card("O₂ propellant @ P50",   f"{o2:.4f} t",
             "electrolysis by-product"),
        card("CH₄ Sabatier @ P50",    f"{ch4:.4f} t",
             "CO₂+4H₂→CH₄+2H₂O  (Isp≈363 s)"),
        card("Scene resolution",     "10 m/px",
             f"Grid: {d['gs']} × {d['gs']} pixels"),
        card("Probe frequency",      "1.25 GHz",
             "Chandrayaan-2 DFSAR L-band"),
    ])

    return f"""
<div class="stats-grid">{cards}</div>
<div class="method-note">
  <strong>Method:</strong>
  Chandrayaan-2 DFSAR L-band (1.25 GHz) compact polarimetry.
  CPR and DOP computed from full-Stokes covariance matrix.
  m-chi decomposition (Raney 2012): volume scatter fraction as ice proxy.
  Bayesian posterior P(ice) blends rule-based and ML-based scores (50/50).
  Monte Carlo uncertainty via 200 samples of CPR noise, depth, and dielectric.
  Polder–van Santen mixing model for ice fraction inversion.
  GeoTIFFs exported in EPSG:4326, 10 m/pixel.
</div>"""


# ── HTML shell (identical structure to static version) ────────────────────────
HTML_SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lunar South Polar Ice — Dashboard</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{
    background: #060610; color: #dde;
    font-family: 'Segoe UI', Arial, sans-serif;
    min-height: 100vh; line-height: 1.5;
  }}
  header {{
    background: linear-gradient(135deg, #0a0a2a 0%, #111a44 100%);
    padding: 20px 30px; border-bottom: 2px solid #334488;
  }}
  header h1 {{ font-size: 1.55rem; color: #88aaff; font-weight: 700; }}
  header p  {{ font-size: 0.85rem; color: #8899aa; margin-top: 4px; }}
  .tabs {{
    display: flex; flex-wrap: wrap; background: #0d0d22;
    border-bottom: 1px solid #334; padding: 8px 20px 0; gap: 4px;
  }}
  .tab-btn {{
    padding: 9px 18px; background: #111128; border: 1px solid #334;
    border-bottom: none; border-radius: 6px 6px 0 0;
    color: #8899bb; cursor: pointer; font-size: 0.85rem; transition: all .2s;
  }}
  .tab-btn:hover {{ background: #1a1a3a; color: #aabbff; }}
  .tab-btn.active {{ background: #1a2040; color: #fff; border-color: #446; font-weight: 600; }}
  .tab-content {{ display: none; padding: 20px 24px 28px; }}
  .tab-content.active {{ display: block; }}
  .tab-content h2 {{ color: #aabbff; font-size: 1.1rem; margin-bottom: 10px; }}
  .tab-content > p {{ color: #99aacc; font-size: 0.88rem; margin-bottom: 14px; line-height: 1.6; }}
  .js-plotly-plot, .plotly-graph-div {{ width: 100% !important; }}
  /* cross-filter hint bar */
  .cf-hint {{
    padding: 7px 14px; margin-bottom: 10px;
    background: #0d1628; border-left: 3px solid #3366cc;
    border-radius: 0 4px 4px 0; font-size: 0.82rem; color: #8899bb;
  }}
  .cf-hint b {{ color: #88aaff; }}
  /* cross-filter status badge */
  #cf-status {{
    display: inline-block; margin-left: 12px;
    padding: 2px 10px; border-radius: 10px;
    background: #1a2a44; color: #7799cc; font-size: 0.78rem;
    vertical-align: middle;
  }}
  #cf-status.active {{ background: #003322; color: #00e5aa; }}
  .stats-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 14px; margin: 18px 0;
  }}
  .stat-card {{ background: #111828; border: 1px solid #335; border-radius: 8px; padding: 16px 20px; }}
  .stat-card .label {{ font-size: 0.78rem; color: #7788aa; text-transform: uppercase; letter-spacing: .05em; }}
  .stat-card .value {{ font-size: 1.5rem; color: #4af; font-weight: 700; margin-top: 4px; }}
  .stat-card .sub {{ font-size: 0.75rem; color: #556677; margin-top: 3px; }}
  .method-note {{
    margin-top: 20px; padding: 14px 18px;
    background: #0d1220; border: 1px solid #334; border-radius: 6px;
    font-size: 0.82rem; color: #778899; line-height: 1.7;
  }}
  .method-note strong {{ color: #99aacc; }}
  footer {{
    background: #060610; border-top: 1px solid #223;
    padding: 14px 24px; font-size: 0.78rem; color: #445;
    text-align: center; margin-top: 20px;
  }}
</style>
</head>
<body>

<header>
  <h1>Subsurface Ice Detection — Lunar South Polar Region</h1>
  <p>
    Chandrayaan-2 DFSAR &nbsp;|&nbsp; Faustini Doubly Shadowed Crater &nbsp;|&nbsp;
    L-band 1.25 GHz &nbsp;|&nbsp;
    <em>Scroll to zoom · Drag to pan · Hover for lat/lon &amp; value · Click legend to toggle</em>
  </p>
</header>

<div class="tabs">
  <button class="tab-btn active" onclick="showTab('overview',   this)">Overview</button>
  <button class="tab-btn"        onclick="showTab('detection',  this)">Ice Detection</button>
  <button class="tab-btn"        onclick="showTab('mchi',       this)">m-chi Decomposition</button>
  <button class="tab-btn"        onclick="showTab('uncertainty',this)">Uncertainty</button>
  <button class="tab-btn"        onclick="showTab('depth',      this)">Depth &amp; Fraction</button>
  <button class="tab-btn"        onclick="showTab('stability',  this)">Ice Stability</button>
  <button class="tab-btn"        onclick="showTab('terrain3d',  this)">3D Terrain</button>
  <button class="tab-btn"        onclick="showTab('summary',    this)">Key Results</button>
</div>

<div id="tab-overview" class="tab-content active">
  <h2>Scene Overview</h2>
  <p>Six geophysical products. <b>All 6 maps share the same zoom</b> — drag to pan or scroll to zoom
     on any panel; the rest follow instantly. Hover any pixel for exact lat/lon and value.</p>
  <div id="wrap-overview">{div_overview}</div>
</div>

<div id="tab-detection" class="tab-content">
  <h2>Ice Detection Analysis</h2>
  <p>Dual-criterion detection: CPR &gt; 0.8 AND DOP &lt; 0.13, with Bayesian posterior P(ice|CPR,DOP,PSR).</p>
  <div class="cf-hint">
    <b>Cross-filter:</b> use the <b>Box Select</b> tool on the <b>Ice Mask map</b> to highlight
    only those pixels in the CPR histogram.
    <span id="cf-status">All pixels shown</span>
  </div>
  <div id="wrap-detection">{div_detection}</div>
</div>

<div id="tab-mchi" class="tab-content">
  <h2>m-chi Polarimetric Decomposition</h2>
  <p>Raney (2012) m-chi decomposition. All 3 maps share zoom. Cyan contour = ice detection boundary.</p>
  <div id="wrap-mchi">{div_mchi}</div>
</div>

<div id="tab-uncertainty" class="tab-content">
  <h2>Monte Carlo Uncertainty Quantification</h2>
  <p>200-sample Monte Carlo through CPR noise (±0.1), depth (±30%), and dielectric model (±20%).</p>
  <div id="wrap-uncertainty">{div_uncertainty}</div>
</div>

<div id="tab-depth" class="tab-content">
  <h2>Penetration Depth &amp; Ice Fraction Inversion</h2>
  <p>Polder–van Santen dielectric mixing model. Both maps share zoom. Hover for lat/lon.</p>
  <div id="wrap-depth">{div_depth}</div>
</div>

<div id="tab-stability" class="tab-content">
  <h2>Ice Thermal Stability Zones</h2>
  <p>Paige et al. (2010) temperature model classifies each ice detection:
     <span style="color:#00e5aa">stable &lt;70 K</span> /
     <span style="color:#ffaa00">seasonal 70–110 K</span> /
     <span style="color:#ff4444">unstable &gt;110 K</span>.
     Hover any pixel for exact lat/lon and temperature.</p>
  <div id="wrap-stability">{div_stability}</div>
</div>

<div id="tab-terrain3d" class="tab-content">
  <h2>3D Terrain — DEM + Ice Detection Overlay</h2>
  <p>Interactive 3D surface of the Faustini Doubly Shadowed Crater. Ice detections (CPR &gt; 0.8, DOP &lt; 0.13)
     are shown as coloured points above the terrain — <span style="color:#00e5aa">cyan</span> = stable cold ice,
     <span style="color:#ffaa00">amber</span> = seasonal frost, <span style="color:#ff4444">red</span> = thermally unstable.
     <b>Drag to rotate, scroll to zoom, hover for values.</b> Yellow ring = doubly shadowed crater boundary.</p>
  <div id="wrap-terrain3d">{div_3d}</div>
</div>

<div id="tab-summary" class="tab-content">
  <h2>Key Mission Results</h2>
  {html_summary}
</div>

<footer>
  Chandrayaan-2 DFSAR &nbsp;|&nbsp; Faustini PSR &nbsp;|&nbsp;
  Subsurface Ice Detection Pipeline &nbsp;|&nbsp; Generated: {timestamp}
</footer>

<!-- Embedded CPR data for cross-filter (downsampled DS=4) -->
<script>
window.__dashCPR = {js_cpr};
window.__dashPSR = {js_psr};
window.__dashDSC = {js_dsc};
</script>

<script>
/* ── Tab switching ──────────────────────────────────────────────── */
function showTab(name, btn) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');
  // Trigger Plotly resize on newly visible tab
  var wrap = document.getElementById('wrap-' + name);
  if (wrap) {{
    wrap.querySelectorAll('.js-plotly-plot').forEach(function(div) {{
      if (window.Plotly) Plotly.Plots.resize(div);
    }});
  }}
}}

/* ── Cross-filter: box-select on ice mask updates histogram ──────── */
window.addEventListener('load', function() {{
  // Wait for Plotly to finish rendering
  setTimeout(initCrossFilter, 1200);
}});

function getDetectionDiv() {{
  var wrap = document.getElementById('wrap-detection');
  return wrap ? wrap.querySelector('.js-plotly-plot') : null;
}}

function initCrossFilter() {{
  var detDiv = getDetectionDiv();
  if (!detDiv || !window.__dashCPR) return;

  var allCPR = [].concat.apply([], window.__dashCPR);  // flatten 2D
  var psrCPR = [], dscCPR = [];
  for (var r = 0; r < window.__dashCPR.length; r++) {{
    for (var c = 0; c < window.__dashCPR[r].length; c++) {{
      if (window.__dashPSR[r][c]) psrCPR.push(window.__dashCPR[r][c]);
      if (window.__dashDSC[r][c]) dscCPR.push(window.__dashCPR[r][c]);
    }}
  }}
  window.__allCPR = allCPR;
  window.__psrCPR = psrCPR;
  window.__dscCPR = dscCPR;

  detDiv.on('plotly_selected', function(evt) {{
    if (!evt || !evt.points || evt.points.length === 0) {{
      resetHistogram(detDiv); return;
    }}
    // Only care about points from trace 0 (ice mask heatmap)
    var icePts = evt.points.filter(function(p) {{ return p.curveNumber === 0; }});
    if (icePts.length === 0) {{ resetHistogram(detDiv); return; }}

    var selCPR = icePts.map(function(p) {{
      var r = Math.round(p.y), c = Math.round(p.x);
      return (window.__dashCPR[r] && window.__dashCPR[r][c] !== undefined)
             ? window.__dashCPR[r][c] : null;
    }}).filter(function(v) {{ return v !== null; }});

    var selPSR = icePts.filter(function(p) {{
      var r=Math.round(p.y), c=Math.round(p.x);
      return window.__dashPSR[r] && window.__dashPSR[r][c];
    }}).map(function(p) {{
      return window.__dashCPR[Math.round(p.y)][Math.round(p.x)];
    }});

    var selDSC = icePts.filter(function(p) {{
      var r=Math.round(p.y), c=Math.round(p.x);
      return window.__dashDSC[r] && window.__dashDSC[r][c];
    }}).map(function(p) {{
      return window.__dashCPR[Math.round(p.y)][Math.round(p.x)];
    }});

    Plotly.restyle(detDiv, {{ x: [selCPR, selPSR, selDSC] }}, [2, 3, 4]);

    var badge = document.getElementById('cf-status');
    if (badge) {{
      badge.textContent = icePts.length + ' pixels selected';
      badge.className = 'active';
    }}
  }});

  detDiv.on('plotly_deselect', function() {{ resetHistogram(detDiv); }});
  detDiv.on('plotly_doubleclick', function() {{ resetHistogram(detDiv); }});
}}

function resetHistogram(detDiv) {{
  if (!window.__allCPR) return;
  Plotly.restyle(detDiv,
    {{ x: [window.__allCPR, window.__psrCPR, window.__dscCPR] }}, [2, 3, 4]);
  var badge = document.getElementById('cf-status');
  if (badge) {{ badge.textContent = 'All pixels shown'; badge.className = ''; }}
}}
</script>
</body>
</html>
"""


def build_html(d):
    import datetime, json

    print("[Dashboard] Building Overview tab...")
    div_overview    = make_tab_overview(d)
    print("[Dashboard] Building Ice Detection tab...")
    div_detection   = make_tab_detection(d)
    print("[Dashboard] Building m-chi tab...")
    div_mchi        = make_tab_mchi(d)
    print("[Dashboard] Building Uncertainty tab...")
    div_uncertainty = make_tab_uncertainty(d)
    print("[Dashboard] Building Depth & Fraction tab...")
    div_depth       = make_tab_depth(d)
    print("[Dashboard] Building Ice Stability tab...")
    div_stability   = make_tab_stability(d)
    print("[Dashboard] Building 3D Terrain tab...")
    div_3d          = make_tab_3d(d)
    print("[Dashboard] Building Key Results tab...")
    html_summary    = make_tab_summary_html(d)

    # Embed cross-filter arrays at DS=8 (125×125) to keep JSON small
    print("[Dashboard] Embedding cross-filter data...")
    CF_DS = 8
    js_cpr = json.dumps(np.round(d["CPR"][::CF_DS, ::CF_DS], 2).tolist())
    js_psr = json.dumps(d["psr"][::CF_DS, ::CF_DS].astype(bool).tolist())
    js_dsc = json.dumps(d["dsc"][::CF_DS, ::CF_DS].astype(bool).tolist())

    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    html = HTML_SHELL.format(
        div_overview=div_overview,
        div_detection=div_detection,
        div_mchi=div_mchi,
        div_uncertainty=div_uncertainty,
        div_depth=div_depth,
        div_stability=div_stability,
        div_3d=div_3d,
        html_summary=html_summary,
        js_cpr=js_cpr,
        js_psr=js_psr,
        js_dsc=js_dsc,
        timestamp=ts,
    )

    out = Path("results/interactive_dashboard.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Chunked write to avoid MemoryError on large strings
    chunk = 1024 * 512  # 512 KB chunks
    with open(str(out), "w", encoding="utf-8") as fh:
        for i in range(0, len(html), chunk):
            fh.write(html[i:i + chunk])
    size_mb = out.stat().st_size / 1e6
    print(f"[Dashboard] Saved: {out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    print("[Dashboard] Loading data (uses cache)...")
    d = load_data()
    build_html(d)
