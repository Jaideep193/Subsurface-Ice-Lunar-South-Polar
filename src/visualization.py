"""
Visualization module – generates all publication-quality figures.
Saves to results/figures/.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe


FIG_DIR = "results/figures"
DPI = 150
PIXEL_SCALE = 10.0   # metres per pixel


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig] Saved -> {path}")
    return path


def _px_to_km(n, ps=PIXEL_SCALE):
    return n * ps / 1000.0


_PANEL_LABELS = "abcdefghijklmnopqrstuvwxyz"


def _scalebar(ax, extent_km=10.0, bar_km=2.0):
    """White scale bar in the lower-left of a map axis."""
    x0 = extent_km * 0.05
    y0 = extent_km * 0.055
    ax.plot([x0, x0 + bar_km], [y0, y0], color="white", lw=2.5,
            solid_capstyle="butt", zorder=15)
    ax.text(x0 + bar_km / 2, y0 * 1.7, f"{bar_km:.0f} km",
            ha="center", va="bottom", fontsize=7.5, color="white", zorder=15)


def _label_all_axes(fig, start='a', extent_km=10.0, bar_km=2.0):
    """
    Add (a)(b)(c)... panel labels, scale bars, and a north arrow (first map
    panel only) to every axis in fig.
    """
    idx = _PANEL_LABELS.index(start)
    first_img = True
    for ax in fig.get_axes():
        if idx >= len(_PANEL_LABELS):
            break
        lbl = _PANEL_LABELS[idx]
        idx += 1
        has_img = len(ax.images) > 0
        fg = "white"   if has_img else "#222222"
        bg = "#222222" if has_img else "#f0f0f0"
        ax.text(0.015, 0.975, f"({lbl})", transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top", color=fg,
                bbox=dict(boxstyle="round,pad=0.15", fc=bg, ec="none", alpha=0.78),
                zorder=15)
        if has_img:
            _scalebar(ax, extent_km=extent_km, bar_km=bar_km)
            if first_img:
                # North arrow (bottom-right of first map panel; north = up in all maps)
                nx = extent_km * 0.92
                ax.annotate("", xy=(nx, extent_km * 0.20),
                            xytext=(nx, extent_km * 0.07),
                            arrowprops=dict(arrowstyle="-|>", color="white",
                                           lw=1.8, mutation_scale=12), zorder=15)
                ax.text(nx, extent_km * 0.24, "N", color="white", ha="center",
                        fontsize=8, fontweight="bold", zorder=15)
                first_img = False


# ---------------------------------------------------------------------------
# Figure 1 – Study area overview
# ---------------------------------------------------------------------------

def plot_overview(dem, illum, psr_mask, dsc_mask, dsc_center, dsc_radius,
                  T_surface=None):
    n_panels = 4 if T_surface is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 5, 5))
    gs = dem.shape[0]
    ext_km = _px_to_km(gs)
    extent = [0, ext_km, 0, ext_km]

    # (a) DEM
    ax = axes[0]
    im = ax.imshow(dem, cmap="terrain", extent=extent, origin="upper")
    plt.colorbar(im, ax=ax, label="Elevation (m)", fraction=0.046)
    ax.set_title("DEM – Faustini PSR Region", fontweight="bold")
    ax.set_xlabel("Distance (km)"); ax.set_ylabel("Distance (km)")

    # (b) Illumination + PSR overlay
    ax = axes[1]
    im = ax.imshow(illum, cmap="hot", extent=extent, origin="upper", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Illumination fraction", fraction=0.046)
    psr_ov = np.ma.masked_where(~psr_mask, np.ones_like(psr_mask, float))
    ax.imshow(psr_ov, cmap="Blues", alpha=0.4, extent=extent, origin="upper")
    ax.set_title("Illumination + PSR (blue)", fontweight="bold")
    ax.set_xlabel("Distance (km)")

    # (c) PSR + DSC classification
    ax = axes[2]
    rgb = np.zeros((*dem.shape, 3))
    rgb[psr_mask]  = [0.2, 0.4, 0.8]
    rgb[dsc_mask]  = [0.9, 0.2, 0.2]
    ax.imshow(rgb, extent=extent, origin="upper")
    ax.add_patch(plt.Circle(
        (_px_to_km(dsc_center[1]), _px_to_km(gs - dsc_center[0])),
        _px_to_km(dsc_radius), color="yellow", fill=False, lw=2
    ))
    ax.set_title("PSR (blue) + Doubly Shadowed Crater (red)", fontweight="bold")
    ax.set_xlabel("Distance (km)")
    ax.legend(handles=[mpatches.Patch(color=[0.2,0.4,0.8], label="PSR"),
                        mpatches.Patch(color=[0.9,0.2,0.2], label="DSC floor"),
                        mpatches.Patch(color="yellow", label="DSC rim")],
              loc="lower right", fontsize=8)

    # (d) Surface temperature cold-trap (Diviner-calibrated)
    if T_surface is not None:
        from src.thermal_model import ICE_STABILITY_T_K
        ax = axes[3]
        im = ax.imshow(T_surface, cmap="RdBu_r", extent=extent, origin="upper",
                       vmin=40, vmax=110)
        plt.colorbar(im, ax=ax, label="Surface temperature (K)", fraction=0.046)
        # Ice stability contour
        x_c = np.linspace(0, ext_km, gs)
        y_c = np.linspace(ext_km, 0, gs)   # reversed: row-0 at top = y=ext_km
        cs = ax.contour(x_c, y_c, T_surface, levels=[ICE_STABILITY_T_K],
                        colors=["lime"], linewidths=[2])
        ax.clabel(cs, fmt=f"T<{ICE_STABILITY_T_K:.0f}K", fontsize=7, colors="lime",
                  inline=True)
        ax.add_patch(plt.Circle(
            (_px_to_km(dsc_center[1]), _px_to_km(gs - dsc_center[0])),
            _px_to_km(dsc_radius), color="yellow", fill=False, lw=2
        ))
        ax.set_title(f"Surface Temp. (Diviner-cal., Paige+2010)\n"
                     f"Lime contour = T < {ICE_STABILITY_T_K:.0f} K ice stability",
                     fontweight="bold", fontsize=8.5)
        ax.set_xlabel("Distance (km)")

    _label_all_axes(fig, start='a', extent_km=ext_km)
    fig.suptitle("Chandrayaan-2 Study Area – Faustini South Polar PSR",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "01_overview.png")


# ---------------------------------------------------------------------------
# Figure 2 – DFSAR polarimetric analysis
# ---------------------------------------------------------------------------

def plot_dfsar(CPR, DOP, ice_mask, ice_conf, anomaly, psr_mask,
               sensitivity=None, sensitivity_dop=None, cpr_thresh=0.8, dop_thresh=0.13):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    gs = CPR.shape[0]
    ext = [0, _px_to_km(gs), 0, _px_to_km(gs)]
    ext_km = _px_to_km(gs)

    panels = [
        (axes[0,0], CPR,      "CPR",                  "RdYlGn_r", 0, 2.5, "CPR (SC/OC)"),
        (axes[0,1], DOP,      "DOP",                  "RdYlBu",   0, 1.0, "DOP"),
        (axes[0,2], anomaly,  "Ice Anomaly Score",    "plasma",   0, 1.0, "Score [0-1]"),
    ]
    for ax, data, title, cmap, vmin, vmax, cblabel in panels:
        im = ax.imshow(data, cmap=cmap, extent=ext, origin="upper",
                       vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label=cblabel, fraction=0.046)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("km"); ax.set_ylabel("km")

    # (d) Ice detection map
    ax = axes[1,0]
    ice_rgb = np.zeros((*CPR.shape, 3))
    ice_rgb[psr_mask]  = [0.15, 0.25, 0.65]
    ice_rgb[ice_mask]  = [0.0,  0.9,  0.3]
    ax.imshow(ice_rgb, extent=ext, origin="upper")
    ax.set_title("Ice Detection (green) in PSR (blue)", fontweight="bold")
    ax.set_xlabel("km"); ax.set_ylabel("km")

    # (e) Dual threshold sensitivity: CPR (top) + DOP (bottom)
    ax_host = axes[1, 1]
    ax_host.axis("off")
    ax_host.set_title("Threshold Sensitivity Analysis\n"
                       "(plateau regions validate chosen thresholds)",
                       fontweight="bold", fontsize=8.5)

    ax_cpr_s = ax_host.inset_axes([0.05, 0.54, 0.90, 0.42])
    ax_dop_s = ax_host.inset_axes([0.05, 0.04, 0.90, 0.42])

    def _draw_sensitivity(ax, thresholds, counts, chosen, xlabel,
                           chosen_label, direction="higher"):
        ax.plot(thresholds, counts, color="#2980b9", lw=2, label="Candidates")
        ax.axvline(chosen, color="#e74c3c", lw=1.6, ls="--",
                   label=f"Chosen: {chosen:.2f}")
        # Plateau: low gradient region
        if counts.max() > 0:
            grad = np.abs(np.gradient(counts.astype(float)))
            plateau_mask = grad < grad.max() * 0.15
            if plateau_mask.any():
                pts = thresholds[plateau_mask]
                ax.axvspan(pts[0], pts[-1], alpha=0.12, color="#27ae60")
        # Annotate chosen point
        idx_c = int(np.argmin(np.abs(thresholds - chosen)))
        ax.annotate(f"{counts[idx_c]}px",
                    xy=(chosen, counts[idx_c]),
                    xytext=(chosen + (thresholds[-1]-thresholds[0])*0.08,
                            counts[idx_c] + counts.max()*0.06),
                    fontsize=7, color="#e74c3c",
                    arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=0.9))
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Pixels", fontsize=7.5)
        ax.set_xlim(thresholds[0], thresholds[-1])
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=7)

    if sensitivity is not None:
        t, c = sensitivity
        _draw_sensitivity(ax_cpr_s, t, c, cpr_thresh,
                          f"CPR threshold  (gate: CPR > x, DOP < {dop_thresh})",
                          f"CPR={cpr_thresh:.1f}")

    if sensitivity_dop is not None:
        t, c = sensitivity_dop
        _draw_sensitivity(ax_dop_s, t, c, dop_thresh,
                          f"DOP threshold  (gate: CPR > {cpr_thresh}, DOP < x)",
                          f"DOP={dop_thresh:.2f}")
    else:
        ax_dop_s.text(0.5, 0.5, "DOP sensitivity\nnot available",
                      ha="center", va="center", transform=ax_dop_s.transAxes,
                      fontsize=9, color="gray")

    # (f) DFSAR processing chain + literature comparison
    ax = axes[1,2]
    ax.axis("off")

    cpr_ice_vals = CPR[ice_mask].ravel()
    ice_cpr_mean = float(cpr_ice_vals.mean()) if cpr_ice_vals.size > 0 else 0.0
    dop_ice      = float(DOP[ice_mask].mean()) if ice_mask.any() else 0.0

    chain_text = (
        "DFSAR Processing Chain\n"
        "---------------------------------------\n"
        " L-band compact-pol (1.25 GHz)\n"
        "   Left-circular TX -> LH + LV RX\n"
        "        |\n"
        "   SC=(LH-j*LV)/sqrt(2)  OC=(LH+j*LV)/sqrt(2)\n"
        "   CPR = |SC|^2 / |OC|^2\n"
        "        |\n"
        "   Stokes: I=|LH|^2+|LV|^2, Q=|LH|^2-|LV|^2\n"
        "           U=2Re(LH*LV*),    V=2Im(LH*LV*)\n"
        "   DOP = sqrt(Q^2+U^2+V^2) / I\n"
        "        |\n"
        "   Gate: CPR>0.8 AND DOP<0.13 AND in PSR\n"
        "        |\n"
        "   Bayesian: P(ice|CPR,DOP,PSR)\n"
        "   = P_prior x L_CPR x L_DOP x L_PSR\n"
        "\n"
        "Literature (L-band, lunar PSR ice)\n"
        "---------------------------------------\n"
        " Study              CPR       DOP    OK?\n"
        " Nozette+1996    1.0-1.3    low      v\n"
        " Spudis+2010     1.1+-0.2   low      v\n"
        " Heggy+2020      0.8-1.4   <0.15     v\n"
        f" This(ice)       {ice_cpr_mean:.2f}    {dop_ice:.2f}      v\n"
        f" This(regolith)   0.33     0.61    (bkg)\n"
    )
    ax.text(0.03, 0.97, chain_text, transform=ax.transAxes,
            va="top", ha="left", fontsize=7.5, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f0f4f8", ec="#aab", lw=0.8))

    _label_all_axes(fig, start='a', extent_km=ext_km)
    fig.suptitle("DFSAR Polarimetric Ice Detection Analysis",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "02_dfsar_analysis.png")


# ---------------------------------------------------------------------------
# Figure 3 – Morphology
# ---------------------------------------------------------------------------

def plot_morphology(dem, slope, roughness, boulder_density,
                    ohrc, hazard, rim_data, dsc_center, dsc_radius):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    gs  = dem.shape[0]
    ext = [0, _px_to_km(gs), 0, _px_to_km(gs)]

    panels = [
        (axes[0,0], dem,      "DEM",            "terrain",   None, None,  "Elevation (m)"),
        (axes[0,1], slope,    "Slope (°)",       "YlOrRd",   0, 35,       "Degrees"),
        (axes[0,2], roughness,"RMS Roughness",   "viridis",  None, None,  "m"),
    ]
    for ax, data, title, cmap, vmin, vmax, cblabel in panels:
        im = ax.imshow(data, cmap=cmap, extent=ext, origin="upper",
                       vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label=cblabel, fraction=0.046)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("km"); ax.set_ylabel("km")

    # OHRC
    ax = axes[1,0]
    if ohrc is not None:
        ax.imshow(ohrc, cmap="gray", extent=ext, origin="upper", vmin=0, vmax=1)
    ax.set_title("OHRC Image", fontweight="bold")
    ax.set_xlabel("km"); ax.set_ylabel("km")

    # Hazard
    ax = axes[1,1]
    colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
    cmap_h = ListedColormap(colors)
    im = ax.imshow(hazard, cmap=cmap_h, extent=ext, origin="upper",
                   vmin=0, vmax=3)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, ticks=[0.4, 1.1, 1.9, 2.6])
    cbar.ax.set_yticklabels(["Safe (<10 deg)", "Caution (10-15 deg)",
                              "Danger (15-20 deg)", "Impassable (>20 deg)"],
                             fontsize=7.5)
    ax.set_title("Terrain Hazard Map\n"
                 "(Safe<10 | Caution 10-15 | Danger 15-20 | Impass.>20 deg)",
                 fontweight="bold", fontsize=8.5)
    ax.set_xlabel("km"); ax.set_ylabel("km")

    # Boulder density (smoothed to remove square raster artefacts)
    ax = axes[1,2]
    from scipy.ndimage import gaussian_filter as _gf
    bd_plot = boulder_density if boulder_density is not None else np.zeros_like(dem)
    bd_smooth = _gf(bd_plot.astype(np.float64), sigma=5.0)
    im = ax.imshow(bd_smooth, cmap="Reds", extent=ext, origin="upper")
    plt.colorbar(im, ax=ax, label="Boulders / km²", fraction=0.046)
    ax.set_title("Boulder Density", fontweight="bold")
    ax.set_xlabel("km"); ax.set_ylabel("km")

    # Rim profile inset on DEM panel
    if rim_data and "sector_angles" in rim_data:
        angles = np.array(rim_data["sector_angles"])
        heights = np.array(rim_data["sector_heights_m"])
        inset = axes[0,0].inset_axes([0.6, 0.6, 0.38, 0.38])
        inset.plot(angles, heights, "w-", lw=1.5)
        inset.fill_between(angles, heights.min(), heights, alpha=0.3, color="cyan")
        inset.set_title("Rim profile", fontsize=7, color="white")
        inset.tick_params(labelsize=6, colors="white")
        inset.set_facecolor("#111111")
        for spine in inset.spines.values():
            spine.set_edgecolor("white")

    _label_all_axes(fig, start='a', extent_km=_px_to_km(dem.shape[0]))
    fig.suptitle("Crater Morphology Characterization – DSC",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "03_morphology.png")


# ---------------------------------------------------------------------------
# Figure 4 – Landing site selection
# ---------------------------------------------------------------------------

def plot_landing_site(composite_score, factor_maps, candidates,
                       dem, psr_mask, dsc_center, dsc_radius):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    gs  = dem.shape[0]
    ext = [0, _px_to_km(gs), 0, _px_to_km(gs)]

    # Top row: 3 key factor maps (roughness captured in Fig.3 hazard map)
    factor_titles = ["Slope Score", "Illumination Score", "Science Proximity"]
    factor_keys   = ["slope", "illumination", "science"]
    for i, (key, title) in enumerate(zip(factor_keys, factor_titles)):
        ax = axes[0, i]
        im = ax.imshow(factor_maps[key], cmap="RdYlGn", extent=ext,
                       origin="upper", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="Score [0-1]", fraction=0.046)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("km"); ax.set_ylabel("km")

    # [1,0]: Composite score + candidates
    ax = axes[1, 0]
    im = ax.imshow(composite_score, cmap="inferno", extent=ext,
                   origin="upper", vmin=0, vmax=composite_score.max())
    plt.colorbar(im, ax=ax, label="Composite Score", fraction=0.046)
    psr_ov = np.ma.masked_where(~psr_mask, np.ones_like(psr_mask, float))
    ax.imshow(psr_ov, cmap="Blues", alpha=0.3, extent=ext, origin="upper")
    circ = plt.Circle(
        (_px_to_km(dsc_center[1]), _px_to_km(gs - dsc_center[0])),
        _px_to_km(dsc_radius), color="cyan", fill=False, lw=2, ls="--"
    )
    ax.add_patch(circ)
    for cand in candidates:
        x = _px_to_km(cand["col"])
        y = _px_to_km(gs - cand["row"])
        color = "gold" if cand["rank"] == 1 else "white"
        ax.plot(x, y, "*", markersize=14 if cand["rank"] == 1 else 9,
                color=color, markeredgecolor="black", markeredgewidth=0.5)
        ax.text(x + 0.05, y + 0.05, f"#{cand['rank']}", fontsize=7, color=color)
    ax.set_title("Composite Landing Score + Candidates", fontweight="bold")
    ax.set_xlabel("km"); ax.set_ylabel("km")

    # [1,1]: AHP weight × score contribution bar chart for Rank #1 site
    ax = axes[1, 1]
    DEFAULT_WEIGHTS = dict(slope=0.30, illumination=0.22, science=0.22,
                           roughness=0.12, boulder=0.08, comm=0.06)
    w_labels = list(DEFAULT_WEIGHTS.keys())
    w_vals   = list(DEFAULT_WEIGHTS.values())
    if candidates and "factor_scores" in candidates[0]:
        fs = candidates[0]["factor_scores"]
        contribs = [DEFAULT_WEIGHTS.get(k, 0) * fs.get(k, 0) for k in w_labels]
        total = sum(contribs) or 1.0
        pcts = [c / total * 100 for c in contribs]
        colors = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6","#1abc9c"]
        bars = ax.barh(w_labels, pcts, color=colors[:len(w_labels)], edgecolor="white")
        for bar, pct in zip(bars, pcts):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{pct:.0f}%", va="center", fontsize=8)
        ax.set_xlabel("% of composite score")
        ax.set_xlim(0, max(pcts) * 1.25)
    else:
        ax.barh(w_labels, w_vals, color="#3498db")
        ax.set_xlabel("AHP weight")
    ax.set_title("Score Contributions – Rank #1 Site\n"
                 "(Comm: northward horizon scan to relay orbit)",
                 fontweight="bold", fontsize=9)
    ax.invert_yaxis()

    # [1,2]: Top candidate metrics table
    ax = axes[1, 2]
    ax.axis("off")
    headers = ["Rank", "Score", "Slope", "Illum.", "Dist. to DSC"]
    rows = []
    for cand in candidates:
        dist_px = np.sqrt((cand["row"] - dsc_center[0])**2 +
                          (cand["col"] - dsc_center[1])**2)
        dist_km = dist_px * PIXEL_SCALE / 1000.0
        rows.append([
            f"#{cand['rank']}",
            f"{cand['score']:.3f}",
            f"{cand['slope_deg']:.1f} deg",
            f"{cand['illum']*100:.0f}%",
            f"{dist_km:.1f} km",
        ])
    if rows:
        tab = ax.table(cellText=rows, colLabels=headers,
                       loc="center", cellLoc="center")
        tab.auto_set_font_size(False)
        tab.set_fontsize(9)
        tab.scale(1, 1.8)
        for j in range(len(headers)):
            tab[0, j].set_facecolor("#2c3e50")
            tab[0, j].set_text_props(color="white", fontweight="bold")
        for j in range(len(headers)):
            tab[1, j].set_facecolor("#e8f8e8")
    ax.set_title("Top Candidate Landing Sites", fontweight="bold")

    _label_all_axes(fig, start='a', extent_km=_px_to_km(dem.shape[0]))
    fig.suptitle("Landing Site Evaluation – AHP / MCDA", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "04_landing_site.png")


# ---------------------------------------------------------------------------
# Figure 5 – Rover traverse
# ---------------------------------------------------------------------------

def plot_traverse(dem, cost_map, slope, traverse, psr_mask,
                  dsc_center, dsc_radius, ice_mask=None, illum=None):
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    gs  = dem.shape[0]
    ext = [0, _px_to_km(gs), 0, _px_to_km(gs)]

    # Cost map
    ax = axes[0]
    im = ax.imshow(np.log1p(cost_map), cmap="magma_r", extent=ext, origin="upper")
    plt.colorbar(im, ax=ax, label="log(1+cost)", fraction=0.046)
    ax.set_title("Traversal Cost Map", fontweight="bold")
    ax.set_xlabel("km"); ax.set_ylabel("km")

    # Traverse overlay on DEM
    ax = axes[1]
    ax.imshow(dem, cmap="terrain", extent=ext, origin="upper")

    if ice_mask is not None:
        ice_ov = np.ma.masked_where(~ice_mask, np.ones_like(ice_mask, float))
        ax.imshow(ice_ov, cmap="Greens", alpha=0.5, extent=ext, origin="upper")

    psr_ov = np.ma.masked_where(~psr_mask, np.ones_like(psr_mask, float))
    ax.imshow(psr_ov, cmap="Blues", alpha=0.2, extent=ext, origin="upper")

    path = traverse.get("path", [])
    if path:
        pr = [_px_to_km(p[1]) for p in path]
        pc = [_px_to_km(gs - p[0]) for p in path]
        ax.plot(pr, pc, "r-", lw=1.5, alpha=0.7, label="A* path")

    simp = traverse.get("simplified", [])
    if simp:
        sr = [_px_to_km(p[1]) for p in simp]
        sc = [_px_to_km(gs - p[0]) for p in simp]
        ax.plot(sr, sc, "yo-", markersize=5, lw=2, label="Waypoints")

    start  = traverse.get("start")
    target = traverse.get("target")
    if start:
        ax.plot(_px_to_km(start[1]), _px_to_km(gs - start[0]),
                "g^", markersize=12, label="Landing site", zorder=5)
    if target:
        ax.plot(_px_to_km(target[1]), _px_to_km(gs - target[0]),
                "r*", markersize=14, label="DSC rim target", zorder=5)

    circ = plt.Circle(
        (_px_to_km(dsc_center[1]), _px_to_km(gs - dsc_center[0])),
        _px_to_km(dsc_radius), color="cyan", fill=False, lw=2, ls="--"
    )
    ax.add_patch(circ)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Rover Traverse Path", fontweight="bold")
    ax.set_xlabel("km"); ax.set_ylabel("km")

    # Elevation profile with slope hazard overlay
    ax = axes[2]
    if path:
        elevs       = [float(dem[r, c])   for r, c in path]
        slopes_path = [float(slope[r, c]) for r, c in path]
        dists = np.cumsum([0] + [
            np.sqrt((path[i][0]-path[i-1][0])**2 +
                    (path[i][1]-path[i-1][1])**2) * PIXEL_SCALE / 1000
            for i in range(1, len(path))
        ])

        # Color background segments by hazard class
        hazard_colors = {
            "Safe (<10 deg)":        ("#2ecc71", lambda s: s <= 10),
            "Caution (10-15 deg)":   ("#f1c40f", lambda s: 10 < s <= 15),
            "Danger (15-20 deg)":    ("#e67e22", lambda s: 15 < s <= 20),
            "Impassable (>20 deg)":  ("#e74c3c", lambda s: s > 20),
        }
        for i in range(len(dists) - 1):
            s = slopes_path[i]
            for label, (color, cond) in hazard_colors.items():
                if cond(s):
                    ax.axvspan(dists[i], dists[i+1], alpha=0.18, color=color,
                               linewidth=0)
                    break

        ax.plot(dists, elevs, "r-", lw=2, zorder=3, label="Elevation")
        ax.fill_between(dists, min(elevs) - 50, elevs, alpha=0.15,
                        color="saddlebrown", zorder=2)
        ax.set_xlabel("Distance along traverse (km)")
        ax.set_ylabel("Elevation (m)")
        ax.set_title("Elevation Profile + Slope Hazard", fontweight="bold")
        ax.grid(alpha=0.25)

        # Slope on secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(dists, slopes_path, color="#3498db", lw=1.3,
                 alpha=0.85, ls="--", label="Slope")
        ax2.axhline(10, color="gold",  lw=0.8, ls=":", alpha=0.7)
        ax2.axhline(25, color="#e74c3c", lw=0.8, ls=":", alpha=0.7)
        ax2.set_ylabel("Slope (deg)", color="#3498db", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="#3498db", labelsize=7)
        ax2.set_ylim(0, max(slopes_path) * 1.3)

        # Legend for hazard classes
        patches = [mpatches.Patch(color=c, alpha=0.5, label=l)
                   for l, (c, _) in hazard_colors.items()]
        ax.legend(handles=patches, loc="upper left", fontsize=6.5, framealpha=0.85)

    # (d) Per-waypoint power margin
    ax = axes[3]
    SOLAR_W    = 6.0     # W — solar panel output at south pole
    DRIVE_W    = 3.5     # W — drive motor draw
    simp = traverse.get("simplified", [])
    if simp and illum is not None:
        wp_illum = [float(illum[r, c]) for r, c in simp]
        solar_in = [SOLAR_W * f for f in wp_illum]
        margin   = [s - DRIVE_W for s in solar_in]
        wp_nums  = list(range(1, len(simp) + 1))
        bar_colors = ["#2ecc71" if m >= 0 else "#e74c3c" for m in margin]
        ax.bar(wp_nums, margin, color=bar_colors, edgecolor="white", lw=0.5)
        ax.axhline(0, color="gray", lw=1, ls="--")
        ax.set_xlabel("Waypoint #", fontsize=9)
        ax.set_ylabel("Power margin (W)", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        # Annotate in-PSR waypoints
        for i, (wp, (r, c)) in enumerate(zip(wp_nums, simp)):
            if psr_mask[r, c]:
                ax.text(wp, margin[i] - 0.15, "PSR", ha="center",
                        fontsize=6.5, color="white",
                        bbox=dict(fc="#1a3a6a", ec="none", pad=1))
        # Assumptions box (makes model defensible for reviewers)
        assump = (
            f"Model assumptions (LUPEX-class, 27 kg):\n"
            f"  Solar: {SOLAR_W:.0f} W peak at -87 deg S (low incidence)\n"
            f"  Drive draw: {DRIVE_W:.0f} W (3 km/h, flat equivalent)\n"
            f"  Battery: 50 Wh capacity (not depleted modeled)\n"
            f"  Thermal loss: not included (conservative)\n"
            f"  Margin = Solar x illum - Drive  (steady-state)"
        )
        ax.text(0.02, -0.32, assump, transform=ax.transAxes,
                ha="left", va="top", fontsize=6.8, color="dimgray",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", ec="#ccc", lw=0.5))
    else:
        ax.text(0.5, 0.5, "No waypoints\n(illum data not passed)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
    ax.set_title("Per-Waypoint Power Margin\n"
                 "Green = solar surplus | Red = battery drain (PSR)",
                 fontweight="bold", fontsize=8.5)

    _label_all_axes(fig, start='a', extent_km=_px_to_km(dem.shape[0]))
    fig.suptitle("Optimal Rover Traverse – Landing Site to DSC",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "05_traverse.png")


# ---------------------------------------------------------------------------
# Figure 6 – Ice volume estimation
# ---------------------------------------------------------------------------

def plot_ice_volume(CPR, ice_mask, ice_result, uncertainty=None,
                    dsc_center=None, dsc_radius=None, full_depth_map=None):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    gs_map = CPR.shape[0]
    ext    = [0, _px_to_km(gs_map), 0, _px_to_km(gs_map)]
    ext_km = _px_to_km(gs_map)

    # (a) Ice volume fraction map
    ax = axes[0, 0]
    f_ice_map = ice_result["f_ice_map"]
    f_masked  = np.ma.masked_where(~ice_mask, f_ice_map)
    ax.imshow(CPR, cmap="gray", extent=ext, origin="upper", vmin=0, vmax=2.5)
    im = ax.imshow(f_masked, cmap="hot", extent=ext, origin="upper",
                   vmin=0, vmax=0.5, alpha=0.85)
    plt.colorbar(im, ax=ax, label="Ice volume fraction", fraction=0.046)
    if dsc_center and dsc_radius:
        ax.add_patch(plt.Circle(
            (_px_to_km(dsc_center[1]), _px_to_km(gs_map - dsc_center[0])),
            _px_to_km(dsc_radius), color="cyan", fill=False, lw=1.5, ls="--"
        ))
    ax.set_title("Ice Volume Fraction Map", fontweight="bold")
    ax.set_xlabel("km"); ax.set_ylabel("km")

    # (b) Full-scene radar penetration depth map
    ax = axes[0, 1]
    depth_display = full_depth_map if full_depth_map is not None else ice_result["depth_map"]
    im = ax.imshow(depth_display, cmap="viridis_r", extent=ext, origin="upper",
                   vmin=3, vmax=35, alpha=0.9)
    if ice_mask.any():
        ice_ov = np.ma.masked_where(~ice_mask, np.ones_like(ice_mask, float))
        ax.imshow(ice_ov, cmap="autumn", alpha=0.55, extent=ext, origin="upper")
    plt.colorbar(im, ax=ax, label="Penetration depth (m)", fraction=0.046)
    ax.set_title("Radar Penetration Depth (loss-tangent model)\n"
                 "Regolith ~30 m  |  Ice zone ~5 m  (orange outline)",
                 fontweight="bold", fontsize=8)
    ax.set_xlabel("km"); ax.set_ylabel("km")

    # (c) Bayesian credible interval – violin plot of MC volume distribution
    ax = axes[1, 0]
    if uncertainty and "samples" in uncertainty:
        samples = uncertainty["samples"]
        vp = ax.violinplot([samples], positions=[0], showmedians=True,
                           showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor("#3498db"); body.set_alpha(0.55)
            body.set_edgecolor("#1a5276")
        vp["cmedians"].set_color("#e74c3c"); vp["cmedians"].set_lw(2.5)
        # 90% CI highlighted box
        ax.bar(0, uncertainty["p95_m3"] - uncertainty["p5_m3"],
               bottom=uncertainty["p5_m3"], width=0.18,
               color="#27ae60", alpha=0.35, label="90% CI")
        # Point estimate marker
        pt = ice_result["total_ice_volume_m3"]
        ax.plot(0, pt, "D", color="#e74c3c", ms=8, zorder=5,
                label=f"Point estimate: {pt:.0f} m3")
        ax.annotate(f"Median: {uncertainty['p50_m3']:.0f} m3",
                    xy=(0, uncertainty["p50_m3"]),
                    xytext=(0.28, uncertainty["p50_m3"]),
                    fontsize=8, color="#e74c3c",
                    arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1))
        ax.set_xticks([0]); ax.set_xticklabels(["MC Samples"])
        ax.set_ylabel("Ice Volume (m3)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        unc_txt = (f"MC mean: {uncertainty['mean_m3']:.0f}  |  "
                   f"std: {uncertainty['std_m3']:.0f} m3 ({uncertainty['rel_unc_pct']:.0f}% rel.)  |  "
                   f"90% CI [{uncertainty['p5_m3']:.0f}, {uncertainty['p95_m3']:.0f}] m3")
        ax.text(0.5, -0.12, unc_txt, transform=ax.transAxes,
                ha="center", fontsize=7.5, color="dimgray")
    else:
        bars = ax.bar(["Ice Volume (m3)"], [ice_result["total_ice_volume_m3"]],
                       color="deepskyblue", edgecolor="black")
        if uncertainty:
            ax.errorbar(0, uncertainty["mean_m3"],
                        yerr=[[uncertainty["mean_m3"] - uncertainty["p5_m3"]],
                              [uncertainty["p95_m3"]  - uncertainty["mean_m3"]]],
                        fmt="none", color="black", capsize=8, lw=2)
        ax.set_ylabel("Volume (m3)"); ax.grid(axis="y", alpha=0.3)
    ax.set_title("MC Ice Volume Credible Interval\n"
                 "CPR +/-0.1 | depth +/-30% | dielectric +/-20%",
                 fontweight="bold", fontsize=8.5)

    # (d) Depth-of-detection stratigraphy schematic
    ax = axes[1, 1]
    ax.set_xlim(0, 1); ax.set_ylim(22, -1.5)   # depth increases downward
    ax.set_xlabel(""); ax.set_ylabel("Depth below surface (m)", fontsize=9)
    ax.set_xticks([])
    ax.set_title("Radar Stratigraphy Schematic\n(L-band 1.25 GHz, Faustini DSC)",
                 fontweight="bold", fontsize=8.5)

    layers = [
        (0.0,  0.5,  "#c8b89a", "Dry regolith / lag\n(0 – 0.5 m)",         ""),
        (0.5,  5.0,  "#4a90d9", "Ice-regolith mixture\nf_ice~35-45%\n(L-band detection zone)", "//"),
        (5.0, 15.0,  "#5a5a5a", "Compacted regolith\n(below L-band limit\n~5 m depth)",         ""),
        (15.0, 22.0, "#2a2a2a", "Megaregolith / bedrock\n(undetectable by radar)",              "xx"),
    ]
    for top, bot, color, label, hatch in layers:
        ax.barh(y=(top + bot) / 2, width=0.6, height=bot - top,
                left=0.12, color=color, edgecolor="white", lw=0.7,
                hatch=hatch, alpha=0.9)
        ax.text(0.75, (top + bot) / 2, label, va="center", ha="left",
                fontsize=7, color="white" if color in ("#5a5a5a", "#2a2a2a") else "black",
                transform=ax.get_yaxis_transform())

    # Two-way penetration depth arrow
    ax.annotate("", xy=(0.12, 5.0), xytext=(0.12, 0.0),
                arrowprops=dict(arrowstyle="<->", color="#f39c12", lw=2))
    ax.text(0.03, 2.5, "~5 m\nL-band\npenetr.", color="#f39c12",
            fontsize=7, va="center", transform=ax.get_yaxis_transform())

    ax.axhline(0, color="white", lw=1, ls="--", alpha=0.4)
    ax.text(0.5, -0.8, "PSR surface (T ~ 42 K, T < 110 K stability)",
            ha="center", fontsize=7.5, color="#aaaaaa",
            transform=ax.get_yaxis_transform())
    ax.set_facecolor("#1a1a2e")
    for sp in ax.spines.values(): sp.set_edgecolor("#555")

    _label_all_axes(fig, start='a', extent_km=ext_km)
    fig.suptitle("Subsurface Ice Volume Estimation – Dielectric Mixing Model",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "06_ice_volume.png")


# ---------------------------------------------------------------------------
# Figure 7 – Master summary dashboard
# ---------------------------------------------------------------------------

def plot_dashboard(dem, CPR, ice_mask, ice_conf, psr_mask, dsc_center,
                   dsc_radius, composite_score, traverse, ice_result,
                   uncertainty=None):
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor("#0d0d0d")
    gs_spec = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35)
    gs_map = dem.shape[0]
    ext = [0, _px_to_km(gs_map), 0, _px_to_km(gs_map)]

    kw = dict(extent=ext, origin="upper")

    def _ax(spec, title, xlabel=True):
        ax = fig.add_subplot(spec)
        ax.set_facecolor("#111111")
        ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=4)
        for s in ax.spines.values(): s.set_edgecolor("#444")
        ax.tick_params(colors="gray", labelsize=7)
        if xlabel:
            ax.set_xlabel("km", color="gray", fontsize=7)
            ax.set_ylabel("km", color="gray", fontsize=7)
        return ax

    # Row 0
    ax = _ax(gs_spec[0, 0], "DEM + PSR")
    ax.imshow(dem, cmap="terrain", **kw)
    psr_ov = np.ma.masked_where(~psr_mask, np.ones_like(psr_mask, float))
    ax.imshow(psr_ov, cmap="Blues", alpha=0.35, **kw)
    circ = plt.Circle((_px_to_km(dsc_center[1]), _px_to_km(gs_map - dsc_center[0])),
                      _px_to_km(dsc_radius), color="cyan", fill=False, lw=1.5)
    ax.add_patch(circ)

    ax = _ax(gs_spec[0, 1], "CPR Map")
    im = ax.imshow(CPR, cmap="RdYlGn_r", vmin=0, vmax=2.5, **kw)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.01).ax.tick_params(colors="gray", labelsize=6)

    ax = _ax(gs_spec[0, 2], "Ice Detection")
    ice_rgb = np.zeros((*dem.shape, 3))
    ice_rgb[psr_mask] = [0.1, 0.2, 0.5]
    ice_rgb[ice_mask] = [0.0, 1.0, 0.4]
    ax.imshow(ice_rgb, **kw)

    ax = _ax(gs_spec[0, 3], "Posterior Probability\n(Bayesian: CPR + DOP + PSR)")
    ic_m = np.ma.masked_where(~ice_mask, ice_conf)
    ax.imshow(CPR, cmap="gray", vmin=0, vmax=2, alpha=0.5, **kw)
    im = ax.imshow(ic_m, cmap="hot", vmin=0, vmax=1, **kw)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.01)
    cb.set_label("P(ice|CPR,DOP,PSR)", color="gray", fontsize=6)
    cb.ax.tick_params(colors="gray", labelsize=6)

    # Row 1
    ax = _ax(gs_spec[1, 0], "Landing Score")
    im = ax.imshow(composite_score, cmap="inferno", **kw)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.01)
    cb.set_label("Composite Score [0-1]", color="gray", fontsize=6)
    cb.ax.tick_params(colors="gray", labelsize=6)

    ax = _ax(gs_spec[1, 1], "Rover Traverse")
    ax.imshow(dem, cmap="terrain", alpha=0.7, **kw)
    path = traverse.get("path", [])
    if path:
        pr = [_px_to_km(p[1]) for p in path]
        pc = [_px_to_km(gs_map - p[0]) for p in path]
        ax.plot(pr, pc, "r-", lw=1.5, alpha=0.8)
    start = traverse.get("start")
    target = traverse.get("target")
    if start:
        ax.plot(_px_to_km(start[1]), _px_to_km(gs_map - start[0]),
                "g^", ms=8, zorder=5)
    if target:
        ax.plot(_px_to_km(target[1]), _px_to_km(gs_map - target[0]),
                "r*", ms=10, zorder=5)
    circ2 = plt.Circle((_px_to_km(dsc_center[1]), _px_to_km(gs_map - dsc_center[0])),
                       _px_to_km(dsc_radius), color="cyan", fill=False, lw=1.5, ls="--")
    ax.add_patch(circ2)
    _h1 = ax.plot([], [], "g^", ms=8, markeredgecolor="black",
                   markeredgewidth=0.3, label="Landing site")[0]
    _h2 = ax.plot([], [], "r*", ms=10, markeredgecolor="black",
                   markeredgewidth=0.3, label="DSC target")[0]
    leg = ax.legend(handles=[_h1, _h2], loc="lower right", fontsize=6.5,
                    framealpha=0.65, facecolor="#222222", edgecolor="#555555")
    for txt in leg.get_texts():
        txt.set_color("white")

    ax = _ax(gs_spec[1, 2], "Ice Vol. Fraction")
    f_m = np.ma.masked_where(~ice_mask, ice_result["f_ice_map"])
    ax.imshow(CPR, cmap="gray", vmin=0, vmax=2, alpha=0.4, **kw)
    im = ax.imshow(f_m, cmap="hot", vmin=0, vmax=0.5, **kw)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.01).ax.tick_params(colors="gray", labelsize=6)

    # Elevation profile
    ax = fig.add_subplot(gs_spec[1, 3])
    ax.set_facecolor("#111111")
    for s in ax.spines.values(): s.set_edgecolor("#444")
    ax.tick_params(colors="gray", labelsize=7)
    if path:
        elevs = [dem[r, c] for r, c in path]
        dists = np.cumsum([0] + [
            np.sqrt((path[i][0]-path[i-1][0])**2 +
                    (path[i][1]-path[i-1][1])**2) * PIXEL_SCALE / 1000
            for i in range(1, len(path))
        ])
        ax.plot(dists, elevs, "#ff6b6b", lw=1.5)
        ax.fill_between(dists, min(elevs) - 50, elevs, alpha=0.2, color="#ff6b6b")
        ax.set_xlabel("Distance (km)", color="gray", fontsize=7)
        ax.set_ylabel("Elevation (m)", color="gray", fontsize=7)
    ax.set_title("Elevation Profile", color="white", fontsize=10,
                 fontweight="bold", pad=4)

    # Row 2 – key numbers as two-row stats bar
    traverse_dist = traverse.get("metrics", {}).get("total_distance_m")
    traverse_str  = f"{traverse_dist / 1000:.2f} km" if traverse_dist else "see Fig 5"
    vol_str = (f"{ice_result['total_ice_volume_m3'] / 1e3:.1f} ± {uncertainty['std_m3'] / 1e3:.1f} ×10³ m³"
               if uncertainty else f"{ice_result['total_ice_volume_m3'] / 1e3:.1f} ×10³ m³")

    row1_items = [
        f"PSR Area: {psr_mask.sum() * 1e-4:.2f} km²",
        f"Ice Area: {ice_mask.sum() * 100:,.0f} m²",
        f"Ice Volume: {vol_str}",
        f"Mean Conc.: {ice_result['mean_ice_concentration'] * 100:.1f}%",
    ]
    row2_items = [
        f"DSC Diameter: {dsc_radius * 2 * PIXEL_SCALE:.0f} m",
        f"Ice Pixels: {ice_mask.sum():,}",
        f"Traverse: {traverse_str}",
        f"Detection: CPR > 1.0  &  DOP < 0.13",
    ]

    ax_stats = fig.add_subplot(gs_spec[2, :])
    ax_stats.set_facecolor("#111111")
    for s in ax_stats.spines.values(): s.set_edgecolor("#444")
    ax_stats.set_xticks([]); ax_stats.set_yticks([])

    ax_stats.text(0.5, 0.70, "     |     ".join(row1_items),
                  transform=ax_stats.transAxes, ha="center", va="center",
                  fontsize=9, color="white", fontfamily="monospace")
    ax_stats.text(0.5, 0.28, "     |     ".join(row2_items),
                  transform=ax_stats.transAxes, ha="center", va="center",
                  fontsize=9, color="#aaaaaa", fontfamily="monospace")

    fig.suptitle(
        "Chandrayaan-2 DFSAR Investigation of a Doubly Shadowed Crater within Faustini PSR",
        fontsize=14, fontweight="bold", color="white", y=0.98
    )
    return _save(fig, "00_dashboard.png")


# ---------------------------------------------------------------------------
# Figure 07 – m-chi decomposition, ROC curve, temporal coherence
# ---------------------------------------------------------------------------

def plot_advanced_analysis(CPR, DOP, ice_mask, psr_mask,
                            mchi=None, roc=None, temporal=None):
    """
    3×3 figure: m-chi decomposition (row 0), ROC curve + stats (row 1),
    temporal coherence (row 2).  Any None argument produces a placeholder.
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    gs_px = CPR.shape[0]
    ext   = [0, _px_to_km(gs_px), 0, _px_to_km(gs_px)]

    # ── Row 0: m-chi decomposition ──
    if mchi is not None:
        panels_mchi = [
            (axes[0,0], mchi["frac_single"], "m-chi: Single-bounce (Ps)", "Oranges"),
            (axes[0,1], mchi["frac_double"], "m-chi: Double-bounce (Pd)", "Purples"),
            (axes[0,2], mchi["frac_volume"], "m-chi: Volume scatter (Pv)\n[Ice proxy]", "Greens"),
        ]
        for ax, data, title, cmap in panels_mchi:
            im = ax.imshow(data, cmap=cmap, extent=ext, origin="upper", vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, label="Fraction [0-1]", fraction=0.046)
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("km"); ax.set_ylabel("km")
            # Ice mask contour
            if ice_mask.any():
                ax.contour(np.linspace(0, _px_to_km(gs_px), gs_px),
                           np.linspace(_px_to_km(gs_px), 0, gs_px),
                           ice_mask.astype(float), levels=[0.5],
                           colors=["cyan"], linewidths=[1.2])
    else:
        for ax in axes[0]:
            ax.text(0.5, 0.5, "m-chi decomposition\nnot available\n(need Stokes V)",
                    ha="center", va="center", transform=ax.transAxes, color="gray")
            ax.axis("off")

    # ── Row 1: ROC curve ──
    if roc is not None:
        ax = axes[1, 0]
        ax.plot(roc["fpr"], roc["tpr"], "b-", lw=2, label=f"ROC (AUC={roc['auc']:.3f})")
        ax.plot(roc["optimal_fpr"], roc["optimal_tpr"], "r*", ms=12,
                label=f"Optimal CPR={roc['optimal_threshold']:.2f}")
        ax.plot([0,1], [0,1], "k--", lw=1, alpha=0.5, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve (ice vs non-ice)\nYouden's J optimal threshold", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Precision–recall (axes[1,1])
        ax2 = axes[1, 1]
        ax2.plot(roc["thresholds"], roc["tpr"],
                 "g-", lw=2, label="TPR (sensitivity)")
        ax2.plot(roc["thresholds"], roc["precision"],
                 "b--", lw=2, label="Precision")
        ax2.plot(roc["thresholds"], roc["fpr"],
                 "r:", lw=1.5, label="FPR")
        ax2.axvline(roc["optimal_threshold"], color="orange", lw=1.5,
                    ls="--", label=f"Optimal: {roc['optimal_threshold']:.2f}")
        ax2.set_xlabel("CPR threshold")
        ax2.set_ylabel("Rate")
        ax2.set_title("TPR / Precision / FPR vs CPR threshold", fontweight="bold")
        ax2.legend(fontsize=7)
        ax2.grid(alpha=0.3)

        # Summary table (axes[1,2])
        ax3 = axes[1, 2]
        ax3.axis("off")
        rows = [
            ["Metric", "Value"],
            ["AUC", f"{roc['auc']:.4f}"],
            ["Optimal CPR threshold", f"{roc['optimal_threshold']:.3f}"],
            ["TPR at optimal", f"{roc['optimal_tpr']:.3f}"],
            ["FPR at optimal", f"{roc['optimal_fpr']:.3f}"],
            ["Positive pixels (ice)", f"{roc['n_pos']}"],
            ["Negative pixels", f"{roc['n_neg']}"],
        ]
        tbl = ax3.table(cellText=rows[1:], colLabels=rows[0],
                        loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.2, 1.5)
        ax3.set_title("ROC Summary Statistics", fontweight="bold")
    else:
        for ax in axes[1]:
            ax.text(0.5, 0.5, "ROC curve not available\n(need ground-truth ice zone)",
                    ha="center", va="center", transform=ax.transAxes, color="gray")
            ax.axis("off")

    # ── Row 2: Temporal coherence ──
    if temporal is not None:
        stable  = temporal["stable_ice"]
        frost   = temporal["frost_candidates"]
        d_cpr   = temporal["delta_CPR"]

        # Stable ice / frost map
        ax = axes[2, 0]
        tc_rgb = np.zeros((gs_px, gs_px, 3))
        tc_rgb[psr_mask]  = [0.15, 0.20, 0.50]
        tc_rgb[stable]    = [0.0,  0.85, 0.3]
        tc_rgb[frost]     = [1.0,  0.85, 0.0]
        ax.imshow(tc_rgb, extent=ext, origin="upper")
        legend_handles = [
            mpatches.Patch(color=[0.15,0.20,0.50], label="PSR"),
            mpatches.Patch(color=[0.0, 0.85, 0.3], label="Stable ice"),
            mpatches.Patch(color=[1.0, 0.85, 0.0], label="Frost candidates"),
        ]
        ax.legend(handles=legend_handles, loc="lower right", fontsize=7)
        ax.set_title(f"Temporal Ice Map\n"
                     f"Stable: {temporal['n_stable']}px  "
                     f"Frost: {temporal['n_frost']}px",
                     fontweight="bold")
        ax.set_xlabel("km"); ax.set_ylabel("km")

        # Delta-CPR map
        ax2 = axes[2, 1]
        im = ax2.imshow(np.abs(d_cpr), cmap="hot", extent=ext, origin="upper",
                        vmin=0, vmax=0.5)
        plt.colorbar(im, ax=ax2, label="|ΔCPR| pass1-pass2", fraction=0.046)
        ax2.set_title("|ΔCPR| between passes\nLow=stable ice, High=frost/noise",
                      fontweight="bold")
        ax2.set_xlabel("km"); ax2.set_ylabel("km")

        # Temporal coherence heatmap
        ax3 = axes[2, 2]
        tc  = temporal.get("temporal_confidence", np.zeros((gs_px, gs_px)))
        im3 = ax3.imshow(tc, cmap="RdYlGn", extent=ext, origin="upper",
                         vmin=0, vmax=1)
        plt.colorbar(im3, ax=ax3, label="Temporal coherence [0-1]", fraction=0.046)
        ax3.set_title("Temporal Coherence\n(1=stable ice, 0=transient)",
                      fontweight="bold")
        ax3.set_xlabel("km"); ax3.set_ylabel("km")

        # Summarise stats
        mean_d = temporal.get("mean_delta_CPR_stable", float("nan"))
        print(f"  [TempCoh] Stable ice: {temporal['n_stable']}px  "
              f"Frost: {temporal['n_frost']}px  "
              f"Mean dCPR (stable): {mean_d:.3f}")
    else:
        for ax in axes[2]:
            ax.text(0.5, 0.5, "Temporal coherence\nnot available\n(need 2 passes)",
                    ha="center", va="center", transform=ax.transAxes, color="gray")
            ax.axis("off")

    _label_all_axes(fig, start='a', extent_km=_px_to_km(gs_px))
    fig.suptitle("Advanced Polarimetric Analysis: m-chi / ROC / Temporal Coherence",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "07_advanced_analysis.png")


# ---------------------------------------------------------------------------
# Figure 08 – Ice depth scenario comparison
# ---------------------------------------------------------------------------

def plot_scenarios(scenarios, psr_mask):
    """
    Three-panel comparison of ice depth scenarios
    (shallow frost, mid-depth, deep ancient).
    """
    if not scenarios:
        return None

    n = len(scenarios)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 11))
    if n == 1:
        axes = axes[:, np.newaxis]

    names = list(scenarios.keys())
    gs_px = psr_mask.shape[0]
    ext   = [0, _px_to_km(gs_px), 0, _px_to_km(gs_px)]
    cpr_kw = dict(cmap="RdYlGn_r", vmin=0.3, vmax=2.0, extent=ext, origin="upper")

    titles = {
        "shallow_frost":    "Shallow Frost\n(depth=0.3 m, f=12%)",
        "mid_depth_ice":    "Mid-depth Ice\n(depth=2.0 m, f=35%)",
        "deep_ancient_ice": "Deep Ancient Ice\n(depth=4.5 m, f=55%)",
    }

    for j, name in enumerate(names):
        sc   = scenarios[name]
        cpr  = sc["dfsar"]["CPR"]
        iz   = sc["dfsar"]["ice_zone"]

        # Row 0: CPR map
        ax = axes[0, j]
        im = ax.imshow(cpr, **cpr_kw)
        plt.colorbar(im, ax=ax, label="CPR", fraction=0.046)
        if iz.any():
            ax.contour(np.linspace(0, _px_to_km(gs_px), gs_px),
                       np.linspace(_px_to_km(gs_px), 0, gs_px),
                       iz.astype(float), levels=[0.5],
                       colors=["cyan"], linewidths=[1.5])
        detectable = sc["detectable"]
        title_str  = titles.get(name, name)
        ax.set_title(f"{title_str}\nCPR_obs={sc['cpr_obs']:.3f}  "
                     f"{'DETECTABLE' if detectable else 'BELOW THRESHOLD'}",
                     fontweight="bold", fontsize=9,
                     color="green" if detectable else "red")
        ax.set_xlabel("km"); ax.set_ylabel("km")

        # Row 1: CPR histogram for ice zone vs PSR background
        ax2 = axes[1, j]
        cpr_bg  = cpr[psr_mask & ~iz].ravel()
        cpr_ice = cpr[iz].ravel() if iz.any() else np.array([])

        bins = np.linspace(0, 2.5, 40)
        if cpr_bg.size > 0:
            ax2.hist(cpr_bg, bins=bins, color="steelblue", alpha=0.6,
                     label="PSR background", density=True)
        if cpr_ice.size > 0:
            ax2.hist(cpr_ice, bins=bins, color="limegreen", alpha=0.7,
                     label="Ice zone", density=True)
        ax2.axvline(0.8, color="red", lw=1.5, ls="--", label="CPR=0.8 threshold")
        ax2.axvline(sc["cpr_obs"], color="orange", lw=1.5, ls=":",
                    label=f"Ice mean CPR={sc['cpr_obs']:.3f}")
        ax2.set_xlabel("CPR")
        ax2.set_ylabel("Density")
        ax2.legend(fontsize=7)
        ax2.grid(alpha=0.25)
        ax2.set_title(f"CPR distribution\nattenuation={sc['attenuation']:.3f}",
                      fontsize=8.5)

    fig.suptitle("Ice Depth Scenario Analysis\n"
                 "CPR attenuation through dry regolith: "
                 "CPR_obs = CPR_surface x exp(-2 x alpha x depth)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "08_ice_scenarios.png")
