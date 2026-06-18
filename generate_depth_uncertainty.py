"""
Depth-Uncertainty Cross-Plot
=============================
Scatter of ice volume fraction (f_ice) vs radar penetration depth per pixel,
coloured by surface temperature.  Highlights which ice is deep + cold
(most resource-valuable, lowest sublimation risk).

Also shows:
  - Uncertainty ellipses per thermal zone
  - 2D histogram density overlay
  - Marginal distributions
  - Annotated quadrants: high f_ice / deep depth = prime ISRU targets

Output: results/figures/16_depth_uncertainty.png
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

T_STABLE   = 70.0
T_SEASONAL = 110.0

# ── ISRU value score: high f_ice AND deep AND cold ─────────────────────────
def isru_value(f_ice, depth, T):
    """Composite value index [0,1]: higher = more extractable resource."""
    v_fice  = np.clip(f_ice / 0.4, 0, 1)
    v_depth = np.clip(depth / 5.0, 0, 1)
    v_cold  = np.clip(1.0 - (T - 40.0) / 80.0, 0, 1)
    return 0.4 * v_fice + 0.35 * v_depth + 0.25 * v_cold


def load_data():
    from src.data_generator import load_all
    from src.dfsar_analysis import run_analysis
    from src.thermal_model  import compute_surface_temperature
    from src.ice_volume     import (estimate_ice_volume, monte_carlo_uncertainty,
                                    compute_penetration_depth_map)

    data  = load_all(cache=True)
    psr   = data["psr_mask"]; slope = data["slope"]; illum = data["illum"]
    dfsar = data["dfsar"]; dem = data["dem"]; meta = data["meta"]
    gs    = meta["grid_size"]; ps = meta["pixel_scale"]
    dsc_c = meta["dsc_center"]; dsc_r = meta["dsc_radius"]

    res  = run_analysis(dfsar, psr, slope, cpr_thresh=0.8, dop_thresh=0.13)
    CPR  = res["CPR"]; ice = res["ice_mask"]
    T    = compute_surface_temperature(illum)

    iv   = estimate_ice_volume(CPR, ice, pixel_scale=ps)
    unc  = monte_carlo_uncertainty(CPR, ice, pixel_scale=ps, n_samples=300)
    full_depth = compute_penetration_depth_map(CPR)

    return dict(CPR=CPR, ice=ice, T=T, f_ice_map=iv["f_ice_map"],
                depth_map=full_depth, ice_conf=res["ice_conf"],
                psr=psr, gs=gs, ps=ps, dsc_c=dsc_c, dsc_r=dsc_r,
                unc=unc, iv=iv)


def confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    """Draw n_std-sigma covariance ellipse for points (x, y)."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    pearson = cov[0, 1] / (np.sqrt(cov[0, 0]) * np.sqrt(cov[1, 1]) + 1e-12)
    rx = np.sqrt(1 + pearson) * n_std * np.sqrt(cov[0, 0])
    ry = np.sqrt(1 - pearson) * n_std * np.sqrt(cov[1, 1])
    angle = np.degrees(np.arctan2(cov[0, 1], cov[0, 0] + 1e-12))
    ell = Ellipse(xy=(np.mean(x), np.mean(y)), width=2*rx, height=2*ry,
                  angle=angle, **kwargs)
    ax.add_patch(ell)


def make_figure(d):
    ice    = d["ice"]
    f_ice  = d["f_ice_map"][ice]
    depth  = d["depth_map"][ice]
    T_arr  = d["T"][ice]
    CPR_arr= d["CPR"][ice]
    conf   = d["ice_conf"][ice]
    unc    = d["unc"]

    stable   = T_arr < T_STABLE
    seasonal = (T_arr >= T_STABLE) & (T_arr < T_SEASONAL)
    unstable = T_arr >= T_SEASONAL

    ZONE_COLORS = {"stable": "#00e5aa", "seasonal": "#ffaa00", "unstable": "#ff4444"}

    value  = isru_value(f_ice, depth, T_arr)

    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor("#080810")
    fig.suptitle(
        "Ice Fraction vs Penetration Depth — Per-Pixel Resource Analysis  |  "
        "Faustini PSR  |  Chandrayaan-2 DFSAR\n"
        "Point color = surface temperature  |  Size ∝ radar confidence  |  "
        "Top-right quadrant = prime ISRU targets",
        color="white", fontsize=12, fontweight="bold")

    def ax_style(ax):
        ax.set_facecolor("#0d0d1a")
        for sp in ax.spines.values(): sp.set_color("#334")
        ax.tick_params(colors="#888", labelsize=8)
        return ax

    gs_fig = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.38)

    # ── Panel 1 (large): Main scatter f_ice vs depth, color = T ─────────
    ax_main = ax_style(fig.add_subplot(gs_fig[0:2, 0:2]))

    sc = ax_main.scatter(
        depth, f_ice,
        c=T_arr, cmap="RdYlBu_r", vmin=40, vmax=160,
        s=np.clip(conf * 60, 4, 60), alpha=0.65, edgecolors="none", zorder=3
    )
    cb = fig.colorbar(sc, ax=ax_main, fraction=0.046, pad=0.03)
    cb.set_label("Surface Temperature (K)", color="white", fontsize=9)
    cb.ax.tick_params(colors="w", labelsize=7)

    # Quadrant shading
    f_med  = float(np.median(f_ice));  d_med = float(np.median(depth))
    ax_main.fill_betweenx([f_med, 1], d_med, 20, color="#003322", alpha=0.25, zorder=1)
    ax_main.text(d_med + 0.2, f_med + 0.02,
                 "PRIME ISRU\n(deep + high f_ice)", color="#00e5aa",
                 fontsize=9, fontweight="bold", va="bottom", alpha=0.85)

    # Confidence ellipses per zone
    for mask, label, col in [
        (stable,   "stable",   ZONE_COLORS["stable"]),
        (seasonal, "seasonal", ZONE_COLORS["seasonal"]),
        (unstable, "unstable", ZONE_COLORS["unstable"]),
    ]:
        if mask.sum() >= 5:
            confidence_ellipse(depth[mask], f_ice[mask], ax_main,
                               n_std=2.0, facecolor=col, alpha=0.12,
                               edgecolor=col, lw=1.5, ls="--", zorder=2)
            ax_main.scatter(depth[mask].mean(), f_ice[mask].mean(),
                            color=col, s=80, marker="X", zorder=4, edgecolors="white", lw=0.5)

    ax_main.axvline(d_med, color="#556", lw=1, ls=":", alpha=0.7)
    ax_main.axhline(f_med, color="#556", lw=1, ls=":", alpha=0.7)
    ax_main.set_xlabel("Radar Penetration Depth (m)", color="#aaa", fontsize=9)
    ax_main.set_ylabel("Ice Volume Fraction f_ice", color="#aaa", fontsize=9)
    ax_main.set_title("(a) f_ice vs Penetration Depth\n(size ∝ confidence, color = T, ellipses = 2σ per zone)",
                      color="white", fontsize=10, fontweight="bold")

    patches_z = [
        mpatches.Patch(color=ZONE_COLORS["stable"],   label=f"Stable T<70 K ({stable.sum()} px)"),
        mpatches.Patch(color=ZONE_COLORS["seasonal"], label=f"Seasonal 70-110 K ({seasonal.sum()} px)"),
        mpatches.Patch(color=ZONE_COLORS["unstable"], label=f"Unstable T>110 K ({unstable.sum()} px)"),
    ]
    ax_main.legend(handles=patches_z, fontsize=8, facecolor="#111",
                   edgecolor="#334", labelcolor="white", loc="upper right")

    # ── Panel 2: 2D histogram density ────────────────────────────────────
    ax_hex = ax_style(fig.add_subplot(gs_fig[0:2, 2]))
    if len(depth) > 5:
        hb = ax_hex.hexbin(depth, f_ice, gridsize=28, cmap="inferno",
                           mincnt=1, bins="log")
        fig.colorbar(hb, ax=ax_hex, fraction=0.046,
                     label="log10(count)").ax.tick_params(colors="w", labelsize=7)
    ax_hex.set_xlabel("Penetration Depth (m)", color="#aaa", fontsize=8)
    ax_hex.set_ylabel("f_ice", color="#aaa", fontsize=8)
    ax_hex.set_title("(b) 2D Density\n(hexbin log-count)", color="white",
                     fontsize=10, fontweight="bold")

    # ── Panel 3: ISRU value map ──────────────────────────────────────────
    ax_val = ax_style(fig.add_subplot(gs_fig[0:2, 3]))
    val_arr = ax_val.scatter(depth, f_ice, c=value, cmap="plasma",
                             vmin=0, vmax=1, s=8, alpha=0.7, edgecolors="none")
    fig.colorbar(val_arr, ax=ax_val, fraction=0.046,
                 label="ISRU Value Index").ax.tick_params(colors="w", labelsize=7)
    # Top-20 highest value pixels
    top_idx = np.argsort(value)[-20:]
    ax_val.scatter(depth[top_idx], f_ice[top_idx], color="white", s=30,
                   marker="*", zorder=5, label="Top-20 ISRU pixels")
    ax_val.legend(fontsize=7.5, facecolor="#111", edgecolor="#334", labelcolor="white")
    ax_val.set_xlabel("Penetration Depth (m)", color="#aaa", fontsize=8)
    ax_val.set_ylabel("f_ice", color="#aaa", fontsize=8)
    ax_val.set_title("(c) ISRU Value Index\n(0.4·f_ice + 0.35·depth + 0.25·cold)",
                     color="white", fontsize=10, fontweight="bold")

    # ── Panel 4: f_ice marginal by zone ─────────────────────────────────
    ax_fi = ax_style(fig.add_subplot(gs_fig[2, 0]))
    bins_fi = np.linspace(0, float(f_ice.max()) + 0.01, 40)
    for mask, label, col in [
        (stable,   "Stable (<70 K)",    ZONE_COLORS["stable"]),
        (seasonal, "Seasonal (70-110K)",ZONE_COLORS["seasonal"]),
        (unstable, "Unstable (>110 K)", ZONE_COLORS["unstable"]),
    ]:
        if mask.sum() >= 2:
            ax_fi.hist(f_ice[mask], bins=bins_fi, color=col, alpha=0.75,
                       density=True, label=label, histtype="stepfilled", edgecolor="none")
    ax_fi.axvline(float(np.median(f_ice)), color="white", lw=1.2, ls="--",
                  alpha=0.7, label="Median")
    ax_fi.set_xlabel("Ice Volume Fraction f_ice", color="#aaa", fontsize=8)
    ax_fi.set_ylabel("Density", color="#aaa", fontsize=8)
    ax_fi.set_title("(d) f_ice Distribution by Zone", color="white",
                    fontsize=10, fontweight="bold")
    ax_fi.legend(fontsize=7.5, facecolor="#111", edgecolor="#334", labelcolor="white")

    # ── Panel 5: depth marginal by zone ─────────────────────────────────
    ax_dep = ax_style(fig.add_subplot(gs_fig[2, 1]))
    bins_d = np.linspace(0, float(depth.max()) + 0.1, 40)
    for mask, label, col in [
        (stable,   "Stable (<70 K)",    ZONE_COLORS["stable"]),
        (seasonal, "Seasonal (70-110K)",ZONE_COLORS["seasonal"]),
        (unstable, "Unstable (>110 K)", ZONE_COLORS["unstable"]),
    ]:
        if mask.sum() >= 2:
            ax_dep.hist(depth[mask], bins=bins_d, color=col, alpha=0.75,
                        density=True, label=label, histtype="stepfilled", edgecolor="none")
    ax_dep.axvline(float(np.median(depth)), color="white", lw=1.2, ls="--",
                   alpha=0.7, label="Median")
    ax_dep.set_xlabel("Penetration Depth (m)", color="#aaa", fontsize=8)
    ax_dep.set_ylabel("Density", color="#aaa", fontsize=8)
    ax_dep.set_title("(e) Depth Distribution by Zone", color="white",
                     fontsize=10, fontweight="bold")
    ax_dep.legend(fontsize=7.5, facecolor="#111", edgecolor="#334", labelcolor="white")

    # ── Panel 6: T vs ISRU value ─────────────────────────────────────────
    ax_tv = ax_style(fig.add_subplot(gs_fig[2, 2]))
    sc2 = ax_tv.scatter(T_arr, value, c=f_ice, cmap="YlOrRd",
                        vmin=0, vmax=0.4, s=8, alpha=0.6, edgecolors="none")
    fig.colorbar(sc2, ax=ax_tv, fraction=0.046,
                 label="f_ice").ax.tick_params(colors="w", labelsize=7)
    ax_tv.axvline(T_STABLE,   color=ZONE_COLORS["stable"],   lw=1.2, ls=":", alpha=0.8)
    ax_tv.axvline(T_SEASONAL, color=ZONE_COLORS["seasonal"], lw=1.2, ls=":", alpha=0.8)
    ax_tv.set_xlabel("Surface Temperature (K)", color="#aaa", fontsize=8)
    ax_tv.set_ylabel("ISRU Value Index", color="#aaa", fontsize=8)
    ax_tv.set_title("(f) T vs ISRU Value\n(color = f_ice)", color="white",
                    fontsize=10, fontweight="bold")

    # ── Panel 7: Summary statistics table ────────────────────────────────
    ax_tbl = fig.add_subplot(gs_fig[2, 3])
    ax_tbl.axis("off")

    def stat_rows(mask, name):
        if not mask.any():
            return [name, "0", "—", "—", "—", "—"]
        return [
            name,
            str(mask.sum()),
            f"{f_ice[mask].mean():.3f}±{f_ice[mask].std():.3f}",
            f"{depth[mask].mean():.2f}±{depth[mask].std():.2f}",
            f"{T_arr[mask].mean():.1f}",
            f"{value[mask].mean():.3f}",
        ]

    col_h = ["Zone", "Px", "f_ice (mean±σ)", "Depth m (mean±σ)", "T mean K", "ISRU val"]
    rows  = [
        stat_rows(stable,   "Stable\n<70 K"),
        stat_rows(seasonal, "Seasonal\n70-110 K"),
        stat_rows(unstable, "Unstable\n>110 K"),
        ["ALL ice",
         str(len(f_ice)),
         f"{f_ice.mean():.3f}±{f_ice.std():.3f}",
         f"{depth.mean():.2f}±{depth.std():.2f}",
         f"{T_arr.mean():.1f}",
         f"{value.mean():.3f}"],
    ]
    tbl = ax_tbl.table(cellText=rows, colLabels=col_h,
                       loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    tbl.scale(1.1, 2.0)
    zone_cs = [ZONE_COLORS["stable"], ZONE_COLORS["seasonal"],
               ZONE_COLORS["unstable"], "#aabbcc"]
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#1a2040" if r == 0 else "#0d1520")
        cell.set_text_props(color=zone_cs[r-1] if r > 0 else "white")
        cell.set_edgecolor("#2a3a55")
    ax_tbl.set_title("(g) Per-Zone Statistics", color="white",
                     fontsize=10, fontweight="bold", pad=8)

    fig.tight_layout()
    out = Path("results/figures/16_depth_uncertainty.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[DepthUnc] Saved: {out}")

    # Print top ISRU pixels
    top5 = np.argsort(value)[-5:][::-1]
    print(f"[DepthUnc] Top-5 ISRU pixels by value index:")
    for i, idx in enumerate(top5):
        print(f"  #{i+1}  f_ice={f_ice[idx]:.3f}  depth={depth[idx]:.2f}m  "
              f"T={T_arr[idx]:.1f}K  value={value[idx]:.3f}")


if __name__ == "__main__":
    print("[DepthUnc] Loading data...")
    d = load_data()
    n = d["ice"].sum()
    print(f"[DepthUnc] Ice pixels: {n}")
    make_figure(d)
