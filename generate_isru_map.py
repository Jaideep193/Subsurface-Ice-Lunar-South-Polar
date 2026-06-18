"""
ISRU Resource Priority Map
==========================
Shows extractable water-equivalent mass per km² derived from the
Polder-van Santen ice fraction inversion, with contoured mining priority zones.

Output: results/figures/09_isru_resource_map.png
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

ICE_DENSITY_KG_M3   = 917.0    # pure water ice
WATER_ICE_RATIO     = 0.997    # mass water per mass ice (essentially 1)
KG_PER_TONNE        = 1000.0


def load_data():
    from src.data_generator import load_all
    from src.dfsar_analysis import run_analysis
    from src.ice_volume     import estimate_ice_volume, monte_carlo_uncertainty, \
                                    compute_penetration_depth_map
    from src.morphology     import run_morphology
    from src.thermal_model  import compute_surface_temperature

    data = load_all(cache=True)
    dem=data["dem"]; slope=data["slope"]; illum=data["illum"]
    psr=data["psr_mask"]; dsc=data["dsc_mask"]
    dfsar=data["dfsar"]; ohrc=data["ohrc"]; meta=data["meta"]
    gs=meta["grid_size"]; ps=meta["pixel_scale"]
    dsc_c=meta["dsc_center"]; dsc_r=meta["dsc_radius"]

    res   = run_analysis(dfsar, psr, slope, cpr_thresh=0.8, dop_thresh=0.13)
    ice_result = estimate_ice_volume(res["CPR"], res["ice_mask"], pixel_scale=ps)
    unc        = monte_carlo_uncertainty(res["CPR"], res["ice_mask"], pixel_scale=ps,
                                         n_samples=200)
    depth_map  = compute_penetration_depth_map(res["CPR"])

    return dict(dem=dem, slope=slope, illum=illum, psr=psr, dsc=dsc,
                CPR=res["CPR"], ice_mask=res["ice_mask"],
                ice_conf=res["ice_conf"], f_ice=ice_result["f_ice_map"],
                depth_map=depth_map, ice_result=ice_result, unc=unc,
                dsc_c=dsc_c, dsc_r=dsc_r, gs=gs, ps=ps, meta=meta)


def make_isru_map(d):
    gs=d["gs"]; ps=d["ps"]; KM=ps/1000.0
    f_ice     = d["f_ice"]        # fraction per pixel
    depth_map = d["depth_map"]    # metres
    ice_mask  = d["ice_mask"]
    psr       = d["psr"]
    dsc_c     = d["dsc_c"]
    dsc_r     = d["dsc_r"]

    # ── Water-equivalent mass map ─────────────────────────────────────────────
    # mass (kg) per pixel = f_ice * pixel_area_m2 * depth_m * density
    pixel_area_m2   = ps ** 2
    mass_kg_px      = f_ice * pixel_area_m2 * np.minimum(depth_map, 5.0) * ICE_DENSITY_KG_M3
    # Convert to tonnes per km²
    px_per_km2      = (1000.0 / ps) ** 2
    mass_t_per_km2  = mass_kg_px / KG_PER_TONNE * px_per_km2

    # Only show within ice mask (zero elsewhere for clean display)
    display_map = np.where(ice_mask, mass_t_per_km2, 0.0).astype(np.float32)

    # Total resource
    total_t = float(mass_kg_px[ice_mask].sum()) / KG_PER_TONNE if ice_mask.any() else 0.0

    ext = [0, gs*KM, gs*KM, 0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0a0a12")
    for ax in axes:
        ax.set_facecolor("#0d0d1a")
        for sp in ax.spines.values(): sp.set_color("#334")

    # ── Panel 1: Ice confidence + PSR context ─────────────────────────────────
    ax = axes[0]
    conf_disp = np.where(psr, d["ice_conf"], 0.0)
    im0 = ax.imshow(conf_disp, cmap="plasma", vmin=0, vmax=1,
                    extent=ext, origin="upper", aspect="auto")
    cb0 = plt.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
    cb0.set_label("P(ice | CPR, DOP, PSR)", color="white", fontsize=8)
    cb0.ax.tick_params(colors="white", labelsize=7)
    ax.add_patch(plt.Circle((dsc_c[1]*KM, dsc_c[0]*KM), dsc_r*KM,
                              color="yellow", fill=False, lw=1.5, ls="--"))
    ax.set_title("(a) Bayesian Ice Posterior\nwithin PSR boundary",
                 color="white", fontsize=9, fontweight="bold")
    ax.set_xlabel("Distance East (km)", color="#aaa", fontsize=8)
    ax.set_ylabel("Distance South (km)", color="#aaa", fontsize=8)
    ax.tick_params(colors="#888", labelsize=7)

    # ── Panel 2: Water-equivalent resource map + contours ────────────────────
    ax = axes[1]
    # Custom colourmap: dark for 0, vivid plasma for resources
    im1 = ax.imshow(display_map, cmap="YlOrRd",
                    vmin=0, vmax=max(float(display_map.max()), 1e-3),
                    extent=ext, origin="upper", aspect="auto")
    cb1 = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    cb1.set_label("Water-equivalent (t/km²)", color="white", fontsize=8)
    cb1.ax.tick_params(colors="white", labelsize=7)

    # Mining priority contours
    levels = [10, 50, 100, 200]
    x_km   = np.linspace(0, gs*KM, gs)
    y_km   = np.linspace(0, gs*KM, gs)
    cs = ax.contour(x_km, y_km, display_map, levels=levels,
                    colors=["#ffff00","#ff8800","#ff4400","#ff0000"],
                    linewidths=[0.9, 1.1, 1.3, 1.5])
    ax.clabel(cs, fmt="%d t/km²", fontsize=6.5, colors="white", inline=True)

    # Priority zone labels
    zone_labels = {
        10:  ("Low priority",      "#ffff80"),
        50:  ("Medium priority",   "#ffaa55"),
        100: ("High priority",     "#ff6644"),
        200: ("Prime extraction",  "#ff3333"),
    }
    for lvl, (lbl, col) in zone_labels.items():
        if float(display_map.max()) >= lvl:
            ax.text(0.98, 0.03 + list(zone_labels).index(lvl)*0.05, lbl,
                    transform=ax.transAxes, color=col, fontsize=6.5,
                    ha="right", fontweight="bold")

    ax.add_patch(plt.Circle((dsc_c[1]*KM, dsc_c[0]*KM), dsc_r*KM,
                              color="cyan", fill=False, lw=1.5, ls="--"))
    ax.set_title("(b) ISRU Resource Map\nWater-equivalent mass density",
                 color="white", fontsize=9, fontweight="bold")
    ax.set_xlabel("Distance East (km)", color="#aaa", fontsize=8)
    ax.set_ylabel("Distance South (km)", color="#aaa", fontsize=8)
    ax.tick_params(colors="#888", labelsize=7)

    # ── Panel 3: Resource summary ─────────────────────────────────────────────
    ax = axes[2]
    ax.axis("off")

    # Monte Carlo uncertainty bands
    unc    = d["unc"]
    p10    = unc["p5_m3"]  * ICE_DENSITY_KG_M3 / KG_PER_TONNE
    p50    = unc["p50_m3"] * ICE_DENSITY_KG_M3 / KG_PER_TONNE
    p90    = unc["p95_m3"] * ICE_DENSITY_KG_M3 / KG_PER_TONNE
    p_mean = unc["mean_m3"]* ICE_DENSITY_KG_M3 / KG_PER_TONNE

    # ── Rigorous propellant mass budget ──────────────────────────────────────
    # Water mass fractions: H2O → 11.19% H by mass, 88.81% O by mass
    # PEM electrolysis system efficiency η = 0.70 (includes BoP, compression,
    #   thermal management losses; vs η=0.85 for SOEC at elevated T)
    # Reference: Elias & Shafii 2019; Ingham et al. 2022 (lunar ISRU review)
    ETA_PEM   = 0.70
    H2_FRAC   = 0.1119   # H mass fraction in water
    O2_FRAC   = 0.8881   # O mass fraction in water

    h2_p50    = p50 * H2_FRAC * ETA_PEM    # tonnes H2
    o2_p50    = p50 * O2_FRAC * ETA_PEM    # tonnes O2 (from electrolysis)

    # Sabatier reaction: CO2 + 4H2 → CH4 + 2H2O
    # Mass ratio: CH4 (16) produced per 4×H2 (8) → 2.0 kg CH4 per kg H2
    # Requires CO2 from local atmosphere (∼negligible on Moon) or ISRU capture
    ch4_p50   = h2_p50 * 2.0               # tonnes CH4
    # O2 to combust CH4: CH4 + 2O2 → CO2 + 2H2O, MW: 16 + 64 → 4.0 kg O2 per kg CH4
    o2_ch4_p50 = ch4_p50 * 4.0             # tonnes O2 needed for CH4 combustion

    # LOX/LH2 reference (Isp=450 s): propellant ratio O:H = 6:1 by mass
    # LOX/CH4 reference (Isp=363 s): propellant ratio O:CH4 = 3.5:1 by mass
    # Delta-v context: lunar orbit → surface ≈ 1.7 km/s
    dv_lnd    = 1.70   # km/s
    isp_lh2   = 450    # s
    isp_ch4   = 363    # s

    crew_days_p50 = p50 * 1000.0 / 1.83   # 1.83 L water/person/day → days

    summary_rows = [
        ["Resource Estimate",                  "Value"],
        ["P10 water-equiv.",                   f"{p10:.1f} t"],
        ["P50 water-equiv. (median)",          f"{p50:.1f} t"],
        ["P90 water-equiv.",                   f"{p90:.1f} t"],
        ["Mean estimate",                      f"{p_mean:.1f} t"],
        ["MC relative uncertainty",            f"{unc['rel_unc_pct']:.0f}%"],
        ["",""],
        ["PEM Electrolysis (η=0.70)",          "P50 scenario"],
        ["H₂ propellant",                      f"{h2_p50:.3f} t"],
        ["O₂ propellant",                      f"{o2_p50:.3f} t"],
        ["",""],
        ["Sabatier Path (CO₂+4H₂→CH₄+2H₂O)", "P50 scenario"],
        ["CH₄ yield",                          f"{ch4_p50:.3f} t"],
        ["O₂ for CH₄ combustion",             f"{o2_ch4_p50:.3f} t"],
        ["",""],
        ["Δv context (lunar orbit→surface)",   f"{dv_lnd:.1f} km/s"],
        ["LOX/LH₂ Isp",                        f"{isp_lh2} s"],
        ["LOX/CH₄ Isp",                        f"{isp_ch4} s"],
        ["",""],
        ["Crew water @ 1.83 L/p/day",          f"{crew_days_p50/1e6:.2f}M person-days"],
        ["Ice pixels",                         f"{ice_mask.sum()}"],
        ["Ice-bearing area",                   f"{d['ice_result']['ice_bearing_area_km2']:.4f} km²"],
        ["Mean ice conc. (f_ice)",             f"{d['ice_result']['mean_ice_concentration']*100:.1f}%"],
    ]

    tbl = ax.table(cellText=[[r[1]] for r in summary_rows],
                   rowLabels=[r[0] for r in summary_rows],
                   colLabels=["Value"], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.3, 1.6)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#111827" if row % 2 == 0 else "#0d1420")
        cell.set_text_props(color="white")
        cell.set_edgecolor("#334455")

    ax.set_title("(c) ISRU Resource Summary\nMonte Carlo probabilistic estimate",
                 color="white", fontsize=9, fontweight="bold", pad=12)

    fig.suptitle(
        "ISRU Resource Assessment  |  "
        "Water-Ice Extraction Potential  |  Faustini PSR Doubly Shadowed Crater",
        color="white", fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()

    out = Path("results/figures/09_isru_resource_map.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[ISRU] Saved: {out}")


if __name__ == "__main__":
    print("[ISRU] Loading data...")
    d = load_data()
    make_isru_map(d)
