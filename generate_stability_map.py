"""
Ice Stability Zone Map
=======================
Classifies detected ice pixels into thermal stability regimes using the
Diviner-calibrated surface temperature model (Paige et al. 2010):

  Zone 1 — Stable ancient ice    : T < 70 K   (thermodynamically stable, >1 Ga)
  Zone 2 — Seasonal frost         : 70-110 K   (seasonal condensation/sublimation)
  Zone 3 — Thermally unstable     : T >= 110 K (active sublimation)

Output:
  results/figures/12_ice_stability_zones.png
  results/export/stability_zones.geojson
"""
import sys, os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

CENTER_LAT = -87.0
CENTER_LON =   0.0
M_PER_DEG  = 111_320.0

T_STABLE   = 70.0    # K
T_SEASONAL = 110.0   # K


def load_data():
    from src.data_generator import load_all
    from src.dfsar_analysis import run_analysis
    from src.thermal_model  import compute_surface_temperature

    data  = load_all(cache=True)
    psr   = data["psr_mask"]; slope = data["slope"]; illum = data["illum"]
    dfsar = data["dfsar"]; meta = data["meta"]; dem = data["dem"]
    gs    = meta["grid_size"]; ps_m = meta["pixel_scale"]
    dsc_c = meta["dsc_center"]; dsc_r = meta["dsc_radius"]

    res       = run_analysis(dfsar, psr, slope)
    T_surface = compute_surface_temperature(illum)

    return dict(dem=dem, slope=slope, illum=illum, T_surface=T_surface,
                psr=psr, CPR=res["CPR"], ice_mask=res["ice_mask"],
                ice_conf=res["ice_conf"], gs=gs, ps=ps_m,
                dsc_c=dsc_c, dsc_r=dsc_r)


def compute_stability(d):
    T   = d["T_surface"]
    ice = d["ice_mask"]

    stable   = ice & (T < T_STABLE)
    seasonal = ice & (T >= T_STABLE)   & (T < T_SEASONAL)
    unstable = ice & (T >= T_SEASONAL)
    no_ice   = ~ice

    # Categorical integer map
    zone_map = np.zeros(T.shape, dtype=np.int8)
    zone_map[stable]   = 1
    zone_map[seasonal] = 2
    zone_map[unstable] = 3

    return zone_map, dict(
        n_stable=int(stable.sum()),
        n_seasonal=int(seasonal.sum()),
        n_unstable=int(unstable.sum()),
        n_no_ice=int(no_ice.sum()),
        frac_stable=float(stable.sum() / max(ice.sum(), 1) * 100),
        frac_seasonal=float(seasonal.sum() / max(ice.sum(), 1) * 100),
        frac_unstable=float(unstable.sum() / max(ice.sum(), 1) * 100),
    )


def make_figure(d, zone_map, stats):
    gs = d["gs"]; ps_m = d["ps"]
    KM  = ps_m / 1000.0
    ext = [0, gs*KM, gs*KM, 0]
    dsc_c = d["dsc_c"]; dsc_r = d["dsc_r"]

    # Discrete colormap
    cmap_zones = mcolors.ListedColormap([
        "#0d0d1a",   # 0 = no ice (dark bg)
        "#00e5aa",   # 1 = stable ancient (cyan-green)
        "#ffaa00",   # 2 = seasonal frost (amber)
        "#ff4444",   # 3 = thermally unstable (red)
    ])
    norm_zones = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap_zones.N)

    fig = plt.figure(figsize=(18, 7))
    fig.patch.set_facecolor("#080810")
    fig.suptitle(
        "Ice Thermal Stability Classification  |  Chandrayaan-2 DFSAR  |  Faustini PSR\n"
        f"Based on Diviner surface temperature model (Paige et al. 2010)",
        color="white", fontsize=12, fontweight="bold")

    # ── Panel 1: Stability zone map ───────────────────────────────────────────
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_facecolor("#0d0d1a")
    for sp in ax1.spines.values(): sp.set_color("#334")
    ax1.imshow(zone_map, cmap=cmap_zones, norm=norm_zones,
               extent=ext, origin="upper", aspect="auto", interpolation="nearest")
    ax1.add_patch(plt.Circle((dsc_c[1]*KM, dsc_c[0]*KM), dsc_r*KM,
                              color="white", fill=False, lw=1.5, ls="--"))

    # PSR boundary
    x_km = np.linspace(0, gs*KM, gs)
    y_km = np.linspace(0, gs*KM, gs)
    ax1.contour(x_km, y_km, d["psr"].astype(float),
                levels=[0.5], colors=["#888"], linewidths=[0.8], linestyles=["--"])

    patches = [
        mpatches.Patch(color="#0d0d1a", label="No ice detected"),
        mpatches.Patch(color="#00e5aa", label=f"Stable ancient ice  (T < {T_STABLE:.0f} K)  [{stats['n_stable']} px]"),
        mpatches.Patch(color="#ffaa00", label=f"Seasonal frost  ({T_STABLE:.0f}–{T_SEASONAL:.0f} K)  [{stats['n_seasonal']} px]"),
        mpatches.Patch(color="#ff4444", label=f"Thermally unstable  (T > {T_SEASONAL:.0f} K)  [{stats['n_unstable']} px]"),
    ]
    ax1.legend(handles=patches, loc="upper left", fontsize=7.5,
               facecolor="#111", edgecolor="#445", framealpha=0.9)
    ax1.set_title("(a) Stability Zone Map", color="white", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Distance East (km)", color="#aaa", fontsize=8)
    ax1.set_ylabel("Distance South (km)", color="#aaa", fontsize=8)
    ax1.tick_params(colors="#888", labelsize=7)

    # ── Panel 2: Temperature vs CPR scatter ───────────────────────────────────
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_facecolor("#0d0d1a")
    for sp in ax2.spines.values(): sp.set_color("#334")
    ax2.tick_params(colors="#888", labelsize=7)

    T_ice = d["T_surface"][d["ice_mask"]]
    CPR_ice = d["CPR"][d["ice_mask"]]

    colors_scatter = np.where(T_ice < T_STABLE, "#00e5aa",
                    np.where(T_ice < T_SEASONAL, "#ffaa00", "#ff4444"))

    ax2.scatter(T_ice, CPR_ice, c=colors_scatter, s=18, alpha=0.7, edgecolors="none")
    ax2.axhline(0.8, color="white", lw=1.2, ls="--", alpha=0.7, label="CPR = 0.8 threshold")
    ax2.axvline(T_STABLE,   color="#00e5aa", lw=1.2, ls=":", alpha=0.8,
                label=f"T = {T_STABLE:.0f} K")
    ax2.axvline(T_SEASONAL, color="#ffaa00", lw=1.2, ls=":", alpha=0.8,
                label=f"T = {T_SEASONAL:.0f} K")
    ax2.set_xlabel("Surface Temperature (K)", color="#aaa", fontsize=8)
    ax2.set_ylabel("CPR (SC/OC)", color="#aaa", fontsize=8)
    ax2.set_title("(b) Temperature vs CPR\n(ice pixels only)",
                  color="white", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=7.5, facecolor="#111", edgecolor="#445", labelcolor="white")

    # ── Panel 3: Zone breakdown pie + stats table ─────────────────────────────
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_facecolor("#0d0d1a")
    for sp in ax3.spines.values(): sp.set_color("#334")
    ax3.axis("off")

    n_ice = stats["n_stable"] + stats["n_seasonal"] + stats["n_unstable"]
    if n_ice > 0:
        sizes  = [stats["n_stable"], stats["n_seasonal"], stats["n_unstable"]]
        colors = ["#00e5aa", "#ffaa00", "#ff4444"]
        labels = [f"Stable\n{stats['frac_stable']:.1f}%",
                  f"Seasonal\n{stats['frac_seasonal']:.1f}%",
                  f"Unstable\n{stats['frac_unstable']:.1f}%"]
        # Only include non-zero sectors
        sizes_nz  = [(s,c,l) for s,c,l in zip(sizes, colors, labels) if s > 0]
        if sizes_nz:
            sz, co, la = zip(*sizes_nz)
            wedges, texts, autotexts = ax3.pie(
                sz, colors=co, labels=la, autopct="%1.0f%%",
                pctdistance=0.7, startangle=90,
                textprops=dict(color="white", fontsize=8),
                wedgeprops=dict(linewidth=0.5, edgecolor="#111"),
            )
            for at in autotexts:
                at.set_color("white"); at.set_fontsize(8)

    ax3.set_title("(c) Ice Zone Breakdown", color="white",
                  fontsize=10, fontweight="bold", pad=20)

    # Stats annotation
    stat_text = (
        f"Total ice pixels:   {n_ice}\n"
        f"  Stable (<{T_STABLE:.0f} K):     {stats['n_stable']}  ({stats['frac_stable']:.1f}%)\n"
        f"  Seasonal ({T_STABLE:.0f}-{T_SEASONAL:.0f} K):  {stats['n_seasonal']}  ({stats['frac_seasonal']:.1f}%)\n"
        f"  Unstable (>{T_SEASONAL:.0f} K):  {stats['n_unstable']}  ({stats['frac_unstable']:.1f}%)\n\n"
        f"Stable ice temp:   {T_ice[T_ice < T_STABLE].mean():.1f} K (mean)"
        if n_ice > 0 and (T_ice < T_STABLE).any()
        else f"Total ice pixels: {n_ice}"
    )
    ax3.text(0.5, -0.08, stat_text, transform=ax3.transAxes,
             color="#aabbcc", fontsize=8.5, ha="center", va="top",
             fontfamily="monospace",
             bbox=dict(fc="#0d1420", ec="#334455", pad=6))

    fig.tight_layout()
    out = Path("results/figures/12_ice_stability_zones.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Stability] Saved PNG: {out}")


def write_geojson(d, zone_map):
    """Export stability zones as GeoJSON points."""
    gs = d["gs"]; ps_m = d["ps"]
    ice_rows, ice_cols = np.where(d["ice_mask"])
    cos_lat = np.cos(np.radians(CENTER_LAT))
    half_lat = (ps_m / 2.0) / M_PER_DEG
    half_lon = (ps_m / 2.0) / (M_PER_DEG * cos_lat)

    ZONE_NAMES = {0:"no_ice", 1:"stable_ancient", 2:"seasonal_frost", 3:"unstable"}
    features = []
    for r, c in zip(ice_rows, ice_cols):
        lat = CENTER_LAT + (gs/2.0 - r) * ps_m / M_PER_DEG
        lon = CENTER_LON + (c - gs/2.0) * ps_m / (M_PER_DEG * cos_lat)
        z   = int(zone_map[r, c])
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[
                [lon-half_lon, lat-half_lat],
                [lon+half_lon, lat-half_lat],
                [lon+half_lon, lat+half_lat],
                [lon-half_lon, lat+half_lat],
                [lon-half_lon, lat-half_lat],
            ]]},
            "properties": {
                "zone_id":   z,
                "zone_name": ZONE_NAMES[z],
                "T_surface": round(float(d["T_surface"][r, c]), 2),
                "CPR":       round(float(d["CPR"][r, c]), 4),
                "P_ice":     round(float(d["ice_conf"][r, c]), 4),
            }
        })

    out = Path("results/export/stability_zones.geojson")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "type": "FeatureCollection",
        "name": "Ice Stability Zones",
        "crs": {"type":"name","properties":{"name":"EPSG:4326"}},
        "features": features,
    }, indent=2), encoding="utf-8")
    print(f"[Stability] Saved GeoJSON: {out}  ({len(features)} features)")


if __name__ == "__main__":
    print("[Stability] Loading data...")
    d = load_data()
    zone_map, stats = compute_stability(d)
    make_figure(d, zone_map, stats)
    write_geojson(d, zone_map)
