"""
Landing Site Ranker
====================
Scores every pixel by a 4-criterion MCDA and returns the top-5 candidate
landing sites with minimum separation of 500 m (50 pixels).

Criteria (AHP-style weights):
  30% — slope safety      (slope < 15 deg preferred)
  25% — solar illumination (higher = more power)
  25% — ice access        (proximity to nearest ice pixel)
  20% — radar confidence   (P(ice|CPR,DOP,PSR) as science value proxy)

Outputs:
  results/figures/13_landing_sites.png
  results/export/landing_sites.csv
  results/export/landing_sites.geojson
"""
import sys, os
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

CENTER_LAT = -87.0
CENTER_LON =   0.0
M_PER_DEG  = 111_320.0
MIN_SEP_PX = 50        # minimum separation between sites in pixels
SLOPE_MAX  = 15.0      # deg — hard exclusion above this
MIN_SCORE  = 0.20      # minimum composite score to be considered


def load_data():
    from src.data_generator import load_all
    from src.dfsar_analysis import run_analysis
    from src.thermal_model  import compute_surface_temperature

    data  = load_all(cache=True)
    psr   = data["psr_mask"]; slope = data["slope"]; illum = data["illum"]
    dfsar = data["dfsar"]; dem = data["dem"]; meta = data["meta"]
    gs    = meta["grid_size"]; ps_m = meta["pixel_scale"]
    dsc_c = meta["dsc_center"]; dsc_r = meta["dsc_radius"]

    res = run_analysis(dfsar, psr, slope)
    T   = compute_surface_temperature(illum)

    return dict(dem=dem, slope=slope, illum=illum, T_surface=T,
                psr=psr, ice_mask=res["ice_mask"], ice_conf=res["ice_conf"],
                gs=gs, ps=ps_m, dsc_c=dsc_c, dsc_r=dsc_r)


def score_map(d):
    from scipy.ndimage import distance_transform_edt

    slope   = d["slope"]
    illum   = d["illum"]
    ice_conf= d["ice_conf"]
    ice_mask= d["ice_mask"]

    # 1. Slope safety score (linear, clamped at 30 deg)
    slope_score = np.clip(1.0 - slope / SLOPE_MAX, 0.0, 1.0)

    # 2. Illumination score (already normalised 0-1)
    illum_score = np.clip(illum, 0.0, 1.0)

    # 3. Ice proximity score — exp decay from nearest ice pixel
    #    distance_transform_edt gives distance in pixels to nearest True pixel
    if ice_mask.any():
        dist_px = distance_transform_edt(~ice_mask)
    else:
        dist_px = np.ones(ice_mask.shape) * 9999
    ice_prox_score = np.exp(-dist_px / 200.0)   # 200 px = 2 km half-decay

    # 4. Radar confidence score
    conf_score = np.clip(ice_conf, 0.0, 1.0)

    # Composite
    composite = (0.30 * slope_score +
                 0.25 * illum_score +
                 0.25 * ice_prox_score +
                 0.20 * conf_score)

    # Hard exclusion: steep terrain
    composite[slope > SLOPE_MAX] = 0.0

    return composite, dict(slope=slope_score, illum=illum_score,
                           ice_prox=ice_prox_score, conf=conf_score)


def find_top_sites(composite, n=5):
    from scipy.ndimage import maximum_filter, label as nd_label

    # Find local maxima: pixel must be the maximum within a MIN_SEP_PX window
    local_max = maximum_filter(composite, size=MIN_SEP_PX)
    candidates = np.argwhere((composite == local_max) & (composite >= MIN_SCORE))
    if len(candidates) == 0:
        return []

    scores = composite[candidates[:, 0], candidates[:, 1]]
    order  = np.argsort(scores)[::-1]
    candidates = candidates[order]
    scores     = scores[order]

    # Enforce minimum separation greedily
    selected = []
    for idx, (r, c) in enumerate(candidates):
        too_close = False
        for sr, sc in selected:
            if np.hypot(r - sr, c - sc) < MIN_SEP_PX:
                too_close = True; break
        if not too_close:
            selected.append((r, c))
        if len(selected) >= n:
            break

    return [(r, c, float(composite[r, c])) for r, c in selected]


def pixel_to_latlon(r, c, gs, ps_m):
    lat = CENTER_LAT + (gs/2.0 - r) * ps_m / M_PER_DEG
    lon = CENTER_LON + (c - gs/2.0) * ps_m / (M_PER_DEG * np.cos(np.radians(CENTER_LAT)))
    return float(lat), float(lon)


def make_figure(d, composite, sub_scores, sites):
    gs = d["gs"]; ps_m = d["ps"]
    KM  = ps_m / 1000.0
    ext = [0, gs*KM, gs*KM, 0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#080810")
    fig.suptitle(
        "Landing Site Ranking  |  MCDA 4-Criterion Score  |  Faustini PSR\n"
        "Weights: 30% slope · 25% illumination · 25% ice proximity · 20% radar confidence",
        color="white", fontsize=11, fontweight="bold")

    site_colors = ["#ff3333","#ff8800","#ffff00","#00ff88","#00aaff"]
    site_markers= ["*", "D", "^", "s", "P"]

    for ax in axes:
        ax.set_facecolor("#0d0d1a")
        for sp in ax.spines.values(): sp.set_color("#334")
        ax.tick_params(colors="#888", labelsize=7)
        ax.add_patch(plt.Circle(
            (d["dsc_c"][1]*KM, d["dsc_c"][0]*KM), d["dsc_r"]*KM,
            color="white", fill=False, lw=1.2, ls="--"))
        ax.contour(np.linspace(0, gs*KM, gs), np.linspace(0, gs*KM, gs),
                   d["psr"].astype(float),
                   levels=[0.5], colors=["#555"], linewidths=[0.7], linestyles=["--"])

    # Panel 1: Composite score map
    ax = axes[0]
    im = ax.imshow(composite, cmap="RdYlGn", vmin=0, vmax=1,
                   extent=ext, origin="upper", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(
        colors="white", labelsize=6)
    for rank, (r, c, sc) in enumerate(sites):
        ax.plot(c*KM, r*KM, site_markers[rank], color=site_colors[rank],
                ms=14 - rank, mew=1.5, mec="white",
                label=f"#{rank+1}  score={sc:.3f}")
    ax.legend(fontsize=7.5, facecolor="#111", edgecolor="#445",
              labelcolor="white", loc="lower right")
    ax.set_title("(a) Composite Landing Score", color="white",
                 fontsize=9, fontweight="bold")
    ax.set_xlabel("East (km)", color="#aaa", fontsize=8)
    ax.set_ylabel("South (km)", color="#aaa", fontsize=8)

    # Panel 2: Ice proximity score
    ax = axes[1]
    im2 = ax.imshow(sub_scores["ice_prox"], cmap="plasma", vmin=0, vmax=1,
                    extent=ext, origin="upper", aspect="auto")
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(
        colors="white", labelsize=6)
    # Ice pixels overlay
    ice_disp = np.zeros((*d["ice_mask"].shape, 4), dtype=np.float32)
    ice_disp[d["ice_mask"]] = [0.0, 0.9, 0.55, 0.85]
    ax.imshow(ice_disp, extent=ext, origin="upper", aspect="auto")
    for rank, (r, c, sc) in enumerate(sites):
        ax.plot(c*KM, r*KM, site_markers[rank], color=site_colors[rank],
                ms=12 - rank, mew=1.5, mec="white")
    ax.set_title("(b) Ice Proximity Score\n(cyan = detected ice)",
                 color="white", fontsize=9, fontweight="bold")
    ax.set_xlabel("East (km)", color="#aaa", fontsize=8)
    ax.set_ylabel("South (km)", color="#aaa", fontsize=8)

    # Panel 3: Table
    ax = axes[2]
    ax.axis("off")

    if sites:
        col_labels = ["Rank", "Score", "Slope", "Illum", "Ice prox", "Conf",
                      "Lat", "Lon"]
        rows_data  = []
        for rank, (r, c, sc) in enumerate(sites):
            lat, lon = pixel_to_latlon(r, c, d["gs"], d["ps"])
            rows_data.append([
                f"#{rank+1}",
                f"{sc:.3f}",
                f"{d['slope'][r,c]:.1f}°",
                f"{d['illum'][r,c]*100:.0f}%",
                f"{sub_scores['ice_prox'][r,c]:.3f}",
                f"{d['ice_conf'][r,c]:.3f}",
                f"{lat:.4f}°",
                f"{lon:.4f}°",
            ])

        tbl = ax.table(cellText=rows_data, colLabels=col_labels,
                       loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
        tbl.scale(1.0, 2.0)
        row_colors = ["#0d1622","#111a2a"]
        for (row, col), cell in tbl.get_celld().items():
            bg = "#1a2040" if row == 0 else row_colors[(row-1) % 2]
            cell.set_facecolor(bg)
            cell.set_text_props(color=site_colors[row-1] if row > 0 else "white")
            cell.set_edgecolor("#2a3a55")
    else:
        ax.text(0.5, 0.5, "No candidate sites found\n(check slope/score thresholds)",
                ha="center", va="center", color="white", fontsize=10,
                transform=ax.transAxes)

    ax.set_title("(c) Top-5 Landing Sites", color="white",
                 fontsize=9, fontweight="bold", pad=10)

    fig.tight_layout()
    out = Path("results/figures/13_landing_sites.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Sites] Saved PNG: {out}")


def write_csv(d, sub_scores, sites):
    out = Path("results/export/landing_sites.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank","score","slope_deg","illumination_frac",
                    "ice_prox_score","confidence","lat_deg","lon_deg",
                    "pixel_row","pixel_col"])
        for rank, (r, c, sc) in enumerate(sites):
            lat, lon = pixel_to_latlon(r, c, d["gs"], d["ps"])
            w.writerow([rank+1, f"{sc:.4f}", f"{d['slope'][r,c]:.2f}",
                        f"{d['illum'][r,c]:.4f}",
                        f"{sub_scores['ice_prox'][r,c]:.4f}",
                        f"{d['ice_conf'][r,c]:.4f}",
                        f"{lat:.6f}", f"{lon:.6f}", r, c])
    print(f"[Sites] Saved CSV: {out}")


def write_geojson(d, sub_scores, sites):
    features = []
    for rank, (r, c, sc) in enumerate(sites):
        lat, lon = pixel_to_latlon(r, c, d["gs"], d["ps"])
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "rank":            rank + 1,
                "composite_score": round(sc, 4),
                "slope_deg":       round(float(d["slope"][r, c]), 2),
                "illumination":    round(float(d["illum"][r, c]), 4),
                "ice_prox_score":  round(float(sub_scores["ice_prox"][r, c]), 4),
                "radar_confidence":round(float(d["ice_conf"][r, c]), 4),
                "lat":             round(lat, 6),
                "lon":             round(lon, 6),
            }
        })
    out = Path("results/export/landing_sites.geojson")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "type": "FeatureCollection",
        "name": "Top Landing Sites",
        "crs": {"type":"name","properties":{"name":"EPSG:4326"}},
        "features": features,
    }, indent=2), encoding="utf-8")
    print(f"[Sites] Saved GeoJSON: {out}")


if __name__ == "__main__":
    print("[Sites] Loading data...")
    d = load_data()
    print("[Sites] Computing score map...")
    composite, sub_scores = score_map(d)
    print("[Sites] Finding top-5 landing sites...")
    sites = find_top_sites(composite, n=5)

    print(f"\n[Sites] Top-{len(sites)} candidate landing sites:")
    for rank, (r, c, sc) in enumerate(sites):
        lat, lon = pixel_to_latlon(r, c, d["gs"], d["ps"])
        print(f"  #{rank+1}  score={sc:.3f}  slope={d['slope'][r,c]:.1f}deg  "
              f"illum={d['illum'][r,c]*100:.0f}%  lat={lat:.4f}  lon={lon:.4f}")

    make_figure(d, composite, sub_scores, sites)
    write_csv(d, sub_scores, sites)
    write_geojson(d, sub_scores, sites)
