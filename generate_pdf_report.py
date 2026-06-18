"""
PDF Science Report Generator
=============================
Produces a 4-page A4-landscape PDF report with key figures and statistics.

Output: results/science_report.pdf
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

PAGE = (16.53, 11.69)   # A4 landscape inches
DARK  = "#080810"
PANEL = "#0d0d1a"


def load_data():
    from src.data_generator import load_all
    from src.dfsar_analysis import run_analysis
    from src.ice_volume     import estimate_ice_volume, monte_carlo_uncertainty, \
                                    compute_penetration_depth_map
    from src.thermal_model  import compute_surface_temperature

    data  = load_all(cache=True)
    dem   = data["dem"]; slope = data["slope"]; illum = data["illum"]
    psr   = data["psr_mask"]; dsc = data["dsc_mask"]
    dfsar = data["dfsar"]; meta = data["meta"]
    gs    = meta["grid_size"]; ps_m = meta["pixel_scale"]
    dsc_c = meta["dsc_center"]; dsc_r = meta["dsc_radius"]

    res       = run_analysis(dfsar, psr, slope)
    T_surface = compute_surface_temperature(illum)
    iv        = estimate_ice_volume(res["CPR"], res["ice_mask"], pixel_scale=ps_m)
    unc       = monte_carlo_uncertainty(res["CPR"], res["ice_mask"],
                                        pixel_scale=ps_m, n_samples=300)
    depth_map = compute_penetration_depth_map(res["CPR"])

    return dict(dem=dem, slope=slope, illum=illum, T_surface=T_surface,
                psr=psr, dsc=dsc, CPR=res["CPR"], DOP=res["DOP"],
                ice_mask=res["ice_mask"], ice_conf=res["ice_conf"],
                mchi=res.get("mchi"), iv=iv, unc=unc, depth_map=depth_map,
                dsc_c=dsc_c, dsc_r=dsc_r, gs=gs, ps=ps_m)


def _ext(gs, ps): return [0, gs*ps/1000, gs*ps/1000, 0]
def _km(v, ps): return v * ps / 1000


def _dsc(ax, dsc_c, dsc_r, ps, color="yellow"):
    ax.add_patch(plt.Circle((_km(dsc_c[1], ps), _km(dsc_c[0], ps)), _km(dsc_r, ps),
                             color=color, fill=False, lw=1.3, ls="--"))


def _cax(fig, ax):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    div = make_axes_locatable(ax)
    return div.append_axes("right", size="4%", pad=0.05)


def page1_cover(pdf, d):
    """Title page with mission context."""
    import datetime
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor(DARK)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(DARK)
    ax.axis("off")

    # Title block
    fig.text(0.5, 0.76, "Detection and Characterisation of Subsurface Ice",
             ha="center", color="#88aaff", fontsize=28, fontweight="bold")
    fig.text(0.5, 0.69, "in Lunar South Polar Regions",
             ha="center", color="#88aaff", fontsize=28, fontweight="bold")
    fig.text(0.5, 0.61, "Using Chandrayaan-2 DFSAR Compact Polarimetry",
             ha="center", color="#aabbdd", fontsize=18)

    fig.add_artist(plt.Line2D([0.1, 0.9], [0.58, 0.58], color="#334488",
                               lw=2, transform=fig.transFigure))

    # Key parameters box
    params = [
        ("Instrument",      "Chandrayaan-2 DFSAR (Dual-Frequency SAR)"),
        ("Frequency",       "L-band 1.25 GHz (S-band 2.50 GHz)"),
        ("Polarimetry",     "Compact polarimetry — full Stokes reconstruction"),
        ("Ice criterion",   "CPR > 0.8 AND DOP < 0.13 (dual threshold)"),
        ("Target",          "Faustini Doubly Shadowed Crater (PSR), 87°S"),
        ("L-band pen. depth","~4.5 m in dry regolith"),
        ("Analysis",        "m-chi decomposition · Bayesian posterior · Monte Carlo"),
        ("Scene resolution","10 m/pixel  |  10 km × 10 km coverage"),
    ]
    y0 = 0.50
    for label, val in params:
        fig.text(0.22, y0, f"{label}:", color="#7799bb", fontsize=12, ha="right")
        fig.text(0.23, y0, val,         color="#ccddff", fontsize=12)
        y0 -= 0.055

    fig.add_artist(plt.Line2D([0.1, 0.9], [0.10, 0.10], color="#223344",
                               lw=1, transform=fig.transFigure))
    fig.text(0.5, 0.06,
             f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d')}  |  "
             "Chandrayaan-2 DFSAR Science Pipeline  |  Faustini PSR Subsurface Ice Detection",
             ha="center", color="#445566", fontsize=9)

    pdf.savefig(fig, facecolor=DARK); plt.close(fig)
    print("[PDF] Page 1 — cover")


def page2_overview(pdf, d):
    """Six-panel scene overview."""
    gs = d["gs"]; ps = d["ps"]; ext = _ext(gs, ps)
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor(DARK)
    fig.suptitle("Scene Overview — Chandrayaan-2 DFSAR | Faustini PSR",
                 color="white", fontsize=14, fontweight="bold", y=0.98)

    products = [
        (d["CPR"],       "inferno",  0,    2.5,  "CPR (SC/OC)",             "CPR"),
        (d["DOP"],       "RdYlBu",   0,    1.0,  "DOP",                     "DOP"),
        (d["ice_conf"],  "plasma",   0,    1.0,  "P(ice|CPR,DOP,PSR)",      "P(ice)"),
        (d["dem"],       "terrain",  None, None, "Elevation (m)",            "DEM"),
        (d["T_surface"], "RdBu_r",   40,   110,  "Temperature (K)",          "Temp"),
        (d["ice_mask"].astype(float),"RdYlGn", 0, 1, "Ice detection mask",  "Ice"),
    ]

    axes = []
    for i in range(6):
        axes.append(fig.add_subplot(2, 3, i+1))

    for ax, (arr, cmap, vmin, vmax, label, short) in zip(axes, products):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_color("#334")
        vn = vmin if vmin is not None else np.nanmin(arr)
        vx = vmax if vmax is not None else np.nanmax(arr)
        im = ax.imshow(arr, cmap=cmap, vmin=vn, vmax=vx,
                       extent=ext, origin="upper", aspect="auto")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(label, color="white", fontsize=7)
        cb.ax.tick_params(colors="white", labelsize=6)
        _dsc(ax, d["dsc_c"], d["dsc_r"], ps)
        ax.set_title(short, color="white", fontsize=9, fontweight="bold")
        ax.set_xlabel("East (km)", color="#aaa", fontsize=7)
        ax.set_ylabel("South (km)", color="#aaa", fontsize=7)
        ax.tick_params(colors="#888", labelsize=7)

    # CPR ice threshold contour on CPR panel
    axes[0].contour(d["CPR"], levels=[0.8], colors=["cyan"], linewidths=[0.9],
                    extent=ext, origin="upper")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, facecolor=DARK); plt.close(fig)
    print("[PDF] Page 2 — overview")


def page3_detection(pdf, d):
    """Ice detection + uncertainty analysis."""
    gs = d["gs"]; ps = d["ps"]; ext = _ext(gs, ps)
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor(DARK)
    fig.suptitle("Ice Detection & Uncertainty Quantification",
                 color="white", fontsize=14, fontweight="bold", y=0.98)

    gs_fig = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    def sax(sp):
        ax = fig.add_subplot(sp)
        ax.set_facecolor(PANEL)
        for s in ax.spines.values(): s.set_color("#334")
        ax.tick_params(colors="#888", labelsize=7)
        return ax

    # Ice mask
    ax = sax(gs_fig[0, 0])
    bg = np.full((*d["ice_mask"].shape, 3), [0.10, 0.10, 0.15])
    bg[d["ice_mask"]] = [0.0, 0.9, 0.55]
    ax.imshow(bg, extent=ext, origin="upper", aspect="auto")
    _dsc(ax, d["dsc_c"], d["dsc_r"], ps)
    n = int(d["ice_mask"].sum())
    ax.text(0.02, 0.02, f"N={n} ice pixels\n({n*100/d['ice_mask'].size:.3f}%)",
            transform=ax.transAxes, color="#00e5aa", fontsize=8,
            bbox=dict(fc="#111", ec="#00e5aa", pad=2))
    ax.set_title("Ice Detection Mask", color="white", fontsize=9, fontweight="bold")
    ax.set_xlabel("East (km)", color="#aaa", fontsize=7)
    ax.set_ylabel("South (km)", color="#aaa", fontsize=7)

    # Ice confidence
    ax = sax(gs_fig[0, 1])
    im = ax.imshow(np.where(d["psr"], d["ice_conf"], 0), cmap="plasma",
                   vmin=0, vmax=1, extent=ext, origin="upper", aspect="auto")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(colors="white", labelsize=6)
    _dsc(ax, d["dsc_c"], d["dsc_r"], ps)
    ax.set_title("Bayesian P(ice) Posterior", color="white", fontsize=9, fontweight="bold")
    ax.set_xlabel("East (km)", color="#aaa", fontsize=7)
    ax.set_ylabel("South (km)", color="#aaa", fontsize=7)

    # m-chi volume if available
    ax = sax(gs_fig[0, 2])
    if d.get("mchi"):
        im2 = ax.imshow(d["mchi"]["frac_volume"], cmap="Greens", vmin=0, vmax=1,
                        extent=ext, origin="upper", aspect="auto")
        cb2 = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.03)
        cb2.ax.tick_params(colors="white", labelsize=6)
        ax.contour(d["ice_mask"].astype(float), levels=[0.5], colors=["cyan"],
                   linewidths=[0.8], extent=ext, origin="upper")
        _dsc(ax, d["dsc_c"], d["dsc_r"], ps)
        ax.set_title("m-chi Volume Scatter Pv", color="white", fontsize=9, fontweight="bold")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "m-chi not available", transform=ax.transAxes,
                ha="center", color="white")
    ax.set_xlabel("East (km)", color="#aaa", fontsize=7)
    ax.set_ylabel("South (km)", color="#aaa", fontsize=7)

    # Monte Carlo PDF
    unc = d["unc"]
    samples_t = unc["samples"] * 917 / 1000
    p5  = unc["p5_m3"]  * 917 / 1000
    p50 = unc["p50_m3"] * 917 / 1000
    p95 = unc["p95_m3"] * 917 / 1000

    ax = sax(gs_fig[1, 0])
    ax.hist(samples_t, bins=40, color="#3355aa", edgecolor="#6688cc",
            density=True, alpha=0.85)
    for q, col, lbl in [(p5,"#ffff00","P5"),(p50,"#ff8800","P50"),(p95,"#ff3333","P95")]:
        ax.axvline(q, color=col, lw=1.5, ls="--", label=f"{lbl}={q:.3f} t")
    ax.set_xlabel("Water-equiv. (t)", color="#aaa", fontsize=8)
    ax.set_ylabel("Density", color="#aaa", fontsize=8)
    ax.legend(fontsize=7, facecolor="#111", edgecolor="#334", labelcolor="white")
    ax.set_title("Ice Volume PDF (N=300)", color="white", fontsize=9, fontweight="bold")

    # CDF
    ax = sax(gs_fig[1, 1])
    sorted_t = np.sort(samples_t)
    ax.plot(sorted_t, np.linspace(0, 100, len(sorted_t)),
            color="#4488ee", lw=2.0)
    for q, col, lbl in [(p5,"#ffff00","P5"),(p50,"#ff8800","P50"),(p95,"#ff3333","P95")]:
        ax.axvline(q, color=col, lw=1.2, ls="--")
        ax.text(q, 2, lbl, color=col, fontsize=7, ha="center")
    ax.set_xlabel("Water-equiv. (t)", color="#aaa", fontsize=8)
    ax.set_ylabel("CDF (%)", color="#aaa", fontsize=8)
    ax.set_title("Ice Volume CDF", color="white", fontsize=9, fontweight="bold")

    # CPR histogram
    ax = sax(gs_fig[1, 2])
    bins = dict(range=(0, 3), bins=70, density=True)
    ax.hist(d["CPR"].ravel(), **bins, color="#334466", alpha=0.5, label="All")
    cpr_psr = d["CPR"][d["psr"]]
    if cpr_psr.size:
        ax.hist(cpr_psr, **bins, color="#6688cc", alpha=0.8, label="PSR")
    cpr_dsc = d["CPR"][d["dsc"]]
    if cpr_dsc.size:
        ax.hist(cpr_dsc, **{**bins,"bins":40}, color="#00e5ff", alpha=0.9, label="DSC")
    ax.axvline(0.8, color="red", lw=1.5, ls="--", label="CPR=0.8")
    ax.set_xlabel("CPR", color="#aaa", fontsize=8)
    ax.set_ylabel("Density", color="#aaa", fontsize=8)
    ax.legend(fontsize=7, facecolor="#111", edgecolor="#334", labelcolor="white")
    ax.set_title("CPR Distribution by Zone", color="white", fontsize=9, fontweight="bold")

    pdf.savefig(fig, facecolor=DARK); plt.close(fig)
    print("[PDF] Page 3 — detection & uncertainty")


def page4_isru(pdf, d):
    """ISRU resource assessment summary page."""
    unc = d["unc"]; iv = d["iv"]; gs = d["gs"]; ps = d["ps"]
    ext = _ext(gs, ps)

    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor(DARK)
    fig.suptitle("ISRU Resource Assessment & Mission Summary",
                 color="white", fontsize=14, fontweight="bold", y=0.98)

    gs_fig = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.36)

    def sax(sp):
        ax = fig.add_subplot(sp)
        ax.set_facecolor(PANEL)
        for s in ax.spines.values(): s.set_color("#334")
        ax.tick_params(colors="#888", labelsize=7)
        return ax

    # Depth map
    ax = sax(gs_fig[0, 0])
    depth_disp = np.where(d["ice_mask"], d["depth_map"], np.nan)
    im = ax.imshow(depth_disp, cmap="magma_r", vmin=0, vmax=5,
                   extent=ext, origin="upper", aspect="auto")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Depth (m)", color="white", fontsize=7)
    cb.ax.tick_params(colors="white", labelsize=6)
    _dsc(ax, d["dsc_c"], d["dsc_r"], ps)
    ax.set_title("Radar Penetration Depth", color="white", fontsize=9, fontweight="bold")
    ax.set_xlabel("East (km)", color="#aaa", fontsize=7)
    ax.set_ylabel("South (km)", color="#aaa", fontsize=7)

    # Ice fraction
    ax = sax(gs_fig[0, 1])
    f_disp = np.where(d["ice_mask"], d["iv"]["f_ice_map"], np.nan)
    im2 = ax.imshow(f_disp, cmap="YlGn", vmin=0, vmax=0.8,
                    extent=ext, origin="upper", aspect="auto")
    cb2 = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.03)
    cb2.set_label("f_ice", color="white", fontsize=7)
    cb2.ax.tick_params(colors="white", labelsize=6)
    _dsc(ax, d["dsc_c"], d["dsc_r"], ps)
    ax.set_title("Ice Volume Fraction", color="white", fontsize=9, fontweight="bold")
    ax.set_xlabel("East (km)", color="#aaa", fontsize=7)
    ax.set_ylabel("South (km)", color="#aaa", fontsize=7)

    # Propellant bar chart
    ax = sax(gs_fig[0, 2])
    qs = ["P5", "P25", "P50", "P75", "P95"]
    qv = [unc["p5_m3"], unc["p25_m3"], unc["p50_m3"],
          unc["p75_m3"], unc["p95_m3"]]
    h2 = [v * 917/1000 * 1000 * 0.112 * 0.83 / 1000 for v in qv]
    o2 = [v * 917/1000 * 1000 * 0.888 * 0.83 / 1000 for v in qv]
    x  = np.arange(len(qs))
    ax.bar(x - 0.2, h2, 0.35, color="#4af", label="H2")
    ax.bar(x + 0.2, o2, 0.35, color="#f84", label="O2")
    ax.set_xticks(x); ax.set_xticklabels(qs, fontsize=8, color="#aaa")
    ax.set_ylabel("Propellant mass (t)", color="#aaa", fontsize=8)
    ax.legend(fontsize=8, facecolor="#111", edgecolor="#334", labelcolor="white")
    ax.set_title("ISRU Propellant Potential\n(electrolysis 83%)",
                 color="white", fontsize=9, fontweight="bold")

    # Summary table
    ax_tbl = fig.add_subplot(gs_fig[1, :])
    ax_tbl.set_facecolor(PANEL)
    ax_tbl.axis("off")

    def electro(t): return t*1000*0.112*0.83/1000, t*1000*0.888*0.83/1000
    p5t  = unc["p5_m3"]  * 917/1000
    p50t = unc["p50_m3"] * 917/1000
    p95t = unc["p95_m3"] * 917/1000
    h2_50, o2_50 = electro(p50t)

    rows = [
        ["Ice detections",             f"{d['ice_mask'].sum()} pixels",
         f"({d['ice_mask'].sum()*100/d['ice_mask'].size:.4f}% of scene)"],
        ["Ice-bearing area",           f"{iv['ice_bearing_area_km2']:.5f} km²",     "within 10 km × 10 km scene"],
        ["Mean ice fraction (f_ice)",  f"{iv['mean_ice_concentration']*100:.1f}%",   "Polder-van Santen inversion"],
        ["P5 water-equiv. mass",       f"{p5t:.4f} t",                               "Monte Carlo 5th percentile"],
        ["P50 water-equiv. mass",      f"{p50t:.4f} t",                              "Monte Carlo median"],
        ["P95 water-equiv. mass",      f"{p95t:.4f} t",                              "Monte Carlo 95th percentile"],
        ["Relative uncertainty",       f"{unc['rel_unc_pct']:.1f}%",                 "CPR noise + depth + dielectric"],
        ["H2 propellant (P50)",        f"{h2_50:.5f} t",                             "electrolysis η=83%, mass ratio H:O = 1:8"],
        ["O2 propellant (P50)",        f"{o2_50:.5f} t",                             "electrolysis by-product"],
        ["Radar frequency",            "L-band 1.25 GHz",                            "Chandrayaan-2 DFSAR"],
        ["Penetration depth",          "~4.5 m",                                     "dry regolith loss tangent 0.01"],
    ]

    tbl = ax_tbl.table(
        cellText=[[r[1], r[2]] for r in rows],
        rowLabels=[r[0] for r in rows],
        colLabels=["Value", "Notes"],
        loc="center", cellLoc="left",
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1.1, 1.65)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#111827" if row % 2 == 0 else "#0d1420")
        cell.set_text_props(color="white")
        cell.set_edgecolor("#2a3a55")
    ax_tbl.set_title("Mission Key Results", color="white",
                     fontsize=11, fontweight="bold", pad=10)

    pdf.savefig(fig, facecolor=DARK); plt.close(fig)
    print("[PDF] Page 4 — ISRU & summary")


def main():
    print("[PDF] Loading data...")
    d = load_data()

    out = Path("results/science_report.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(out)) as pdf:
        page1_cover(pdf, d)
        page2_overview(pdf, d)
        page3_detection(pdf, d)
        page4_isru(pdf, d)

        info = pdf.infodict()
        info["Title"]   = "Subsurface Ice Detection — Lunar South Polar"
        info["Subject"] = "Chandrayaan-2 DFSAR Compact Polarimetry Analysis"
        info["Keywords"]= "lunar ice, DFSAR, CPR, PSR, Faustini, ISRU"

    size_mb = out.stat().st_size / 1e6
    print(f"[PDF] Saved: {out}  ({size_mb:.1f} MB,  4 pages)")


if __name__ == "__main__":
    main()
