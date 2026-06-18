"""
L-band vs S-band Detection Comparison
======================================
Side-by-side comparison of L-band (CPR > 0.8, ~4.5 m penetration) vs
S-band (CPR > 1.0, ~1.2 m penetration) ice detection thresholds within
the Faustini Doubly Shadowed Crater.

Output: results/figures/10_lband_sband_comparison.png
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

# Band parameters
BANDS = {
    "L-band": dict(
        freq_GHz=1.25,
        cpr_thresh=0.8,
        color_ice="#00e5ff",       # cyan
        color_bg="#1a2a2a",
        pen_depth_m=4.5,           # L-band dry-regolith penetration
        label="DFSAR L-band\n1.25 GHz | CPR > 0.8\nPenetration: ~4.5 m",
    ),
    "S-band": dict(
        freq_GHz=2.40,
        cpr_thresh=1.0,
        color_ice="#ff6b35",       # orange
        color_bg="#2a1a1a",
        pen_depth_m=1.2,           # S-band (estimated)
        label="Simulated S-band\n2.40 GHz | CPR > 1.0\nPenetration: ~1.2 m",
    ),
}


def simulate_sband_cpr(cpr_l, pen_l=4.5, pen_s=1.2):
    """
    Approximate S-band CPR from L-band CPR.
    Shallower penetration -> weaker ice signal; scale relative to depth ratio.
    Add noise representative of different frequency scattering.
    """
    rng = np.random.default_rng(42)
    ratio = pen_s / pen_l
    cpr_s = cpr_l * ratio + rng.normal(0, 0.07, cpr_l.shape)
    return np.clip(cpr_s, 0, 3.5).astype(np.float32)


def load_data():
    from src.data_generator import load_all
    from src.dfsar_analysis import run_analysis

    data = load_all(cache=True)
    dem = data["dem"]; slope = data["slope"]
    psr = data["psr_mask"]; dsc = data["dsc_mask"]
    dfsar = data["dfsar"]; meta = data["meta"]
    gs = meta["grid_size"]; ps = meta["pixel_scale"]
    dsc_c = meta["dsc_center"]; dsc_r = meta["dsc_radius"]

    res = run_analysis(dfsar, psr, slope, cpr_thresh=0.8, dop_thresh=0.13)
    CPR_L = res["CPR"].astype(np.float32)
    DOP   = res["DOP"].astype(np.float32)
    CPR_S = simulate_sband_cpr(CPR_L)

    return dict(dem=dem, psr=psr, dsc=dsc,
                CPR_L=CPR_L, CPR_S=CPR_S, DOP=DOP,
                dsc_c=dsc_c, dsc_r=dsc_r, gs=gs, ps=ps)


def detection_stats(cpr, thresh, psr, dsc):
    mask_all = cpr > thresh
    mask_psr = mask_all & psr
    mask_dsc = mask_all & dsc
    return dict(
        total=int(mask_all.sum()),
        psr=int(mask_psr.sum()),
        dsc=int(mask_dsc.sum()),
        frac_scene=float(mask_all.mean() * 100),
        frac_psr=float(mask_psr.sum() / max(psr.sum(), 1) * 100),
        mask=mask_all,
    )


def make_figure(d):
    gs_n = d["gs"]; ps = d["ps"]; KM = ps / 1000.0

    CPR_L = d["CPR_L"]; CPR_S = d["CPR_S"]
    psr = d["psr"]; dsc_mask = d["dsc"]
    dsc_c = d["dsc_c"]; dsc_r = d["dsc_r"]

    stats = {}
    for band, cfg in BANDS.items():
        cpr = CPR_L if band == "L-band" else CPR_S
        stats[band] = detection_stats(cpr, cfg["cpr_thresh"], psr, dsc_mask)

    # Agreement / disagreement masks
    agree_ice    = stats["L-band"]["mask"] & stats["S-band"]["mask"]
    lband_only   = stats["L-band"]["mask"] & ~stats["S-band"]["mask"]   # deep ice
    sband_only   = stats["S-band"]["mask"] & ~stats["L-band"]["mask"]   # surface roughness
    neither      = ~stats["L-band"]["mask"] & ~stats["S-band"]["mask"]

    ext = [0, gs_n * KM, gs_n * KM, 0]

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 13))
    fig.patch.set_facecolor("#080810")
    gs_fig = gridspec.GridSpec(3, 4, figure=fig,
                               height_ratios=[2.5, 2.5, 1.0],
                               hspace=0.40, wspace=0.35)

    def styled_ax(ax):
        ax.set_facecolor("#0d0d1a")
        for sp in ax.spines.values(): sp.set_color("#334")
        ax.tick_params(colors="#888", labelsize=7)
        return ax

    # ── Row 0: CPR maps ──────────────────────────────────────────────────────
    for col, (band, cfg) in enumerate(BANDS.items()):
        ax = styled_ax(fig.add_subplot(gs_fig[0, col]))
        cpr = CPR_L if band == "L-band" else CPR_S
        im = ax.imshow(cpr, cmap="inferno", vmin=0, vmax=2.5,
                       extent=ext, origin="upper", aspect="auto")
        # Ice threshold contour
        x_km = np.linspace(0, gs_n * KM, gs_n)
        y_km = np.linspace(0, gs_n * KM, gs_n)
        ax.contour(x_km, y_km, cpr, levels=[cfg["cpr_thresh"]],
                   colors=[cfg["color_ice"]], linewidths=[1.2])
        # PSR boundary
        ax.contour(x_km, y_km, psr.astype(float), levels=[0.5],
                   colors=["white"], linewidths=[0.7], linestyles=["--"])
        # DSC circle
        ax.add_patch(plt.Circle((dsc_c[1]*KM, dsc_c[0]*KM), dsc_r*KM,
                                 color="yellow", fill=False, lw=1.5, ls="--"))
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("CPR", color="white", fontsize=7)
        cb.ax.tick_params(colors="white", labelsize=6)
        ax.set_title(f"({['a','b'][col]}) {band}  CPR Map\n{cfg['label']}",
                     color="white", fontsize=8.5, fontweight="bold")
        ax.set_xlabel("Distance East (km)", color="#aaa", fontsize=7)
        ax.set_ylabel("Distance South (km)", color="#aaa", fontsize=7)

    # ── Row 0: Agreement map ─────────────────────────────────────────────────
    ax_agree = styled_ax(fig.add_subplot(gs_fig[0, 2:]))
    composite = np.zeros((*CPR_L.shape, 3), dtype=np.float32)
    composite[neither]    = [0.10, 0.10, 0.16]   # dark
    composite[lband_only] = [0.00, 0.90, 1.00]   # cyan (L-band deep ice)
    composite[sband_only] = [1.00, 0.42, 0.21]   # orange (S-band surface)
    composite[agree_ice]  = [0.00, 0.95, 0.30]   # green (both)
    ax_agree.imshow(composite, extent=ext, origin="upper", aspect="auto")
    ax_agree.contour(x_km, y_km, psr.astype(float), levels=[0.5],
                     colors=["white"], linewidths=[0.7], linestyles=["--"])
    ax_agree.add_patch(plt.Circle((dsc_c[1]*KM, dsc_c[0]*KM), dsc_r*KM,
                                   color="yellow", fill=False, lw=1.5, ls="--"))
    patches = [
        mpatches.Patch(color=[0.00, 0.90, 1.00], label=f"L-band only  ({lband_only.sum()} px) — deep ice"),
        mpatches.Patch(color=[1.00, 0.42, 0.21], label=f"S-band only  ({sband_only.sum()} px) — surface roughness"),
        mpatches.Patch(color=[0.00, 0.95, 0.30], label=f"Both bands   ({agree_ice.sum()} px) — confirmed ice"),
        mpatches.Patch(color=[0.10, 0.10, 0.16], label="No detection"),
    ]
    ax_agree.legend(handles=patches, loc="upper left", fontsize=7,
                    framealpha=0.85, facecolor="#111", edgecolor="#445")
    ax_agree.set_title("(c) Band-Agreement Map\nL-band deep ice vs S-band surface detection",
                        color="white", fontsize=8.5, fontweight="bold")
    ax_agree.set_xlabel("Distance East (km)", color="#aaa", fontsize=7)
    ax_agree.set_ylabel("Distance South (km)", color="#aaa", fontsize=7)

    # ── Row 1: CPR histograms ────────────────────────────────────────────────
    hist_axes = [styled_ax(fig.add_subplot(gs_fig[1, c])) for c in range(2)]
    for col, (band, cfg) in enumerate(BANDS.items()):
        ax = hist_axes[col]
        cpr = CPR_L if band == "L-band" else CPR_S
        # All pixels
        ax.hist(cpr.ravel(), bins=80, range=(0, 3), color="#445577",
                alpha=0.6, label="All pixels", density=True)
        # PSR pixels
        cpr_psr = cpr[psr]
        if cpr_psr.size:
            ax.hist(cpr_psr, bins=80, range=(0, 3), color="#8899cc",
                    alpha=0.8, label="PSR pixels", density=True)
        # DSC pixels
        cpr_dsc = cpr[dsc_mask]
        if cpr_dsc.size:
            ax.hist(cpr_dsc, bins=40, range=(0, 3), color=cfg["color_ice"],
                    alpha=0.9, label="DSC floor", density=True)
        ax.axvline(cfg["cpr_thresh"], color="red", lw=1.5, ls="--",
                   label=f"Threshold = {cfg['cpr_thresh']}")
        ax.set_title(f"({['d','e'][col]}) {band} CPR Distribution",
                     color="white", fontsize=8.5, fontweight="bold")
        ax.set_xlabel("CPR", color="#aaa", fontsize=8)
        ax.set_ylabel("Density", color="#aaa", fontsize=8)
        ax.legend(fontsize=7, framealpha=0.7, facecolor="#111", edgecolor="#334")

    # Row 1 col 2: scatter CPR_L vs CPR_S
    ax_scat = styled_ax(fig.add_subplot(gs_fig[1, 2]))
    subsamp = slice(None, None, 5)  # every 5th pixel for speed
    cl = CPR_L.ravel()[::5]; cs = CPR_S.ravel()[::5]
    ax_scat.hexbin(cl, cs, gridsize=60, cmap="hot", mincnt=1,
                   extent=(0, 3, 0, 3), bins="log")
    ax_scat.axvline(BANDS["L-band"]["cpr_thresh"], color="cyan", lw=1, ls="--")
    ax_scat.axhline(BANDS["S-band"]["cpr_thresh"], color="orange", lw=1, ls="--")
    ax_scat.plot([0, 3], [0, 3], color="#555", lw=0.7)   # 1:1 line
    ax_scat.set_xlabel("L-band CPR", color="#aaa", fontsize=8)
    ax_scat.set_ylabel("S-band CPR", color="#aaa", fontsize=8)
    ax_scat.set_title("(f) L-band vs S-band CPR\nHexbin density",
                       color="white", fontsize=8.5, fontweight="bold")

    # Row 1 col 3: detection fraction vs threshold with ±1σ uncertainty bands
    # Bands derived from CPR noise Monte Carlo (σ_CPR = 0.10 per pixel)
    ax_roc = styled_ax(fig.add_subplot(gs_fig[1, 3]))
    thresh_range = np.linspace(0.3, 2.5, 200)
    rng_unc = np.random.default_rng(123)
    N_BOOT  = 50   # bootstrap resamples for ±1σ band

    for band, cfg in BANDS.items():
        cpr_base = CPR_L if band == "L-band" else CPR_S
        cpr_psr  = cpr_base[psr]

        # Central estimate
        frac_central = np.array([(cpr_psr > t).mean() * 100 for t in thresh_range])

        # Bootstrap: add radiometric noise to CPR each replicate
        cpr_noise_std = 0.10 if band == "L-band" else 0.10
        boot_fracs = np.zeros((N_BOOT, len(thresh_range)))
        for b in range(N_BOOT):
            noisy = cpr_psr + rng_unc.normal(0, cpr_noise_std, cpr_psr.shape)
            noisy = np.clip(noisy, 0, 3.5)
            boot_fracs[b] = [(noisy > t).mean() * 100 for t in thresh_range]

        sigma_lo = frac_central - boot_fracs.std(axis=0)
        sigma_hi = frac_central + boot_fracs.std(axis=0)

        ax_roc.fill_between(thresh_range, sigma_lo, sigma_hi,
                            color=cfg["color_ice"], alpha=0.20)
        ax_roc.plot(thresh_range, frac_central,
                    color=cfg["color_ice"], lw=1.8, label=f"{band} (±1σ CPR noise)")
        ax_roc.axvline(cfg["cpr_thresh"], color=cfg["color_ice"],
                       lw=1.0, ls="--", alpha=0.7)

    ax_roc.set_xlabel("CPR threshold", color="#aaa", fontsize=8)
    ax_roc.set_ylabel("% PSR pixels detected", color="#aaa", fontsize=8)
    ax_roc.set_title("(g) Detection Fraction vs Threshold\n±1σ from CPR radiometric noise (σ=0.10)",
                      color="white", fontsize=8.5, fontweight="bold")
    ax_roc.legend(fontsize=8, facecolor="#111", edgecolor="#334")

    # ── Row 2: Statistics table ──────────────────────────────────────────────
    ax_tbl = styled_ax(fig.add_subplot(gs_fig[2, :]))
    ax_tbl.axis("off")

    rows_data = []
    col_headers = ["Band", "Freq (GHz)", "CPR thresh", "Penetration",
                   "All detections", "PSR detections", "DSC detections",
                   "% of scene", "% of PSR"]
    for band, cfg in BANDS.items():
        st = stats[band]
        rows_data.append([
            band,
            f"{cfg['freq_GHz']:.2f}",
            f"{cfg['cpr_thresh']:.1f}",
            f"{cfg['pen_depth_m']:.1f} m",
            str(st["total"]),
            str(st["psr"]),
            str(st["dsc"]),
            f"{st['frac_scene']:.3f}%",
            f"{st['frac_psr']:.2f}%",
        ])
    # Agreement row
    rows_data.append([
        "Agreement",
        "—", "—", "—",
        str(agree_ice.sum()),
        str((agree_ice & psr).sum()),
        str((agree_ice & dsc_mask).sum()),
        f"{agree_ice.mean()*100:.3f}%",
        f"{(agree_ice & psr).sum() / max(psr.sum(),1)*100:.2f}%",
    ])

    tbl = ax_tbl.table(cellText=rows_data, colLabels=col_headers,
                        loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.8)

    header_color = "#1a2a4a"
    row_colors   = ["#0d1520", "#111827", "#141e30"]
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor(header_color)
        elif row <= 2:
            cell.set_facecolor(row_colors[row - 1])
        else:
            cell.set_facecolor("#1a1a2a")  # agreement row
        cell.set_text_props(color="white")
        cell.set_edgecolor("#334455")

    ax_tbl.set_title("(h) Band Comparison Statistics", color="white",
                      fontsize=9, fontweight="bold", pad=6)

    # ── Suptitle ──────────────────────────────────────────────────────────────
    fig.suptitle(
        "L-band vs S-band Radar Detection Comparison  |  "
        "Chandrayaan-2 DFSAR  |  Faustini Doubly Shadowed Crater\n"
        "L-band (1.25 GHz, 4.5 m penetration)  vs  Simulated S-band (2.40 GHz, 1.2 m penetration)",
        color="white", fontsize=11, fontweight="bold")

    out = Path("results/figures/10_lband_sband_comparison.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[L/S-band] Saved: {out}")


if __name__ == "__main__":
    print("[L/S-band] Loading data...")
    d = load_data()
    make_figure(d)
