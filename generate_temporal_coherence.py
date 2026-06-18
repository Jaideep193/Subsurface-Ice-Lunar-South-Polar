"""
Temporal Coherence Analysis
============================
Compares two synthetic SAR passes to discriminate stable subsurface ice
from transient surface frost.

Physics:
  - Subsurface ice: buried below thermal cycling depth (~1 m) → CPR stable
    across orbital passes separated by weeks/months.
  - Surface frost:  seasonal condensate → CPR variable between passes
    (evaporates during local thermal noon, redeposits near perihelion).

Two-pass model:
  Pass 1 – nominal DFSAR L-band CPR
  Pass 2 – Pass 1 + Gaussian noise (σ=0.08) to mimic orbital repeat
            plus a frost perturbation patch that is present only in pass 1

Outputs:
  results/figures/14_temporal_coherence.png
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

CPR_THRESH   = 0.8
DOP_THRESH   = 0.13
FROST_NOISE  = 0.08    # CPR std between passes for stable ice
FROST_DELTA  = 0.35    # extra CPR in pass-1 frost patch (transient signal)


def load_data():
    from src.data_generator import load_all
    from src.dfsar_analysis import run_analysis, temporal_coherence

    data  = load_all(cache=True)
    psr   = data["psr_mask"]; slope = data["slope"]
    dfsar = data["dfsar"];    meta  = data["meta"]
    gs    = meta["grid_size"]; ps   = meta["pixel_scale"]
    dsc_c = meta["dsc_center"]; dsc_r = meta["dsc_radius"]

    res  = run_analysis(dfsar, psr, slope, cpr_thresh=CPR_THRESH, dop_thresh=DOP_THRESH)
    CPR1 = res["CPR"].copy().astype(np.float32)
    DOP1 = res["DOP"].copy().astype(np.float32)

    # ── Synthetic second pass ─────────────────────────────────────────────
    rng  = np.random.default_rng(77)
    CPR2 = np.clip(CPR1 + rng.normal(0, FROST_NOISE, CPR1.shape), 0, 3.5).astype(np.float32)
    DOP2 = np.clip(DOP1 + rng.normal(0, 0.01,        DOP1.shape), 0, 1.0).astype(np.float32)

    # Inject transient frost patch: a small PSR region high-CPR in pass-1 only
    gs_h = gs // 2
    r0, c0 = gs_h - 40, gs_h + 60      # offset from centre
    rr, cc = np.ogrid[:gs, :gs]
    frost_patch = (np.sqrt((rr-r0)**2 + (cc-c0)**2) < 25) & psr
    CPR1_frost = CPR1.copy()
    CPR1_frost[frost_patch] = np.clip(CPR1_frost[frost_patch] + FROST_DELTA, 0, 3.5)
    # Pass 2 does NOT have the frost patch (it has evaporated)

    # Re-detect ice on both passes
    from src.dfsar_analysis import detect_ice_pixels
    ice1, _, _ = detect_ice_pixels(CPR1_frost, DOP1, psr, slope,
                                    cpr_threshold=CPR_THRESH, dop_threshold=DOP_THRESH)
    ice2, _, _ = detect_ice_pixels(CPR2,       DOP2, psr, slope,
                                    cpr_threshold=CPR_THRESH, dop_threshold=DOP_THRESH)

    tc = temporal_coherence(CPR1_frost, DOP1, ice1, CPR2, DOP2, ice2)

    return dict(
        CPR1=CPR1_frost, CPR2=CPR2, DOP1=DOP1, DOP2=DOP2,
        ice1=ice1, ice2=ice2, psr=psr, tc=tc,
        frost_patch=frost_patch,
        gs=gs, ps=ps, dsc_c=dsc_c, dsc_r=dsc_r,
    )


def make_figure(d):
    gs  = d["gs"]; ps  = d["ps"]; KM = ps / 1000.0
    dsc_c = d["dsc_c"]; dsc_r = d["dsc_r"]
    ext = [0, gs*KM, gs*KM, 0]
    x_km = np.linspace(0, gs*KM, gs)
    y_km = np.linspace(0, gs*KM, gs)

    tc   = d["tc"]
    CPR1 = d["CPR1"]; CPR2 = d["CPR2"]
    ice1 = d["ice1"]; ice2 = d["ice2"]
    psr  = d["psr"]
    stable   = tc["stable_ice"]
    frost    = tc["frost_candidates"]
    delta    = tc["delta_CPR"]
    conf     = tc["temporal_confidence"]

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#080810")
    fig.suptitle(
        "Temporal Coherence Analysis  |  Two-Pass L-band DFSAR  |  Faustini PSR\n"
        "Stable ice (both passes) vs Transient frost (single pass)",
        color="white", fontsize=12, fontweight="bold")

    def ax_style(ax):
        ax.set_facecolor("#0d0d1a")
        for sp in ax.spines.values(): sp.set_color("#334")
        ax.tick_params(colors="#888", labelsize=7)
        ax.add_patch(plt.Circle((dsc_c[1]*KM, dsc_c[0]*KM), dsc_r*KM,
                                 color="white", fill=False, lw=1.2, ls="--"))
        ax.contour(x_km, y_km, psr.astype(float), levels=[0.5],
                   colors=["#555"], linewidths=[0.7], linestyles=["--"])
        return ax

    # ── Panel 1: Pass-1 CPR ──────────────────────────────────────────────────
    ax1 = ax_style(fig.add_subplot(3, 4, 1))
    im1 = ax1.imshow(CPR1, cmap="inferno", vmin=0, vmax=2.5,
                     extent=ext, origin="upper", aspect="auto")
    # Annotate frost patch
    ax1.contour(x_km, y_km, d["frost_patch"].astype(float),
                levels=[0.5], colors=["#00ffff"], linewidths=[1.5])
    plt.colorbar(im1, ax=ax1, fraction=0.046).ax.tick_params(colors="w", labelsize=6)
    ax1.set_title("(a) Pass-1 CPR\n(includes frost patch, cyan)",
                  color="white", fontsize=9, fontweight="bold")
    ax1.set_xlabel("East (km)", color="#aaa", fontsize=7)
    ax1.set_ylabel("South (km)", color="#aaa", fontsize=7)

    # ── Panel 2: Pass-2 CPR ──────────────────────────────────────────────────
    ax2 = ax_style(fig.add_subplot(3, 4, 2))
    im2 = ax2.imshow(CPR2, cmap="inferno", vmin=0, vmax=2.5,
                     extent=ext, origin="upper", aspect="auto")
    plt.colorbar(im2, ax=ax2, fraction=0.046).ax.tick_params(colors="w", labelsize=6)
    ax2.set_title("(b) Pass-2 CPR\n(frost evaporated — no cyan patch)",
                  color="white", fontsize=9, fontweight="bold")
    ax2.set_xlabel("East (km)", color="#aaa", fontsize=7)
    ax2.set_ylabel("South (km)", color="#aaa", fontsize=7)

    # ── Panel 3: Delta CPR map ───────────────────────────────────────────────
    ax3 = ax_style(fig.add_subplot(3, 4, 3))
    im3 = ax3.imshow(np.where(psr, delta, np.nan), cmap="hot", vmin=0, vmax=0.6,
                     extent=ext, origin="upper", aspect="auto")
    plt.colorbar(im3, ax=ax3, fraction=0.046,
                 label="|CPR₁ − CPR₂|").ax.tick_params(colors="w", labelsize=6)
    ax3.set_title("(c) |ΔCPR| between passes\nHigh = transient frost",
                  color="white", fontsize=9, fontweight="bold")
    ax3.set_xlabel("East (km)", color="#aaa", fontsize=7)
    ax3.set_ylabel("South (km)", color="#aaa", fontsize=7)

    # ── Panel 4: Stable vs frost classification ───────────────────────────────
    ax4 = ax_style(fig.add_subplot(3, 4, 4))
    overlay = np.zeros((*CPR1.shape, 4), dtype=np.float32)
    overlay[stable]   = [0.0,  0.9, 0.4,  0.9]   # green = stable ice (both passes)
    overlay[frost]    = [1.0,  0.6, 0.0,  0.9]   # amber = frost (one pass only)
    # Pixels in ice2 only
    pass2_only = ice2 & ~ice1
    overlay[pass2_only] = [0.4, 0.4, 1.0, 0.7]   # blue = new detections in pass 2
    ax4.imshow(np.zeros_like(CPR1), cmap="gray", vmin=0, vmax=1,
               extent=ext, origin="upper", aspect="auto", alpha=0.0)
    ax4.imshow(overlay, extent=ext, origin="upper", aspect="auto")
    patches = [
        mpatches.Patch(color=[0.0, 0.9, 0.4], label=f"Stable ice – both passes ({stable.sum()} px)"),
        mpatches.Patch(color=[1.0, 0.6, 0.0], label=f"Frost – pass-1 only ({(ice1&~ice2).sum()} px)"),
        mpatches.Patch(color=[0.4, 0.4, 1.0], label=f"New – pass-2 only ({pass2_only.sum()} px)"),
    ]
    ax4.legend(handles=patches, loc="upper left", fontsize=7,
               facecolor="#111", edgecolor="#445", framealpha=0.9)
    ax4.set_title("(d) Temporal Classification\nStable ice vs Transient frost",
                  color="white", fontsize=9, fontweight="bold")
    ax4.set_xlabel("East (km)", color="#aaa", fontsize=7)
    ax4.set_ylabel("South (km)", color="#aaa", fontsize=7)

    # ── Panel 5: Temporal confidence map ────────────────────────────────────
    ax5 = ax_style(fig.add_subplot(3, 4, 5))
    im5 = ax5.imshow(np.where(stable, conf, np.nan), cmap="plasma", vmin=0, vmax=1,
                     extent=ext, origin="upper", aspect="auto")
    plt.colorbar(im5, ax=ax5, fraction=0.046,
                 label="Temporal confidence").ax.tick_params(colors="w", labelsize=6)
    ax5.set_title("(e) Temporal Confidence\n(stable ice pixels only)",
                  color="white", fontsize=9, fontweight="bold")
    ax5.set_xlabel("East (km)", color="#aaa", fontsize=7)
    ax5.set_ylabel("South (km)", color="#aaa", fontsize=7)

    # ── Panel 6: ΔCPR histogram: stable vs frost ─────────────────────────────
    ax6 = fig.add_subplot(3, 4, 6)
    ax6.set_facecolor("#0d0d1a")
    for sp in ax6.spines.values(): sp.set_color("#334")
    ax6.tick_params(colors="#888", labelsize=7)

    if stable.any():
        ax6.hist(delta[stable], bins=40, range=(0, 0.8), color="#00e57a",
                 alpha=0.8, density=True, label=f"Stable ice (n={stable.sum()})")
    if frost.any():
        ax6.hist(delta[frost], bins=40, range=(0, 0.8), color="#ff8800",
                 alpha=0.8, density=True, label=f"Frost candidates (n={frost.sum()})")
    ax6.axvline(0.15, color="white", lw=1.2, ls="--", alpha=0.7,
                label="|ΔCPR| = 0.15 (discriminator)")
    ax6.set_xlabel("|ΔCPR| between passes", color="#aaa", fontsize=8)
    ax6.set_ylabel("Density", color="#aaa", fontsize=8)
    ax6.set_title("(f) |ΔCPR| Distribution\nStable ice vs Frost candidates",
                  color="white", fontsize=9, fontweight="bold")
    ax6.legend(fontsize=7.5, facecolor="#111", edgecolor="#334", labelcolor="white")

    # ── Panel 7: CPR pass-1 vs pass-2 scatter (stable ice pixels) ───────────
    ax7 = fig.add_subplot(3, 4, 7)
    ax7.set_facecolor("#0d0d1a")
    for sp in ax7.spines.values(): sp.set_color("#334")
    ax7.tick_params(colors="#888", labelsize=7)

    if stable.any():
        ax7.hexbin(CPR1[stable], CPR2[stable], gridsize=40, cmap="YlGn",
                   mincnt=1, extent=(0, 3, 0, 3))
    ax7.plot([0, 3], [0, 3], color="white", lw=1, ls="--", alpha=0.6, label="1:1 line")
    ax7.axvline(CPR_THRESH, color="cyan", lw=1, ls=":", alpha=0.7)
    ax7.axhline(CPR_THRESH, color="cyan", lw=1, ls=":", alpha=0.7)
    ax7.set_xlabel("CPR Pass 1", color="#aaa", fontsize=8)
    ax7.set_ylabel("CPR Pass 2", color="#aaa", fontsize=8)
    ax7.set_title("(g) CPR Stability Scatter\n(stable ice pixels, hexbin)",
                  color="white", fontsize=9, fontweight="bold")
    ax7.legend(fontsize=7.5, facecolor="#111", edgecolor="#334", labelcolor="white")

    # ── Panel 8: Statistics table ────────────────────────────────────────────
    ax8 = fig.add_subplot(3, 4, 8)
    ax8.axis("off")
    ax8.set_facecolor("#0d0d1a")

    ice_total = max(ice1.sum() + ice2.sum(), 1)
    rows = [
        ["Metric",                        "Pass 1",      "Pass 2",       "Combined"],
        ["Ice pixels detected",
         str(ice1.sum()), str(ice2.sum()), str(stable.sum())],
        ["Mean CPR (ice pixels)",
         f"{CPR1[ice1].mean():.3f}" if ice1.any() else "—",
         f"{CPR2[ice2].mean():.3f}" if ice2.any() else "—",
         f"{CPR1[stable].mean():.3f}" if stable.any() else "—"],
        ["Stable ice (both passes)",       "—",           "—",           str(stable.sum())],
        ["Frost candidates (1 pass only)", "—",           "—",           str(frost.sum())],
        ["Mean |ΔCPR| stable",             "—",           "—",
         f"{delta[stable].mean():.3f}" if stable.any() else "—"],
        ["Mean temporal confidence",       "—",           "—",
         f"{conf[stable].mean():.3f}" if stable.any() else "—"],
        ["Frost fraction",                 "—",           "—",
         f"{frost.sum()/max(stable.sum()+frost.sum(),1)*100:.1f}%"],
    ]

    tbl = ax8.table(cellText=[r[1:] for r in rows[1:]],
                    rowLabels=[r[0] for r in rows[1:]],
                    colLabels=rows[0][1:],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    tbl.scale(1.0, 1.9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#1a2040" if r == 0 else ("#0d1622" if r % 2 else "#111a2a"))
        cell.set_text_props(color="white")
        cell.set_edgecolor("#2a3a55")
    ax8.set_title("(h) Two-Pass Statistics",
                  color="white", fontsize=9, fontweight="bold", pad=8)

    # ── Panels 9-12: DOP stability row ──────────────────────────────────────
    # DOP pass1 map
    ax9 = ax_style(fig.add_subplot(3, 4, 9))
    im9 = ax9.imshow(np.where(psr, d["DOP1"], np.nan), cmap="viridis_r",
                     vmin=0, vmax=0.5, extent=ext, origin="upper", aspect="auto")
    plt.colorbar(im9, ax=ax9, fraction=0.046).ax.tick_params(colors="w", labelsize=6)
    ax9.contour(x_km, y_km, ice1.astype(float), levels=[0.5],
                colors=["#00ff88"], linewidths=[1.0])
    ax9.set_title("(i) DOP Pass 1 (ice = green)", color="white", fontsize=9, fontweight="bold")
    ax9.set_xlabel("East (km)", color="#aaa", fontsize=7)
    ax9.set_ylabel("South (km)", color="#aaa", fontsize=7)

    # DOP pass2 map
    ax10 = ax_style(fig.add_subplot(3, 4, 10))
    im10 = ax10.imshow(np.where(psr, d["DOP2"], np.nan), cmap="viridis_r",
                       vmin=0, vmax=0.5, extent=ext, origin="upper", aspect="auto")
    plt.colorbar(im10, ax=ax10, fraction=0.046).ax.tick_params(colors="w", labelsize=6)
    ax10.contour(x_km, y_km, ice2.astype(float), levels=[0.5],
                 colors=["#00ff88"], linewidths=[1.0])
    ax10.set_title("(j) DOP Pass 2 (ice = green)", color="white", fontsize=9, fontweight="bold")
    ax10.set_xlabel("East (km)", color="#aaa", fontsize=7)
    ax10.set_ylabel("South (km)", color="#aaa", fontsize=7)

    # DOP delta
    ax11 = ax_style(fig.add_subplot(3, 4, 11))
    im11 = ax11.imshow(np.where(psr, tc["delta_DOP"], np.nan), cmap="Reds",
                       vmin=0, vmax=0.15, extent=ext, origin="upper", aspect="auto")
    plt.colorbar(im11, ax=ax11, fraction=0.046,
                 label="|ΔDOP|").ax.tick_params(colors="w", labelsize=6)
    ax11.set_title("(k) |ΔDOP| between passes\nLow = stable polarimetry",
                   color="white", fontsize=9, fontweight="bold")
    ax11.set_xlabel("East (km)", color="#aaa", fontsize=7)
    ax11.set_ylabel("South (km)", color="#aaa", fontsize=7)

    # DOP stability score for stable ice pixels
    ax12 = fig.add_subplot(3, 4, 12)
    ax12.set_facecolor("#0d0d1a")
    for sp in ax12.spines.values(): sp.set_color("#334")
    ax12.tick_params(colors="#888", labelsize=7)

    if stable.any():
        ax12.scatter(CPR1[stable], tc["dop_stability"][stable],
                     c=delta[stable], cmap="hot", s=8, alpha=0.6,
                     vmin=0, vmax=0.4, label="Stable ice")
        ax12.axhline(0.7, color="#00e57a", lw=1.2, ls="--",
                     label="High stability threshold")
    ax12.set_xlabel("CPR (Pass 1)", color="#aaa", fontsize=8)
    ax12.set_ylabel("DOP Stability Score", color="#aaa", fontsize=8)
    ax12.set_title("(l) DOP Stability vs CPR\nColored by |ΔCPR|",
                   color="white", fontsize=9, fontweight="bold")
    ax12.legend(fontsize=7.5, facecolor="#111", edgecolor="#334", labelcolor="white")

    fig.tight_layout()
    out = Path("results/figures/14_temporal_coherence.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Temporal] Saved: {out}")


if __name__ == "__main__":
    print("[Temporal] Loading data...")
    d = load_data()
    print(f"[Temporal] Stable ice: {d['tc']['n_stable']} px | "
          f"Frost candidates: {d['tc']['n_frost']} px")
    make_figure(d)
