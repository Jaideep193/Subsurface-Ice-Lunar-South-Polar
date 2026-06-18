"""
Ice Sublimation Lifetime Map
=============================
Computes how long subsurface ice survives at each pixel using the
Hertz-Knudsen sublimation equation calibrated to lunar conditions.

Physics (Schorghofer & Taylor 2007; Prem et al. 2018):
  P_vap(T) = P0 * exp(-L_sub/R * (1/T - 1/T0))   [Clausius-Clapeyron]
  J         = alpha * P_vap / sqrt(2*pi*m_H2O*k*T) [Hertz-Knudsen flux, kg/m²/s]
  tau       = rho_ice * d_ice / J                   [sublimation lifetime]

Constants:
  L_sub = 51,058  J/mol  (latent heat of sublimation, water ice)
  R     = 8.314   J/mol/K
  P0    = 611.73  Pa  (triple point)
  T0    = 273.16  K
  alpha = 0.10    (evaporation coefficient on lunar regolith, nominal)
  m_H2O = 3.0e-26 kg  (molecular mass, single molecule)
  k     = 1.381e-23 J/K
  rho_ice = 917  kg/m³
  d_ice   = 0.10 m  (10 cm nominal layer thickness for seasonal frost)

Stability zones follow Paige et al. 2010:
  Stable   : T < 70 K  → tau >> Ga (ancient ice)
  Seasonal : 70-110 K  → tau computable, hours to kyrs
  Unstable : > 110 K   → tau < 1 lunar day

Output:
  results/figures/15_sublimation_lifetime.png
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

# ── Physical constants ──────────────────────────────────────────────────────
L_SUB   = 51_058.0      # J/mol — latent heat of sublimation (water ice)
R_GAS   = 8.314         # J/mol/K
P0_TRIP = 611.73        # Pa  — triple point
T0_TRIP = 273.16        # K   — triple point
ALPHA   = 0.10          # evaporation coefficient (lunar regolith surface)
M_H2O   = 3.0e-26       # kg  — single water molecule mass
K_BOLTZ = 1.381e-23     # J/K
RHO_ICE = 917.0         # kg/m³
D_ICE   = 0.10          # m   — nominal frost layer thickness

# Stability thresholds (Paige et al. 2010)
T_STABLE   = 70.0       # K
T_SEASONAL = 110.0      # K

# Time conversions
SECONDS_PER_HOUR      = 3600.0
SECONDS_PER_LUNAR_DAY = 2_360_448.0   # 27.32 days × 86400
SECONDS_PER_YEAR      = 3.156e7
SECONDS_PER_GYR       = 3.156e16


def sublimation_lifetime(T_K, d_ice=D_ICE, alpha=ALPHA):
    """
    Compute sublimation lifetime (seconds) for a layer of thickness d_ice
    at temperature T_K.  Vectorised — T_K can be an ndarray.
    """
    T = np.asarray(T_K, dtype=np.float64)
    # Clausius-Clapeyron
    P_vap = P0_TRIP * np.exp(-L_SUB / R_GAS * (1.0/T - 1.0/T0_TRIP))
    # Hertz-Knudsen flux (kg/m²/s)
    J_kg  = alpha * P_vap / np.sqrt(2.0 * np.pi * M_H2O * K_BOLTZ * T)
    J_kg  = np.clip(J_kg, 1e-50, 1e10)   # prevent /0 or overflow
    # Lifetime (seconds)
    tau   = RHO_ICE * d_ice / J_kg
    return tau


def load_data():
    from src.data_generator import load_all
    from src.dfsar_analysis import run_analysis
    from src.thermal_model  import compute_surface_temperature

    data  = load_all(cache=True)
    psr   = data["psr_mask"]; slope = data["slope"]; illum = data["illum"]
    dfsar = data["dfsar"]; dem = data["dem"]; meta = data["meta"]
    gs    = meta["grid_size"]; ps = meta["pixel_scale"]
    dsc_c = meta["dsc_center"]; dsc_r = meta["dsc_radius"]

    res       = run_analysis(dfsar, psr, slope, cpr_thresh=0.8, dop_thresh=0.13)
    T_surface = compute_surface_temperature(illum)

    return dict(T=T_surface, ice_mask=res["ice_mask"], CPR=res["CPR"],
                psr=psr, gs=gs, ps=ps, dsc_c=dsc_c, dsc_r=dsc_r)


def make_figure(d):
    T       = d["T"]
    ice     = d["ice_mask"]
    CPR     = d["CPR"]
    psr     = d["psr"]
    gs      = d["gs"]; ps = d["ps"]; KM = ps / 1000.0
    dsc_c   = d["dsc_c"]; dsc_r = d["dsc_r"]
    ext     = [0, gs*KM, gs*KM, 0]
    x_km    = np.linspace(0, gs*KM, gs)
    y_km    = np.linspace(0, gs*KM, gs)

    # ── Stability zones ───────────────────────────────────────────────────
    stable   = ice & (T < T_STABLE)
    seasonal = ice & (T >= T_STABLE) & (T < T_SEASONAL)
    unstable = ice & (T >= T_SEASONAL)

    # ── Lifetime map ──────────────────────────────────────────────────────
    tau_all  = sublimation_lifetime(np.clip(T, 40.0, 200.0))
    tau_ice  = np.where(ice, tau_all, np.nan)
    tau_log  = np.where(ice, np.log10(np.clip(tau_all, 1e-3, SECONDS_PER_GYR*10)), np.nan)

    # ── Temperature sensitivity curve ─────────────────────────────────────
    T_range  = np.linspace(40, 160, 500)
    tau_T    = sublimation_lifetime(T_range)

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#080810")
    fig.suptitle(
        "Ice Sublimation Lifetime Map  |  Hertz-Knudsen Equation  |  Faustini PSR\n"
        "Layer thickness: 10 cm  |  Evaporation coefficient α = 0.10  "
        "(Schorghofer & Taylor 2007; Prem et al. 2018)",
        color="white", fontsize=11, fontweight="bold")

    def ax_style(ax):
        ax.set_facecolor("#0d0d1a")
        for sp in ax.spines.values(): sp.set_color("#334")
        ax.tick_params(colors="#888", labelsize=7)
        return ax

    def map_ax(ax):
        ax_style(ax)
        ax.add_patch(plt.Circle((dsc_c[1]*KM, dsc_c[0]*KM), dsc_r*KM,
                                 color="white", fill=False, lw=1.2, ls="--"))
        ax.contour(x_km, y_km, psr.astype(float), levels=[0.5],
                   colors=["#555"], linewidths=[0.7], linestyles=["--"])
        return ax

    # ── Panel 1: Log10(τ) map ────────────────────────────────────────────
    ax1 = map_ax(fig.add_subplot(3, 4, 1))
    cmap_tau = plt.cm.plasma
    im1 = ax1.imshow(tau_log, cmap=cmap_tau, vmin=0, vmax=18,
                     extent=ext, origin="upper", aspect="auto")
    cb1 = plt.colorbar(im1, ax=ax1, fraction=0.046)
    cb1.set_label("log₁₀(τ) [seconds]", color="white", fontsize=7)
    cb1.ax.tick_params(colors="w", labelsize=6)
    # Reference ticks
    ref_ticks = {
        0:  "1 s",
        3:  "17 min",
        4:  "3 hr",
        6:  "12 days",
        7:  "4 months",
        9:  "31 yrs",
        13: "0.1 Myr",
        16: "1 Gyr",
    }
    cb1.set_ticks(list(ref_ticks.keys()))
    cb1.set_ticklabels(list(ref_ticks.values()))
    cb1.ax.yaxis.set_tick_params(labelcolor="white", labelsize=6)
    ax1.set_title("(a) Sublimation Lifetime\nlog₁₀(τ)", color="white",
                  fontsize=9, fontweight="bold")
    ax1.set_xlabel("East (km)", color="#aaa", fontsize=7)
    ax1.set_ylabel("South (km)", color="#aaa", fontsize=7)

    # ── Panel 2: Stability zone map ──────────────────────────────────────
    ax2 = map_ax(fig.add_subplot(3, 4, 2))
    zone_cmap = mcolors.ListedColormap(["#0d0d1a", "#00e5aa", "#ffaa00", "#ff4444"])
    zone_norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], zone_cmap.N)
    zone_map  = np.zeros(T.shape, dtype=np.int8)
    zone_map[stable]   = 1
    zone_map[seasonal] = 2
    zone_map[unstable] = 3
    ax2.imshow(zone_map, cmap=zone_cmap, norm=zone_norm,
               extent=ext, origin="upper", aspect="auto", interpolation="nearest")
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color="#00e5aa", label=f"Stable (T<70 K, τ>1 Gyr)  [{stable.sum()} px]"),
        mpatches.Patch(color="#ffaa00", label=f"Seasonal (70-110 K)  [{seasonal.sum()} px]"),
        mpatches.Patch(color="#ff4444", label=f"Unstable (T>110 K)  [{unstable.sum()} px]"),
    ]
    ax2.legend(handles=patches, loc="upper left", fontsize=6.5,
               facecolor="#111", edgecolor="#445", framealpha=0.9)
    ax2.set_title("(b) Thermal Stability Zones", color="white",
                  fontsize=9, fontweight="bold")
    ax2.set_xlabel("East (km)", color="#aaa", fontsize=7)
    ax2.set_ylabel("South (km)", color="#aaa", fontsize=7)

    # ── Panel 3: Surface temperature map ────────────────────────────────
    ax3 = map_ax(fig.add_subplot(3, 4, 3))
    im3 = ax3.imshow(np.where(psr, T, np.nan), cmap="coolwarm", vmin=40, vmax=160,
                     extent=ext, origin="upper", aspect="auto")
    plt.colorbar(im3, ax=ax3, fraction=0.046,
                 label="T_surface (K)").ax.tick_params(colors="w", labelsize=6)
    ax3.contour(x_km, y_km, ice.astype(float), levels=[0.5],
                colors=["#00ff88"], linewidths=[1.0])
    ax3.set_title("(c) Surface Temperature\n(ice pixels = green contour)",
                  color="white", fontsize=9, fontweight="bold")
    ax3.set_xlabel("East (km)", color="#aaa", fontsize=7)
    ax3.set_ylabel("South (km)", color="#aaa", fontsize=7)

    # ── Panel 4: Temperature sensitivity curve ───────────────────────────
    ax4 = ax_style(fig.add_subplot(3, 4, 4))
    ax4.semilogy(T_range, tau_T / SECONDS_PER_YEAR, color="#00ccff", lw=2)
    ax4.axhline(1,           color="#00e5aa", lw=1.2, ls="--", alpha=0.8, label="1 yr")
    ax4.axhline(1e3,         color="#ffaa00", lw=1.2, ls="--", alpha=0.8, label="1 kyr")
    ax4.axhline(1e6,         color="#ff8800", lw=1.2, ls="--", alpha=0.8, label="1 Myr")
    ax4.axhline(1e9,         color="#ff4444", lw=1.2, ls="--", alpha=0.8, label="1 Gyr")
    ax4.axvline(T_STABLE,   color="#00e5aa", lw=1.0, ls=":", alpha=0.7)
    ax4.axvline(T_SEASONAL, color="#ffaa00", lw=1.0, ls=":", alpha=0.7)
    ax4.fill_betweenx([1e-5, 1e15], T_STABLE, T_SEASONAL,
                      color="#ffaa00", alpha=0.10)
    ax4.set_ylim(1e-5, 1e12)
    ax4.set_xlim(40, 160)
    ax4.set_xlabel("Surface Temperature (K)", color="#aaa", fontsize=8)
    ax4.set_ylabel("Sublimation Lifetime (years)", color="#aaa", fontsize=8)
    ax4.set_title("(d) τ vs Temperature\n(Clausius-Clapeyron + Hertz-Knudsen)",
                  color="white", fontsize=9, fontweight="bold")
    ax4.legend(fontsize=7.5, facecolor="#111", edgecolor="#334", labelcolor="white",
               loc="lower left")
    ax4.yaxis.set_minor_locator(mticker.LogLocator(subs="all"))

    # ── Panel 5: Seasonal zone: tau histogram ────────────────────────────
    ax5 = ax_style(fig.add_subplot(3, 4, 5))
    if seasonal.any():
        tau_seas_yr = tau_all[seasonal] / SECONDS_PER_YEAR
        ax5.hist(np.log10(np.clip(tau_seas_yr, 1e-5, 1e12)), bins=50,
                 color="#ffaa00", alpha=0.85, edgecolor="#aa7700")
        ax5.axvline(0,  color="white", lw=1.2, ls="--", label="1 yr")
        ax5.axvline(3,  color="#ff8800", lw=1.2, ls="--", label="1 kyr")
        ax5.axvline(6,  color="#ff4444", lw=1.2, ls="--", label="1 Myr")
    ax5.set_xlabel("log₁₀(τ) [years]", color="#aaa", fontsize=8)
    ax5.set_ylabel("Pixel count", color="#aaa", fontsize=8)
    ax5.set_title("(e) Seasonal Zone Lifetime Distribution\n(70–110 K ice pixels)",
                  color="white", fontsize=9, fontweight="bold")
    ax5.legend(fontsize=7.5, facecolor="#111", edgecolor="#334", labelcolor="white")

    # ── Panel 6: τ vs CPR scatter ────────────────────────────────────────
    ax6 = ax_style(fig.add_subplot(3, 4, 6))
    if ice.any():
        tau_ice_yr = tau_all[ice] / SECONDS_PER_YEAR
        colors_z   = np.where(T[ice] < T_STABLE, "#00e5aa",
                     np.where(T[ice] < T_SEASONAL, "#ffaa00", "#ff4444"))
        ax6.scatter(CPR[ice], np.log10(np.clip(tau_ice_yr, 1e-5, 1e15)),
                    c=colors_z, s=10, alpha=0.6, edgecolors="none")
        ax6.axhline(0,   color="white",   lw=1, ls="--", alpha=0.6, label="1 yr")
        ax6.axhline(9,   color="#aaaaaa", lw=1, ls="--", alpha=0.6, label="1 Gyr")
        ax6.axvline(0.8, color="cyan",    lw=1, ls=":",  alpha=0.6, label="CPR thr")
    ax6.set_xlabel("CPR", color="#aaa", fontsize=8)
    ax6.set_ylabel("log₁₀(τ) [years]", color="#aaa", fontsize=8)
    ax6.set_title("(f) Sublimation Lifetime vs CPR\n(colored by thermal zone)",
                  color="white", fontsize=9, fontweight="bold")
    import matplotlib.patches as mpatches
    patches6 = [
        mpatches.Patch(color="#00e5aa", label="Stable (T<70 K)"),
        mpatches.Patch(color="#ffaa00", label="Seasonal (70-110 K)"),
        mpatches.Patch(color="#ff4444", label="Unstable (>110 K)"),
    ]
    ax6.legend(handles=patches6, fontsize=7, facecolor="#111",
               edgecolor="#334", framealpha=0.8)

    # ── Panel 7: Alpha sensitivity (evaporation coefficient) ─────────────
    ax7 = ax_style(fig.add_subplot(3, 4, 7))
    T_fixed = 90.0   # representative seasonal zone temperature
    alpha_range = np.logspace(-3, 0, 200)
    for alpha_val, color, lbl in [
        (0.01,  "#4488ff", "α=0.01 (wet regolith)"),
        (0.10,  "#00e5aa", "α=0.10 (nominal)"),
        (1.00,  "#ff4444", "α=1.00 (bare ice face)"),
    ]:
        tau_a = sublimation_lifetime(T_fixed, d_ice=D_ICE, alpha=alpha_val)
        ax7.axhline(tau_a / SECONDS_PER_YEAR, color=color, lw=1.5, ls="--",
                    alpha=0.8, label=lbl)
    taus_a = np.array([sublimation_lifetime(T_fixed, alpha=a) for a in alpha_range])
    ax7.semilogx(alpha_range, taus_a / SECONDS_PER_YEAR, color="#ffffff", lw=2)
    ax7.set_xlabel("Evaporation coefficient α", color="#aaa", fontsize=8)
    ax7.set_ylabel("τ at T=90 K (years)", color="#aaa", fontsize=8)
    ax7.set_title("(g) α Sensitivity at T=90 K\n(seasonal zone representative)",
                  color="white", fontsize=9, fontweight="bold")
    ax7.legend(fontsize=7.5, facecolor="#111", edgecolor="#334", labelcolor="white")

    # ── Panel 8: Layer thickness sensitivity ─────────────────────────────
    ax8 = ax_style(fig.add_subplot(3, 4, 8))
    d_range = np.logspace(-3, 1, 200)
    for T_val, color, lbl in [(70, "#00e5aa","T=70 K"), (90, "#ffaa00","T=90 K"),
                               (110, "#ff4444","T=110 K")]:
        taus_d = np.array([sublimation_lifetime(T_val, d_ice=d) for d in d_range])
        ax8.loglog(d_range * 100, taus_d / SECONDS_PER_YEAR,
                   color=color, lw=2, label=lbl)
    ax8.axvline(D_ICE * 100, color="white", lw=1, ls="--", alpha=0.6, label="10 cm (nominal)")
    ax8.set_xlabel("Ice layer thickness (cm)", color="#aaa", fontsize=8)
    ax8.set_ylabel("Sublimation lifetime (years)", color="#aaa", fontsize=8)
    ax8.set_title("(h) Thickness Sensitivity\n(τ ∝ d, Hertz-Knudsen)",
                  color="white", fontsize=9, fontweight="bold")
    ax8.legend(fontsize=7.5, facecolor="#111", edgecolor="#334", labelcolor="white")

    # ── Panels 9-12: summary statistics ─────────────────────────────────
    # Panel 9: violins per zone
    ax9 = ax_style(fig.add_subplot(3, 4, 9))
    zone_data = []
    zone_labels_v = []
    if stable.any():
        tau_s = np.log10(np.clip(tau_all[stable] / SECONDS_PER_YEAR, 1e-5, 1e15))
        zone_data.append(tau_s);  zone_labels_v.append("Stable\n(<70 K)")
    if seasonal.any():
        tau_se = np.log10(np.clip(tau_all[seasonal] / SECONDS_PER_YEAR, 1e-5, 1e15))
        zone_data.append(tau_se); zone_labels_v.append("Seasonal\n(70-110 K)")
    if unstable.any():
        tau_u = np.log10(np.clip(tau_all[unstable] / SECONDS_PER_YEAR, 1e-5, 1e15))
        zone_data.append(tau_u);  zone_labels_v.append("Unstable\n(>110 K)")
    if zone_data:
        vp = ax9.violinplot(zone_data, positions=range(len(zone_data)),
                            showmedians=True, showextrema=True)
        colors_v = ["#00e5aa", "#ffaa00", "#ff4444"]
        for patch, col in zip(vp["bodies"], colors_v):
            patch.set_facecolor(col); patch.set_alpha(0.7)
        vp["cmedians"].set_color("white")
        ax9.set_xticks(range(len(zone_data)))
        ax9.set_xticklabels(zone_labels_v, color="white", fontsize=7.5)
        ax9.axhline(0, color="white", lw=0.8, ls=":", alpha=0.5)
        ax9.axhline(9, color="#aaaaaa", lw=0.8, ls=":", alpha=0.5)
    ax9.set_ylabel("log₁₀(τ) [years]", color="#aaa", fontsize=8)
    ax9.set_title("(i) Lifetime by Zone (violin)",
                  color="white", fontsize=9, fontweight="bold")

    # Panel 10-12: key metrics table spanning 3 columns
    ax_tbl = fig.add_subplot(3, 4, (10, 12))
    ax_tbl.axis("off")

    def fmt_tau(tau_s):
        yr = tau_s / SECONDS_PER_YEAR
        if yr < 1:    return f"{tau_s/3600:.1f} hr"
        if yr < 1e3:  return f"{yr:.0f} yr"
        if yr < 1e6:  return f"{yr/1e3:.1f} kyr"
        if yr < 1e9:  return f"{yr/1e6:.1f} Myr"
        return f"{yr/1e9:.1f} Gyr"

    rows = [["Zone", "Pixels", "T mean (K)", "τ median", "τ P10", "τ P90"]]
    for zone_mask, zone_name, zone_col in [
        (stable,   "Stable (<70 K)",    "#00e5aa"),
        (seasonal, "Seasonal (70-110K)","#ffaa00"),
        (unstable, "Unstable (>110 K)", "#ff4444"),
    ]:
        if zone_mask.any():
            tz = tau_all[zone_mask]
            rows.append([
                zone_name,
                str(zone_mask.sum()),
                f"{T[zone_mask].mean():.1f} K",
                fmt_tau(float(np.median(tz))),
                fmt_tau(float(np.percentile(tz, 10))),
                fmt_tau(float(np.percentile(tz, 90))),
            ])
        else:
            rows.append([zone_name, "0", "—", "—", "—", "—"])

    # Full-scene row
    tau_full = sublimation_lifetime(np.array([40, 70, 90, 110, 160]))
    rows.append(["Reference temps",
                 "40/70/90/110/160 K", "—",
                 " / ".join(fmt_tau(t) for t in tau_full),
                 "—", "—"])

    tbl = ax_tbl.table(cellText=[r[1:] for r in rows[1:]],
                       rowLabels=[r[0] for r in rows[1:]],
                       colLabels=rows[0][1:],
                       loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    tbl.scale(1.1, 2.0)
    zone_colors = ["#00e5aa", "#ffaa00", "#ff4444", "#aabbcc"]
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#1a2040" if r == 0 else "#0d1520")
        cell.set_text_props(color=zone_colors[r-1] if r > 0 else "white")
        cell.set_edgecolor("#2a3a55")
    ax_tbl.set_title("(j) Sublimation Lifetime Statistics by Stability Zone",
                     color="white", fontsize=9, fontweight="bold", pad=8)

    fig.tight_layout()
    out = Path("results/figures/15_sublimation_lifetime.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Sublimation] Saved: {out}")

    # Print key results
    print(f"[Sublimation] Stable ice: {stable.sum()} px  "
          f"median tau = {fmt_tau(float(np.median(tau_all[stable]))) if stable.any() else 'N/A'}")
    print(f"[Sublimation] Seasonal:   {seasonal.sum()} px  "
          f"median tau = {fmt_tau(float(np.median(tau_all[seasonal]))) if seasonal.any() else 'N/A'}")
    print(f"[Sublimation] Unstable:   {unstable.sum()} px  "
          f"median tau = {fmt_tau(float(np.median(tau_all[unstable]))) if unstable.any() else 'N/A'}")


if __name__ == "__main__":
    print("[Sublimation] Loading data...")
    d = load_data()
    make_figure(d)
