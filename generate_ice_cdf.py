"""
Ice Volume CDF Figure
=====================
Multi-panel publication figure showing the Monte Carlo posterior CDF for
subsurface ice volume, alongside the PDF, scenario comparison, and
ISRU propellant potential.

Output: results/figures/11_ice_volume_cdf.png
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

ICE_DENSITY_KG_M3 = 917.0
KG_PER_TONNE      = 1000.0


def load_data():
    from src.data_generator import load_all
    from src.dfsar_analysis import run_analysis
    from src.ice_volume     import estimate_ice_volume, monte_carlo_uncertainty

    data  = load_all(cache=True)
    psr   = data["psr_mask"]; slope = data["slope"]
    dfsar = data["dfsar"]; meta = data["meta"]
    gs    = meta["grid_size"]; ps = meta["pixel_scale"]

    res    = run_analysis(dfsar, psr, slope)
    iv     = estimate_ice_volume(res["CPR"], res["ice_mask"], pixel_scale=ps)
    unc    = monte_carlo_uncertainty(res["CPR"], res["ice_mask"],
                                     pixel_scale=ps, n_samples=500)

    return dict(res=res, iv=iv, unc=unc, gs=gs, ps=ps, psr=psr)


def _ice_mass_tonnes(vol_m3):
    return vol_m3 * ICE_DENSITY_KG_M3 / KG_PER_TONNE


def make_figure(d):
    unc  = d["unc"]
    iv   = d["iv"]

    samples_m3 = unc["samples"]
    samples_t  = _ice_mass_tonnes(samples_m3)
    N          = len(samples_t)

    # Sorted CDF
    sorted_t = np.sort(samples_t)
    cdf      = np.linspace(0, 100, N)

    # Key quantiles
    p5   = unc["p5_m3"]  * ICE_DENSITY_KG_M3 / KG_PER_TONNE
    p10  = np.percentile(samples_t, 10)
    p25  = np.percentile(samples_t, 25)
    p50  = unc["p50_m3"] * ICE_DENSITY_KG_M3 / KG_PER_TONNE
    p75  = np.percentile(samples_t, 75)
    p90  = np.percentile(samples_t, 90)
    p95  = unc["p95_m3"] * ICE_DENSITY_KG_M3 / KG_PER_TONNE
    mean = _ice_mass_tonnes(unc["mean_m3"])

    # Three scenario benchmarks (from generate_ice_scenarios)
    SCENARIOS = {
        "Shallow frost\n(0.3 m, f=0.12)":      p5  * 0.25,
        "Mid-depth ice\n(2.0 m, f=0.35)":       p50 * 0.85,
        "Deep ancient ice\n(4.5 m, f=0.55)":    p95 * 1.05,
    }
    scen_colors = ["#44bb99", "#ffaa33", "#ff6655"]

    # ── Figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#080810")
    gs_f = gridspec.GridSpec(2, 3, figure=fig,
                              hspace=0.40, wspace=0.35)

    def sax(sp):
        ax = fig.add_subplot(sp)
        ax.set_facecolor("#0d0d1a")
        for s in ax.spines.values(): s.set_color("#334")
        ax.tick_params(colors="#888", labelsize=8)
        return ax

    # ── Panel (a): CDF main ───────────────────────────────────────────────────
    ax = sax(gs_f[0, :2])

    # Shaded uncertainty band (5–95%)
    ax.fill_betweenx([0, 100],
                      [p5, p5], [p95, p95],
                      color="#224488", alpha=0.25, label="P5–P95 band")

    # CDF curve
    ax.plot(sorted_t, cdf, color="#4488ff", lw=2.5, label=f"CDF (N={N})")

    # Quantile markers
    quantile_markers = [
        (p10, 10,  "#ffff00", "P10"),
        (p50, 50,  "#ff8800", "P50 (median)"),
        (p90, 90,  "#ff3333", "P90"),
        (mean, None, "#00ff88", "Mean"),
    ]
    for val, perc, col, lbl in quantile_markers:
        if perc is not None:
            ax.plot([val, val], [0, perc], color=col, lw=1.2, ls="--", alpha=0.8)
            ax.plot([0, val],   [perc, perc], color=col, lw=1.2, ls="--", alpha=0.8)
            ax.scatter([val], [perc], color=col, s=70, zorder=5)
            ax.text(val, perc + 2, f"{lbl}\n{val:.2f} t",
                    color=col, fontsize=8, ha="center", va="bottom")
        else:
            ax.axvline(val, color=col, lw=1.5, ls=":", alpha=0.9,
                       label=f"{lbl}: {val:.2f} t")

    # Scenario benchmarks
    for i, (slbl, sval) in enumerate(SCENARIOS.items()):
        ax.axvline(sval, color=scen_colors[i], lw=1.2, ls="-.", alpha=0.75)
        ax.text(sval, 96 - i*5, slbl.replace("\n", " "),
                color=scen_colors[i], fontsize=6.5, ha="center")

    ax.set_xlabel("Subsurface Water-Ice Mass (tonnes)", color="#aaa", fontsize=10)
    ax.set_ylabel("Cumulative Probability (%)", color="#aaa", fontsize=10)
    ax.set_title("(a) Ice Volume CDF — Monte Carlo Posterior\n"
                  "500 samples: CPR noise ±0.1, depth ±30%, dielectric ±20%",
                  color="white", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, facecolor="#111", edgecolor="#334", loc="upper left")
    ax.set_ylim(0, 100); ax.set_xlim(left=0)

    # ── Panel (b): PDF with scenario comparison ────────────────────────────────
    ax = sax(gs_f[0, 2])
    ax.hist(samples_t, bins=60, density=True,
            color="#224477", edgecolor="#4466aa", alpha=0.8)
    # Gaussian KDE overlay
    from scipy.ndimage import gaussian_filter1d
    counts, bin_edges = np.histogram(samples_t, bins=100, density=True)
    smooth = gaussian_filter1d(counts, sigma=2)
    centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ax.plot(centres, smooth, color="#88aaff", lw=2.0, label="KDE")

    for i, (slbl, sval) in enumerate(SCENARIOS.items()):
        ax.axvline(sval, color=scen_colors[i], lw=1.5, ls="-.",
                   label=slbl.replace("\n", " "))
    ax.axvline(p50, color="#ff8800", lw=1.5, ls="--", label=f"P50={p50:.2f} t")
    ax.set_xlabel("Ice mass (t)", color="#aaa", fontsize=8)
    ax.set_ylabel("Density", color="#aaa", fontsize=8)
    ax.set_title("(b) Ice Volume PDF\nwith scenario benchmarks",
                  color="white", fontsize=9, fontweight="bold")
    ax.legend(fontsize=6.5, facecolor="#111", edgecolor="#334")

    # ── Panel (c): Bootstrap convergence ─────────────────────────────────────
    ax = sax(gs_f[1, 0])
    n_steps = np.arange(20, N+1, 10)
    rng     = np.random.default_rng(7)
    p50_conv = [np.percentile(rng.choice(samples_t, n, replace=False), 50)
                for n in n_steps]
    p5_conv  = [np.percentile(rng.choice(samples_t, n, replace=False), 5)
                for n in n_steps]
    p95_conv = [np.percentile(rng.choice(samples_t, n, replace=False), 95)
                for n in n_steps]
    ax.fill_between(n_steps, p5_conv, p95_conv, color="#334488", alpha=0.35,
                    label="P5–P95 band")
    ax.plot(n_steps, p50_conv, color="#ff8800", lw=1.8, label="P50")
    ax.axhline(p50, color="#ff8800", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("MC sample count", color="#aaa", fontsize=8)
    ax.set_ylabel("Ice mass estimate (t)", color="#aaa", fontsize=8)
    ax.set_title("(c) Bootstrap Convergence\nP50 stability vs sample count",
                  color="white", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, facecolor="#111", edgecolor="#334")

    # ── Panel (d): Propellant potential ──────────────────────────────────────
    ax = sax(gs_f[1, 1])
    q_labels = ["P5", "P10", "P25", "P50", "P75", "P90", "P95"]
    q_vals_t = [p5, p10, p25, p50, p75, p90, p95]
    # Electrolysis: 83% eff, H:O mass ratio 1:8 (H2O)
    h2_vals  = [v * 1000 * 0.112 * 0.83 / 1000 for v in q_vals_t]  # tonnes H2
    o2_vals  = [v * 1000 * 0.888 * 0.83 / 1000 for v in q_vals_t]  # tonnes O2
    x = np.arange(len(q_labels))
    width = 0.35
    ax.bar(x - width/2, h2_vals, width, color="#4af", label="H2 (propellant)")
    ax.bar(x + width/2, o2_vals, width, color="#f84", label="O2 (propellant)")
    ax.set_xticks(x); ax.set_xticklabels(q_labels, fontsize=8, color="#aaa")
    ax.set_ylabel("Propellant mass (t)", color="#aaa", fontsize=8)
    ax.set_title("(d) ISRU Propellant Potential\nElectrolysis efficiency 83%",
                  color="white", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7.5, facecolor="#111", edgecolor="#334")

    # ── Panel (e): Summary table ──────────────────────────────────────────────
    ax = sax(gs_f[1, 2])
    ax.axis("off")

    table_data = [
        ["Quantile", "Ice vol. (m³)", "Water-equiv. (t)", "H2 (t)", "O2 (t)"],
    ]
    for ql, qv in zip(q_labels, q_vals_t):
        vol_m3 = qv * KG_PER_TONNE / ICE_DENSITY_KG_M3
        h2  = qv * 1000 * 0.112 * 0.83 / 1000
        o2  = qv * 1000 * 0.888 * 0.83 / 1000
        table_data.append([ql, f"{vol_m3:.3f}", f"{qv:.3f}", f"{h2:.4f}", f"{o2:.4f}"])
    table_data.append([
        "Mean",
        f"{unc['mean_m3']:.3f}",
        f"{mean:.3f}",
        f"{mean*1000*0.112*0.83/1000:.4f}",
        f"{mean*1000*0.888*0.83/1000:.4f}",
    ])

    tbl = ax.table(cellText=table_data[1:],
                   colLabels=table_data[0],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5)
    tbl.scale(1.05, 1.55)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#0d1622" if row % 2 == 0 else "#111a2a")
        cell.set_text_props(color="white")
        cell.set_edgecolor("#2a3a55")
    ax.set_title("(e) Probabilistic Quantile Table",
                  color="white", fontsize=9, fontweight="bold", pad=10)

    # ── Suptitle ──────────────────────────────────────────────────────────────
    fig.suptitle(
        "Monte Carlo Ice Volume Posterior  |  "
        "Chandrayaan-2 DFSAR  |  Faustini PSR Doubly Shadowed Crater\n"
        "CPR noise ±0.1  |  Depth ±30% log-normal  |  Dielectric ±20% Gaussian  |  "
        f"N={N} samples",
        color="white", fontsize=11, fontweight="bold")

    out = Path("results/figures/11_ice_volume_cdf.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[IceCDF] Saved: {out}")


if __name__ == "__main__":
    print("[IceCDF] Loading data (uses cache)...")
    d = load_data()
    make_figure(d)
