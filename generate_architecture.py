"""
Generate system architecture diagram as a PNG image.
Run once: python generate_architecture.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── palette ──────────────────────────────────────────────────────────────────
BG       = "#0d1117"
C_IN     = "#0d3b6e"   # data inputs (deep blue)
C_S0     = "#1c3a4a"   # step 0 loader
C_S1     = "#0e3b2e"   # step 1 PSR / thermal (dark green)
C_S2     = "#1c2e4a"   # step 2 DFSAR (blue-grey)
C_S3     = "#2e2a0e"   # step 3 morphology (amber)
C_S4     = "#2e1a2e"   # step 4 landing (purple)
C_S5     = "#1a2e3a"   # step 5 traverse (teal)
C_S6     = "#0e2e2e"   # step 6 ice volume (cyan-dark)
C_OUT    = "#1a1a3a"   # output figures
EDGE     = "#30363d"
ARROW    = "#58a6ff"
WHITE    = "#e6edf3"
GRAY     = "#8b949e"
ACCENT   = "#58a6ff"

def box(ax, x, y, w, h, title, lines, color, title_color=ACCENT,
        title_size=8.5, body_size=7.2, corner=0.02):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0",
        linewidth=0.9, edgecolor=EDGE, facecolor=color, zorder=2
    )
    ax.add_patch(rect)
    # title bar (top stripe)
    stripe = FancyBboxPatch(
        (x, y + h - 0.38), w, 0.38,
        boxstyle="round,pad=0",
        linewidth=0, facecolor="#ffffff14", zorder=3
    )
    ax.add_patch(stripe)
    ax.text(x + w / 2, y + h - 0.19, title,
            ha="center", va="center", fontsize=title_size,
            color=title_color, fontweight="bold", zorder=4)
    for i, line in enumerate(lines):
        ax.text(x + 0.12, y + h - 0.58 - i * 0.215, line,
                ha="left", va="center", fontsize=body_size,
                color=WHITE, zorder=4, family="monospace")


def arrow(ax, x0, y0, x1, y1, label=""):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=ARROW,
                                lw=1.4, mutation_scale=12),
                zorder=5)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx + 0.08, my, label, ha="left", va="center",
                fontsize=6.2, color=GRAY, zorder=6)


def hline_arrow(ax, x0, y_mid, x1, y1, label=""):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y_mid),
                arrowprops=dict(
                    arrowstyle="-|>", color=ARROW, lw=1.2,
                    connectionstyle=f"arc3,rad=0.0",
                    mutation_scale=11),
                zorder=5)


# ── figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 22))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 16)
ax.set_ylim(0, 22)
ax.axis("off")

# ── title ────────────────────────────────────────────────────────────────────
ax.text(8, 21.4, "Chandrayaan-2 DFSAR Subsurface Ice Pipeline",
        ha="center", va="center", fontsize=14, fontweight="bold",
        color=WHITE)
ax.text(8, 21.0, "System Architecture & Data Flow",
        ha="center", va="center", fontsize=10, color=GRAY)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 0 — data inputs (3 boxes)
# ─────────────────────────────────────────────────────────────────────────────
Y_IN = 19.4
H_IN = 1.1

box(ax, 0.3,  Y_IN, 4.5, H_IN, "DFSAR  (L-band)",
    ["1.25 GHz compact polarimetry",
     "Left-circular TX, LH+LV RX",
     "Raw SC / OC channels"],
    C_IN, title_color="#79c0ff")

box(ax, 5.7,  Y_IN, 4.6, H_IN, "OHRC  Imagery",
    ["0.32 m/px optical camera",
     "Calibrated .img + .xml",
     "Browse PNG + geometry CSV"],
    C_IN, title_color="#79c0ff")

box(ax, 11.0, Y_IN, 4.7, H_IN, "DEM / Illumination",
    ["10 m/px elevation model",
     "Slope, aspect maps",
     "Illumination fraction [0,1]"],
    C_IN, title_color="#79c0ff")

# ─────────────────────────────────────────────────────────────────────────────
# ROW 1 — Step 0: Data Loading
# ─────────────────────────────────────────────────────────────────────────────
Y_S0 = 17.4
H_S0 = 1.15

arrow(ax, 2.55,  Y_IN,           2.55,  Y_S0 + H_S0)
arrow(ax, 8.0,   Y_IN,           8.0,   Y_S0 + H_S0)
arrow(ax, 13.35, Y_IN,           13.35, Y_S0 + H_S0)

box(ax, 0.3, Y_S0, 15.4, H_S0, "STEP 0  ·  DATA LOADING   (src/data_loader.py  |  src/data_generator.py)",
    ["Synthetic mode: physics-consistent 1000×1000 px scene (DEM, SAR backscatter, OHRC texture) generated on-the-fly",
     "Real mode: parse DFRS ZIPs, SAR .dat/.xml, OHRC .img/.xml; align geometry; expose unified dict"],
    C_S0, title_color="#ffa657")

# ─────────────────────────────────────────────────────────────────────────────
# ROW 2 — Step 1 (PSR) + Step 2 (DFSAR) — parallel
# ─────────────────────────────────────────────────────────────────────────────
Y_S12 = 14.5
H_S12 = 2.05

arrow(ax, 4.0, Y_S0, 3.6, Y_S12 + H_S12)
arrow(ax, 12.0, Y_S0, 12.4, Y_S12 + H_S12)

box(ax, 0.3, Y_S12, 7.0, H_S12,
    "STEP 1  ·  PSR MAPPING + THERMAL MODEL   (src/psr_mapping.py  |  src/thermal_model.py)",
    ["illum < 1% threshold  →  PSR patches via connected-component labelling",
     "Watershed segmentation  →  Doubly Shadowed Crater (DSC) detection inside PSR",
     "Lobate rim score · depth-to-diameter ratio · azimuthal rim height profile",
     "T⁴ = T_sunlit⁴ × f + T_shadow⁴ × (1−f)  [Paige et al. 2010, Diviner-calibrated]",
     "Cold-trap mask:  T < 110 K  →  thermally independent ice stability gate",
     "ISRU: water mass / propellant / crew-days from ice volume estimate"],
    C_S1, title_color="#56d364", title_size=7.8, body_size=6.8)

box(ax, 8.0, Y_S12, 7.7, H_S12,
    "STEP 2  ·  DFSAR POLARIMETRIC ANALYSIS   (src/dfsar_analysis.py)",
    ["Lee-sigma speckle filter  →  CPR = |SC|²/|OC|²",
     "Full Stokes DOP = √(Q²+U²+V²)/I  [fallback: |SC−OC|/(SC+OC)]",
     "Dual gate: CPR>0.8 AND DOP<0.13 AND PSR AND slope≤25° AND cluster≥5 px",
     "Bayesian posterior P(ice|CPR,DOP,PSR): calibrated LR × prior (P_ice=0.05)",
     "False-positive analysis: slope>18° rough-terrain FP · ejecta DOP>0.25 FP",
     "KS test · Mann-Whitney U · Cohen's d  on ice vs non-ice PSR populations",
     "CPR & DOP threshold sensitivity sweep (50 pts each)  →  plateau validation"],
    C_S2, title_color="#79c0ff", title_size=7.8, body_size=6.8)

# PSR → DFSAR feed arrow (horizontal)
ax.annotate("", xy=(8.0, Y_S12 + H_S12 * 0.5), xytext=(7.3, Y_S12 + H_S12 * 0.5),
            arrowprops=dict(arrowstyle="-|>", color="#f78166", lw=1.2, mutation_scale=10),
            zorder=5)
ax.text(7.65, Y_S12 + H_S12 * 0.5 + 0.1, "psr_mask", ha="center", va="bottom",
        fontsize=5.8, color="#f78166")

# ─────────────────────────────────────────────────────────────────────────────
# ROW 3 — Step 3: Morphology
# ─────────────────────────────────────────────────────────────────────────────
Y_S3 = 12.55
H_S3 = 1.15

arrow(ax, 3.8, Y_S12, 5.5, Y_S3 + H_S3)
arrow(ax, 11.9, Y_S12, 10.5, Y_S3 + H_S3)

box(ax, 0.3, Y_S3, 15.4, H_S3,
    "STEP 3  ·  CRATER MORPHOLOGY   (src/morphology.py)",
    ["RMS height roughness σ_h = √(<h²>−<h>²) over 21×21 px window · autocorrelation length l_c",
     "Boulder detection: local brightness contrast (OHRC) → morphological filter → density map (Gaussian KDE)",
     "Slope hazard: Safe ≤10° / Caution 10–15° / Danger 15–20° / Impassable >20°  [JAXA SELENE / Artemis standards]"],
    C_S3, title_color="#e3b341")

# ─────────────────────────────────────────────────────────────────────────────
# ROW 4 — Steps 4, 5, 6 — parallel
# ─────────────────────────────────────────────────────────────────────────────
Y_S456 = 9.8
H_S456 = 1.95

arrow(ax, 3.8,  Y_S3, 2.4,  Y_S456 + H_S456)
arrow(ax, 8.0,  Y_S3, 8.0,  Y_S456 + H_S456)
arrow(ax, 12.2, Y_S3, 13.2, Y_S456 + H_S456)

box(ax, 0.3, Y_S456, 4.7, H_S456,
    "STEP 4  ·  LANDING SITES\n(src/landing_site.py)",
    ["MCDA / AHP weights (CR=0.04):",
     "  slope 0.30 · illum. 0.22",
     "  science 0.22 · rough. 0.12",
     "  boulder 0.08 · comm 0.06",
     "200 m×200 m ellipse assessment",
     "Greedy NMS · 1 km exclusion",
     "→ 5 geographically diverse sites"],
    C_S4, title_color="#d2a8ff", title_size=7.8, body_size=6.8)

box(ax, 5.65, Y_S456, 4.7, H_S456,
    "STEP 5  ·  ROVER TRAVERSE\n(src/rover_traverse.py)",
    ["A* path planning, 8-connectivity",
     "Cost = w_s·slope² + w_r·rough",
     "     + w_e·energy + 3×PSR penalty",
     "Rover: 27 kg, 6 W solar, 5 cm/s",
     "Energy: m·g·h + m·g·cos(θ)·μ·d",
     "Power margin per waypoint:",
     "  6W × illum_fraction − 3.5W"],
    C_S5, title_color="#56d4dd", title_size=7.8, body_size=6.8)

box(ax, 11.0, Y_S456, 4.7, H_S456,
    "STEP 6  ·  ICE VOLUME + ISRU\n(src/ice_volume.py  |  thermal_model.py)",
    ["Polder-van Santen mixing model:",
     "  ε_eff = ε_h + f·Δε·ε_eff/denom",
     "CPR → f_ice via 601-point LUT",
     "Vol = f_ice × area × δ_penetration",
     "Monte Carlo 500 samples:",
     "  (1) CPR noise +-0.1 (Gaussian)",
     "  (2) Depth +-30% (log-normal)",
     "  (3) Dielectric +-20% (Gaussian)"],
    C_S6, title_color="#39d353", title_size=7.8, body_size=6.8)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 5 — Step 7: Visualisation
# ─────────────────────────────────────────────────────────────────────────────
Y_S7 = 8.0
H_S7 = 1.0

arrow(ax, 2.65, Y_S456, 4.5,  Y_S7 + H_S7)
arrow(ax, 8.0,  Y_S456, 8.0,  Y_S7 + H_S7)
arrow(ax, 13.35, Y_S456, 11.5, Y_S7 + H_S7)

box(ax, 0.3, Y_S7, 15.4, H_S7,
    "STEP 7  ·  VISUALISATION   (src/visualization.py)",
    ["7 publication-quality PNG figures + master dashboard  →  results/figures/"],
    C_S0, title_color="#ffa657")

# ─────────────────────────────────────────────────────────────────────────────
# ROW 6 — output figure tiles
# ─────────────────────────────────────────────────────────────────────────────
Y_FIG = 5.6
H_FIG = 1.6

arrow(ax, 8.0, Y_S7, 8.0, Y_FIG + H_FIG)

fig_defs = [
    ("00_dashboard.png",   "#ffa657", "Master 8-panel\nsummary + stats"),
    ("01_overview.png",    "#79c0ff", "DEM · illum\nPSR · temp (K)"),
    ("02_dfsar_analysis.png","#79c0ff","CPR · DOP · ice\nposterior · sensitivity"),
    ("03_morphology.png",  "#e3b341", "Slope · rough.\nhazard · rim"),
    ("04_landing_site.png","#d2a8ff", "MCDA score\nAHP breakdown"),
    ("05_traverse.png",    "#56d4dd", "Path · power\nmargin · elev."),
    ("06_ice_volume.png",  "#39d353", "f_ice · depth\nviolin · strat."),
]

tile_w = 2.1
gap    = 0.12
total  = len(fig_defs) * tile_w + (len(fig_defs) - 1) * gap
x0     = (16 - total) / 2

for i, (fname, color, desc) in enumerate(fig_defs):
    tx = x0 + i * (tile_w + gap)
    rect = FancyBboxPatch((tx, Y_FIG), tile_w, H_FIG,
                          boxstyle="round,pad=0",
                          linewidth=1.0, edgecolor=color, facecolor="#161b22",
                          zorder=2)
    ax.add_patch(rect)
    stripe = FancyBboxPatch((tx, Y_FIG + H_FIG - 0.32), tile_w, 0.32,
                            boxstyle="round,pad=0",
                            linewidth=0, facecolor=color + "33", zorder=3)
    ax.add_patch(stripe)
    ax.text(tx + tile_w / 2, Y_FIG + H_FIG - 0.16, fname,
            ha="center", va="center", fontsize=5.4,
            color=color, fontweight="bold", zorder=4)
    ax.text(tx + tile_w / 2, Y_FIG + H_FIG * 0.38, desc,
            ha="center", va="center", fontsize=6.2,
            color=WHITE, zorder=4, linespacing=1.4)

# ─────────────────────────────────────────────────────────────────────────────
# Key data-flow labels on main arrows
# ─────────────────────────────────────────────────────────────────────────────
flow_labels = [
    (8.0, 17.1, "dem, illum, slope, dfsar, ohrc, meta"),
    (8.0, 14.2, "psr_mask, dsc_center, T_surface, ice_mask, CPR, DOP, ice_conf"),
    (8.0, 12.2, "slope, roughness, hazard, boulders, composite_score, traverse"),
    (8.0,  9.5, "all results"),
    (8.0,  7.7, "figure arrays"),
]
for (fx, fy, flabel) in flow_labels:
    ax.text(fx, fy, flabel, ha="center", va="center",
            fontsize=6.0, color=GRAY, style="italic", zorder=6)

# ─────────────────────────────────────────────────────────────────────────────
# Legend
# ─────────────────────────────────────────────────────────────────────────────
legend_items = [
    (C_IN,    "Raw data input"),
    (C_S1,    "PSR / thermal module"),
    (C_S2,    "Polarimetric analysis"),
    (C_S3,    "Morphology module"),
    (C_S4,    "Landing site MCDA"),
    (C_S5,    "Rover traverse A*"),
    (C_S6,    "Ice volume / ISRU"),
]
lx, ly = 0.3, 4.8
ax.text(lx, ly + 0.35, "Module colour key", fontsize=7, color=GRAY,
        fontweight="bold")
for i, (col, label) in enumerate(legend_items):
    rx = lx + i * 2.2
    rect = FancyBboxPatch((rx, ly - 0.22), 0.32, 0.28,
                          boxstyle="round,pad=0",
                          linewidth=0.6, edgecolor=EDGE, facecolor=col, zorder=2)
    ax.add_patch(rect)
    ax.text(rx + 0.40, ly - 0.08, label, fontsize=6.2, color=GRAY, va="center")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
ax.text(8, 0.35,
        "Chandrayaan-2 DFSAR Investigation  ·  Faustini PSR Doubly Shadowed Crater  ·  "
        "L-band 1.25 GHz compact polarimetry  ·  10 m/px  ·  Python 3.9+",
        ha="center", va="center", fontsize=6.5, color="#444c56")

fig.tight_layout(pad=0.2)
fig.savefig("results/figures/architecture.png", dpi=180,
            bbox_inches="tight", facecolor=BG)
print("Saved: results/figures/architecture.png")
