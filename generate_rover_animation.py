"""
Rover Traverse Animation  — enhanced with:
  1. Battery state-of-charge panel
  2. Full A* path (smooth movement, no simplified jumps)
  3. Radar cross-section panel (L-band subsurface view)

Layout  (18×10 figure, 2-row × 3-col grid):
  Col 0 (full height) : Terrain + hazard map
  Col 1 top           : Elevation profile
  Col 1 bottom        : Solar power margin
  Col 2 top           : L-band radar cross-section
  Col 2 bottom        : Battery state-of-charge

Output: results/figures/rover_animation.gif
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines   as mlines
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

# ── Rover parameters ──────────────────────────────────────────────────────────
ROVER_MASS_KG       = 27.0
SOLAR_POWER_W       = 6.0
DRIVE_POWER_W       = 4.5
SPEED_FLAT_MPS      = 0.05
LUNAR_G             = 1.62
BATTERY_CAPACITY_WH = 50.0       # realistic for a small lunar rover
BATTERY_CAPACITY_J  = BATTERY_CAPACITY_WH * 3600.0
PIXEL_SCALE_M       = 10.0
LBAND_PENETRATION_M = 4.5        # L-band max penetration in dry regolith

N_FRAMES = 90
FPS      = 12
DPI      = 110


# ── Data loading ──────────────────────────────────────────────────────────────

def load_pipeline_data():
    from src.data_generator import load_all
    from src.dfsar_analysis  import run_analysis
    from src.morphology      import run_morphology
    from src.landing_site    import composite_landing_score, select_landing_site
    from src.rover_traverse  import build_cost_map, plan_traverse
    from src.thermal_model   import compute_surface_temperature

    print("[Anim] Loading cached data...")
    data = load_all(cache=True)
    dem=data["dem"]; slope=data["slope"]; illum=data["illum"]
    psr_mask=data["psr_mask"]; dsc_mask=data["dsc_mask"]
    dfsar=data["dfsar"]; ohrc=data["ohrc"]; meta=data["meta"]
    gs=meta["grid_size"]; ps=meta["pixel_scale"]
    dsc_c=meta["dsc_center"]; dsc_r=meta["dsc_radius"]

    print("[Anim] Running analysis & path planning...")
    res   = run_analysis(dfsar, psr_mask, slope, cpr_thresh=0.8, dop_thresh=0.13)
    morph = run_morphology(dem, ohrc, dsc_c, dsc_r, ps)

    comp, fmaps = composite_landing_score(
        slope, morph["roughness"], morph["boulder_density"],
        illum, dem, psr_mask, dsc_c, dsc_r, ps)
    candidates = select_landing_site(comp, psr_mask, slope, illum,
                                      n_candidates=5, exclusion_radius=100,
                                      factor_maps=fmaps)
    best = candidates[0] if candidates else dict(row=50, col=gs//2)

    cost_map = build_cost_map(slope, morph["roughness"], psr_mask, illum, pixel_scale=ps)
    traverse = plan_traverse(cost_map, dem, slope, best, dsc_c, dsc_r,
                              roughness=morph["roughness"],
                              illum_fraction=illum, pixel_scale=ps)
    return dict(
        dem=dem, slope=slope, illum=illum, psr_mask=psr_mask,
        dsc_mask=dsc_mask, hazard=morph["hazard"],
        ice_mask=res["ice_mask"], CPR=res["CPR"],
        traverse=traverse, dsc_c=dsc_c, dsc_r=dsc_r, gs=gs, ps=ps,
    )


# ── Physics helpers ───────────────────────────────────────────────────────────

def compute_power_and_battery(path, slope, illum):
    solar, drive, margin, soc = [], [], [], []
    charge = BATTERY_CAPACITY_J
    for i, (r, c) in enumerate(path):
        illum_f = float(illum[r, c])
        sl      = float(slope[r, c])
        sol = SOLAR_POWER_W * illum_f
        drv = DRIVE_POWER_W * (1 + 0.4 * np.sin(np.radians(sl)))
        net = sol - drv
        solar.append(sol); drive.append(drv); margin.append(net)

        # Step time (seconds at nominal speed)
        if i > 0:
            pr, pc = path[i-1]
            dist = np.sqrt((r-pr)**2 + (c-pc)**2) * PIXEL_SCALE_M
            spd  = max(SPEED_FLAT_MPS * np.cos(np.radians(sl)), 0.001)
            dt   = dist / spd
        else:
            dt = 0.0
        charge = np.clip(charge + net * dt, 0, BATTERY_CAPACITY_J)
        soc.append(charge / BATTERY_CAPACITY_J * 100.0)

    return (np.array(solar), np.array(drive),
            np.array(margin), np.array(soc))


def build_terrain_rgb(dem, hazard, psr_mask, dsc_mask, ice_mask, illum):
    dy, dx = np.gradient(dem.astype(np.float64))
    hs = np.clip((1 - dx*0.4 - dy*0.3 - (1 - dx*0.4 - dy*0.3).min()) /
                 ((1 - dx*0.4 - dy*0.3).ptp() + 1e-6), 0, 1)
    rgb = np.stack([hs, hs, hs], axis=-1)
    rgb = np.where(psr_mask[:,:,None], rgb*0.3 + np.array([0.05,0.10,0.35]), rgb)
    for cls, col in {1:[0.9,0.85,0.1], 2:[0.9,0.5,0.1], 3:[0.85,0.1,0.1]}.items():
        m = hazard == cls
        if m.any():
            rgb = np.where(m[:,:,None], rgb*0.55 + np.array(col)*0.45, rgb)
    if dsc_mask.any():
        rgb[dsc_mask] = rgb[dsc_mask]*0.6 + np.array([0.1,0.2,0.5])*0.4
    if ice_mask.any():
        rgb[ice_mask] = [0.0, 1.0, 0.6]
    return np.clip(rgb, 0, 1).astype(np.float32)


def build_radar_xsec(path, dem, CPR, ice_mask, ps):
    """
    Build a 2-D radar cross-section image along the traverse path.
    Rows = depth (0 at surface → LBAND_PENETRATION_M deep).
    Cols = path waypoints.
    Values: 0=regolith, 1=radar cone, 2=ice anomaly.
    Also returns the surface elevation profile and normalised CPR along path.
    """
    N        = len(path)
    N_DEPTH  = 60       # depth pixels
    xsec     = np.zeros((N_DEPTH, N), dtype=np.float32)
    elev     = np.array([float(dem[r, c]) for r, c in path])
    cpr_path = np.array([float(CPR[r, c])  for r, c in path])
    is_ice   = np.array([ice_mask[r, c]    for r, c in path])

    # Fill: regolith = 0.3 (base colour); ice anomaly at depth if CPR > 0.8
    xsec[:] = 0.3
    for j, ((r, c), cpr_val, ice_flag) in enumerate(zip(path, cpr_path, is_ice)):
        # Ice / anomaly layer: depth proportional to penetration
        if cpr_val > 0.8:
            # Detected anomaly depth: shallower for higher CPR
            ice_depth_frac = max(0.1, 0.6 - (cpr_val - 0.8) * 0.3)
            d0 = int(N_DEPTH * ice_depth_frac)
            d1 = min(N_DEPTH, d0 + max(3, int(N_DEPTH * 0.18)))
            xsec[d0:d1, j] = 0.85   # ice layer highlight

    return xsec, elev, cpr_path, is_ice


# ── Main animation builder ────────────────────────────────────────────────────

def make_animation(data):
    path_full = data["traverse"]["path"]
    metrics   = data["traverse"]["metrics"]
    dem       = data["dem"]
    slope     = data["slope"]
    illum     = data["illum"]
    psr_mask  = data["psr_mask"]
    dsc_mask  = data["dsc_mask"]
    hazard    = data["hazard"]
    ice_mask  = data["ice_mask"]
    CPR       = data["CPR"]
    dsc_c     = data["dsc_c"]
    dsc_r     = data["dsc_r"]
    gs        = data["gs"]
    ps        = data["ps"]

    if not path_full:
        print("[Anim] No traverse path found."); return

    path = np.array(path_full)
    N    = len(path)
    KM   = ps / 1000.0
    frame_idx = np.linspace(0, N-1, N_FRAMES, dtype=int)

    elev_arr                       = np.array([dem[r,c]   for r,c in path_full])
    dist_km                        = np.arange(N) * ps / 1000.0
    solar, drive, power_margin, soc = compute_power_and_battery(path_full, slope, illum)
    xsec, _, cpr_path, is_ice      = build_radar_xsec(path_full, dem, CPR, ice_mask, ps)

    terrain_rgb = build_terrain_rgb(dem, hazard, psr_mask, dsc_mask, ice_mask, illum)

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10), facecolor="#0a0a12")
    gs_spec = gridspec.GridSpec(
        2, 3, figure=fig,
        width_ratios=[1.45, 1, 1],
        hspace=0.38, wspace=0.30,
        left=0.04, right=0.97, top=0.88, bottom=0.08)

    ax_map  = fig.add_subplot(gs_spec[:, 0])
    ax_elev = fig.add_subplot(gs_spec[0, 1])
    ax_pwr  = fig.add_subplot(gs_spec[1, 1])
    ax_xsec = fig.add_subplot(gs_spec[0, 2])
    ax_batt = fig.add_subplot(gs_spec[1, 2])

    for ax in (ax_map, ax_elev, ax_pwr, ax_xsec, ax_batt):
        ax.set_facecolor("#0d0d1a")
        for sp in ax.spines.values(): sp.set_color("#334")

    # ── Terrain map ───────────────────────────────────────────────────────────
    ax_map.imshow(terrain_rgb, origin="upper", aspect="auto",
                  extent=[0, gs*KM, gs*KM, 0], interpolation="bilinear")
    ax_map.add_patch(plt.Circle((dsc_c[1]*KM, dsc_c[0]*KM), dsc_r*KM,
                                 color="yellow", fill=False, lw=1.5, ls="--"))
    ax_map.text(dsc_c[1]*KM, (dsc_c[0]-dsc_r-0.12)*KM,
                "DSC floor", color="yellow", fontsize=7.5, ha="center")
    ax_map.plot(path[:,1]*KM, path[:,0]*KM,
                color="#ffffff", lw=0.5, alpha=0.15, zorder=2)
    ax_map.plot(path[0,1]*KM,  path[0,0]*KM,  "g^", ms=10, zorder=6)
    ax_map.plot(dsc_c[1]*KM,   dsc_c[0]*KM,   "r*", ms=13, zorder=6)
    ax_map.set_xlabel("Distance East (km)",  color="#aaa", fontsize=8)
    ax_map.set_ylabel("Distance South (km)", color="#aaa", fontsize=8)
    ax_map.tick_params(colors="#888", labelsize=7)

    # Full, self-contained legend
    legend_elems = [
        mpatches.Patch(color=[0.05,0.10,0.35], label="PSR — permanent shadow"),
        mpatches.Patch(color=[0.9,0.85,0.1],   label="Caution slope  (10-15 deg)"),
        mpatches.Patch(color=[0.9,0.5,0.1],    label="Danger slope   (15-20 deg)"),
        mpatches.Patch(color=[0.0,1.0,0.6],    label="Ice detection  (CPR>0.8, DOP<0.13)"),
        mlines.Line2D([], [], color="#00ff88", lw=3.2,
                      label="Rover traverse path"),
        mlines.Line2D([], [], marker="^", color="none",
                      markerfacecolor="#27ae60", markeredgecolor="#27ae60",
                      markersize=9,  label="Landing site (optimal MCDA)"),
        mlines.Line2D([], [], marker="*", color="none",
                      markerfacecolor="#e74c3c", markeredgecolor="#e74c3c",
                      markersize=11, label="DSC target (ice deposit)"),
    ]
    ax_map.legend(handles=legend_elems, loc="lower left", fontsize=6.0,
                  facecolor="#05050dcc", edgecolor="#445566",
                  labelcolor="#e8e8e8", framealpha=0.88,
                  handlelength=1.8, handletextpad=0.6, borderpad=0.7)

    # Static mission summary
    total_km   = metrics.get("total_distance_m", 0) / 1000
    total_hr   = metrics.get("estimated_time_hr", 0)
    elev_desc  = elev_arr[-1] - elev_arr[0]
    psr_pct    = metrics.get("psr_fraction", 0) * 100
    e_kj       = metrics.get("total_energy_kJ", 0)
    term_illum = float(illum[path[-1,0], path[-1,1]]) * 100
    term_power = float(power_margin[-1])

    ax_map.text(0.02, 0.985,
        "Mission Profile\n"
        "-------------------------\n"
        f"Traverse distance  : {total_km:.2f} km\n"
        f"Estimated duration : {total_hr:.1f} hr\n"
        f"Elevation descent  : {elev_desc:+.0f} m\n"
        f"Energy required    : {e_kj:.1f} kJ\n"
        f"Path in PSR        : {psr_pct:.0f}%\n"
        f"Terminal illum.    : {term_illum:.0f}%\n"
        f"Terminal power     : {term_power:+.1f} W",
        transform=ax_map.transAxes, color="#e8e8e8",
        fontsize=7.5, va="top", fontfamily="monospace", linespacing=1.55,
        bbox=dict(fc="#05050dcc", ec="#445566", lw=0.9, pad=6,
                  boxstyle="round,pad=0.5"))

    # ── Elevation profile ─────────────────────────────────────────────────────
    ax_elev.fill_between(dist_km, elev_arr, elev_arr.min()-5,
                          color="#2255aa", alpha=0.35)
    ax_elev.plot(dist_km, elev_arr, color="#5599ff", lw=1.4)
    ax_elev.set_xlabel("Distance from landing (km)", color="#aaa", fontsize=7.5)
    ax_elev.set_ylabel("Elevation (m)", color="#aaa", fontsize=7.5)
    ax_elev.set_title("Elevation Profile", color="white", fontsize=8, fontweight="bold")
    ax_elev.set_xlim(0, dist_km[-1]); ax_elev.grid(alpha=0.15, color="#445")
    ax_elev.tick_params(colors="#888", labelsize=7)

    # ── Solar power margin ────────────────────────────────────────────────────
    pos = power_margin >= 0
    ax_pwr.fill_between(dist_km, power_margin, 0, where=pos,
                         color="#27ae60", alpha=0.5, label="Surplus power")
    ax_pwr.fill_between(dist_km, power_margin, 0, where=~pos,
                         color="#e74c3c", alpha=0.5, label="Power deficit")
    ax_pwr.axhline(0, color="#888", lw=0.8, ls="--")
    ax_pwr.set_xlabel("Distance from landing (km)", color="#aaa", fontsize=7.5)
    ax_pwr.set_ylabel("Net power (W)", color="#aaa", fontsize=7.5)
    ax_pwr.set_title("Solar Power Margin", color="white", fontsize=8, fontweight="bold")
    ax_pwr.set_xlim(0, dist_km[-1]); ax_pwr.grid(alpha=0.15, color="#445")
    ax_pwr.legend(fontsize=6.5, facecolor="#111", edgecolor="#444",
                  labelcolor="white", loc="upper right")
    ax_pwr.tick_params(colors="#888", labelsize=7)

    # ── Radar cross-section (static background) ───────────────────────────────
    depth_axis = np.linspace(0, LBAND_PENETRATION_M, xsec.shape[0])
    ax_xsec.imshow(xsec, cmap="plasma", vmin=0, vmax=1, aspect="auto",
                   extent=[0, dist_km[-1], LBAND_PENETRATION_M, 0],
                   origin="upper", interpolation="bilinear")
    # Annotate ice layer region
    ax_xsec.axhline(1.5, color="cyan", lw=0.8, ls=":", alpha=0.6)
    ax_xsec.axhline(3.5, color="cyan", lw=0.8, ls=":", alpha=0.6)
    ax_xsec.text(dist_km[-1]*0.01, 2.5, "Ice anomaly\nzone",
                 color="cyan", fontsize=6, va="center")
    ax_xsec.set_xlabel("Distance (km)", color="#aaa", fontsize=7.5)
    ax_xsec.set_ylabel("Depth below surface (m)", color="#aaa", fontsize=7.5)
    ax_xsec.set_title("L-band Radar Cross-Section\n(1.25 GHz, ~4.5 m penetration)",
                       color="white", fontsize=8, fontweight="bold")
    ax_xsec.set_xlim(0, dist_km[-1])
    ax_xsec.set_ylim(LBAND_PENETRATION_M, 0)   # depth increases downward
    ax_xsec.tick_params(colors="#888", labelsize=7)
    # CPR profile along path (overlaid as line on top)
    cpr_norm = np.clip((cpr_path - 0.2) / 2.0, 0, 1) * LBAND_PENETRATION_M * 0.9
    ax_xsec.plot(dist_km, cpr_norm, color="#ffcc00", lw=1.2, alpha=0.7,
                 label="CPR (scaled)")
    ax_xsec.axhline(0, color="white", lw=0.5, alpha=0.3)
    ax_xsec.legend(fontsize=6, facecolor="#111", edgecolor="#444",
                   labelcolor="white", loc="lower right")

    # ── Battery SoC ───────────────────────────────────────────────────────────
    # Zone colouring: green > 50%, yellow 20-50%, red < 20%
    ax_batt.fill_between(dist_km, soc, 0,
                          where=(soc >= 50), color="#27ae60", alpha=0.55,
                          label="Good  (>50%)")
    ax_batt.fill_between(dist_km, soc, 0,
                          where=(soc >= 20) & (soc < 50), color="#f39c12",
                          alpha=0.55, label="Low   (20-50%)")
    ax_batt.fill_between(dist_km, soc, 0,
                          where=(soc < 20), color="#e74c3c", alpha=0.55,
                          label="Critical (<20%)")
    ax_batt.axhline(20,  color="#e74c3c", lw=0.9, ls="--", alpha=0.7)
    ax_batt.axhline(50,  color="#f39c12", lw=0.9, ls="--", alpha=0.7)
    ax_batt.axhline(100, color="#aaa",    lw=0.5, ls=":",  alpha=0.5)
    ax_batt.set_xlabel("Distance from landing (km)", color="#aaa", fontsize=7.5)
    ax_batt.set_ylabel("Battery SoC (%)", color="#aaa", fontsize=7.5)
    ax_batt.set_title(f"Battery State-of-Charge\n({BATTERY_CAPACITY_WH:.0f} Wh capacity)",
                       color="white", fontsize=8, fontweight="bold")
    ax_batt.set_xlim(0, dist_km[-1])
    ax_batt.set_ylim(0, 105)
    ax_batt.grid(alpha=0.15, color="#445")
    ax_batt.legend(fontsize=6.5, facecolor="#111", edgecolor="#444",
                   labelcolor="white", loc="lower left")
    ax_batt.tick_params(colors="#888", labelsize=7)

    # ── Figure title ──────────────────────────────────────────────────────────
    fig.suptitle(
        "Lunar Polar Rover Traverse  |  "
        "Landing Site to Doubly Shadowed Crater (DSC)  |  "
        f"{total_km:.2f} km  |  {total_hr:.1f} hr",
        color="white", fontsize=10, fontweight="bold", y=0.96)

    # ── Dynamic artists ───────────────────────────────────────────────────────
    trail_line, = ax_map.plot([], [], color="#00ff88", lw=3.2, alpha=0.85, zorder=4)
    rover_dot,  = ax_map.plot([], [], "o", ms=11, color="white",
                               markeredgecolor="#00ff88", markeredgewidth=2.5, zorder=5)

    elev_vline = ax_elev.axvline(0, color="#ffcc00", lw=1.5, ls="--", alpha=0.85)
    elev_dot,  = ax_elev.plot([], [], "o", color="#ffcc00", ms=6, zorder=5)

    pwr_vline  = ax_pwr.axvline(0, color="#ffcc00", lw=1.5, ls="--", alpha=0.85)
    pwr_dot,   = ax_pwr.plot([], [], "o", color="#ffcc00", ms=6, zorder=5)

    # Radar beam cone on cross-section
    xsec_vline = ax_xsec.axvline(0, color="#ffcc00", lw=1.5, ls="--", alpha=0.85)
    _beam_fill_container = [ax_xsec.fill_betweenx(
        [0, LBAND_PENETRATION_M], [0, 0], [0, 0],
        color="white", alpha=0.12)]

    batt_vline = ax_batt.axvline(0, color="#ffcc00", lw=1.5, ls="--", alpha=0.85)
    batt_dot,  = ax_batt.plot([], [], "o", color="#ffcc00", ms=6, zorder=5)

    # ── Update function ───────────────────────────────────────────────────────
    def update(frame):
        idx = frame_idx[frame]
        r, c = path[idx]
        d    = dist_km[idx]

        # Trail
        trail_line.set_data(path[:idx+1, 1]*KM, path[:idx+1, 0]*KM)
        rover_dot.set_data([c*KM], [r*KM])

        # Elevation cursor
        elev_vline.set_xdata([d])
        elev_dot.set_data([d], [elev_arr[idx]])

        # Power cursor
        pwr_vline.set_xdata([d])
        pwr_dot.set_data([d], [power_margin[idx]])

        # Radar cross-section cursor + beam cone
        xsec_vline.set_xdata([d])
        half_bw = 0.05   # km half-beam-width at surface
        beam_left  = np.array([d - half_bw, d - half_bw - LBAND_PENETRATION_M * 0.36])
        beam_right = np.array([d + half_bw, d + half_bw + LBAND_PENETRATION_M * 0.36])
        # Remove previous beam fill and draw a new one
        _beam_fill_container[0].remove()
        _beam_fill_container[0] = ax_xsec.fill_betweenx(
            [0, LBAND_PENETRATION_M],
            [beam_left[0],  beam_left[1]],
            [beam_right[0], beam_right[1]],
            color="white", alpha=0.12, zorder=3)

        # Battery cursor
        batt_vline.set_xdata([d])
        batt_dot.set_data([d], [soc[idx]])

        return (trail_line, rover_dot,
                elev_vline, elev_dot,
                pwr_vline,  pwr_dot,
                xsec_vline,
                batt_vline, batt_dot)

    # ── Save ─────────────────────────────────────────────────────────────────
    anim = FuncAnimation(fig, update, frames=N_FRAMES,
                          interval=1000//FPS, blit=False)
    out = Path("results/figures/rover_animation.gif")
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Anim] Rendering {N_FRAMES} frames at {FPS} fps -> {out}")
    anim.save(str(out), writer=PillowWriter(fps=FPS), dpi=DPI)
    print(f"[Anim] Saved GIF: {out}  ({out.stat().st_size // 1024} KB)")

    # MP4 export — use imageio-ffmpeg binary if system ffmpeg not found
    try:
        import imageio_ffmpeg
        import os
        os.environ["PATH"] = (
            os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
            + os.pathsep + os.environ.get("PATH", "")
        )
    except ImportError:
        pass

    try:
        from matplotlib.animation import FFMpegWriter
        out_mp4 = out.with_suffix(".mp4")
        writer  = FFMpegWriter(fps=FPS, codec="h264",
                               extra_args=["-pix_fmt", "yuv420p", "-crf", "22"])
        anim.save(str(out_mp4), writer=writer, dpi=DPI)
        print(f"[Anim] Saved MP4: {out_mp4}  ({out_mp4.stat().st_size // 1024} KB)")
    except Exception as e:
        print(f"[Anim] MP4 skipped: {e}")

    plt.close(fig)


if __name__ == "__main__":
    data = load_pipeline_data()
    make_animation(data)
