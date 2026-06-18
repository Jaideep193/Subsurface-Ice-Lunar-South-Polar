"""
Rover Traverse Path Planning with Energy and Solar-Power Model
==============================================================

Cost function (per pixel):
    Cost = w_s * slope_cost + w_r * roughness_cost
         + w_e * energy_cost + w_p * power_risk

where:
    slope_cost   = (slope/max_slope)^2          (quadratic penalty)
    roughness_cost = roughness/max_roughness
    energy_cost  = traversal energy ∝ slope + rolling resistance
    power_risk   = shadow_penalty (high inside PSR; low outside)

Rover model (Chandrayaan-3 class):
    Max slope       : 20° (hard limit, same as Pragyan)
    Speed           : 1 cm/s = ~36 m/hr on flat terrain
    Power source    : Solar (6W nominal from 4W/m² at south pole × 1.5 m² panels)
    Shadow penalty  : 3× cost multiplier inside PSR
    Communication   : Available when relay orbiter overhead (simplified: always)

Energy model (from rover dynamics):
    E_traverse (J) = m*g*h_gain + m*g*cos(theta)*mu*d
    where m=27 kg, mu=0.15 (regolith rolling resistance), d=distance
    This gives a cost proportional to slope and distance.

Path planning: A* with 8-connectivity and bi-directional extension.

References:
  Arvidson et al. 2011 (MER rover mobility on planetary surfaces)
  Iagnemma & Dubowsky 2004 (rover locomotion on soft terrain)
  Chandrayaan-3 mission design documents (ISRO)
"""

import heapq
import numpy as np
from scipy.ndimage import gaussian_filter


IMPASSABLE = 1e9

# Rover parameters (Chandrayaan-3 class)
ROVER_MASS_KG       = 27.0
ROLLING_RESISTANCE  = 0.15    # mu (lunar regolith)
LUNAR_G             = 1.62    # m/s²
MAX_SLOPE_DEG       = 20.0
SHADOW_PENALTY      = 3.0     # cost multiplier in PSR
SOLAR_PANEL_W       = 6.0     # Watts available on lit terrain
POWER_DRAW_W        = 3.5     # Watts consumed while moving
SPEED_FLAT_MPS      = 0.05    # m/s on flat (5 cm/s – LUPEX-class rover)


# ---------------------------------------------------------------------------
# Cost map construction
# ---------------------------------------------------------------------------

def build_cost_map(slope, roughness, psr_mask, illum_fraction,
                    max_slope=MAX_SLOPE_DEG, pixel_scale=10.0,
                    weights=None):
    """
    Build physically motivated traversal cost map.

    Each pixel cost = energy to traverse + risk penalty, normalised
    so that a flat lit pixel has cost = 1.0.

    Parameters
    ----------
    slope          : 2-D float  (degrees)
    roughness      : 2-D float  (m RMS)
    psr_mask       : 2-D bool   (True = PSR / shadow)
    illum_fraction : 2-D float  [0,1] illumination fraction
    max_slope      : float      hard cutoff (degrees)
    weights        : dict or None

    Returns
    -------
    cost_map : 2-D float32
    """
    if weights is None:
        weights = dict(slope=0.50, roughness=0.20, energy=0.20, power=0.10)

    # ── Slope cost (quadratic – severe penalty near max slope) ──
    slope_norm  = np.clip(slope / max_slope, 0, 1)
    slope_cost  = slope_norm ** 2   # 0 → 0, max_slope → 1

    # ── Roughness cost ──
    rough_norm  = np.clip(roughness / 5.0, 0, 1)

    # ── Energy cost (from rover dynamics) ──
    # E per pixel = m*g*(sin_theta*d + mu*cos_theta*d)
    theta   = np.radians(slope)
    d_m     = pixel_scale
    e_slope = ROVER_MASS_KG * LUNAR_G * np.sin(theta) * d_m
    e_roll  = ROVER_MASS_KG * LUNAR_G * np.cos(theta) * ROLLING_RESISTANCE * d_m
    e_total = e_slope + e_roll
    e_max   = ROVER_MASS_KG * LUNAR_G * (np.sin(np.radians(max_slope)) +
                                           ROLLING_RESISTANCE) * d_m
    energy_cost = np.clip(e_total / e_max, 0, 1)

    # ── Power risk (solar availability) ──
    # Inside PSR: heavy penalty (rover runs on battery, limited range)
    # Near PSR boundary with low illumination: moderate penalty
    power_score = np.where(
        psr_mask, SHADOW_PENALTY,
        np.where(illum_fraction < 0.3, 1.5, 1.0)
    )

    # ── Composite cost ──
    cost = (1.0
            + weights["slope"]     * slope_cost
            + weights["roughness"] * rough_norm
            + weights["energy"]    * energy_cost) * power_score

    # Hard impassable: slope above maximum (use smoothed slope – rover navigates
    # at >10 m scale so pixel-scale roughness spikes don't block the path)
    slope_smooth = gaussian_filter(slope.astype(np.float64), sigma=2.0)
    cost[slope_smooth > max_slope] = IMPASSABLE

    cost = gaussian_filter(cost.astype(np.float64), sigma=0.5)
    return cost.astype(np.float32)


# ---------------------------------------------------------------------------
# A* path planning
# ---------------------------------------------------------------------------

def astar(cost_map, start, goal, connectivity=8):
    """
    A* search on 2-D cost grid with 8-connectivity.

    Returns (path, total_cost).  path is empty if no path found.
    """
    rows, cols = cost_map.shape

    if connectivity == 4:
        neighbors = [(-1,0),(1,0),(0,-1),(0,1)]
        move_cost = [1.0, 1.0, 1.0, 1.0]
    else:
        neighbors = [(-1,0),(1,0),(0,-1),(0,1),
                     (-1,-1),(-1,1),(1,-1),(1,1)]
        move_cost = [1.0, 1.0, 1.0, 1.0,
                     np.sqrt(2), np.sqrt(2), np.sqrt(2), np.sqrt(2)]

    def h(r, c):
        return np.sqrt((r - goal[0])**2 + (c - goal[1])**2)

    g_score   = np.full((rows, cols), np.inf)
    g_score[start] = 0.0
    came_from = {}
    pq        = [(h(*start), 0.0, start[0], start[1])]

    while pq:
        f, g, r, c = heapq.heappop(pq)
        if (r, c) == goal:
            return _reconstruct_path(came_from, (r, c)), float(g)
        if g > g_score[r, c] + 1e-9:
            continue
        for (dr, dc), mc in zip(neighbors, move_cost):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            tile  = cost_map[nr, nc]
            if tile >= IMPASSABLE:
                continue
            new_g = g + tile * mc
            if new_g < g_score[nr, nc]:
                g_score[nr, nc] = new_g
                came_from[(nr, nc)] = (r, c)
                heapq.heappush(pq, (new_g + h(nr, nc), new_g, nr, nc))

    return [], float("inf")


def _reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]


# ---------------------------------------------------------------------------
# Waypoint simplification (Ramer-Douglas-Peucker)
# ---------------------------------------------------------------------------

def rdp_simplify(path, epsilon=2.0):
    """Reduce waypoints while preserving path shape."""
    if len(path) < 3:
        return path
    pts = np.array(path, dtype=np.float64)

    def _rdp(pts, eps):
        if len(pts) < 3:
            return pts.tolist()
        start, end = pts[0], pts[-1]
        line   = end - start
        lenl   = np.linalg.norm(line)
        if lenl == 0:
            dists = np.linalg.norm(pts - start, axis=1)
        else:
            t     = np.dot(pts - start, line) / lenl**2
            proj  = start + np.outer(np.clip(t, 0, 1), line)
            dists = np.linalg.norm(pts - proj, axis=1)
        idx = np.argmax(dists)
        if dists[idx] > eps:
            return _rdp(pts[:idx+1], eps)[:-1] + _rdp(pts[idx:], eps)
        return [pts[0].tolist(), pts[-1].tolist()]

    simp = _rdp(pts, epsilon)
    return [tuple(map(int, p)) for p in simp]


# ---------------------------------------------------------------------------
# Path metrics with energy model
# ---------------------------------------------------------------------------

def path_metrics(path, cost_map, dem, slope, roughness=None,
                  illum_fraction=None, pixel_scale=10.0):
    """
    Compute traverse statistics including energy budget and power analysis.
    """
    if not path:
        return dict(total_distance_m=0, total_cost=0,
                    max_slope_deg=0, elevation_change_m=0,
                    n_waypoints=0, total_energy_J=0,
                    psr_fraction=0, waypoints=[])

    pts   = np.array(path)
    diffs = np.diff(pts, axis=0)
    step_dists = np.linalg.norm(diffs, axis=1) * pixel_scale

    total_dist    = float(step_dists.sum())
    total_cost    = float(sum(cost_map[r, c] for r, c in path))
    max_slope_deg = float(max(slope[r, c] for r, c in path))
    elevations    = [float(dem[r, c]) for r, c in path]
    elev_range    = max(elevations) - min(elevations)

    # Energy budget per step
    thetas  = np.radians([slope[r, c] for r, c in path])
    e_steps = (ROVER_MASS_KG * LUNAR_G *
               (np.sin(thetas) + ROLLING_RESISTANCE * np.cos(thetas)))
    # Combine with step distances (prepend zero for first node)
    step_d_all = np.concatenate([[0], step_dists])
    total_E_J  = float((e_steps * step_d_all).sum())

    # Time (seconds at 1 cm/s, scaling by slope penalty)
    speed_factors = np.cos(thetas)   # slower on slopes
    step_times    = step_d_all / np.clip(SPEED_FLAT_MPS * speed_factors, 0.001, None)
    total_time_hr = float(step_times.sum()) / 3600.0

    # Power balance
    psr_pixels = sum(1 for r, c in path
                     if illum_fraction is not None and illum_fraction[r, c] < 0.01)
    psr_frac   = psr_pixels / len(path)

    return dict(
        total_distance_m   = total_dist,
        total_cost         = total_cost,
        max_slope_deg      = max_slope_deg,
        elevation_change_m = elev_range,
        n_waypoints        = len(path),
        total_energy_J     = total_E_J,
        total_energy_kJ    = total_E_J / 1000,
        estimated_time_hr  = total_time_hr,
        psr_fraction       = psr_frac,
        elevations_m       = elevations,
        waypoints          = [(int(r), int(c)) for r, c in path],
    )


# ---------------------------------------------------------------------------
# DSC rim target selection
# ---------------------------------------------------------------------------

def _nearest_rim_point(cost_map, dsc_center, dsc_radius, start=None):
    """
    Find the passable DSC rim point closest to start (same quadrant),
    to avoid the A* having to cross the deep crater bowl.
    """
    gs = cost_map.shape[0]
    cr, cc = int(dsc_center[0]), int(dsc_center[1])

    candidates = []
    for angle in np.linspace(0, 2 * np.pi, 360, endpoint=False):
        r = cr + int(dsc_radius * 1.1 * np.sin(angle))
        c = cc + int(dsc_radius * 1.1 * np.cos(angle))
        if 0 <= r < gs and 0 <= c < gs and cost_map[r, c] < IMPASSABLE:
            candidates.append((r, c))

    if not candidates:
        return (max(0, cr - int(dsc_radius * 1.2)), cc)

    if start is not None:
        sr, sc = start
        return min(candidates, key=lambda p: (p[0]-sr)**2 + (p[1]-sc)**2)

    return min(candidates, key=lambda p: cost_map[p[0], p[1]])


# ---------------------------------------------------------------------------
# High-level planner
# ---------------------------------------------------------------------------

def plan_traverse(cost_map, dem, slope, landing_site, dsc_center,
                   dsc_radius=20, roughness=None, illum_fraction=None,
                   pixel_scale=10.0):
    """
    Plan traverse from landing site to DSC rim.

    Strategy: rover stops at DSC rim (does NOT descend into PSR on first pass).
    Subsequent sorties can explore deeper under separate planning.
    """
    start  = (int(landing_site["row"]), int(landing_site["col"]))
    target = _nearest_rim_point(cost_map, dsc_center, dsc_radius, start=start)

    print(f"[Rover] Planning A* from {start} -> DSC rim {target} ...")
    path, total_cost = astar(cost_map, start, target)

    if not path:
        print("[Rover] No path on standard cost map – relaxing slope constraint...")
        relaxed = cost_map.copy()
        relaxed[relaxed >= IMPASSABLE] = 10.0
        path, total_cost = astar(relaxed, start, target)
        if not path:
            print("[Rover] WARNING: No path found.")
            return dict(path=[], simplified=[], metrics={},
                        start=start, target=target)

    simplified = rdp_simplify(path, epsilon=2.0)
    metrics    = path_metrics(path, cost_map, dem, slope, roughness,
                               illum_fraction, pixel_scale)

    print(f"[Rover] Path found: {metrics['total_distance_m']:.0f} m, "
          f"{len(simplified)} waypoints, "
          f"energy: {metrics['total_energy_kJ']:.2f} kJ, "
          f"time: {metrics['estimated_time_hr']:.2f} hr, "
          f"PSR fraction: {metrics['psr_fraction']*100:.1f}%")

    return dict(path=path, simplified=simplified, metrics=metrics,
                start=start, target=target)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def traverse_summary(traverse):
    m = traverse.get("metrics", {})
    s = traverse.get("start") or ("N/A", "N/A")
    t = traverse.get("target") or ("N/A", "N/A")
    d   = m.get('total_distance_m', 0)
    e_kj = m.get('total_energy_kJ', 0)

    # Representative values for annotating the energy formula
    theta_rep = 5.0   # representative mean slope in degrees
    e_slope_kj = ROVER_MASS_KG * LUNAR_G * np.sin(np.radians(theta_rep)) * d / 1e3
    e_roll_kj  = ROVER_MASS_KG * LUNAR_G * ROLLING_RESISTANCE * np.cos(
                     np.radians(theta_rep)) * d / 1e3

    lines = [
        "=" * 60,
        "ROVER TRAVERSE SUMMARY  (LUPEX-class rover, 27 kg)",
        "=" * 60,
        f"  Start                  : {s}",
        f"  Target (DSC rim)       : {t}",
        f"  Total distance         : {d:.0f} m",
        f"  Elevation change       : {m.get('elevation_change_m', 0):.0f} m",
        f"  Max slope on path      : {m.get('max_slope_deg', 0):.1f} deg",
        f"  Estimated traverse time: {m.get('estimated_time_hr', 0):.2f} hr",
        f"  Total energy needed    : {e_kj:.2f} kJ",
        "",
        "  Energy derivation (Arvidson et al. 2011 rover dynamics):",
        "    E_per_step = m * g * (sin(theta) + mu*cos(theta)) * d_step",
        f"    m={ROVER_MASS_KG:.0f} kg,  g={LUNAR_G:.2f} m/s2 (lunar),  mu={ROLLING_RESISTANCE:.2f}",
        f"    At representative mean slope ~{theta_rep:.0f} deg over {d:.0f} m:",
        f"      E_slope = {ROVER_MASS_KG:.0f}*{LUNAR_G:.2f}*sin({theta_rep:.0f}deg)*{d:.0f}m"
          f" = {e_slope_kj:.1f} kJ",
        f"      E_roll  = {ROVER_MASS_KG:.0f}*{LUNAR_G:.2f}*{ROLLING_RESISTANCE:.2f}"
          f"*cos({theta_rep:.0f}deg)*{d:.0f}m = {e_roll_kj:.1f} kJ",
        f"      Total ~{e_kj:.1f} kJ (integrated step-by-step over actual path)",
        "",
        f"  Solar power available  : {SOLAR_PANEL_W:.1f} W",
        f"  Power draw (moving)    : {POWER_DRAW_W:.1f} W",
        f"  PSR fraction on path   : {m.get('psr_fraction', 0)*100:.1f}%",
        f"  Path nodes             : {m.get('n_waypoints', 0)}",
        f"  Simplified waypoints   : {len(traverse.get('simplified', []))}",
        "=" * 60,
    ]
    return "\n".join(lines)
