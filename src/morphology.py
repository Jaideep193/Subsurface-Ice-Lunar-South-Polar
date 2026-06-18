"""
Crater Morphology Characterization (OHRC + DEM)
------------------------------------------------
* Surface roughness (RMS height, correlation length)
* Boulder detection and density mapping
* Slope hazard map
* Lobate-rim morphology scoring
* Depth-to-diameter ratio analysis
"""

import numpy as np
from scipy.ndimage import (gaussian_filter, label, binary_dilation,
                            uniform_filter, maximum_filter, minimum_filter)
from scipy.signal import correlate2d


# ---------------------------------------------------------------------------
# Surface roughness
# ---------------------------------------------------------------------------

def rms_roughness(dem, window=21):
    """
    RMS height roughness over a sliding window.
    σ_h = sqrt( <h²> - <h>² )
    """
    mean  = uniform_filter(dem.astype(np.float64), size=window)
    mean2 = uniform_filter(dem.astype(np.float64)**2, size=window)
    var   = np.clip(mean2 - mean**2, 0, None)
    return np.sqrt(var).astype(np.float32)


def autocorrelation_length(dem, pixel_scale=10.0):
    """
    Estimate spatial autocorrelation length (correlation length l_c)
    where the autocorrelation function first drops to 1/e.
    Returns a single float in metres.
    """
    profile = dem[dem.shape[0] // 2, :].astype(np.float64)
    profile -= profile.mean()
    acf = np.correlate(profile, profile, mode="full")
    acf = acf[acf.size // 2:]
    acf /= acf[0] if acf[0] != 0 else 1.0
    idx = np.where(acf < 1 / np.e)[0]
    l_c = idx[0] * pixel_scale if idx.size > 0 else len(acf) * pixel_scale
    return float(l_c)


# ---------------------------------------------------------------------------
# Slope and aspect
# ---------------------------------------------------------------------------

def compute_slope_aspect(dem, pixel_scale=10.0):
    """Return slope (degrees) and aspect (degrees from North, 0–360)."""
    dy, dx = np.gradient(dem.astype(np.float64), pixel_scale)
    slope  = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    aspect = np.degrees(np.arctan2(-dx, dy)) % 360
    return slope.astype(np.float32), aspect.astype(np.float32)


def slope_hazard_map(slope, thresholds=(10, 15, 20), smooth_sigma=3.0):
    """
    Categorise slope into hazard levels for rover safety.
    Returns integer map: 0=safe, 1=caution, 2=danger, 3=impassable.

    Thresholds (JAXA SELENE / NASA Artermis mobility standards):
      0 Safe       : slope <= 10 deg
      1 Caution    : 10 < slope <= 15 deg
      2 Danger     : 15 < slope <= 20 deg
      3 Impassable : slope > 20 deg

    Applies Gaussian smoothing (sigma=3 px = 30 m) BEFORE classification
    so the output consists of coherent terrain units, not per-pixel noise.
    """
    slope_s = gaussian_filter(slope.astype(np.float64), sigma=smooth_sigma)
    hazard = np.zeros_like(slope, dtype=np.uint8)
    hazard[slope_s > thresholds[0]] = 1
    hazard[slope_s > thresholds[1]] = 2
    hazard[slope_s > thresholds[2]] = 3
    return hazard


# ---------------------------------------------------------------------------
# Boulder detection
# ---------------------------------------------------------------------------

def detect_boulders(ohrc, min_brightness=0.7, min_size=2, max_size=50):
    """
    Detect boulders as bright compact features in OHRC image.
    Uses local contrast enhancement + morphological filtering.

    Returns
    -------
    boulder_mask   : bool ndarray
    boulder_labels : int  ndarray
    boulder_stats  : list of dicts (area, centroid, brightness)
    """
    # Normalise
    img = ohrc.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-9)

    # Local contrast: pixel significantly brighter than neighbourhood
    local_mean = uniform_filter(img, size=11)
    contrast   = img - local_mean

    # Threshold on bright contrast
    thresh = contrast > (contrast.std() * 2.5)

    lbl, n = label(thresh)
    boulder_mask   = np.zeros_like(thresh)
    boulder_labels = np.zeros_like(thresh, dtype=int)
    boulder_stats  = []
    valid_id = 1

    for i in range(1, n + 1):
        patch = lbl == i
        area  = patch.sum()
        if not (min_size <= area <= max_size):
            continue
        rows, cols  = np.where(patch)
        mean_bright = float(img[patch].mean())
        boulder_mask  |= patch
        boulder_labels[patch] = valid_id
        boulder_stats.append(dict(
            id=valid_id,
            area_pixels=int(area),
            centroid=(float(rows.mean()), float(cols.mean())),
            brightness=mean_bright,
        ))
        valid_id += 1

    return boulder_mask, boulder_labels, boulder_stats


def boulder_density_map(boulder_labels, window=51):
    """
    Gaussian-KDE boulder density (boulders / km²).
    Replaces box-filter uniform_filter to eliminate grid / square-block artefacts.
    sigma = window/5 gives roughly equivalent spatial support with smooth falloff.
    """
    binary  = (boulder_labels > 0).astype(np.float32)
    sigma   = window / 5.0          # ~10 px at default window=51
    density = gaussian_filter(binary, sigma=sigma)
    area_km2 = (window * 10) ** 2 / 1e6
    return (density / area_km2).astype(np.float32)


# ---------------------------------------------------------------------------
# Crater rim morphology
# ---------------------------------------------------------------------------

def lobate_rim_analysis(dem, center, radius, n_sectors=36, pixel_scale=10.0):
    """
    Azimuthal rim-height profile + lobate-rim score.

    Returns
    -------
    dict with sector_angles, sector_heights, lrs (lobate rim score),
         height_variability_m, is_lobate (bool)
    """
    cr, cc = int(center[0]), int(center[1])
    gs     = dem.shape[0]
    angles = np.linspace(0, 2 * np.pi, n_sectors, endpoint=False)

    # Sample rim height over a radial band [0.9R, 1.3R]
    r_inner = max(1, int(radius * 0.9))
    r_outer = int(radius * 1.3)

    sector_heights = []
    for angle in angles:
        heights = []
        for r in range(r_inner, r_outer + 1):
            row = cr + int(r * np.sin(angle))
            col = cc + int(r * np.cos(angle))
            if 0 <= row < gs and 0 <= col < gs:
                heights.append(float(dem[row, col]))
        sector_heights.append(np.mean(heights) if heights else 0.0)

    sector_heights = np.array(sector_heights)
    mean_h = sector_heights.mean()
    std_h  = sector_heights.std()
    lrs    = float(std_h / (abs(mean_h) + 1e-6))

    # Lobate if std > 10% of mean rim height relative to floor
    floor_z = float(dem[cr, cc])
    rim_height_above_floor = mean_h - floor_z
    is_lobate = std_h > 0.15 * abs(rim_height_above_floor)

    return dict(
        sector_angles        = np.degrees(angles).tolist(),
        sector_heights_m     = sector_heights.tolist(),
        lobate_rim_score     = lrs,
        rim_mean_elev_m      = float(mean_h),
        rim_std_m            = float(std_h),
        floor_elev_m         = floor_z,
        rim_height_m         = float(rim_height_above_floor),
        is_lobate            = bool(is_lobate),
    )


def depth_diameter_analysis(dem, center, radius, pixel_scale=10.0):
    """
    Compute depth-to-diameter ratio and compare to lunar crater scaling.

    Lunar fresh craters: d/D ~ 0.2 (simple), 0.1–0.15 (complex)
    Ice-modified craters may show anomalously shallow floors: d/D < 0.1
    """
    cr, cc = int(center[0]), int(center[1])
    gs     = dem.shape[0]

    xx, yy = np.meshgrid(np.arange(gs), np.arange(gs))
    r_map  = np.sqrt((xx - cc)**2 + (yy - cr)**2)

    interior = r_map < radius
    exterior_rim = (r_map >= radius) & (r_map < radius * 1.3)

    if interior.sum() == 0 or exterior_rim.sum() == 0:
        return {}

    floor_z = float(np.percentile(dem[interior], 5))
    rim_z   = float(np.percentile(dem[exterior_rim], 95))
    depth   = rim_z - floor_z
    diameter_m = 2 * radius * pixel_scale

    d_to_D = depth / diameter_m

    # Classification
    if d_to_D >= 0.15:
        crater_type = "fresh_simple"
    elif d_to_D >= 0.08:
        crater_type = "modified_complex"
    else:
        crater_type = "ice_modified_or_degraded"

    return dict(
        diameter_m      = float(diameter_m),
        depth_m         = float(depth),
        depth_to_diam   = float(d_to_D),
        crater_type     = crater_type,
        floor_elev_m    = floor_z,
        rim_elev_m      = rim_z,
    )


# ---------------------------------------------------------------------------
# Full morphology report
# ---------------------------------------------------------------------------

def run_morphology(dem, ohrc, dsc_center, dsc_radius, pixel_scale=10.0):
    """
    Run complete morphology characterisation for a doubly shadowed crater.
    """
    print("[Morphology] Computing slope & aspect...")
    slope, aspect = compute_slope_aspect(dem, pixel_scale)

    print("[Morphology] Computing surface roughness...")
    roughness = rms_roughness(dem)
    l_c       = autocorrelation_length(dem, pixel_scale)

    print("[Morphology] Detecting boulders...")
    if ohrc is not None:
        b_mask, b_labels, b_stats = detect_boulders(ohrc)
        b_density = boulder_density_map(b_labels)
    else:
        b_mask = b_labels = b_density = None
        b_stats = []

    print("[Morphology] Analysing crater rim morphology...")
    rim_data = lobate_rim_analysis(dem, dsc_center, dsc_radius, pixel_scale=pixel_scale)
    dd_data  = depth_diameter_analysis(dem, dsc_center, dsc_radius, pixel_scale)

    hazard   = slope_hazard_map(slope)

    return dict(
        slope         = slope,
        aspect        = aspect,
        hazard        = hazard,
        roughness     = roughness,
        corr_length_m = l_c,
        boulder_mask  = b_mask,
        boulder_density = b_density,
        boulder_stats = b_stats,
        rim_analysis  = rim_data,
        depth_diameter= dd_data,
    )


def morphology_summary(morph):
    dd  = morph.get("depth_diameter", {})
    rim = morph.get("rim_analysis", {})
    lines = [
        "=" * 60,
        "CRATER MORPHOLOGY SUMMARY",
        "=" * 60,
        f"  Diameter          : {dd.get('diameter_m', 0):.0f} m",
        f"  Depth             : {dd.get('depth_m', 0):.0f} m",
        f"  Depth/Diameter    : {dd.get('depth_to_diam', 0):.3f}",
        f"  Crater type       : {dd.get('crater_type', 'unknown')}",
        f"  Lobate-rim score  : {rim.get('lobate_rim_score', 0):.3f}",
        f"  Rim height        : {rim.get('rim_height_m', 0):.0f} m",
        f"  Is lobate         : {rim.get('is_lobate', False)}",
        f"  Boulders detected : {len(morph.get('boulder_stats', []))}",
        f"  Corr. length      : {morph.get('corr_length_m', 0):.0f} m",
        "=" * 60,
    ]
    return "\n".join(lines)
