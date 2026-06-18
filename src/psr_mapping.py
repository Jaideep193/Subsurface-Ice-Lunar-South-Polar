"""
PSR Mapping and Doubly Shadowed Crater Identification.

Methods
-------
- Illumination-fraction thresholding to define PSR boundaries
- Connected-component labelling to isolate individual PSR patches
- Geometric tests to detect sub-craters fully enclosed in a PSR (doubly shadowed)
- Crater morphology metrics (depth-to-diameter, rim-height, lobate rim score)
"""

import numpy as np
from scipy.ndimage import label, binary_fill_holes, binary_dilation, distance_transform_edt
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


# ---------------------------------------------------------------------------
# PSR identification
# ---------------------------------------------------------------------------

def identify_psrs(illum_fraction, threshold=0.01, min_area_pixels=50):
    """
    Label individual PSR patches from an illumination-fraction map.

    Parameters
    ----------
    illum_fraction : 2-D ndarray   [0, 1] fraction of time each pixel is lit
    threshold      : float         pixels below this are considered PSR
    min_area_pixels: int           discard PSR patches smaller than this

    Returns
    -------
    psr_mask     : bool ndarray  (True = PSR)
    psr_labels   : int  ndarray  (0 = non-PSR, >0 = individual PSR label)
    psr_stats    : list of dicts  per-PSR stats
    """
    raw_mask = illum_fraction < threshold
    raw_mask = binary_fill_holes(raw_mask)

    labeled, n_labels = label(raw_mask)

    psr_mask   = np.zeros_like(raw_mask, dtype=bool)
    psr_labels = np.zeros_like(labeled)
    psr_stats  = []
    valid_id   = 1

    for lbl in range(1, n_labels + 1):
        patch = labeled == lbl
        area  = patch.sum()
        if area < min_area_pixels:
            continue
        rows, cols = np.where(patch)
        psr_mask  |= patch
        psr_labels[patch] = valid_id
        psr_stats.append(dict(
            id=valid_id,
            area_pixels=int(area),
            area_km2=float(area * 1e-4),      # 10 m pixel → 100 m² → 1e-4 km²
            centroid=(float(rows.mean()), float(cols.mean())),
            bbox=(int(rows.min()), int(cols.min()),
                  int(rows.max()), int(cols.max())),
        ))
        valid_id += 1

    return psr_mask, psr_labels, psr_stats


# ---------------------------------------------------------------------------
# Doubly shadowed crater detection
# ---------------------------------------------------------------------------

def identify_doubly_shadowed_craters(dem, psr_mask, slope,
                                      min_depth_m=50,
                                      min_diameter_pixels=20,
                                      max_slope_rim_deg=45):
    """
    Detect sub-craters that are:
      1. Fully within a PSR
      2. Have clear topographic bowl signatures (local depth minima)
      3. Have steep rims consistent with impact craters

    Uses watershed segmentation on the inverted DEM restricted to PSR.

    Returns
    -------
    dsc_labels  : int ndarray  (0 = not a DSC, >0 = individual DSC label)
    dsc_stats   : list of dicts
    """
    psr_dem = np.where(psr_mask, dem, np.nan)

    # Fill NaNs with max value so watershed only operates inside PSR
    psr_dem_filled = np.where(psr_mask, dem, dem.max())
    smoothed = gaussian_filter(psr_dem_filled, sigma=3)

    # Local minima = candidate crater centres
    inverted = -smoothed
    inverted_psr = np.where(psr_mask, inverted, inverted.min() - 1)

    # Find local maxima of inverted DEM (= local minima of surface)
    coords = peak_local_max(
        inverted_psr,
        min_distance=min_diameter_pixels // 2,
        threshold_rel=0.05,
    )

    markers = np.zeros_like(dem, dtype=int)
    for idx, (r, c) in enumerate(coords, start=1):
        if psr_mask[r, c]:
            markers[r, c] = idx

    # Watershed
    ws_labels = watershed(smoothed, markers, mask=psr_mask)

    dsc_labels = np.zeros_like(dem, dtype=int)
    dsc_stats  = []
    valid_id   = 1

    for lbl in np.unique(ws_labels):
        if lbl == 0:
            continue
        patch = ws_labels == lbl
        if not psr_mask[patch].all():
            continue   # not entirely within PSR

        rows, cols = np.where(patch)
        area = patch.sum()
        if area < np.pi * (min_diameter_pixels / 2)**2:
            continue

        # Depth = rim elevation - floor elevation
        crater_dem = dem[patch]
        floor_z    = np.percentile(crater_dem, 5)
        rim_z      = np.percentile(crater_dem, 95)
        depth      = rim_z - floor_z

        if depth < min_depth_m:
            continue

        diameter_pixels = 2 * np.sqrt(area / np.pi)
        d_to_d_ratio    = depth / (diameter_pixels * 10)   # depth / diameter (metres)

        # Rim slope check
        rim_mask  = binary_dilation(patch, iterations=3) & ~patch
        rim_slope = slope[rim_mask].mean() if rim_mask.sum() > 0 else 0

        if rim_slope > max_slope_rim_deg:
            dsc_labels[patch] = valid_id
            dsc_stats.append(dict(
                id=valid_id,
                centroid=(float(rows.mean()), float(cols.mean())),
                area_pixels=int(area),
                area_m2=float(area * 100),
                diameter_m=float(diameter_pixels * 10),
                depth_m=float(depth),
                depth_to_diameter=float(d_to_d_ratio),
                rim_slope_deg=float(rim_slope),
                floor_elevation_m=float(floor_z),
                rim_elevation_m=float(rim_z),
            ))
            valid_id += 1

    return dsc_labels, dsc_stats


# ---------------------------------------------------------------------------
# Lobate-rim morphology scorer
# ---------------------------------------------------------------------------

def lobate_rim_score(dem, crater_center, crater_radius, pixel_scale=10.0,
                     n_sectors=36):
    """
    Quantify lobate rim morphology by computing azimuthal rim-height variance.
    Higher variance → more lobate.

    Returns
    -------
    lrs : float  [0, 1] normalised lobate rim score
    """
    cr, cc = int(crater_center[0]), int(crater_center[1])
    gs     = dem.shape[0]

    rim_r_inner = int(crater_radius * 0.9)
    rim_r_outer = int(crater_radius * 1.3)

    sector_heights = []
    angles = np.linspace(0, 2 * np.pi, n_sectors, endpoint=False)

    for angle in angles:
        heights = []
        for r in range(rim_r_inner, rim_r_outer + 1):
            row = cr + int(r * np.sin(angle))
            col = cc + int(r * np.cos(angle))
            if 0 <= row < gs and 0 <= col < gs:
                heights.append(dem[row, col])
        sector_heights.append(np.mean(heights) if heights else 0.0)

    sector_heights = np.array(sector_heights)
    mean_h = sector_heights.mean()
    std_h  = sector_heights.std()
    lrs    = std_h / (abs(mean_h) + 1e-6)
    return float(np.clip(lrs, 0, 1))


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def psr_summary(psr_stats, dsc_stats):
    lines = [
        "=" * 60,
        "PSR MAPPING SUMMARY",
        "=" * 60,
        f"  Total PSR patches identified : {len(psr_stats)}",
        f"  Total PSR area               : {sum(s['area_km2'] for s in psr_stats):.2f} km2",
        "",
        "  PSR Detection Method:",
        "    24-azimuth ray-casting: each pixel marked PSR when all 24",
        "    solar rays (south-polar sun elevation 2.8 deg) are blocked",
        "    by higher terrain. Illumination fraction = (unblocked rays)/24.",
        "    PSR threshold: illum_frac < 0.01 (< 1 ray in 24 unblocked).",
        "",
        f"  Doubly Shadowed Craters (DSC): {len(dsc_stats)}",
        "  DSC Detection Pipeline:",
        "    1. Restrict DEM to PSR interior (NaN outside PSR)",
        "    2. Gaussian-smooth DEM (sigma=3 px) to suppress roughness noise",
        "    3. Invert smoothed DEM; find local maxima (= topographic minima)",
        "       via skimage.peak_local_max with min_distance=10 px",
        "    4. Watershed segmentation on smoothed DEM to delineate basins",
        "    5. Filter criteria applied to each basin:",
        "       - Basin fully within PSR (no pixel outside PSR boundary)",
        "       - Area > pi*(min_diam/2)^2  (diameter >= 200 m)",
        "       - Depth (95th-5th pct elevation) > 50 m",
        "       - Mean rim slope > max_slope_rim threshold",
        "    6. Morphology computed: D/d ratio, rim slope, lobate rim score",
    ]
    for ds in dsc_stats:
        lines += [
            f"    DSC #{ds['id']}:",
            f"      Centroid        : ({ds['centroid'][0]:.0f}, {ds['centroid'][1]:.0f}) px",
            f"      Diameter        : {ds['diameter_m']:.0f} m",
            f"      Depth           : {ds['depth_m']:.0f} m",
            f"      Depth/Diameter  : {ds['depth_to_diameter']:.3f}",
            f"      Rim slope       : {ds['rim_slope_deg']:.1f} deg",
        ]
    lines.append("=" * 60)
    return "\n".join(lines)
