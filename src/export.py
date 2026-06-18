"""
GeoTIFF and KML export for pipeline results.

Uses rasterio (already in requirements.txt) to write georeferenced TIFFs
from the ice mask, CPR map, confidence map, and other key outputs.
All products are projected in geographic coordinates (EPSG:4326).

Pixel coordinate convention:
  Upper-left = (center_lon - half_extent, center_lat + half_extent)
  Lower-right = (center_lon + half_extent, center_lat - half_extent)
  Pixel size in degrees computed from pixel_scale_m / 111320 m/deg.
"""

import os
import numpy as np
from pathlib import Path

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# Metres per degree of latitude (approximate, valid near poles)
M_PER_DEG = 111_320.0


def _check_rasterio():
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required for GeoTIFF export. "
                          "Run: pip install rasterio")


def export_geotiff(array, filepath, center_lat, center_lon,
                   pixel_scale_m=10.0, nodata=None, band_names=None):
    """
    Write a 2-D or 3-D numpy array to a georeferenced GeoTIFF.

    Parameters
    ----------
    array        : ndarray  shape (rows, cols) or (bands, rows, cols)
    filepath     : str / Path
    center_lat   : float  scene centre latitude (degrees)
    center_lon   : float  scene centre longitude (degrees)
    pixel_scale_m: float  metres per pixel
    nodata       : scalar  no-data fill value (optional)
    band_names   : list of str  (written as GDAL metadata, optional)
    """
    _check_rasterio()
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if array.ndim == 2:
        arr = array[np.newaxis, :, :]   # (1, rows, cols)
    elif array.ndim == 3:
        arr = array
    else:
        raise ValueError(f"array must be 2D or 3D, got shape {array.shape}")

    n_bands, rows, cols = arr.shape
    half_lat = rows * pixel_scale_m / 2.0 / M_PER_DEG
    half_lon = cols * pixel_scale_m / 2.0 / M_PER_DEG

    west  = center_lon - half_lon
    east  = center_lon + half_lon
    south = center_lat - half_lat
    north = center_lat + half_lat

    transform = from_bounds(west, south, east, north, cols, rows)
    crs = CRS.from_epsg(4326)

    dtype = arr.dtype
    if np.issubdtype(dtype, np.bool_):
        arr   = arr.astype(np.uint8)
        dtype = np.uint8

    profile = dict(
        driver    = "GTiff",
        dtype     = dtype,
        width     = cols,
        height    = rows,
        count     = n_bands,
        crs       = crs,
        transform = transform,
        compress  = "lzw",
    )
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(filepath, "w", **profile) as dst:
        for i in range(n_bands):
            dst.write(arr[i], i + 1)
            if band_names and i < len(band_names):
                dst.update_tags(i + 1, name=band_names[i])

    return str(filepath)


def export_all(results_dir, center_lat, center_lon, pixel_scale_m,
               dem=None, ice_mask=None, CPR=None, DOP=None,
               ice_conf=None, composite_score=None, hazard=None,
               T_surface=None, mchi=None):
    """
    Export all key pipeline outputs as georeferenced GeoTIFFs.

    Outputs written to results_dir/geotiff/:
      ice_mask.tif         — binary ice detection (uint8)
      cpr.tif              — Circular Polarisation Ratio (float32)
      dop.tif              — Degree of Polarisation (float32)
      ice_confidence.tif   — Bayesian posterior P(ice) (float32)
      landing_score.tif    — MCDA composite landing score (float32)
      hazard.tif           — slope hazard classification 0-3 (uint8)
      dem.tif              — digital elevation model (float32)
      temperature.tif      — surface temperature in K (float32)
      mchi_volume.tif      — m-chi volume scatter fraction (float32)

    Returns list of written file paths.
    """
    out_dir = Path(results_dir) / "geotiff"
    out_dir.mkdir(parents=True, exist_ok=True)

    kw = dict(center_lat=center_lat, center_lon=center_lon,
              pixel_scale_m=pixel_scale_m)
    written = []

    exports = [
        (ice_mask,        "ice_mask.tif",        None,  ["ice_detection"]),
        (CPR,             "cpr.tif",              None,  ["CPR"]),
        (DOP,             "dop.tif",              None,  ["DOP"]),
        (ice_conf,        "ice_confidence.tif",   None,  ["P_ice"]),
        (composite_score, "landing_score.tif",    None,  ["MCDA_score"]),
        (hazard,          "hazard.tif",            255,  ["hazard_class"]),
        (dem,             "dem.tif",              None,  ["elevation_m"]),
        (T_surface,       "temperature.tif",      None,  ["T_surface_K"]),
    ]

    if mchi is not None:
        exports.append((mchi["frac_volume"], "mchi_volume.tif", None,
                        ["volume_scatter_fraction"]))
        exports.append((mchi["frac_single"], "mchi_single.tif", None,
                        ["single_bounce_fraction"]))
        exports.append((mchi["frac_double"], "mchi_double.tif", None,
                        ["double_bounce_fraction"]))

    for arr, fname, nodata, bands in exports:
        if arr is None:
            continue
        path = export_geotiff(arr, out_dir / fname, nodata=nodata,
                              band_names=bands, **kw)
        written.append(path)
        print(f"  [Export] {fname}")

    print(f"  [Export] {len(written)} GeoTIFFs written to {out_dir}/")
    return written


def export_summary(written_paths):
    lines = [
        "=" * 60,
        "GEOTIFF EXPORT SUMMARY",
        "=" * 60,
        f"  Files written: {len(written_paths)}",
        "  All products georeferenced EPSG:4326 (geographic)",
        "  Open in QGIS / ArcGIS / SNAP for further analysis.",
        "",
    ]
    for p in written_paths:
        lines.append(f"  {Path(p).name}")
    lines.append("=" * 60)
    return "\n".join(lines)
